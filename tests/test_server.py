import json
import struct

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lipvision.model import Tokenizer
from lipvision.server import create_app, parse_frame

from .conftest import FakeModel, face_image, face_points, speech_openness


def pack(t_ms, gray=None, pts=None, openness=None):
    header = json.dumps({"t": t_ms, "pts": None if pts is None else pts.tolist(), "open": openness}).encode()
    body = b""
    if gray is not None:
        body = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    return struct.pack("<I", len(header)) + header + body


def client_crop(t):
    """Simulate the browser: 192x192 crop around the mouth + points in crop coords."""
    p = face_points()
    o = speech_openness(t, 1.0, 2.5)
    frame = face_image(p, open_amount=o * 3)
    side = 2.6 * np.linalg.norm(p[0] - p[1])
    sx, sy = p[3] - side / 2
    k = 192 / side
    m = np.array([[k, 0, -sx * k], [0, k, -sy * k]], np.float32)
    crop = cv2.warpAffine(frame, m, (192, 192))
    return crop, (p - [sx, sy]) * k, o


def test_parse_frame_roundtrip():
    crop, pts, o = client_crop(0.0)
    t, gray, p, openness = parse_frame(pack(1234.0, crop, pts, o))
    assert t == 1.234 and gray.shape == (192, 192) and np.allclose(p, pts, atol=1e-3)
    t, gray, p, openness = parse_frame(pack(10.0))
    assert gray is None and p is None


def test_websocket_live_session(settings, mean_face):
    model = FakeModel()
    app = create_app(settings, model=model, mean_face=mean_face)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["ready"] is True
        cfg = client.get("/api/config").json()
        assert len(cfg["mesh"]["mouth"]) == 20
        assert "LipVision" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200

        with client.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "ready"
            msgs = []
            for i in range(int(4.5 * 30)):
                t = i / 30
                crop, pts, o = client_crop(t)
                ws.send_bytes(pack(t * 1000, crop, pts, o))
                while True:  # wait for this frame's ack before sending the next (flow control)
                    m = json.loads(ws.receive_text())
                    msgs.append(m)
                    if m["type"] == "frame":
                        break
            while not any(m["type"] == "final" for m in msgs):
                msgs.append(json.loads(ws.receive_text()))
            finals = [m for m in msgs if m["type"] == "final"]
            assert len(finals) == 1
            assert finals[0]["text"] == "HELLO WORLD" and finals[0]["dur"] > 1.0
            assert any(m["type"] == "partial" for m in msgs)
            acks = [m for m in msgs if m["type"] == "frame"]
            assert any(a["speaking"] for a in acks)
            assert any("patch" in a for a in acks)
            assert any(not final for _, final in model.calls)


def test_websocket_reports_model_error(settings, mean_face):
    app = create_app(settings, model=None, mean_face=mean_face)
    app.state.engine.load = lambda: setattr(app.state.engine, "error", "weights missing")
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            m = json.loads(ws.receive_text())
            assert m["type"] == "error" and "weights missing" in m["message"]


def test_tokenizer_decode(tmp_path):
    units = tmp_path / "units.txt"
    units.write_text("<unk> 1\n▁HELLO 2\n▁WOR 3\nLD 4\n")
    tok = Tokenizer(units)
    assert tok.token_list[0] == "<blank>" and tok.token_list[-1] == "<eos>"
    assert tok.decode([2, 3, 4, 5]) == "HELLO WORLD"


def test_decode_error_is_reported_and_session_survives(settings, mean_face):
    class Boom(FakeModel):
        def transcribe(self, patches, final=True):
            if final:
                raise RuntimeError("CUDA out of memory")
            return super().transcribe(patches, final)

    app = create_app(settings, model=Boom(), mean_face=mean_face)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "mode", "mode": "manual"}))
        ws.send_text(json.dumps({"type": "start", "t": 0}))
        for i in range(30):
            crop, pts, o = client_crop(i / 30)
            ws.send_bytes(pack(i / 30 * 1000, crop, pts, o))
        ws.send_text(json.dumps({"type": "stop", "t": 1000}))
        msgs = []
        while not any(m["type"] == "error" for m in msgs):
            msgs.append(json.loads(ws.receive_text()))
        err = next(m for m in msgs if m["type"] == "error")
        assert "out of memory" in err["message"] and err["utt"] == 1
        ws.send_bytes(pack(2000.0))  # still alive: frames are acknowledged
        while json.loads(ws.receive_text())["type"] != "frame":
            pass
