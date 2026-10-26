"""Real-time lip-reading server.

    python -m lipvision.server [--port 8000]

The browser tracks the face locally (MediaPipe JS), sends a small JPEG crop of
the lower face plus 4 landmark points per frame over a WebSocket, and receives
live partial and final transcripts. The GPU does only the heavy part.

Wire protocol (client -> server)
  binary: uint32 little-endian header length | JSON header | JPEG bytes (optional)
          header = {"t": ms, "pts": [[x, y] x4] | null, "open": float | null}
  text:   {"type": "mode", "mode": "auto"|"manual"} | {"type": "start"|"stop"|"reset"}
(server -> client) JSON: ready | frame (ack) | partial | final | discard | error
"""

import argparse
import asyncio
import base64
import json
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, assets
from .config import Settings, client_config
from .stream import DecodeJob, LiveSession

log = logging.getLogger("lipvision.server")
WEB_DIR = Path(__file__).parent / "web"


def load_mean_face(settings: Settings) -> np.ndarray:
    root = assets.auto_avsr_dir(settings)
    return np.load(root / "preparation" / "detectors" / "mediapipe" / "20words_mean_face.npy")


def parse_frame(data: bytes):
    (n,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + n])
    gray = None
    if len(data) > 4 + n:
        buf = np.frombuffer(data, np.uint8, offset=4 + n)
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    pts = header.get("pts")
    pts = np.asarray(pts, np.float32).reshape(4, 2) if pts is not None and gray is not None else None
    return header.get("t", 0.0) / 1000.0, gray, pts, header.get("open")


class Engine:
    """Owns the model and a single inference thread shared by all sessions."""

    def __init__(self, settings: Settings, model=None, mean_face=None):
        self.settings = settings
        self.model = model
        self.mean_face = mean_face
        self.error = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vsr")

    def load(self):
        try:
            if self.mean_face is None:
                self.mean_face = load_mean_face(self.settings)
            if self.model is None:
                from .model import VSRModel
                self.model = VSRModel(self.settings)
                self.model.warmup()
        except Exception as e:  # keep serving the page so the user sees the reason
            log.exception("model load failed")
            self.error = str(e)

    async def transcribe(self, job: DecodeJob):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, self.model.transcribe, job.patches, job.final)

    def info(self) -> dict:
        m = self.model
        return {
            "version": __version__,
            "ready": m is not None and self.error is None,
            "error": self.error,
            "device": str(getattr(m, "device", "n/a")),
            "fp16": getattr(m, "fp16", False),
            "checkpoint": Path(self.settings.checkpoint).name,
            "beam": self.settings.beam_size,
        }


class Connection:
    """One browser tab. Final jobs always run; partials keep only the newest."""

    def __init__(self, ws: WebSocket, engine: Engine):
        self.ws, self.engine = ws, engine
        self.session = LiveSession(engine.settings, engine.mean_face)
        self.finals: asyncio.Queue = asyncio.Queue()
        self.partial: DecodeJob | None = None
        self.wake = asyncio.Event()
        self.done_utts: set = set()
        self.frames = 0

    async def send(self, msg: dict):
        await self.ws.send_text(json.dumps(msg))

    def submit(self, jobs):
        for job in jobs:
            if job.final:
                self.finals.put_nowait(job)
                self.partial = None
            else:
                self.partial = job
            self.wake.set()

    async def decoder_loop(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            while not self.finals.empty() or self.partial is not None:
                if not self.finals.empty():
                    job = self.finals.get_nowait()
                    self.done_utts.add(job.utterance)
                    if len(job.patches) == 0:
                        await self.send({"type": "discard", "utt": job.utterance})
                        continue
                else:
                    job, self.partial = self.partial, None
                    if job.utterance in self.done_utts:
                        continue
                try:
                    res = await self.engine.transcribe(job)
                except Exception as e:  # e.g. CUDA OOM: report it, keep the session alive
                    log.exception("decode failed")
                    if job.final:
                        await self.send({"type": "error", "message": f"decode failed: {e}", "utt": job.utterance})
                    continue
                if not job.final and job.utterance in self.done_utts:
                    continue  # a final for this utterance overtook us
                await self.send({
                    "type": "final" if job.final else "partial", "utt": job.utterance,
                    "text": res.text, "ms": round(res.latency_ms), "dur": round(job.duration, 2),
                })

    async def handle_frame(self, data: bytes):
        t, gray, pts, openness = parse_frame(data)
        r = self.session.push(t, gray, pts, openness)
        self.submit(r.jobs)
        self.frames += 1
        msg = {"type": "frame", "t": t * 1000, "speaking": r.speaking,
               "energy": round(r.energy, 4), "thr": round(self.session.vad.on_threshold, 4)}
        if r.patch is not None and self.frames % 3 == 0:  # "what the model sees" preview
            ok, jpg = cv2.imencode(".jpg", r.patch, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                msg["patch"] = base64.b64encode(jpg.tobytes()).decode()
        await self.send(msg)

    async def handle_control(self, msg: dict):
        now = time.monotonic()
        kind = msg.get("type")
        if kind == "mode":
            self.session.set_mode(msg.get("mode", "auto"))
        elif kind == "start":
            self.session.start(msg.get("t", now * 1000) / 1000)
        elif kind == "stop":
            self.submit(self.session.stop(msg.get("t", now * 1000) / 1000))
        elif kind == "reset":
            self.session.reset()

    async def run(self):
        await self.send({"type": "ready", **self.engine.info()})
        worker = asyncio.create_task(self.decoder_loop())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await self.handle_frame(msg["bytes"])
                elif msg.get("text"):
                    await self.handle_control(json.loads(msg["text"]))
                if worker.done():
                    worker.result()  # surface decoder errors
        except WebSocketDisconnect:
            pass
        finally:
            worker.cancel()


def create_app(settings: Settings | None = None, model=None, mean_face=None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = Engine(settings, model, mean_face)

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.get_running_loop().run_in_executor(None, engine.load)
        log.info("engine: %s", engine.info())
        yield

    app = FastAPI(title="LipVision", version=__version__, lifespan=lifespan)
    app.state.engine = engine

    @app.get("/api/health")
    def health():
        return engine.info()

    @app.get("/api/config")
    def config():
        return JSONResponse(client_config(settings))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        if engine.model is None:
            await ws.send_text(json.dumps({"type": "error", "message": engine.error or "model still loading"}))
            await ws.close()
            return
        await Connection(ws, engine).run()

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def main():
    p = argparse.ArgumentParser(description="LipVision real-time server")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--beam-size", type=int)
    p.add_argument("--device")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    s = Settings.from_env(host=a.host, port=a.port, beam_size=a.beam_size, device=a.device)
    import uvicorn
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info", ws_max_size=4 << 20)


if __name__ == "__main__":
    main()
