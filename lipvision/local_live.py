"""Live lip reading from a local webcam in an OpenCV window (needs a local GPU for real time).

    python -m lipvision.local_live [--camera 0]

On a laptop without a GPU use the Kaggle notebook + browser client instead.
"""

import argparse
import logging
import queue
import threading
import time

import cv2

from .config import MESH_LIP_INNER, MESH_LIP_OUTER, Settings
from .landmarks import FaceTracker
from .model import VSRModel
from .server import load_mean_face
from .stream import LiveSession


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="camera index or video file")
    p.add_argument("--manual", action="store_true", help="hold SPACE while speaking instead of auto-detect")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = Settings.from_env()
    model = VSRModel(s)
    model.warmup()
    tracker = FaceTracker(s)
    session = LiveSession(s, load_mean_face(s), mode="manual" if a.manual else "auto")

    jobs: queue.Queue = queue.Queue()
    shown = {"text": "Ready - mouth a sentence", "partial": False}

    def worker():
        done = set()
        while True:
            job = jobs.get()
            if not job.final and (job.utterance in done or not jobs.empty()):
                continue  # stale partial
            if job.final:
                done.add(job.utterance)
                if len(job.patches) == 0:
                    continue
            r = model.transcribe(job.patches, final=job.final)
            shown.update(text=r.text or "[unclear]", partial=not job.final)
            if job.final:
                print(f"> {r.text}   ({job.duration:.1f}s, {r.latency_ms:.0f} ms)")

    threading.Thread(target=worker, daemon=True).start()

    cap = cv2.VideoCapture(int(a.camera) if a.camera.isdigit() else a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic() - t0
        mesh, pts, openness = tracker(frame, t)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = session.push(t, gray, pts, openness)
        for job in r.jobs:
            jobs.put(job)

        view = cv2.flip(frame, 1)
        w = view.shape[1]
        if mesh is not None:
            color = (182, 114, 244) if r.speaking else (212, 234, 94)
            for ring in (MESH_LIP_OUTER, MESH_LIP_INNER):
                poly = mesh[ring].copy()
                poly[:, 0] = w - 1 - poly[:, 0]
                cv2.polylines(view, [poly.astype("int32")], True, color, 2, cv2.LINE_AA)
        if r.patch is not None:
            view[10:106, w - 106:w - 10] = cv2.cvtColor(r.patch, cv2.COLOR_GRAY2BGR)
        label = "SPEAKING" if r.speaking else ("NO FACE" if pts is None else "idle")
        cv2.putText(view, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (80, 80, 255) if r.speaking else (200, 200, 200), 2, cv2.LINE_AA)
        cv2.rectangle(view, (0, view.shape[0] - 52), (w, view.shape[0]), (0, 0, 0), -1)
        cv2.putText(view, shown["text"], (16, view.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200) if shown["partial"] else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("LipVision (q = quit, SPACE = push-to-talk with --manual)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if a.manual and key == 32:
            if session.manual_on:
                for job in session.stop(t):
                    jobs.put(job)
            else:
                session.start(t)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
