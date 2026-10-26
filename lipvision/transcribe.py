"""Transcribe silent video files.

    python -m lipvision.transcribe clip.mp4 [more.mp4 ...] [--save-crops out.mp4]
"""

import argparse
import logging
import time

import cv2
import numpy as np

from .config import Settings
from .landmarks import FaceTracker
from .model import VSRModel
from .preprocess import align_video, resample_indices
from .server import load_mean_face


def read_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def video_to_patches(path: str, settings: Settings, tracker: FaceTracker) -> np.ndarray:
    frames, fps = read_video(path)
    ts = np.arange(len(frames)) / fps
    idx = resample_indices(ts, settings.fps)  # any source fps -> 25 fps
    frames = [frames[i] for i in idx]
    points, grays = [], []
    for i, f in enumerate(frames):
        _, pts, _ = tracker(f, i / settings.fps)
        points.append(pts)
        grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    patches = align_video(grays, points, load_mean_face(settings), crop_size=settings.crop_size)
    if patches is None:
        raise RuntimeError(f"No face found in {path}")
    return patches


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+")
    p.add_argument("--beam-size", type=int)
    p.add_argument("--device")
    p.add_argument("--save-crops", help="write the aligned mouth crops of the first video here")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = Settings.from_env(beam_size=a.beam_size, device=a.device)
    model = VSRModel(s)
    for path in a.videos:
        tracker = FaceTracker(s)  # fresh tracker per file: VIDEO-mode timestamps restart
        t0 = time.perf_counter()
        patches = video_to_patches(path, s, tracker)
        tracker.close()
        if a.save_crops:
            out = cv2.VideoWriter(a.save_crops, cv2.VideoWriter_fourcc(*"mp4v"), s.fps, patches.shape[1:][::-1], False)
            for fr in patches:
                out.write(fr)
            out.release()
            a.save_crops = None
        res = model.transcribe(patches, final=True)
        print(f"{path}\t{res.text}\t({len(patches) / s.fps:.1f}s video, "
              f"{(time.perf_counter() - t0):.2f}s total, decode {res.latency_ms:.0f} ms)")


if __name__ == "__main__":
    main()
