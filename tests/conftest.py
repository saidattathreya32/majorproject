import cv2
import numpy as np
import pytest

from lipvision.config import Settings
from lipvision.model import Transcript

# Stable points of the Auto-AVSR mean face (right eye, left eye, nose, mouth).
REF = np.array([[102.07, 94.27], [156.36, 93.58], [129.0, 135.9], [129.31, 157.82]], np.float32)


@pytest.fixture
def mean_face():
    """Synthetic 68-point face whose group means equal the real reference points."""
    mf = np.zeros((68, 2), np.float32)
    for sl, p in ((slice(36, 42), REF[0]), (slice(42, 48), REF[1]),
                  (slice(31, 36), REF[2]), (slice(48, 68), REF[3])):
        mf[sl] = p
    return mf


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=tmp_path, device="cpu", partial_interval=0.4)


def face_image(points, size=(480, 640), open_amount=0.0):
    """Gray frame with eyes/nose/mouth blobs at `points`; mouth opens with open_amount (0..1)."""
    img = np.full(size, 90, np.uint8)
    re, le, nose, mouth = points
    iod = np.linalg.norm(le - re)
    for p in (re, le):
        cv2.circle(img, tuple(int(v) for v in p), int(iod * 0.12), 30, -1)
    cv2.circle(img, tuple(int(v) for v in nose), int(iod * 0.08), 60, -1)
    w = int(iod * 0.45)
    h = max(2, int(iod * (0.06 + 0.3 * open_amount)))
    cv2.ellipse(img, tuple(int(v) for v in mouth), (w, h), 0, 0, 360, 200, -1)
    cv2.ellipse(img, tuple(int(v) for v in mouth), (int(w * 0.8), max(1, h - 4)), 0, 0, 360, 10, -1)
    return img


def face_points(center=(320, 240), scale=1.6, angle_deg=0.0):
    c = REF.mean(0)
    th = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    return ((REF - c) * scale) @ rot.T + np.asarray(center, np.float32)


class FakeModel:
    device = "cpu"
    fp16 = False

    def __init__(self):
        self.calls = []

    def transcribe(self, patches, final=True):
        self.calls.append((len(patches), final))
        return Transcript("HELLO WORLD" if final else "HELLO", len(patches), 1.0, final)

    def warmup(self):
        pass


def speech_openness(t, onset, offset):
    """Still lips with tiny jitter, oscillating 4 Hz opening while 'speaking'."""
    jitter = 0.002 * np.sin(t * 37.0)
    if onset <= t < offset:
        return 0.15 + 0.15 * np.sin(2 * np.pi * 4.0 * (t - onset)) + jitter
    return 0.02 + jitter
