"""Mouth-region alignment matching the Auto-AVSR training pipeline.

Each frame is mapped with a similarity transform that puts the eyes, nose and
mouth onto the Auto-AVSR mean face (256x256 reference), then a 96x96 patch is
cut around the mouth. Doing exactly what the model saw in training is the
single biggest accuracy factor; the old project fed raw dlib crops instead.
"""

from collections import deque

import cv2
import numpy as np

from .config import MESH_LEFT_EYE, MESH_LIP_OPEN, MESH_MOUTH, MESH_NOSE, MESH_RIGHT_EYE


def stable_reference(mean_face: np.ndarray) -> np.ndarray:
    """Right eye, left eye, nose, mouth centres of the 68-point mean face."""
    return np.vstack([
        mean_face[36:42].mean(0), mean_face[42:48].mean(0),
        mean_face[31:36].mean(0), mean_face[48:68].mean(0),
    ]).astype(np.float32)


def mesh_to_points(mesh: np.ndarray) -> np.ndarray:
    """(478, 2) MediaPipe face-mesh pixel coords -> (4, 2) stable points."""
    return np.vstack([
        mesh[MESH_RIGHT_EYE].mean(0), mesh[MESH_LEFT_EYE].mean(0),
        mesh[MESH_NOSE].mean(0), mesh[MESH_MOUTH].mean(0),
    ]).astype(np.float32)


def mouth_openness(mesh: np.ndarray) -> float:
    top, bottom, left, right = (mesh[i] for i in MESH_LIP_OPEN)
    return float(np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6))


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:  # degenerate points; fall back to plain least squares
        m, _ = cv2.estimateAffinePartial2D(src, dst, method=0)
    return m


class MouthAligner:
    """Turns (frame, 4 stable points) into an aligned uint8 96x96 grayscale patch.

    `window` frames of landmarks are averaged to remove jitter. Live mode is
    causal (past frames only); offline mode uses a centred window like training.
    """

    def __init__(self, mean_face: np.ndarray, crop_size: int = 96, window: int = 5):
        self.ref = stable_reference(mean_face)
        self.crop = crop_size
        self.history: deque = deque(maxlen=max(1, window))

    def reset(self):
        self.history.clear()

    def warp(self, gray: np.ndarray, points: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
        # keep the smoothed shape but anchor it at the current position
        smoothed = smoothed + (points.mean(0) - smoothed.mean(0))
        m = _similarity(smoothed, self.ref)
        mouth = m[:, :2] @ smoothed[3] + m[:, 2]
        half = self.crop / 2.0
        m = m.copy()
        m[:, 2] -= np.round(mouth - half)  # crop the patch directly in one warp
        return cv2.warpAffine(gray, m, (self.crop, self.crop), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def __call__(self, gray: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float32)
        self.history.append(points)
        return self.warp(gray, points, np.mean(self.history, axis=0))


def interpolate_missing(points: list) -> list | None:
    """Linear interpolation over frames where no face was found (offline mode)."""
    valid = [i for i, p in enumerate(points) if p is not None]
    if not valid:
        return None
    out = list(points)
    for a, b in zip(valid, valid[1:]):
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            out[k] = out[a] + t * (out[b] - out[a])
    for k in range(valid[0]):
        out[k] = out[valid[0]]
    for k in range(valid[-1] + 1, len(out)):
        out[k] = out[valid[-1]]
    return out


def align_video(gray_frames: list, points: list, mean_face: np.ndarray,
                window: int = 12, crop_size: int = 96) -> np.ndarray | None:
    """Offline alignment with a centred smoothing window (identical to Auto-AVSR)."""
    points = interpolate_missing(points)
    if points is None:
        return None
    aligner = MouthAligner(mean_face, crop_size)
    pts = np.asarray(points, np.float32)
    n, half = len(pts), window // 2
    patches = []
    for i, gray in enumerate(gray_frames):
        m = min(half, i, n - 1 - i)
        patches.append(aligner.warp(gray, pts[i], pts[i - m:i + m + 1].mean(0)))
    return np.stack(patches)


def resample_indices(timestamps: np.ndarray, fps: float) -> np.ndarray:
    """Indices that turn frames with arbitrary timestamps (s) into a constant-fps sequence."""
    if len(timestamps) < 2:
        return np.arange(len(timestamps))
    grid = np.arange(timestamps[0], timestamps[-1] + 1e-6, 1.0 / fps)
    idx = np.searchsorted(timestamps, grid)
    idx = np.clip(idx, 1, len(timestamps) - 1)
    left_closer = (grid - timestamps[idx - 1]) < (timestamps[idx] - grid)
    return np.where(left_closer, idx - 1, idx)
