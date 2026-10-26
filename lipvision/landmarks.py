"""MediaPipe Tasks face landmarker (the legacy mp.solutions API no longer exists)."""

import cv2
import numpy as np

from . import assets
from .config import Settings
from .preprocess import mesh_to_points, mouth_openness


class FaceTracker:
    def __init__(self, settings: Settings):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(assets.face_landmarker_path(settings))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def __call__(self, bgr: np.ndarray, t: float):
        """Returns (mesh (478,2) px, 4 stable points, mouth openness) or (None, None, None)."""
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(image, ms)
        if not res.face_landmarks:
            return None, None, None
        h, w = bgr.shape[:2]
        mesh = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float32)
        return mesh, mesh_to_points(mesh), mouth_openness(mesh)

    def close(self):
        self.landmarker.close()
