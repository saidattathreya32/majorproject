"""Central settings. Every value can be overridden with an env var LIPVISION_<NAME>."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = Path(os.environ.get("LIPVISION_HOME", ROOT / ".cache"))

# MediaPipe FaceMesh (478 pts) indices for the classic 68-point layout the
# Auto-AVSR mean face uses. Only the groups needed for alignment are kept.
MESH_RIGHT_EYE = [33, 160, 158, 133, 153, 144]       # 68-pt 36..41
MESH_LEFT_EYE = [362, 385, 387, 263, 373, 380]       # 68-pt 42..47
MESH_NOSE = [75, 97, 2, 326, 305]                    # 68-pt 31..35
MESH_MOUTH = [61, 39, 37, 0, 267, 269, 291, 405, 314, 17,
              84, 181, 78, 82, 13, 312, 308, 317, 14, 87]  # 68-pt 48..67
# Inner-lip points used for the mouth-opening signal: top, bottom, left, right.
MESH_LIP_OPEN = [13, 14, 78, 308]
# Outer + inner lip contours, used only for drawing.
MESH_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                  375, 321, 405, 314, 17, 84, 181, 91, 146]
MESH_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                  324, 318, 402, 317, 14, 87, 178, 88, 95]

# Pinned upstream Auto-AVSR commit (model code, tokenizer, mean face).
AUTO_AVSR_REPO = "https://github.com/mpc001/auto_avsr"
AUTO_AVSR_COMMIT = "182b62837773ab01052d4ac21ef1d2203ea7d267"

# Official Auto-AVSR VSR checkpoints (Google Drive ids, md5 prefix, LRS3 WER).
CHECKPOINTS = {
    "vsr_trlrs2lrs3vox2avsp_base.pth": ("1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_", "49f77", 20.3),
    "vsr_trlrs3vox2_base.pth": ("1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i", "774a6", 24.6),
    "vsr_trlrs3_base.pth": ("12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd", "c00a7", 36.0),
}
DEFAULT_CHECKPOINT = "vsr_trlrs2lrs3vox2avsp_base.pth"

FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/1/face_landmarker.task")


@dataclass
class Settings:
    cache_dir: Path = DEFAULT_CACHE
    checkpoint: str = DEFAULT_CHECKPOINT
    device: str = "auto"            # auto | cuda | cpu
    fp16: bool = True               # half-precision encoder on GPU

    # Decoding
    beam_size: int = 10             # final hypothesis; larger = slower, slightly better
    ctc_weight: float = 0.1         # joint CTC/attention weight used in training
    partial_interval: float = 0.6   # seconds between live partial transcripts

    # Video
    fps: float = 25.0               # model was trained on 25 fps video
    crop_size: int = 96             # aligned mouth patch fed to the model (center-cropped to 88)
    smooth_window: int = 5          # causal landmark smoothing for live video (frames)

    # Visual speech activity detection
    vad_on: float = 0.012           # lip-motion energy that starts an utterance
    vad_off: float = 0.006          # energy below which silence is counted
    vad_hangover: float = 0.6       # seconds of stillness that ends an utterance
    pre_roll: float = 0.32          # seconds kept before speech onset
    min_utterance: float = 0.4      # shorter bursts are ignored (blinks, swallows)
    max_utterance: float = 10.0     # force a final decode on long monologues

    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls, **overrides):
        s = cls()
        for f in fields(cls):
            raw = os.environ.get(f"LIPVISION_{f.name.upper()}")
            if raw is None:
                continue
            cur = getattr(s, f.name)
            if isinstance(cur, bool):
                val = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, Path):
                val = Path(raw)
            else:
                val = type(cur)(raw)
            setattr(s, f.name, val)
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        s.cache_dir = Path(s.cache_dir)
        return s


def client_config(settings: Settings) -> dict:
    """Values the browser needs so both sides agree on landmark semantics."""
    return {
        "fps": settings.fps,
        "mesh": {
            "rightEye": MESH_RIGHT_EYE,
            "leftEye": MESH_LEFT_EYE,
            "nose": MESH_NOSE,
            "mouth": MESH_MOUTH,
            "lipOpen": MESH_LIP_OPEN,
            "lipOuter": MESH_LIP_OUTER,
            "lipInner": MESH_LIP_INNER,
        },
        "cropSize": 192,
    }
