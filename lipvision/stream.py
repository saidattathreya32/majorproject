"""Live session: frames in, decode jobs out. Shared by the web server and local webcam mode."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .preprocess import MouthAligner, resample_indices
from .vad import LipActivityDetector


@dataclass
class DecodeJob:
    utterance: int
    patches: np.ndarray      # (T, 96, 96) uint8 at the model frame rate
    final: bool
    duration: float


@dataclass
class FrameResult:
    speaking: bool
    energy: float
    patch: np.ndarray | None
    jobs: list = field(default_factory=list)


class LiveSession:
    """Segments a live stream into utterances and schedules partial/final decodes.

    mode="auto"   : the lip-activity detector starts/ends utterances.
    mode="manual" : the client calls start()/stop() (push-to-talk, most accurate).
    """

    def __init__(self, settings: Settings, mean_face: np.ndarray, mode: str = "auto"):
        self.s = settings
        self.aligner = MouthAligner(mean_face, settings.crop_size, settings.smooth_window)
        self.vad = LipActivityDetector(settings)
        self.mode = mode
        self.pre = deque()            # (t, patch) kept for pre-roll
        self.utt: list | None = None  # (t, patch) of the active utterance
        self.utt_id = 0
        self.utt_start = self.last_partial = 0.0
        self.manual_on = False
        self.lost_since = None

    # ------------------------------------------------------------ controls
    def set_mode(self, mode: str):
        self.mode = mode if mode in ("auto", "manual") else "auto"
        self.manual_on = False

    def start(self, t: float):
        self.manual_on = True
        if self.utt is None:
            self._open(t)

    def stop(self, t: float) -> list:
        self.manual_on = False
        return self._close(t, trim=False)

    def reset(self):
        self.aligner.reset()
        self.vad.reset()
        self.pre.clear()
        self.utt = None

    # ------------------------------------------------------------ internals
    def _open(self, t: float):
        self.utt_id += 1
        self.utt = [x for x in self.pre if x[0] >= t - self.s.pre_roll]
        self.utt_start = self.last_partial = t

    def _job(self, frames: list, final: bool) -> DecodeJob | None:
        if len(frames) < 2:
            return None
        ts = np.array([f[0] for f in frames])
        idx = resample_indices(ts, self.s.fps)
        return DecodeJob(self.utt_id, np.stack([frames[i][1] for i in idx]), final, ts[-1] - ts[0])

    def _close(self, t: float, trim: bool) -> list:
        if self.utt is None:
            return []
        frames, self.utt = self.utt, None
        speech_end = t
        if trim:  # drop most of the trailing stillness the detector waited through
            speech_end = t - self.s.vad_hangover
            frames = [f for f in frames if f[0] <= speech_end + 0.2] or frames
        job = self._job(frames, final=True)
        if job is None or speech_end - self.utt_start < self.s.min_utterance:
            return [DecodeJob(self.utt_id, np.zeros((0, 96, 96), np.uint8), True, 0.0)]  # discard
        return [job]

    # ------------------------------------------------------------ main entry
    def push(self, t: float, gray: np.ndarray | None, points: np.ndarray | None,
             openness: float | None) -> FrameResult:
        """Feed one frame. `t` in seconds, `points` = 4 stable points or None if no face."""
        jobs: list = []
        if points is None or gray is None:
            self.lost_since = t if self.lost_since is None else self.lost_since
            self.vad.update(t, None)
            if self.utt is not None and t - self.lost_since > 0.5:
                jobs += self._close(t, trim=False)
            return FrameResult(self.utt is not None, self.vad.energy, None, jobs)
        self.lost_since = None

        patch = self.aligner(gray, points)
        item = (t, patch)
        self.pre.append(item)
        while self.pre and self.pre[0][0] < t - self.s.pre_roll - 0.1:
            self.pre.popleft()

        speaking = self.vad.update(t, openness)
        active = self.manual_on if self.mode == "manual" else speaking

        if self.utt is None:
            if active:
                self._open(t)
        else:
            self.utt.append(item)
            if not active:
                jobs += self._close(t, trim=self.mode == "auto")
            elif t - self.utt[0][0] >= self.s.max_utterance:
                jobs += self._close(t, trim=False)
                self._open(t)
            elif t - self.last_partial >= (self.s.partial_interval or 0.6) and len(self.utt) > 8:
                self.last_partial = t
                job = self._job(self.utt, final=False)
                if job:
                    jobs.append(job)
        return FrameResult(self.utt is not None, self.vad.energy, patch, jobs)
