"""Visual voice-activity detection: decides when the lips are speaking.

Uses the *motion* of the inner-lip opening rather than a single open/closed
threshold (the old approach), so closed-lip sounds (m, b, p) don't end an
utterance and a still, open mouth doesn't start one. The noise floor adapts
to each person / camera.
"""

from .config import Settings


class LipActivityDetector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.reset()

    def reset(self):
        self.prev_open = None
        self.prev_t = None
        self.energy = 0.0
        self.noise = self.s.vad_off / 2
        self.speaking = False
        self.above_since = None
        self.still_since = None

    @property
    def on_threshold(self) -> float:
        return max(self.s.vad_on, 3.0 * self.noise)

    @property
    def off_threshold(self) -> float:
        return max(self.s.vad_off, 1.5 * self.noise)

    def update(self, t: float, openness: float | None) -> bool:
        """Feed one frame (time in seconds). Returns whether speech is active."""
        if openness is None:
            self.prev_open = None
            return self.speaking
        if self.prev_open is not None and t > self.prev_t:
            # normalise to a 25 fps step so the camera frame rate doesn't matter
            d = abs(openness - self.prev_open) * (1.0 / self.s.fps) / (t - self.prev_t)
            self.energy += 0.35 * (d - self.energy)
        self.prev_open, self.prev_t = openness, t

        if not self.speaking:
            self.noise += 0.02 * (min(self.energy, self.s.vad_on) - self.noise)
            if self.energy > self.on_threshold:
                self.above_since = t if self.above_since is None else self.above_since
                if t - self.above_since >= 0.06:  # ~2 frames of motion
                    self.speaking, self.still_since = True, None
            else:
                self.above_since = None
        else:
            if self.energy < self.off_threshold:
                self.still_since = t if self.still_since is None else self.still_since
                if t - self.still_since >= self.s.vad_hangover:
                    self.speaking, self.above_since = False, None
            else:
                self.still_since = None
        return self.speaking
