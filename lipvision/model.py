"""Auto-AVSR visual speech recognition model (Conformer encoder + Transformer decoder).

LRS3 sentence-level WER ~20% with the default checkpoint, open vocabulary.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import assets
from .config import Settings

log = logging.getLogger("lipvision.model")

MEAN, STD = 0.421, 0.165  # Auto-AVSR video normalisation
INPUT_SIZE = 88           # center crop of the 96x96 aligned patch


@dataclass
class Transcript:
    text: str
    frames: int
    latency_ms: float
    final: bool


class Tokenizer:
    def __init__(self, units_path):
        units = open(units_path, encoding="utf8").read().splitlines()
        self.token_list = ["<blank>"] + [u.split()[0] for u in units] + ["<eos>"]

    def decode(self, ids) -> str:
        pieces = [self.token_list[i] for i in ids if 0 < i < len(self.token_list) - 1]
        return "".join(pieces).replace("▁", " ").replace("<unk>", "").strip()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class VSRModel:
    """Thread-unsafe; callers serialise access (the server uses one worker thread)."""

    def __init__(self, settings: Settings, net=None, tokenizer=None):
        self.settings = settings
        self.device = _resolve_device(settings.device)
        self.fp16 = settings.fp16 and self.device.type == "cuda"
        if net is None:
            root = assets.auto_avsr_dir(settings)
            from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

            tokenizer = Tokenizer(root / "spm" / "unigram" / "unigram5000_units.txt")
            net = E2E(len(tokenizer.token_list), "video", ctc_weight=settings.ctc_weight)
            ckpt = assets.checkpoint_path(settings)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:  # lightning checkpoint
                state = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
            net.load_state_dict(state)
            log.info("Loaded %s on %s (fp16=%s)", ckpt.name, self.device, self.fp16)
        self.net = net.to(self.device).eval()
        self.tokenizer = tokenizer
        self._beam = None
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))
        else:
            torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------- helpers
    def _beam_search(self):
        if self._beam is None:
            from espnet.nets.batch_beam_search import BatchBeamSearch
            from espnet.nets.scorers.length_bonus import LengthBonus

            w = self.settings.ctc_weight
            scorers = self.net.scorers()
            scorers["length_bonus"] = LengthBonus(len(self.tokenizer.token_list))
            self._beam = BatchBeamSearch(
                beam_size=self.settings.beam_size,
                vocab_size=len(self.tokenizer.token_list),
                weights={"decoder": 1.0 - w, "ctc": w, "length_bonus": 0.0},
                scorers=scorers,
                sos=self.net.sos, eos=self.net.eos,
                token_list=self.tokenizer.token_list,
                pre_beam_score_key=None if w == 1.0 else "decoder",
            ).to(self.device).eval()
        return self._beam

    def prepare(self, patches: np.ndarray) -> torch.Tensor:
        """uint8 (T, 96, 96) aligned grayscale mouth patches -> normalised (1, T, 1, 88, 88)."""
        x = torch.from_numpy(np.ascontiguousarray(patches)).to(self.device, non_blocking=True)
        h, w = x.shape[-2:]
        top, left = (h - INPUT_SIZE) // 2, (w - INPUT_SIZE) // 2
        x = x[:, top:top + INPUT_SIZE, left:left + INPUT_SIZE].float().div_(255.0)
        x = (x - MEAN) / STD
        return x[None, :, None]

    @torch.inference_mode()
    def encode(self, patches: np.ndarray) -> torch.Tensor:
        x = self.prepare(patches)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            x = self.net.frontend(x)
            x = self.net.proj_encoder(x)
            enc, _ = self.net.encoder(x, None)
        return enc[0].float()  # (T, 768)

    # ------------------------------------------------------------- decoding
    @torch.inference_mode()
    def transcribe(self, patches: np.ndarray, final: bool = True) -> Transcript:
        """Partial results use greedy CTC (a few ms); finals use joint CTC/attention beam search."""
        t0 = time.perf_counter()
        if len(patches) < 3:
            return Transcript("", len(patches), 0.0, final)
        enc = self.encode(patches)
        if final:
            hyps = self._beam_search()(enc, maxlenratio=0.0, minlenratio=0.0)
            ids = [int(i) for i in hyps[0].yseq[1:]] if hyps else []
        else:
            best = self.net.ctc.ctc_lo(enc).argmax(-1).tolist()
            ids, prev = [], -1
            for i in best:
                if i != prev and i != 0:
                    ids.append(i)
                prev = i
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        text = self.tokenizer.decode(ids)
        return Transcript(text, len(patches), (time.perf_counter() - t0) * 1000, final)

    def warmup(self):
        """Run once so CUDA kernels / cudnn autotune don't hit the first real utterance."""
        dummy = np.full((25, 96, 96), 110, np.uint8)
        self.transcribe(dummy, final=False)
        self.transcribe(dummy, final=True)
