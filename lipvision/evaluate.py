"""Measure the system on labelled clips and write a results folder.

    python -m lipvision.evaluate manifest.csv --out results/baseline --label baseline

manifest.csv columns: path, speaker, reference (the sentence actually mouthed).
Writes clips.csv (one row per clip) and summary.json (pooled WER/CER with a 95%
interval over speakers, latency percentiles, and the exact code/model settings).
A clip that fails (no face, unreadable file) counts as fully wrong, never skipped.
"""

import argparse
import csv
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import ROOT, Settings
from .metrics import bootstrap_ci, char_errors, word_errors

FIELDS = ["path", "speaker", "reference", "hypothesis", "wer", "cer", "frames",
          "latency_ms", "error"]


def load_patches(path: str, settings: Settings, cache: dict) -> np.ndarray:
    """Video file -> aligned mouth patches, with the same code the live app uses."""
    from .landmarks import FaceTracker
    from .transcribe import video_to_patches

    tracker = FaceTracker(settings)  # VIDEO-mode timestamps restart per file
    try:
        return video_to_patches(path, settings, tracker)
    finally:
        tracker.close()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _pct(values, q):
    return float(np.percentile(values, q)) if values else None


def run(manifest, out_dir, settings: Settings, model=None, label: str = "run") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        from .model import VSRModel
        model = VSRModel(settings)
        model.warmup()
    items = list(csv.DictReader(open(manifest, encoding="utf8")))
    base = Path(manifest).resolve().parent
    rows, wer_rows, cer_rows, latencies = [], [], [], []
    for item in items:
        ref = item["reference"]
        row = {"path": item["path"], "speaker": item["speaker"], "reference": ref,
               "hypothesis": "", "frames": 0, "latency_ms": "", "error": ""}
        try:
            clip = Path(item["path"])
            patches = load_patches(str(clip if clip.is_absolute() else base / clip), settings, {})
            res = model.transcribe(patches, final=True)
            row.update(hypothesis=res.text, frames=res.frames, latency_ms=round(res.latency_ms, 1))
            latencies.append(res.latency_ms)
        except Exception as exc:  # recorded per clip; one bad file must not stop the run
            row["error"] = str(exc)[:200]
        we, wn = word_errors(ref, row["hypothesis"])
        ce, cn = char_errors(ref, row["hypothesis"])
        row.update(wer=round(we / max(wn, 1), 4), cer=round(ce / max(cn, 1), 4))
        rows.append(row)
        wer_rows.append((item["speaker"], we, wn))
        cer_rows.append((item["speaker"], ce, cn))
        logging.info("%s | ref: %s | hyp: %s | WER %.2f", item["path"], ref, row["hypothesis"], row["wer"])

    with open(out_dir / "clips.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    pooled = lambda rs: sum(r[1] for r in rs) / max(sum(r[2] for r in rs), 1)  # noqa: E731
    summary = {
        "label": label,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": str(manifest),
        "clips": len(rows),
        "speakers": len({r["speaker"] for r in rows}),
        "failed_clips": sum(1 for r in rows if r["error"]),
        "wer": pooled(wer_rows),
        "wer_ci95": list(bootstrap_ci(wer_rows, n=1000, seed=0)),
        "cer": pooled(cer_rows),
        "cer_ci95": list(bootstrap_ci(cer_rows, n=1000, seed=0)),
        "exact_match": sum(1 for r in rows if r["wer"] == 0) / max(len(rows), 1),
        "latency_ms_p50": _pct(latencies, 50),
        "latency_ms_p95": _pct(latencies, 95),
        "git_commit": _git_commit(),
        "checkpoint": settings.checkpoint,
        "beam_size": settings.beam_size,
        "ctc_weight": settings.ctc_weight,
        "device": str(getattr(model, "device", settings.device)),
        "note": "Measured on our own clips. Not comparable to published LRS3 numbers.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def find_manifest(roots=("/kaggle/input",)) -> Path | None:
    """First manifest.csv inside any attached Kaggle dataset (the clips sit beside it)."""
    for root in roots:
        hits = sorted(Path(root).rglob("manifest.csv")) if Path(root).is_dir() else []
        if hits:
            return hits[0]
    return None


def run_in_notebook(label: str = "baseline", out_root: str = "/kaggle/working/results"):
    """Notebook entry point: find the attached manifest, run, and print a short table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = find_manifest()
    if manifest is None:
        raise FileNotFoundError("No manifest.csv found. Attach the clips dataset with 'Add Input'.")
    s = Settings.from_env()
    summary = run(manifest, Path(out_root) / label, s, label=label)
    print(f"\nWER {summary['wer']:.1%} (95% CI {summary['wer_ci95'][0]:.1%} to {summary['wer_ci95'][1]:.1%}), "
          f"CER {summary['cer']:.1%}, {summary['clips']} clips from {summary['speakers']} speakers, "
          f"{summary['failed_clips']} failed. Results saved in {Path(out_root) / label}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--out", default="results/run")
    p.add_argument("--label", default="run")
    p.add_argument("--beam-size", type=int)
    p.add_argument("--device")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = Settings.from_env(beam_size=a.beam_size, device=a.device)
    summary = run(a.manifest, a.out, s, label=a.label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
