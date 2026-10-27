import csv
import json

import numpy as np

from lipvision import evaluate
from tests.conftest import FakeModel


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "speaker", "reference"])
        w.writeheader()
        w.writerows(rows)


def test_evaluate_writes_per_clip_rows_and_summary(tmp_path, settings, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [
        {"path": "a.mp4", "speaker": "s1", "reference": "hello world"},
        {"path": "b.mp4", "speaker": "s2", "reference": "call the nurse"},
    ])
    monkeypatch.setattr(evaluate, "load_patches",
                        lambda path, s, cache: np.zeros((40, 96, 96), np.uint8))
    out = tmp_path / "run"
    summary = evaluate.run(manifest, out, settings, model=FakeModel(), label="baseline")

    rows = list(csv.DictReader(open(out / "clips.csv", encoding="utf8")))
    assert [r["path"] for r in rows] == ["a.mp4", "b.mp4"]
    assert rows[0]["hypothesis"] == "HELLO WORLD"
    assert float(rows[0]["wer"]) == 0.0
    assert float(rows[1]["wer"]) == 1.0  # 3 subs/ins vs "call the nurse"

    saved = json.loads((out / "summary.json").read_text())
    assert saved == summary
    assert summary["label"] == "baseline"
    assert summary["clips"] == 2 and summary["speakers"] == 2
    # pooled WER: (0 + 3 edits) / (2 + 3 words)
    assert abs(summary["wer"] - 3 / 5) < 1e-9
    assert len(summary["wer_ci95"]) == 2
    for key in ("cer", "latency_ms_p50", "latency_ms_p95", "git_commit", "checkpoint", "beam_size"):
        assert key in summary


def test_relative_clip_paths_resolve_beside_the_manifest(tmp_path, settings, monkeypatch):
    data = tmp_path / "input" / "lipvision-clips"
    data.mkdir(parents=True)
    write_manifest(data / "manifest.csv", [{"path": "s1_01.mp4", "speaker": "s1", "reference": "hello world"}])
    seen = []
    monkeypatch.setattr(evaluate, "load_patches",
                        lambda path, s, cache: seen.append(path) or np.zeros((30, 96, 96), np.uint8))
    assert evaluate.find_manifest([str(tmp_path / "input")]) == data / "manifest.csv"
    evaluate.run(data / "manifest.csv", tmp_path / "r", settings, model=FakeModel())
    assert seen == [str(data / "s1_01.mp4")]


def test_find_manifest_returns_none_when_missing(tmp_path):
    assert evaluate.find_manifest([str(tmp_path)]) is None


def test_evaluate_records_failed_clips_instead_of_crashing(tmp_path, settings, monkeypatch):
    manifest = tmp_path / "m.csv"
    write_manifest(manifest, [{"path": "noface.mp4", "speaker": "s1", "reference": "hi there"}])

    def boom(path, s, cache):
        raise RuntimeError("No face found")

    monkeypatch.setattr(evaluate, "load_patches", boom)
    summary = evaluate.run(manifest, tmp_path / "r", settings, model=FakeModel())
    rows = list(csv.DictReader(open(tmp_path / "r" / "clips.csv", encoding="utf8")))
    assert rows[0]["error"].startswith("No face found")
    assert float(rows[0]["wer"]) == 1.0  # a failed clip counts as fully wrong, never skipped
    assert summary["failed_clips"] == 1
