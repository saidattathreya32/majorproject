"""Word and character error rates, and bootstrap confidence intervals over speakers.

No third-party dependencies, so the numbers in the report can be re-derived anywhere.
"""

import random
import re

_KEEP = re.compile(r"[^a-z0-9' ]+")


def normalise(text: str) -> str:
    """Lower-case, drop punctuation (keep apostrophes), collapse spaces."""
    return " ".join(_KEEP.sub(" ", text.lower()).split())


def edit_distance(ref, hyp) -> int:
    """Levenshtein distance between two sequences (strings or lists)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, reference word count) after normalisation."""
    r, h = normalise(ref).split(), normalise(hyp).split()
    return edit_distance(r, h), len(r)


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(character edits, reference character count), spaces ignored."""
    r, h = normalise(ref).replace(" ", ""), normalise(hyp).replace(" ", "")
    return edit_distance(r, h), len(r)


def wer(ref: str, hyp: str) -> float:
    e, n = word_errors(ref, hyp)
    return e / max(n, 1)


def cer(ref: str, hyp: str) -> float:
    e, n = char_errors(ref, hyp)
    return e / max(n, 1)


def bootstrap_ci(rows, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """95% interval of a pooled error rate, resampling whole speakers.

    rows: iterable of (speaker_id, errors, reference_length). Resampling speakers,
    not utterances, keeps one talkative speaker from making the interval look narrow.
    """
    by_spk = {}
    for spk, e, total in rows:
        a = by_spk.setdefault(spk, [0, 0])
        a[0] += e
        a[1] += total
    groups = list(by_spk.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        pick = [rng.choice(groups) for _ in groups]
        stats.append(sum(g[0] for g in pick) / max(sum(g[1] for g in pick), 1))
    stats.sort()
    return stats[int(n * alpha / 2)], stats[min(n - 1, int(n * (1 - alpha / 2)))]
