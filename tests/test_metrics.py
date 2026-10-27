import pytest

from lipvision.metrics import bootstrap_ci, cer, edit_distance, normalise, wer


def test_normalise_strips_case_and_punctuation():
    assert normalise("  Hello, WORLD!  It's  me. ") == "hello world it's me"


def test_edit_distance_basic():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_wer_counts_word_edits_over_reference_length():
    # one substitution + one deletion over 4 reference words
    assert wer("i need water now", "i need walter") == pytest.approx(2 / 4)
    assert wer("call the nurse", "Call the nurse.") == 0.0


def test_wer_empty_hypothesis_is_full_error():
    assert wer("call the nurse", "") == 1.0


def test_cer_ignores_spaces():
    assert cer("ab cd", "abcd") == 0.0
    assert cer("abcd", "abce") == pytest.approx(0.25)


def test_bootstrap_ci_groups_by_speaker_and_is_reproducible():
    rows = [("s1", 1, 4), ("s1", 0, 4), ("s2", 2, 4), ("s2", 2, 4), ("s3", 0, 4)]
    lo1, hi1 = bootstrap_ci(rows, n=500, seed=7)
    lo2, hi2 = bootstrap_ci(rows, n=500, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    point = sum(r[1] for r in rows) / sum(r[2] for r in rows)
    assert lo1 <= point <= hi1
    assert 0.0 <= lo1 <= hi1 <= 1.0
