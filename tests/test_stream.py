import numpy as np

from lipvision.stream import LiveSession
from lipvision.vad import LipActivityDetector

from .conftest import face_image, face_points, speech_openness

FPS = 30.0


def run(session, seconds, onset, offset):
    jobs, speaking = [], []
    p = face_points()
    for i in range(int(seconds * FPS)):
        t = i / FPS
        o = speech_openness(t, onset, offset)
        r = session.push(t, face_image(p, open_amount=o * 3), p, o)
        jobs += r.jobs
        speaking.append((t, r.speaking))
    return jobs, speaking


def test_vad_detects_speech_window(settings):
    vad = LipActivityDetector(settings)
    states = [(t, vad.update(t, speech_openness(t, 2.0, 3.5))) for t in np.arange(0, 6, 1 / FPS)]
    on = [t for t, s in states if s]
    assert on, "speech never detected"
    assert 2.0 <= on[0] < 2.3
    assert 3.5 < on[-1] < 3.5 + settings.vad_hangover + 0.3


def test_vad_ignores_still_open_mouth(settings):
    vad = LipActivityDetector(settings)
    assert not any(vad.update(t, 0.35) for t in np.arange(0, 3, 1 / FPS))


def test_auto_session_emits_partials_then_one_final(settings, mean_face):
    s = LiveSession(settings, mean_face)
    jobs, _ = run(s, 6.0, 2.0, 4.0)
    finals = [j for j in jobs if j.final]
    partials = [j for j in jobs if not j.final]
    assert len(finals) == 1 and len(partials) >= 3
    f = finals[0]
    # pre-roll + 2 s speech + short tail, resampled to 25 fps
    assert 2.0 < f.duration < 2.9
    assert abs(len(f.patches) - f.duration * settings.fps) <= 2
    assert f.patches.shape[1:] == (96, 96)
    assert all(p.utterance == f.utterance for p in partials)


def test_short_burst_is_discarded(settings, mean_face):
    settings.min_utterance = 0.8
    s = LiveSession(settings, mean_face)
    jobs, _ = run(s, 4.0, 1.0, 1.25)
    finals = [j for j in jobs if j.final]
    assert len(finals) == 1 and len(finals[0].patches) == 0


def test_manual_mode_uses_push_to_talk(settings, mean_face):
    s = LiveSession(settings, mean_face, mode="manual")
    p = face_points()
    jobs = []
    for i in range(int(3 * FPS)):
        t = i / FPS
        if i == 15:
            s.start(t)
        if i == 75:
            jobs += s.stop(t)
        jobs += s.push(t, face_image(p), p, 0.02).jobs  # lips still: VAD alone would never fire
    finals = [j for j in jobs if j.final]
    assert len(finals) == 1 and abs(finals[0].duration - 2.0) < 0.5


def test_face_loss_finalises_utterance(settings, mean_face):
    s = LiveSession(settings, mean_face)
    p = face_points()
    jobs = []
    for i in range(int(4 * FPS)):
        t = i / FPS
        o = speech_openness(t, 1.0, 4.0)
        if t < 2.5:
            jobs += s.push(t, face_image(p), p, o).jobs
        else:
            jobs += s.push(t, None, None, None).jobs
    assert any(j.final and len(j.patches) for j in jobs)


def test_max_utterance_splits_long_speech(settings, mean_face):
    settings.max_utterance = 2.0
    s = LiveSession(settings, mean_face)
    jobs, _ = run(s, 7.0, 0.5, 6.0)
    finals = [j for j in jobs if j.final and len(j.patches)]
    assert len(finals) >= 2
