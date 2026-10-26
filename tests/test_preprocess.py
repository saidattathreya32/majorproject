import numpy as np

from lipvision.preprocess import MouthAligner, align_video, interpolate_missing, resample_indices

from .conftest import face_image, face_points


def test_patch_is_invariant_to_head_roll_scale_and_position(mean_face):
    aligner = MouthAligner(mean_face, window=1)
    p0 = face_points()
    base = aligner(face_image(p0, open_amount=0.5), p0).astype(float)
    for kw in ({"angle_deg": 15}, {"scale": 1.2}, {"center": (250, 300)}):
        aligner.reset()
        p = face_points(**kw)
        patch = aligner(face_image(p, open_amount=0.5), p).astype(float)
        assert patch.shape == (96, 96)
        assert np.abs(patch - base).mean() < 12, kw


def test_patch_is_centred_on_mouth(mean_face):
    p = face_points()
    patch = MouthAligner(mean_face, window=1)(face_image(p, open_amount=1.0), p)
    ys, xs = np.nonzero(patch > 150)  # bright lip ring
    assert abs(xs.mean() - 48) < 3 and abs(ys.mean() - 48) < 3


def test_causal_smoothing_follows_current_position(mean_face):
    aligner = MouthAligner(mean_face, window=5)
    for c in [(300, 240)] * 4:
        p = face_points(center=c)
        aligner(face_image(p), p)
    p = face_points(center=(340, 240))  # sudden move: patch must still be on the mouth
    patch = aligner(face_image(p, open_amount=1.0), p)
    ys, xs = np.nonzero(patch > 150)
    assert abs(xs.mean() - 48) < 3


def test_resample_30fps_to_25fps():
    ts = np.arange(0, 2.0, 1 / 30)
    idx = resample_indices(ts, 25.0)
    assert abs(len(idx) - 50) <= 1
    assert np.all(np.diff(idx) >= 0)


def test_resample_irregular_timestamps():
    ts = np.array([0.0, 0.03, 0.2, 0.21, 0.4])
    idx = resample_indices(ts, 25.0)
    assert len(idx) == 11 and idx[0] == 0 and idx[-1] == 4


def test_interpolate_missing():
    a, b = np.zeros((4, 2), np.float32), np.ones((4, 2), np.float32)
    out = interpolate_missing([None, a, None, b, None])
    assert np.allclose(out[0], a) and np.allclose(out[2], 0.5) and np.allclose(out[4], b)
    assert interpolate_missing([None, None]) is None


def test_align_video(mean_face):
    pts = [face_points(center=(320 + i, 240)) for i in range(10)]
    frames = [face_image(p) for p in pts]
    pts[3] = None
    out = align_video(frames, pts, mean_face)
    assert out.shape == (10, 96, 96) and out.dtype == np.uint8
