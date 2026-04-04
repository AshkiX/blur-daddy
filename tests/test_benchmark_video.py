"""Tests for benchmarks/metrics/video.py — temporal consistency and flicker."""

from benchmarks.detectors.base import Detection
from benchmarks.metrics.video import compute_flicker_rate, compute_temporal_consistency

# ── Temporal consistency ─────────────────────────────────────────────────

class TestTemporalConsistency:
    def test_stable_detections(self):
        """Same face barely moves → high consistency."""
        frames = [
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(11, 11, 51, 51), confidence=0.9)],
            [Detection(box=(12, 12, 52, 52), confidence=0.9)],
        ]
        r = compute_temporal_consistency(frames)
        assert r["match_rate"] == 1.0
        assert r["mean_iou"] > 0.85

    def test_no_overlap_between_frames(self):
        """Face jumps to completely different location each frame."""
        frames = [
            [Detection(box=(0, 0, 10, 10), confidence=0.9)],
            [Detection(box=(100, 100, 110, 110), confidence=0.9)],
        ]
        r = compute_temporal_consistency(frames)
        assert r["match_rate"] == 0.0

    def test_single_frame(self):
        frames = [[Detection(box=(0, 0, 10, 10), confidence=0.9)]]
        r = compute_temporal_consistency(frames)
        assert r["num_transitions"] == 0

    def test_empty_frames(self):
        frames = [[], [], []]
        r = compute_temporal_consistency(frames)
        assert r["match_rate"] == 0.0

    def test_appearing_face(self):
        """Face appears in second frame — no match from first (empty) frame."""
        frames = [
            [],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(11, 11, 51, 51), confidence=0.9)],
        ]
        r = compute_temporal_consistency(frames)
        # First transition: 0 dets → can't match. Second: should match.
        assert r["match_rate"] > 0.0

    def test_multiple_faces(self):
        frames = [
            [
                Detection(box=(0, 0, 100, 100), confidence=0.9),
                Detection(box=(200, 200, 300, 300), confidence=0.8),
            ],
            [
                Detection(box=(1, 1, 101, 101), confidence=0.9),
                Detection(box=(201, 201, 301, 301), confidence=0.8),
            ],
        ]
        r = compute_temporal_consistency(frames)
        assert r["match_rate"] == 1.0
        assert r["mean_iou"] > 0.9


# ── Flicker rate ─────────────────────────────────────────────────────────

class TestFlickerRate:
    def test_no_flicker(self):
        """Constant detection count → no flicker."""
        frames = [
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
        ]
        r = compute_flicker_rate(frames)
        assert r["flicker_rate"] == 0.0
        assert r["total_flickers"] == 0

    def test_single_flicker(self):
        """Detection drops for one frame then comes back."""
        frames = [
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [],  # dropped
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
        ]
        r = compute_flicker_rate(frames)
        assert r["total_flickers"] == 1
        assert r["flicker_rate"] > 0.0

    def test_too_few_frames(self):
        frames = [[Detection(box=(0, 0, 10, 10), confidence=0.9)]]
        r = compute_flicker_rate(frames)
        assert r["flicker_rate"] == 0.0

    def test_two_frames(self):
        frames = [
            [Detection(box=(0, 0, 10, 10), confidence=0.9)],
            [],
        ]
        r = compute_flicker_rate(frames)
        assert r["flicker_rate"] == 0.0  # Need 3+ frames to detect flicker

    def test_permanent_change_not_flicker(self):
        """Face appears and stays → not a flicker."""
        frames = [
            [],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
        ]
        r = compute_flicker_rate(frames)
        assert r["total_flickers"] == 0

    def test_multiple_flickers(self):
        """Flicker = count changes then reverts on the very next frame."""
        frames = [
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [],  # flicker 1 (1→0, next reverts to 1)
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
            [],  # flicker 2 (1→0, next reverts to 1)
            [Detection(box=(10, 10, 50, 50), confidence=0.9)],
        ]
        r = compute_flicker_rate(frames)
        # frames[1], [2], [3] are all interior frames checked for flicker
        assert r["total_flickers"] >= 2
