"""Tests for benchmarks/metrics/detection.py — IoU, matching, AP, evaluation."""

import pytest
from benchmarks.metrics.detection import (
    _match_detections_scored,
    compute_ap,
    compute_iou,
    evaluate_dataset,
    evaluate_image,
)

# ── IoU ──────────────────────────────────────────────────────────────────

class TestComputeIoU:
    def test_identical_boxes(self):
        assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        # 5x5 intersection, each box is 10x10 → IoU = 25 / (100+100-25) = 25/175
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert abs(iou - 25 / 175) < 1e-6

    def test_one_box_inside_another(self):
        # Inner box 4x4=16, outer 10x10=100 → IoU = 16 / 100
        iou = compute_iou((0, 0, 10, 10), (3, 3, 7, 7))
        assert abs(iou - 16 / 100) < 1e-6

    def test_touching_edges(self):
        # Boxes share an edge but no area
        assert compute_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_zero_area_box(self):
        assert compute_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_float_coordinates(self):
        iou = compute_iou((0.5, 0.5, 10.5, 10.5), (0.5, 0.5, 10.5, 10.5))
        assert iou == 1.0


# ── Match detections ─────────────────────────────────────────────────────

class TestMatchDetectionsScored:
    def test_perfect_match(self):
        dets = [((0, 0, 10, 10), 0.9)]
        gts = [(0, 0, 10, 10)]
        scored = _match_detections_scored(dets, gts, 0.5)
        assert scored == [(0.9, True)]

    def test_no_match_low_iou(self):
        dets = [((0, 0, 10, 10), 0.9)]
        gts = [(50, 50, 60, 60)]
        scored = _match_detections_scored(dets, gts, 0.5)
        assert scored == [(0.9, False)]

    def test_empty_detections(self):
        assert _match_detections_scored([], [(0, 0, 10, 10)], 0.5) == []

    def test_empty_ground_truths(self):
        scored = _match_detections_scored([((0, 0, 10, 10), 0.9)], [], 0.5)
        assert scored == [(0.9, False)]

    def test_multiple_dets_one_gt(self):
        """Higher confidence detection should claim the GT; lower becomes FP."""
        dets = [((0, 0, 10, 10), 0.9), ((0, 0, 10, 10), 0.5)]
        gts = [(0, 0, 10, 10)]
        scored = _match_detections_scored(dets, gts, 0.5)
        assert scored[0] == (0.9, True)
        assert scored[1] == (0.5, False)

    def test_one_det_multiple_gts(self):
        dets = [((0, 0, 10, 10), 0.9)]
        gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
        scored = _match_detections_scored(dets, gts, 0.5)
        assert scored == [(0.9, True)]

    def test_strict_iou_threshold(self):
        """Partial overlap doesn't pass high IoU threshold."""
        dets = [((0, 0, 10, 10), 0.9)]
        gts = [(5, 5, 15, 15)]
        # IoU ≈ 0.14, threshold 0.5 → false positive
        scored = _match_detections_scored(dets, gts, 0.5)
        assert scored == [(0.9, False)]


# ── AP ───────────────────────────────────────────────────────────────────

class TestComputeAP:
    def test_perfect_ap(self):
        precisions = [1.0, 1.0, 1.0]
        recalls = [0.33, 0.66, 1.0]
        assert compute_ap(precisions, recalls) == pytest.approx(1.0, abs=0.01)

    def test_zero_ap(self):
        assert compute_ap([], []) == 0.0

    def test_decreasing_precision(self):
        precisions = [1.0, 0.5, 0.33]
        recalls = [0.25, 0.5, 0.75]
        ap = compute_ap(precisions, recalls)
        assert 0.0 < ap < 1.0


# ── Evaluate image ───────────────────────────────────────────────────────

class TestEvaluateImage:
    def test_all_correct(self):
        dets = [((0, 0, 10, 10), 0.9), ((20, 20, 30, 30), 0.8)]
        gts = [(0, 0, 10, 10), (20, 20, 30, 30)]
        r = evaluate_image(dets, gts, 0.5)
        assert r["tp"] == 2
        assert r["fp"] == 0
        assert r["fn"] == 0
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0

    def test_all_false_positives(self):
        dets = [((0, 0, 10, 10), 0.9)]
        gts = [(50, 50, 60, 60)]
        r = evaluate_image(dets, gts, 0.5)
        assert r["tp"] == 0
        assert r["fp"] == 1
        assert r["fn"] == 1

    def test_missed_detections(self):
        dets = []
        gts = [(0, 0, 10, 10)]
        r = evaluate_image(dets, gts, 0.5)
        assert r["tp"] == 0
        assert r["fn"] == 1
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0

    def test_empty_both(self):
        r = evaluate_image([], [], 0.5)
        assert r["tp"] == 0
        assert r["fp"] == 0
        assert r["fn"] == 0


# ── Evaluate dataset ─────────────────────────────────────────────────────

class TestEvaluateDataset:
    def test_perfect_detection(self):
        all_dets = [
            [((0, 0, 10, 10), 0.9)],
            [((20, 20, 30, 30), 0.8)],
        ]
        all_gts = [
            [(0, 0, 10, 10)],
            [(20, 20, 30, 30)],
        ]
        r = evaluate_dataset(all_dets, all_gts, 0.5)
        assert r["ap"] == pytest.approx(1.0, abs=0.01)
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0
        assert r["total_tp"] == 2
        assert r["total_gt"] == 2

    def test_mixed_results(self):
        all_dets = [
            [((0, 0, 10, 10), 0.9), ((50, 50, 55, 55), 0.3)],  # 1 TP, 1 FP
            [],  # miss
        ]
        all_gts = [
            [(0, 0, 10, 10)],
            [(20, 20, 30, 30)],
        ]
        r = evaluate_dataset(all_dets, all_gts, 0.5)
        assert r["total_tp"] == 1
        assert r["total_fp"] == 1
        assert r["total_fn"] == 1
        assert r["precision"] == 0.5
        assert r["recall"] == 0.5

    def test_empty_dataset(self):
        r = evaluate_dataset([], [], 0.5)
        assert r["ap"] == 0.0
        assert r["total_gt"] == 0

    def test_multiple_iou_thresholds(self):
        """Higher IoU threshold should produce lower or equal AP."""
        all_dets = [[((0, 0, 10, 10), 0.9)]]
        all_gts = [[(1, 1, 11, 11)]]  # slight offset
        r_50 = evaluate_dataset(all_dets, all_gts, 0.5)
        r_75 = evaluate_dataset(all_dets, all_gts, 0.75)
        assert r_50["ap"] >= r_75["ap"]
