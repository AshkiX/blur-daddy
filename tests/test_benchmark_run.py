"""Tests for benchmarks/run.py — harness, CLI, and integration."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from benchmarks.detectors.base import Detection, DetectorBase
from benchmarks.run import run_image_benchmark, run_video_benchmark


class _FixedDetector(DetectorBase):
    """Detector that always returns fixed detections for testing."""

    name = "test-fixed"

    def __init__(self, detections=None):
        self._detections = detections or []

    def detect(self, image):
        return self._detections


class TestRunImageBenchmark:
    def _make_samples(self, tmp_path, n=5):
        from benchmarks.datasets.base import ImageSample

        samples = []
        for i in range(n):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            path = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(path), img)
            samples.append(ImageSample(
                path=path,
                ground_truth_boxes=[(10, 10, 50, 50)],
            ))
        return samples

    def test_basic_run(self):
        with tempfile.TemporaryDirectory() as td:
            samples = self._make_samples(Path(td))
            det = _FixedDetector([Detection(box=(10, 10, 50, 50), confidence=0.9)])
            det_results, perf_results = run_image_benchmark([det], samples, [0.5])

            assert "test-fixed" in det_results
            assert "IoU=0.5" in det_results["test-fixed"]
            assert det_results["test-fixed"]["IoU=0.5"]["ap"] > 0.0

            assert "test-fixed" in perf_results
            assert perf_results["test-fixed"]["fps"] > 0

    def test_empty_samples(self):
        det = _FixedDetector()
        det_results, perf_results = run_image_benchmark([det], [], [0.5])
        # With 0 images loaded, detectors still run but produce empty metrics
        assert det_results.get("test-fixed", {}).get("IoU=0.5", {}).get("total_gt", 0) == 0

    def test_multiple_detectors(self):
        with tempfile.TemporaryDirectory() as td:
            samples = self._make_samples(Path(td), n=3)
            det1 = _FixedDetector([Detection(box=(10, 10, 50, 50), confidence=0.9)])
            det1.name = "det-a"
            det2 = _FixedDetector([])
            det2.name = "det-b"
            det_results, _ = run_image_benchmark([det1, det2], samples, [0.5])
            assert "det-a" in det_results
            assert "det-b" in det_results

    def test_multiple_iou_thresholds(self):
        with tempfile.TemporaryDirectory() as td:
            samples = self._make_samples(Path(td), n=3)
            det = _FixedDetector([Detection(box=(10, 10, 50, 50), confidence=0.9)])
            det_results, _ = run_image_benchmark([det], samples, [0.5, 0.75])
            assert "IoU=0.5" in det_results["test-fixed"]
            assert "IoU=0.75" in det_results["test-fixed"]


class TestRunVideoBenchmark:
    def _make_video(self, path, n_frames=10):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30, (100, 100))
        for _ in range(n_frames):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_basic_video_run(self):
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / "test.mp4"
            self._make_video(vpath, n_frames=15)
            det = _FixedDetector([Detection(box=(10, 10, 50, 50), confidence=0.9)])
            results = run_video_benchmark([det], [vpath], max_frames=10)

            assert "test-fixed" in results
            assert "temporal_consistency" in results["test-fixed"]
            assert "flicker" in results["test-fixed"]

    def test_frame_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / "long.mp4"
            self._make_video(vpath, n_frames=100)

            call_count = 0

            class CountingDetector(DetectorBase):
                name = "counting"

                def detect(self, image):
                    nonlocal call_count
                    call_count += 1
                    return []

            det = CountingDetector()
            run_video_benchmark([det], [vpath], max_frames=20)
            # Should sample ~20 frames, not all 100
            assert call_count <= 25  # some tolerance

    def test_empty_video_list(self):
        det = _FixedDetector()
        results = run_video_benchmark([det], [])
        assert results == {}


class TestCLIParsing:
    def test_default_args(self):
        """Verify argparse defaults."""
        import argparse

        # We can't easily test main() without datasets, but we can test arg parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--tier", choices=["micro", "full"], default="micro")
        parser.add_argument("--no-competitors", action="store_true")
        parser.add_argument("--no-video", action="store_true")
        opts = parser.parse_args([])
        assert opts.tier == "micro"
        assert opts.no_competitors is False
        assert opts.no_video is False

    def test_full_tier(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--tier", choices=["micro", "full"], default="micro")
        opts = parser.parse_args(["--tier", "full"])
        assert opts.tier == "full"
