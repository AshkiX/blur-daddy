"""Tests for benchmarks/metrics/performance.py — FPS and memory tracking."""

import numpy as np
from benchmarks.metrics.performance import MemoryTracker, measure_fps


class TestMemoryTracker:
    def test_start_stop(self):
        mt = MemoryTracker()
        mt.start()
        delta = mt.stop()
        assert isinstance(delta, float)

    def test_peak_mb_property(self):
        mt = MemoryTracker()
        mt.start()
        _ = [i for i in range(10000)]
        mt.stop()
        assert mt.peak_mb > 0

    def test_double_stop(self):
        """Calling stop twice shouldn't crash."""
        mt = MemoryTracker()
        mt.start()
        mt.stop()
        mt.stop()  # should not raise


class TestMeasureFPS:
    def test_basic_fps(self):
        class DummyDetector:
            def detect(self, image):
                return []

        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
        result = measure_fps(DummyDetector(), images, warmup_runs=1)

        assert result["fps"] > 0
        assert result["num_images"] == 10
        assert result["total_time_s"] >= 0
        assert result["avg_ms_per_image"] >= 0

    def test_single_image(self):
        class DummyDetector:
            def detect(self, image):
                return []

        images = [np.zeros((50, 50, 3), dtype=np.uint8)]
        result = measure_fps(DummyDetector(), images, warmup_runs=0)
        assert result["num_images"] == 1
        assert result["fps"] > 0

    def test_empty_images(self):
        class DummyDetector:
            def detect(self, image):
                return []

        result = measure_fps(DummyDetector(), [], warmup_runs=0)
        assert result["num_images"] == 0
