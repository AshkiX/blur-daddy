"""Tests for benchmark detector wrappers."""

import numpy as np
from benchmarks.detectors.base import Detection, DetectorBase


class TestDetection:
    def test_creation(self):
        d = Detection(box=(0, 0, 10, 10), confidence=0.9)
        assert d.box == (0, 0, 10, 10)
        assert d.confidence == 0.9

    def test_equality(self):
        d1 = Detection(box=(0, 0, 10, 10), confidence=0.9)
        d2 = Detection(box=(0, 0, 10, 10), confidence=0.9)
        assert d1 == d2


class TestDetectorBase:
    def test_warmup_calls_detect(self):
        class FakeDetector(DetectorBase):
            name = "fake"
            call_count = 0

            def detect(self, image):
                self.call_count += 1
                return []

        det = FakeDetector()
        det.warmup()
        assert det.call_count == 1


class TestBlurDaddyYOLO:
    def test_detect_returns_detections(self):
        from benchmarks.detectors.blur_daddy_yolo import BlurDaddyYOLO

        det = BlurDaddyYOLO()
        # Use a real-ish image so YOLO doesn't crash
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = det.detect(img)
        assert isinstance(result, list)
        for d in result:
            assert isinstance(d, Detection)
            assert len(d.box) == 4
            assert 0 <= d.confidence <= 1

    def test_detect_blank_image(self):
        from benchmarks.detectors.blur_daddy_yolo import BlurDaddyYOLO

        det = BlurDaddyYOLO()
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        result = det.detect(img)
        assert isinstance(result, list)


class TestBlurDaddyMTCNN:
    def test_detect_returns_detections(self):
        from benchmarks.detectors.blur_daddy_mtcnn import BlurDaddyMTCNN

        det = BlurDaddyMTCNN()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = det.detect(img)
        assert isinstance(result, list)
        for d in result:
            assert isinstance(d, Detection)

    def test_detect_blank_image(self):
        from benchmarks.detectors.blur_daddy_mtcnn import BlurDaddyMTCNN

        det = BlurDaddyMTCNN()
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        result = det.detect(img)
        assert isinstance(result, list)


class TestGetAllDetectors:
    def test_loads_blur_daddy_detectors(self):
        from benchmarks.run import get_all_detectors

        dets, errors = get_all_detectors(include_competitors=False)
        names = [d.name for d in dets]
        assert "blur-daddy-yolo" in names
        assert "blur-daddy-mtcnn" in names
        assert len(dets) == 2

    def test_competitors_skipped_gracefully(self):
        from benchmarks.run import get_all_detectors

        dets, errors = get_all_detectors(include_competitors=True)
        # blur-daddy detectors should always be present
        names = [d.name for d in dets]
        assert "blur-daddy-yolo" in names
        # Competitors may or may not be available; errors should be populated for missing ones
        assert isinstance(errors, dict)
