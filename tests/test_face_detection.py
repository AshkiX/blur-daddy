"""Tests for face detection (YOLO and MTCNN)."""

import cv2

from utils.face_utils import detect_faces_mtcnn, detect_faces_yolo


class TestYOLODetection:
    def test_detects_faces_in_sample_image(self, sample_image_rgb):
        boxes, confs, _ = detect_faces_yolo(sample_image_rgb)
        assert boxes is not None, "YOLO should detect at least one face in sample image"
        assert len(boxes) > 0

    def test_boxes_are_valid_coordinates(self, sample_image_rgb):
        boxes, _, _ = detect_faces_yolo(sample_image_rgb)
        assert boxes is not None
        for box in boxes:
            x1, y1, x2, y2 = box
            assert x1 < x2, "x1 should be less than x2"
            assert y1 < y2, "y1 should be less than y2"

    def test_confidences_are_valid(self, sample_image_rgb):
        _, confs, _ = detect_faces_yolo(sample_image_rgb)
        assert confs is not None
        for conf in confs:
            assert 0.0 <= conf <= 1.0

    def test_no_faces_in_blank_image(self, blank_image):
        image_rgb = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        boxes, confs, _ = detect_faces_yolo(image_rgb)
        assert boxes is None
        assert confs is None

    def test_returns_three_element_tuple(self, sample_image_rgb):
        result = detect_faces_yolo(sample_image_rgb)
        assert len(result) == 3, "Should return (boxes, confs, landmarks)"


class TestMTCNNDetection:
    def test_detects_faces_in_sample_image(self, sample_image_rgb):
        boxes, probs, landmarks = detect_faces_mtcnn(sample_image_rgb)
        assert boxes is not None, "MTCNN should detect at least one face in sample image"
        assert len(boxes) > 0

    def test_returns_landmarks(self, sample_image_rgb):
        _, _, landmarks = detect_faces_mtcnn(sample_image_rgb)
        assert landmarks is not None, "MTCNN should return landmarks"

    def test_no_faces_in_blank_image(self, blank_image):
        image_rgb = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = detect_faces_mtcnn(image_rgb)
        assert boxes is None
