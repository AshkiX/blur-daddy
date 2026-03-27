"""Tests for blur algorithms."""

import numpy as np
import pytest

from utils.blur_utils import (
    apply_elliptical_gaussian_blur,
    apply_rect_gaussian_blur,
    apply_rect_pixelation,
    blend_images,
    get_padded_clamped_box,
)
from utils.face_utils import get_face_angle

RNG = np.random.default_rng(42)


@pytest.fixture
def image_with_box():
    """An image and a bounding box in the center."""
    img = RNG.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    boxes = [[100, 100, 200, 200]]
    return img.copy(), boxes


@pytest.fixture
def image_with_edge_box():
    """An image with a bounding box near the edge (tests clamping)."""
    img = RNG.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    boxes = [[0, 0, 50, 50]]
    return img.copy(), boxes


class TestGaussianBlur:
    def test_changes_pixels_in_face_region(self, image_with_box):
        original, boxes = image_with_box
        before = original.copy()
        result = apply_rect_gaussian_blur(original, boxes)
        x1, y1, x2, y2 = get_padded_clamped_box(result.shape, boxes[0])
        assert not np.array_equal(before[y1:y2, x1:x2], result[y1:y2, x1:x2])

    def test_preserves_pixels_outside_region(self, image_with_box):
        original, boxes = image_with_box
        before = original.copy()
        result = apply_rect_gaussian_blur(original, boxes)
        x1, y1, x2, y2 = get_padded_clamped_box(result.shape, boxes[0])
        # Check pixels above the blurred region are unchanged
        assert np.array_equal(before[:y1, :], result[:y1, :])

    def test_preserves_image_dimensions(self, image_with_box):
        original, boxes = image_with_box
        result = apply_rect_gaussian_blur(original, boxes)
        assert result.shape == (300, 300, 3)

    def test_handles_edge_box(self, image_with_edge_box):
        original, boxes = image_with_edge_box
        result = apply_rect_gaussian_blur(original, boxes)
        assert result.shape == original.shape

    def test_handles_multiple_boxes(self):
        img = RNG.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        before = img.copy()
        boxes = [[50, 50, 100, 100], [150, 150, 250, 250]]
        result = apply_rect_gaussian_blur(img, boxes)
        # Verify both regions were blurred
        for box in boxes:
            x1, y1, x2, y2 = get_padded_clamped_box(result.shape, box)
            assert not np.array_equal(before[y1:y2, x1:x2], result[y1:y2, x1:x2])

    def test_empty_boxes_returns_unchanged(self):
        img = RNG.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        before = img.copy()
        result = apply_rect_gaussian_blur(img, [])
        assert np.array_equal(before, result)


class TestPixelation:
    def test_changes_pixels_in_face_region(self, image_with_box):
        original, boxes = image_with_box
        before = original.copy()
        result = apply_rect_pixelation(original, boxes)
        x1, y1, x2, y2 = get_padded_clamped_box(result.shape, boxes[0])
        assert not np.array_equal(before[y1:y2, x1:x2], result[y1:y2, x1:x2])

    def test_preserves_image_dimensions(self, image_with_box):
        original, boxes = image_with_box
        result = apply_rect_pixelation(original, boxes)
        assert result.shape == (300, 300, 3)

    def test_handles_edge_box(self, image_with_edge_box):
        original, boxes = image_with_edge_box
        result = apply_rect_pixelation(original, boxes)
        assert result.shape == original.shape

    def test_empty_boxes_returns_unchanged(self):
        img = RNG.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        before = img.copy()
        result = apply_rect_pixelation(img, [])
        assert np.array_equal(before, result)


class TestEllipticalBlur:
    def test_changes_pixels_in_face_region(self, image_with_box):
        original, boxes = image_with_box
        before = original.copy()
        result = apply_elliptical_gaussian_blur(original, boxes, None)
        assert not np.array_equal(before, result)

    def test_preserves_image_dimensions(self, image_with_box):
        original, boxes = image_with_box
        result = apply_elliptical_gaussian_blur(original, boxes, None)
        assert result.shape == (300, 300, 3)

    def test_with_landmarks(self, image_with_box):
        original, boxes = image_with_box
        landmarks = [np.array([[120, 130], [180, 130], [150, 160], [130, 180], [170, 180]])]
        result = apply_elliptical_gaussian_blur(original, boxes, landmarks)
        assert result.shape == (300, 300, 3)

    def test_handles_none_landmarks_in_list(self, image_with_box):
        original, boxes = image_with_box
        result = apply_elliptical_gaussian_blur(original, boxes, [None])
        assert result.shape == (300, 300, 3)


class TestBlendImages:
    def test_all_zero_mask_returns_original(self):
        original = RNG.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = blend_images(original, blurred, mask)
        assert np.array_equal(result, original)

    def test_all_255_mask_returns_blurred(self):
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        blurred = RNG.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = blend_images(original, blurred, mask)
        assert np.array_equal(result, blurred)


class TestGetFaceAngle:
    def test_horizontal_eyes_returns_zero(self):
        landmarks = np.array([[100, 100], [200, 100]])
        angle = get_face_angle(landmarks)
        assert angle == pytest.approx(0.0)

    def test_tilted_eyes_returns_positive(self):
        landmarks = np.array([[100, 100], [200, 200]])
        angle = get_face_angle(landmarks)
        assert angle == pytest.approx(45.0)

    def test_tilted_eyes_returns_negative(self):
        landmarks = np.array([[100, 200], [200, 100]])
        angle = get_face_angle(landmarks)
        assert angle == pytest.approx(-45.0)


class TestPaddedClampedBox:
    def test_adds_padding(self):
        x1, y1, x2, y2 = get_padded_clamped_box((500, 500, 3), [100, 100, 200, 200])
        assert x1 == 85  # 100 - 15
        assert y1 == 85
        assert x2 == 215  # 200 + 15
        assert y2 == 215

    def test_clamps_to_image_bounds(self):
        x1, y1, x2, y2 = get_padded_clamped_box((200, 200, 3), [0, 0, 200, 200])
        assert x1 == 0
        assert y1 == 0
        assert x2 == 200
        assert y2 == 200
