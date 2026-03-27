"""Tests for image I/O utilities."""

import os

import numpy as np
import pytest

from utils.image_utils import read_image, resize_image, save_image


class TestReadImage:
    def test_reads_sample_image(self, sample_image_path):
        img = read_image(sample_image_path)
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3

    def test_returns_none_for_missing_file(self):
        img = read_image("/nonexistent/path.jpg")
        assert img is None


class TestSaveImage:
    def test_saves_and_reads_back(self, tmp_output_path):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        save_image(img, tmp_output_path)
        assert os.path.exists(tmp_output_path)
        loaded = read_image(tmp_output_path)
        assert loaded is not None
        assert loaded.shape == img.shape


class TestResizeImage:
    def test_resize_by_width(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        resized = resize_image(img, width=200)
        assert resized.shape[1] == 200
        assert resized.shape[0] == 100  # aspect ratio preserved

    def test_resize_by_height(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        resized = resize_image(img, height=100)
        assert resized.shape[0] == 100
        assert resized.shape[1] == 200  # aspect ratio preserved

    def test_no_resize_returns_same(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = resize_image(img)
        assert np.array_equal(result, img)

    def test_both_dimensions_raises(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_image(img, width=100, height=100)
