"""Tests for video processing utilities."""

import os

import numpy as np
import pytest

from utils.video_utils import extract_frames, get_video_metadata, write_video


class TestExtractFrames:
    def test_returns_correct_frame_count(self, tiny_video):
        frames = extract_frames(tiny_video)
        assert len(frames) == 3

    def test_frames_are_numpy_arrays(self, tiny_video):
        frames = extract_frames(tiny_video)
        for frame in frames:
            assert isinstance(frame, np.ndarray)

    def test_frames_have_correct_shape(self, tiny_video):
        frames = extract_frames(tiny_video)
        for frame in frames:
            assert frame.shape == (200, 200, 3)

    def test_with_real_video(self, sample_video_path):
        frames = extract_frames(sample_video_path)
        assert len(frames) > 0

    def test_invalid_path_returns_empty(self):
        frames = extract_frames("/nonexistent/video.mp4")
        assert frames == []


class TestGetVideoMetadata:
    def test_returns_fps_and_size(self, tiny_video):
        fps, size = get_video_metadata(tiny_video)
        assert fps == pytest.approx(10.0, abs=1.0)
        assert size == (200, 200)

    def test_with_real_video(self, sample_video_path):
        fps, size = get_video_metadata(sample_video_path)
        assert fps > 0
        assert size[0] > 0 and size[1] > 0


class TestWriteVideo:
    def test_produces_valid_file(self, tmp_video_output_path):
        frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(5)]
        write_video(frames, tmp_video_output_path, 10, (200, 200))
        assert os.path.exists(tmp_video_output_path)
        assert os.path.getsize(tmp_video_output_path) > 0

    def test_preserves_frame_count(self, tmp_video_output_path):
        frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(5)]
        write_video(frames, tmp_video_output_path, 10, (200, 200))
        result_frames = extract_frames(tmp_video_output_path)
        assert len(result_frames) == 5

    def test_preserves_resolution(self, tmp_video_output_path):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        write_video(frames, tmp_video_output_path, 30, (640, 480))
        _, size = get_video_metadata(tmp_video_output_path)
        assert size == (640, 480)
