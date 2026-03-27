"""Tests for CLI end-to-end processing."""

import os
import subprocess
import sys

import cv2

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BLUR_FACES = os.path.join(PROJECT_ROOT, "main", "blur_faces.py")
PYTHON = sys.executable

YOLO_ARGS = ["--model", "yolov8n-face"]


def _run_cli(args, timeout=120):
    """Run blur_faces.py with PYTHONPATH set to project root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(PROJECT_ROOT)
    return subprocess.run(
        [PYTHON, BLUR_FACES] + args,
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=env, timeout=timeout,
    )


def _blur_args(input_path, output_path, method="gaussian"):
    return ["--input", input_path, "--output", output_path, "--method", method] + YOLO_ARGS


class TestCLIImageProcessing:
    def test_gaussian_blur_produces_output(self, sample_image_path, tmp_output_path):
        result = _run_cli(_blur_args(sample_image_path, tmp_output_path, "gaussian"))
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_pixelation_produces_output(self, sample_image_path, tmp_output_path):
        result = _run_cli(_blur_args(sample_image_path, tmp_output_path, "pixelation"))
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_elliptical_produces_output(self, sample_image_path, tmp_output_path):
        result = _run_cli(_blur_args(sample_image_path, tmp_output_path, "elliptical"))
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_mtcnn_model_produces_output(self, sample_image_path, tmp_output_path):
        result = _run_cli(
            ["--input", sample_image_path, "--output", tmp_output_path,
             "--method", "gaussian", "--model", "mtcnn"],
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_output_is_valid_image(self, sample_image_path, tmp_output_path):
        _run_cli(_blur_args(sample_image_path, tmp_output_path))
        img = cv2.imread(tmp_output_path)
        assert img is not None
        assert len(img.shape) == 3


class TestCLIVideoProcessing:
    def test_video_produces_output(self, sample_video_path, tmp_video_output_path):
        result = _run_cli(
            _blur_args(sample_video_path, tmp_video_output_path), timeout=300,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_video_output_path)


class TestCLIErrorHandling:
    def test_invalid_input_file(self, tmp_output_path):
        result = _run_cli(
            ["--input", "/nonexistent/file.jpg", "--output", tmp_output_path],
            timeout=30,
        )
        assert result.returncode != 0

    def test_unsupported_file_type(self, tmp_path, tmp_output_path):
        bad_file = str(tmp_path / "test.txt")
        with open(bad_file, "w") as f:
            f.write("not an image")
        result = _run_cli(
            ["--input", bad_file, "--output", tmp_output_path], timeout=30,
        )
        assert result.returncode != 0
