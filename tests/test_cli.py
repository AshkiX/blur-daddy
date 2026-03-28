"""Tests for CLI end-to-end processing."""

import json
import os
import subprocess
import sys

import cv2

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PYTHON = sys.executable

YOLO_ARGS = ["--model", "yolov8n-face"]


def _run_cli(args, timeout=120):
    """Run blur_daddy.cli as a module with src on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(SRC_DIR)
    return subprocess.run(
        [PYTHON, "-m", "blur_daddy.cli"] + args,
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=env, timeout=timeout,
    )


class TestCLIDetect:
    def test_detect_prints_faces(self, sample_image_path):
        result = _run_cli(["detect", sample_image_path] + YOLO_ARGS)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "face-0" in result.stdout

    def test_detect_saves_preview(self, sample_image_path, tmp_output_path):
        result = _run_cli(["detect", sample_image_path, "-o", tmp_output_path] + YOLO_ARGS)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_detect_json_output(self, sample_image_path):
        result = _run_cli(["detect", sample_image_path, "--json"] + YOLO_ARGS)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # JSON should be parseable and contain detection data
        lines = result.stdout.strip().split("\n")
        # Find JSON block (after the text output)
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("["):
                json_start = i
                break
        assert json_start is not None, f"No JSON found in output: {result.stdout}"
        json_text = "\n".join(lines[json_start:])
        data = json.loads(json_text)
        assert len(data) > 0
        assert "id" in data[0]
        assert "box" in data[0]

    def test_detect_no_faces_on_blank(self, tmp_path):
        blank_path = str(tmp_path / "blank.jpg")
        import numpy as np
        cv2.imwrite(blank_path, np.zeros((100, 100, 3), dtype=np.uint8))
        result = _run_cli(["detect", blank_path] + YOLO_ARGS)
        assert result.returncode == 0
        assert "No detections found" in result.stdout


class TestCLIBlurImage:
    def test_gaussian_blur(self, sample_image_path, tmp_output_path):
        result = _run_cli(["blur", sample_image_path, "-o", tmp_output_path, "--method", "gaussian"] + YOLO_ARGS)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_pixelation_blur(self, sample_image_path, tmp_output_path):
        result = _run_cli(["blur", sample_image_path, "-o", tmp_output_path, "--method", "pixelation"] + YOLO_ARGS)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_elliptical_blur(self, sample_image_path, tmp_output_path):
        result = _run_cli(["blur", sample_image_path, "-o", tmp_output_path, "--method", "elliptical"] + YOLO_ARGS)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_mtcnn_model(self, sample_image_path, tmp_output_path):
        result = _run_cli(["blur", sample_image_path, "-o", tmp_output_path, "--model", "mtcnn"])
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_output_is_valid_image(self, sample_image_path, tmp_output_path):
        _run_cli(["blur", sample_image_path, "-o", tmp_output_path] + YOLO_ARGS)
        img = cv2.imread(tmp_output_path)
        assert img is not None
        assert len(img.shape) == 3

    def test_keep_flag(self, sample_image_path, tmp_output_path):
        result = _run_cli(
            ["blur", sample_image_path, "-o", tmp_output_path, "--keep", "face-0"] + YOLO_ARGS,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_output_path)

    def test_keep_unknown_id_warns(self, sample_image_path, tmp_output_path):
        result = _run_cli(
            ["blur", sample_image_path, "-o", tmp_output_path, "--keep", "face-999"] + YOLO_ARGS,
        )
        assert result.returncode == 0
        assert "unknown IDs" in result.stderr or "Warning" in result.stderr


class TestCLIBlurVideo:
    def test_video_produces_output(self, sample_video_path, tmp_video_output_path):
        result = _run_cli(
            ["blur", sample_video_path, "-o", tmp_video_output_path] + YOLO_ARGS, timeout=300,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(tmp_video_output_path)


class TestCLIErrorHandling:
    def test_no_command_shows_help(self):
        result = _run_cli([], timeout=30)
        assert result.returncode != 0

    def test_invalid_input_file(self, tmp_output_path):
        result = _run_cli(
            ["blur", "/nonexistent/file.jpg", "-o", tmp_output_path], timeout=30,
        )
        assert result.returncode != 0

    def test_unsupported_file_type(self, tmp_path, tmp_output_path):
        bad_file = str(tmp_path / "test.txt")
        with open(bad_file, "w") as f:
            f.write("not an image")
        result = _run_cli(
            ["blur", bad_file, "-o", tmp_output_path], timeout=30,
        )
        assert result.returncode != 0
