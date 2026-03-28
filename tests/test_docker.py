"""Tests for Docker build and execution."""

import os
import subprocess

import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.docker
class TestDocker:
    def test_docker_builds(self):
        result = subprocess.run(
            ["docker", "build", "-t", "blur-daddy-test", "."],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=600,
        )
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

    def test_docker_help(self):
        result = subprocess.run(
            ["docker", "run", "--rm", "blur-daddy-test", "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "--input" in result.stdout

    def test_docker_blurs_image(self, sample_image_path, tmp_path):
        output_path = str(tmp_path / "docker_output.jpg")
        sample_dir = os.path.dirname(os.path.abspath(sample_image_path))
        sample_name = os.path.basename(sample_image_path)
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{sample_dir}:/input:ro",
                "-v", f"{str(tmp_path)}:/output",
                "blur-daddy-test",
                "--input", f"/input/{sample_name}",
                "--output", "/output/docker_output.jpg",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"Docker run failed: {result.stderr}"
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
