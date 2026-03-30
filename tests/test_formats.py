"""Tests for image and video format support using real sample files."""

import os

import cv2
import pytest

from blur_daddy import BlurDaddy
from blur_daddy.video import extract_frames, get_video_metadata, write_video

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "formats")

IMAGE_FORMATS = [
    ("png", ".png"),
    ("jpg", ".jpg"),
    ("bmp", ".bmp"),
    ("webp", ".webp"),
    ("tiff", ".tiff"),
]


@pytest.mark.parametrize("fmt_name,ext", IMAGE_FORMATS, ids=[f[0] for f in IMAGE_FORMATS])
def test_image_format_blur(tmp_path, fmt_name, ext):
    """Blur a real sample image in each format and verify output."""
    input_path = os.path.join(FIXTURES_DIR, f"sample{ext}")
    if not os.path.exists(input_path):
        pytest.skip(f"Fixture missing: {input_path}")

    original = cv2.imread(input_path)
    assert original is not None, f"Failed to read {fmt_name} fixture"

    bd = BlurDaddy()
    result = bd.blur(input_path)

    assert result.image is not None
    assert result.image.shape[:2] == original.shape[:2]
    assert result.image.shape[2] == 3

    # Save in same format and verify roundtrip
    output_path = str(tmp_path / f"output{ext}")
    result.save(output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    reloaded = cv2.imread(output_path)
    assert reloaded is not None, f"Cannot re-read saved {fmt_name} file"
    assert reloaded.shape[:2] == original.shape[:2]


@pytest.mark.parametrize("fmt_name,ext", IMAGE_FORMATS, ids=[f[0] for f in IMAGE_FORMATS])
def test_image_format_detect(fmt_name, ext):
    """Detect faces in each image format."""
    input_path = os.path.join(FIXTURES_DIR, f"sample{ext}")
    if not os.path.exists(input_path):
        pytest.skip(f"Fixture missing: {input_path}")

    bd = BlurDaddy()
    result = bd.detect(input_path)
    assert len(result.faces) > 0, f"No faces detected in {fmt_name} format"


VIDEO_FORMATS = [
    ("mp4", ".mp4"),
    ("avi", ".avi"),
]


@pytest.mark.parametrize("fmt_name,ext", VIDEO_FORMATS, ids=[f[0] for f in VIDEO_FORMATS])
def test_video_format_roundtrip(tmp_path, fmt_name, ext):
    """Process real video in each format frame-by-frame."""
    input_path = os.path.join(FIXTURES_DIR, f"sample{ext}")
    if not os.path.exists(input_path):
        pytest.skip(f"Fixture missing: {input_path}")

    frames = extract_frames(input_path)
    assert len(frames) > 0, f"No frames extracted from {fmt_name}"

    fps, size = get_video_metadata(input_path)

    bd = BlurDaddy()
    blurred_frames = []
    for frame in frames:
        result = bd.blur(frame)
        blurred_frames.append(cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))

    output_path = str(tmp_path / f"output{ext}")
    write_video(blurred_frames, output_path, int(fps), size)

    assert os.path.exists(output_path), f"Output {fmt_name} video not created"
    assert os.path.getsize(output_path) > 0

    out_frames = extract_frames(output_path)
    assert len(out_frames) == len(frames), "Frame count mismatch"
