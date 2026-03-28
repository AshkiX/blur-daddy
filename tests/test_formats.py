"""Tests for image and video format support."""

import os

import cv2
import numpy as np
import pytest

from blur_daddy import BlurDaddy
from blur_daddy.video import extract_frames, get_video_metadata, write_video

IMAGE_FORMATS = [
    ("png", ".png"),
    ("jpg", ".jpg"),
    ("bmp", ".bmp"),
    ("webp", ".webp"),
    ("tiff", ".tiff"),
]


@pytest.mark.parametrize("fmt_name,ext", IMAGE_FORMATS, ids=[f[0] for f in IMAGE_FORMATS])
def test_image_format_roundtrip(sample_image_path, tmp_path, fmt_name, ext):
    """Blur a sample image saved in each format and verify output."""
    # Read original and convert to target format
    original = cv2.imread(sample_image_path)
    assert original is not None, "Failed to read sample image"

    input_path = str(tmp_path / f"input{ext}")
    cv2.imwrite(input_path, original)
    assert os.path.exists(input_path), f"Failed to write {fmt_name} input"

    # Blur
    bd = BlurDaddy()
    result = bd.blur(input_path)

    # Verify result
    assert result.image is not None
    assert result.image.shape[:2] == original.shape[:2], "Output dimensions should match input"
    assert result.image.shape[2] == 3

    # Save in same format and verify
    output_path = str(tmp_path / f"output{ext}")
    result.save(output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0, f"Output {fmt_name} file is empty"

    # Verify the saved file is readable
    reloaded = cv2.imread(output_path)
    assert reloaded is not None, f"Cannot re-read saved {fmt_name} file"
    assert reloaded.shape[:2] == original.shape[:2]


# Video formats: MP4 and AVI are reliably available; others depend on codecs.
VIDEO_FORMATS_RELIABLE = [
    ("mp4", ".mp4", "mp4v"),
    ("avi", ".avi", "XVID"),
]

VIDEO_FORMATS_OPTIONAL = [
    ("mov", ".mov", "mp4v"),
    ("mkv", ".mkv", "X264"),
    ("flv", ".flv", "FLV1"),
    ("wmv", ".wmv", "WMV2"),
    ("webm", ".webm", "VP80"),
]


def _make_synthetic_video(path: str, fourcc_str: str, num_frames: int = 5):
    """Create a small synthetic video with a face-like pattern."""
    size = (200, 200)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, 10, size)
    if not writer.isOpened():
        return False
    for i in range(num_frames):
        frame = np.full((200, 200, 3), 180, dtype=np.uint8)
        # Draw a simple face-like oval
        cv2.ellipse(frame, (100, 100), (40, 55), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(frame, (85, 85), 5, (50, 50, 50), -1)
        cv2.circle(frame, (115, 85), 5, (50, 50, 50), -1)
        writer.write(frame)
    writer.release()
    return os.path.exists(path) and os.path.getsize(path) > 0


@pytest.mark.parametrize(
    "fmt_name,ext,fourcc",
    VIDEO_FORMATS_RELIABLE,
    ids=[f[0] for f in VIDEO_FORMATS_RELIABLE],
)
def test_video_format_roundtrip(tmp_path, fmt_name, ext, fourcc):
    """Process video in MP4/AVI formats frame-by-frame."""
    input_path = str(tmp_path / f"input{ext}")
    assert _make_synthetic_video(input_path, fourcc), f"Failed to create {fmt_name} video"

    # Extract frames and blur each
    frames = extract_frames(input_path)
    assert len(frames) > 0, f"No frames extracted from {fmt_name}"

    bd = BlurDaddy()
    blurred_frames = []
    for frame in frames:
        result = bd.blur(frame)
        # blur returns RGB; convert back to BGR for write_video (OpenCV)
        blurred_frames.append(cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))

    # Write output
    fps, size = get_video_metadata(input_path)
    output_path = str(tmp_path / f"output{ext}")
    write_video(blurred_frames, output_path, int(fps), size)

    assert os.path.exists(output_path), f"Output {fmt_name} video not created"
    assert os.path.getsize(output_path) > 0, f"Output {fmt_name} video is empty"

    # Verify we can read it back
    out_frames = extract_frames(output_path)
    assert len(out_frames) == len(frames), "Frame count mismatch"


@pytest.mark.parametrize(
    "fmt_name,ext,fourcc",
    VIDEO_FORMATS_OPTIONAL,
    ids=[f[0] for f in VIDEO_FORMATS_OPTIONAL],
)
def test_video_format_optional(tmp_path, fmt_name, ext, fourcc):
    """Test optional video formats, skipping if codec is unavailable."""
    input_path = str(tmp_path / f"input{ext}")
    if not _make_synthetic_video(input_path, fourcc):
        pytest.skip(f"Codec for {fmt_name} ({fourcc}) not available")

    frames = extract_frames(input_path)
    if len(frames) == 0:
        pytest.skip(f"Cannot read back {fmt_name} video — codec not fully supported")

    bd = BlurDaddy()
    blurred_frames = []
    for frame in frames:
        result = bd.blur(frame)
        blurred_frames.append(cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))

    fps, size = get_video_metadata(input_path)
    output_path = str(tmp_path / f"output{ext}")
    write_video(blurred_frames, output_path, int(fps), size)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        pytest.skip(f"Writing {fmt_name} ({fourcc}) not supported by this OpenCV build")
