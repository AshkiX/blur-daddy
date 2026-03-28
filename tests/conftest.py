import importlib.util
import os

import cv2
import numpy as np
import pytest

# Load milestone report plugin from adjacent file
_spec = importlib.util.spec_from_file_location(
    "milestone_plugin", os.path.join(os.path.dirname(__file__), "milestone_plugin.py")
)
_plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plugin)

pytest_addoption = _plugin.pytest_addoption
pytest_configure = _plugin.pytest_configure
pytest_runtest_makereport = _plugin.pytest_runtest_makereport
pytest_sessionfinish = _plugin.pytest_sessionfinish

from blur_daddy.video import write_video  # noqa: E402

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_images")
SAMPLE_VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_videos")


@pytest.fixture
def tiny_face_image():
    """A small synthetic image with a simple face-like pattern (skin-colored oval).
    Not guaranteed to trigger detection — use sample images for that."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a skin-colored ellipse as a rough face
    cv2.ellipse(img, (100, 100), (40, 55), 0, 0, 360, (180, 200, 220), -1)
    # Draw eyes
    cv2.circle(img, (85, 85), 5, (50, 50, 50), -1)
    cv2.circle(img, (115, 85), 5, (50, 50, 50), -1)
    return img


@pytest.fixture
def blank_image():
    """A plain image with no faces."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_path():
    """Path to a real sample image with faces."""
    path = os.path.join(SAMPLE_IMAGES_DIR, "sample1.jpg")
    if not os.path.exists(path):
        pytest.skip("sample_images/sample1.jpg not found")
    return path


@pytest.fixture
def sample_video_path():
    """Path to a real sample video with faces."""
    path = os.path.join(SAMPLE_VIDEOS_DIR, "sample1.mp4")
    if not os.path.exists(path):
        pytest.skip("sample_videos/sample1.mp4 not found")
    return path


@pytest.fixture
def tiny_video(tmp_path):
    """A 3-frame synthetic video file."""
    video_path = str(tmp_path / "test_video.mp4")
    size = (200, 200)
    frames = [np.full((200, 200, 3), i * 80, dtype=np.uint8) for i in range(3)]
    write_video(frames, video_path, 10, size)
    return video_path


@pytest.fixture
def sample_image_rgb(sample_image_path):
    """Load sample image as RGB numpy array."""
    image = cv2.imread(sample_image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture
def tmp_output_path(tmp_path):
    """A temporary output path for test results."""
    return str(tmp_path / "output.jpg")


@pytest.fixture
def tmp_video_output_path(tmp_path):
    """A temporary output path for video test results."""
    return str(tmp_path / "output.mp4")
