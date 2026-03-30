"""Visual regression tests comparing current output against golden references.

Golden images must be generated first:
    python tests/generate_goldens.py

Uses SSIM from scikit-image if available, otherwise falls back to
mean absolute pixel difference.
"""

import os

import cv2
import numpy as np
import pytest

from blur_daddy import BlurDaddy

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
SSIM_THRESHOLD = 0.95
MAD_THRESHOLD = 2.0  # mean absolute difference per pixel channel (0-255 scale)

# Try to import SSIM from scikit-image
try:
    from skimage.metrics import structural_similarity as ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def load_golden(name: str) -> np.ndarray:
    """Load a golden PNG as an RGB numpy array, or None if missing."""
    path = os.path.join(GOLDEN_DIR, name)
    if not os.path.exists(path):
        return None
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def assert_images_match(actual: np.ndarray, golden: np.ndarray, label: str) -> None:
    """Assert that actual image closely matches the golden reference."""
    assert actual.shape == golden.shape, (
        f"{label}: shape mismatch {actual.shape} vs {golden.shape}"
    )

    if HAS_SKIMAGE:
        # Use SSIM (more perceptually meaningful)
        score = ssim(golden, actual, channel_axis=2)
        assert score > SSIM_THRESHOLD, (
            f"{label}: SSIM {score:.4f} below threshold {SSIM_THRESHOLD}"
        )
    else:
        # Fallback: mean absolute difference per pixel channel
        mad = np.mean(np.abs(actual.astype(float) - golden.astype(float)))
        assert mad < MAD_THRESHOLD, (
            f"{label}: mean absolute diff {mad:.4f} exceeds threshold {MAD_THRESHOLD}"
        )


SKIP_MSG = "Golden file missing. Run: python tests/generate_goldens.py"


class TestDetectRegression:
    def test_detect_plot(self, sample_image_path):
        golden = load_golden("detect.png")
        if golden is None:
            pytest.skip(SKIP_MSG)

        bd = BlurDaddy()
        result = bd.detect(sample_image_path)
        actual = result.plot()
        assert_images_match(actual, golden, "detect")


BLUR_METHODS = ["gaussian", "pixelation", "elliptical"]


class TestBlurRegression:
    @pytest.mark.parametrize("method", BLUR_METHODS)
    def test_blur_method(self, sample_image_path, method):
        golden = load_golden(f"blur_{method}.png")
        if golden is None:
            pytest.skip(SKIP_MSG)

        bd = BlurDaddy(method=method)
        result = bd.blur(sample_image_path)
        assert_images_match(result.image, golden, f"blur_{method}")


class TestKeepRegression:
    def test_blur_keep(self, sample_image_path):
        golden = load_golden("blur_keep.png")
        if golden is None:
            pytest.skip(SKIP_MSG)

        bd = BlurDaddy()
        faces = bd.detect(sample_image_path).faces
        if not faces:
            pytest.skip("No faces detected in sample image")

        result = bd.blur(sample_image_path, keep=[faces[0]])
        assert_images_match(result.image, golden, "blur_keep")
