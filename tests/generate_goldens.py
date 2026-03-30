"""Generate golden reference images for visual regression tests.

Usage:
    python tests/generate_goldens.py

Generates PNG files in tests/golden/ that serve as the baseline for
visual regression tests in test_regression.py.
"""

import os
import sys

import cv2
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blur_daddy import BlurDaddy

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "..", "sample_images", "sample1.jpg")


def save_rgb_as_png(image: np.ndarray, name: str) -> None:
    """Save an RGB numpy array as a PNG file in the golden directory."""
    path = os.path.join(GOLDEN_DIR, name)
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    print(f"  Saved {path}")


def main() -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    if not os.path.exists(SAMPLE_IMAGE):
        print(f"ERROR: Sample image not found: {SAMPLE_IMAGE}")
        sys.exit(1)

    bd = BlurDaddy()
    print(f"Using sample image: {SAMPLE_IMAGE}")

    # 1. Detection plot
    print("Generating detect golden...")
    det_result = bd.detect(SAMPLE_IMAGE)
    detect_plot = det_result.plot()
    save_rgb_as_png(detect_plot, "detect.png")

    # 2. Blur with each method
    for method in ("gaussian", "pixelation", "elliptical"):
        print(f"Generating blur_{method} golden...")
        bd_m = BlurDaddy(method=method)
        blur_result = bd_m.blur(SAMPLE_IMAGE)
        save_rgb_as_png(blur_result.image, f"blur_{method}.png")

    # 3. Blur with keep (keep first detected face unblurred)
    print("Generating blur_keep golden...")
    faces = det_result.faces
    if faces:
        keep_result = bd.blur(SAMPLE_IMAGE, keep=[faces[0]])
        save_rgb_as_png(keep_result.image, "blur_keep.png")
    else:
        print("  WARNING: No faces detected, skipping blur_keep golden")

    print("Done! Golden images saved to tests/golden/")


if __name__ == "__main__":
    main()
