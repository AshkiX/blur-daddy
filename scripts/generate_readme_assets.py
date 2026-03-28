"""Generate before/after comparison images for the README."""

import os
import sys

import cv2
import numpy as np

from blur_daddy.blur import (
    apply_elliptical_gaussian_blur,
    apply_rect_gaussian_blur,
    apply_rect_pixelation,
)
from blur_daddy.detection import detect_faces_yolo
from blur_daddy.image import read_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_IMAGE = os.path.join(PROJECT_ROOT, "sample_images", "sample1.jpg")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MAX_WIDTH = 400
LABEL_HEIGHT = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
GAP = 4  # pixels between images in the strip


def resize_to_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    """Add a text label bar below the image."""
    h, w = image.shape[:2]
    bar = np.zeros((LABEL_HEIGHT, w, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    text_x = (w - text_size[0]) // 2
    text_y = LABEL_HEIGHT - (LABEL_HEIGHT - text_size[1]) // 2
    cv2.putText(bar, label, (text_x, text_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    return np.vstack([image, bar])


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Load image
    original = read_image(SAMPLE_IMAGE)
    if original is None:
        print(f"ERROR: Could not load {SAMPLE_IMAGE}")
        sys.exit(1)

    print(f"Loaded image: {original.shape[1]}x{original.shape[0]}")

    # Detect faces once on the original
    boxes, probs, landmarks = detect_faces_yolo(original)
    if boxes is None:
        print("ERROR: No faces detected")
        sys.exit(1)
    print(f"Detected {len(boxes)} face(s)")

    # Apply each blur method on a fresh copy
    gaussian = apply_rect_gaussian_blur(original.copy(), boxes)
    pixelated = apply_rect_pixelation(original.copy(), boxes)
    elliptical = apply_elliptical_gaussian_blur(original.copy(), boxes, landmarks)

    # Build labeled panels
    panels = {
        "Original": original,
        "Gaussian": gaussian,
        "Pixelation": pixelated,
        "Elliptical": elliptical,
    }

    labeled = []
    for label, img in panels.items():
        resized = resize_to_max_width(img, MAX_WIDTH)
        labeled.append(add_label(resized, label))

    # Make all panels the same height (pad shorter ones at bottom)
    max_h = max(p.shape[0] for p in labeled)
    padded = []
    for p in labeled:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    # Add gaps between panels
    gap_col = np.zeros((max_h, GAP, 3), dtype=np.uint8)
    strip_parts = []
    for i, p in enumerate(padded):
        if i > 0:
            strip_parts.append(gap_col)
        strip_parts.append(p)

    strip = np.hstack(strip_parts)

    # Save comparison strip
    strip_path = os.path.join(DOCS_DIR, "comparison.png")
    cv2.imwrite(strip_path, strip)
    print(f"Saved comparison strip: {strip_path}")

    # Save individual before/after pairs for GIF creation
    for label, img in panels.items():
        if label == "Original":
            resized = resize_to_max_width(img, MAX_WIDTH)
            path = os.path.join(DOCS_DIR, "original.png")
            cv2.imwrite(path, resized)
            print(f"Saved: {path}")
        else:
            resized = resize_to_max_width(img, MAX_WIDTH)
            name = label.lower()
            path = os.path.join(DOCS_DIR, f"blurred_{name}.png")
            cv2.imwrite(path, resized)
            print(f"Saved: {path}")

    print("\nDone! All assets saved to docs/")


if __name__ == "__main__":
    main()
