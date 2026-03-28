"""High-level API for blur-daddy."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blur_daddy.blur import (
    apply_elliptical_gaussian_blur,
    apply_rect_gaussian_blur,
    apply_rect_pixelation,
)
from blur_daddy.detection import detect_faces_mtcnn, detect_faces_yolo
from blur_daddy.models import BlurResult, Detection, DetectionResult, Face

VALID_MODELS = ("yolov8n-face", "mtcnn")
VALID_METHODS = ("gaussian", "pixelation", "elliptical")
SUPPORTED_TARGETS = ("faces",)


def _load_image(source: str | Path | np.ndarray) -> np.ndarray:
    """Load an image and return it as RGB numpy array."""
    if isinstance(source, np.ndarray):
        # Assume BGR from cv2, convert to RGB
        return cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    path = str(source)
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class BlurDaddy:
    """Main interface for face detection and blurring."""

    def __init__(
        self,
        *,
        model: str = "yolov8n-face",
        method: str = "gaussian",
    ):
        if model not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Choose from: {VALID_MODELS}")
        if method not in VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from: {VALID_METHODS}")
        self.model = model
        self.method = method

    def detect(
        self,
        source: str | Path | np.ndarray,
        *,
        targets: tuple[str, ...] | list[str] = ("faces",),
    ) -> DetectionResult:
        """Detect objects in an image.

        Args:
            source: Image path, Path object, or BGR numpy array.
            targets: What to detect. Currently only "faces" is supported.

        Returns:
            DetectionResult with detected objects.
        """
        for t in targets:
            if t not in SUPPORTED_TARGETS:
                raise NotImplementedError(
                    f"Target '{t}' is not yet supported. Available: {SUPPORTED_TARGETS}"
                )

        image_rgb = _load_image(source)
        detections = self._detect_faces(image_rgb)
        return DetectionResult(image=image_rgb, detections=detections)

    def blur(
        self,
        source: str | Path | np.ndarray,
        output: str | Path | None = None,
        *,
        targets: tuple[str, ...] | list[str] = ("faces",),
        keep: list[Detection] | None = None,
        track: bool = False,
        on_progress=None,
    ) -> BlurResult:
        """Blur detected objects in an image.

        Args:
            source: Image path, Path object, or BGR numpy array.
            output: Optional output path. If provided, saves automatically.
            targets: What to blur. Currently only "faces" is supported.
            keep: List of Detection objects to protect from blurring.
            track: Enable face tracking across video frames (not yet implemented).
            on_progress: Progress callback (not yet implemented).

        Returns:
            BlurResult with blurred image and detection info.
        """
        if track:
            raise NotImplementedError("Face tracking is not yet implemented.")

        for t in targets:
            if t not in SUPPORTED_TARGETS:
                raise NotImplementedError(
                    f"Target '{t}' is not yet supported. Available: {SUPPORTED_TARGETS}"
                )

        image_rgb = _load_image(source)
        detections = self._detect_faces(image_rgb)

        # Filter out kept detections
        keep_ids = {d.id for d in keep} if keep else set()
        to_blur = [d for d in detections if d.id not in keep_ids]

        # Apply blur (blur functions work on any color space — they just blur pixels)
        blurred = self._apply_blur(image_rgb, to_blur)

        result = BlurResult(image=blurred, detections=detections)
        if output is not None:
            result.save(output)
        return result

    def _detect_faces(self, image_rgb: np.ndarray) -> list[Face]:
        """Run face detection and return Face objects."""
        if self.model == "yolov8n-face":
            boxes, confs, _ = detect_faces_yolo(image_rgb)
            landmarks_list = None
        elif self.model == "mtcnn":
            boxes, confs, landmarks_list = detect_faces_mtcnn(image_rgb)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        if boxes is None:
            return []

        faces = []
        for i, box in enumerate(boxes):
            conf = float(confs[i]) if confs is not None else 0.0
            lm = landmarks_list[i] if landmarks_list is not None else None
            faces.append(
                Face(
                    id=f"face-{i}",
                    target_type="face",
                    box=tuple(float(c) for c in box),
                    confidence=conf,
                    landmarks=lm,
                )
            )
        return faces

    def _apply_blur(self, image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Apply blur to detected regions."""
        if not detections:
            return image_rgb.copy()

        boxes = [list(d.box) for d in detections]
        landmarks = None
        if self.method == "elliptical":
            landmarks = [d.landmarks if isinstance(d, Face) else None for d in detections]

        result = image_rgb.copy()
        if self.method == "gaussian":
            result = apply_rect_gaussian_blur(result, boxes)
        elif self.method == "pixelation":
            result = apply_rect_pixelation(result, boxes)
        elif self.method == "elliptical":
            result = apply_elliptical_gaussian_blur(result, boxes, landmarks)

        return result


def blur(
    source: str | Path | np.ndarray,
    output: str | Path | None = None,
    *,
    method: str = "gaussian",
    model: str = "yolov8n-face",
    targets: tuple[str, ...] | list[str] = ("faces",),
    keep: Optional[list[Detection]] = None,
) -> BlurResult:
    """Convenience function: detect and blur in one call.

    Args:
        source: Image path or numpy array.
        output: Optional output path.
        method: Blur method (gaussian, pixelation, elliptical).
        model: Detection model (yolov8n-face, mtcnn).
        targets: What to blur.
        keep: Detections to protect.

    Returns:
        BlurResult with blurred image.
    """
    bd = BlurDaddy(model=model, method=method)
    return bd.blur(source, output, targets=targets, keep=keep)
