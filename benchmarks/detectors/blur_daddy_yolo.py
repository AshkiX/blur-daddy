"""Wrapper for blur-daddy's YOLOv8n-face detector."""

from __future__ import annotations

import numpy as np

from benchmarks.detectors.base import Detection, DetectorBase


class BlurDaddyYOLO(DetectorBase):
    name = "blur-daddy-yolo"

    def detect(self, image: np.ndarray) -> list[Detection]:
        from blur_daddy.detection import detect_faces_yolo

        boxes, confs, _ = detect_faces_yolo(image)
        if boxes is None:
            return []
        return [
            Detection(box=tuple(b), confidence=c)
            for b, c in zip(boxes, confs)
        ]
