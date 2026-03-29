"""Wrapper for blur-daddy's MTCNN detector."""

from __future__ import annotations

import numpy as np
from PIL import Image

from benchmarks.detectors.base import Detection, DetectorBase


class BlurDaddyMTCNN(DetectorBase):
    name = "blur-daddy-mtcnn"

    def detect(self, image: np.ndarray) -> list[Detection]:
        from blur_daddy.detection import detect_faces_mtcnn

        pil_image = Image.fromarray(image[..., ::-1])  # BGR -> RGB
        boxes, probs, _ = detect_faces_mtcnn(pil_image)
        if boxes is None:
            return []
        return [
            Detection(box=tuple(b.tolist()), confidence=float(p))
            for b, p in zip(boxes, probs)
        ]
