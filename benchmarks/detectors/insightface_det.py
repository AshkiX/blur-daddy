"""Wrappers for insightface detectors (RetinaFace, SCRFD)."""

from __future__ import annotations

import numpy as np

from benchmarks.detectors.base import Detection, DetectorBase


class _InsightFaceDetector(DetectorBase):
    """Base class for insightface-based detectors."""

    def __init__(self, model_name: str):
        self._model = None
        self._model_name = model_name

    def _get_model(self):
        if self._model is None:
            try:
                from insightface.app import FaceAnalysis
                self._model = FaceAnalysis(
                    name=self._model_name,
                    allowed_modules=["detection"],
                    providers=["CPUExecutionProvider"],
                )
                self._model.prepare(ctx_id=-1, det_size=(640, 640))
            except ImportError:
                raise ImportError(
                    "insightface not installed. Install with: pip install insightface onnxruntime"
                )
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        model = self._get_model()
        faces = model.get(image)
        return [
            Detection(box=tuple(f.bbox.tolist()), confidence=float(f.det_score))
            for f in faces
        ]



class RetinaFaceDetector(_InsightFaceDetector):
    name = "retinaface"

    def __init__(self):
        super().__init__("buffalo_l")


class SCRFDDetector(_InsightFaceDetector):
    name = "scrfd"

    def __init__(self):
        super().__init__("buffalo_sc")
