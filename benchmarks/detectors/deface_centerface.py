"""Wrapper for deface's CenterFace detector."""

from __future__ import annotations

import numpy as np

from benchmarks.detectors.base import Detection, DetectorBase


class DefaceCenterFace(DetectorBase):
    name = "deface-centerface"

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from deface.centerface import CenterFace
                self._model = CenterFace()
            except ImportError:
                raise ImportError(
                    "deface not installed. Install with: pip install deface"
                )
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        model = self._get_model()
        dets, _ = model(image, threshold=0.5)
        return [
            Detection(box=(d[0], d[1], d[2], d[3]), confidence=float(d[4]))
            for d in dets
        ]

