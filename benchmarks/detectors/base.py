"""Common detector interface for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """A single face detection."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class Detector(Protocol):
    """Protocol all benchmark detectors must implement."""

    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR numpy image. Returns list of Detections."""
        ...

    def warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        ...


class DetectorBase:
    """Base class with shared warmup implementation."""

    name: str = ""

    def warmup(self) -> None:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.detect(dummy)
