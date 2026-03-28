"""Data models for detection results and blur output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """A detected region in an image."""

    id: str
    target_type: str
    box: tuple[float, float, float, float]
    confidence: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def box_int(self) -> tuple[int, int, int, int]:
        return tuple(int(c) for c in self.box)


@dataclass
class Face(Detection):
    """A detected face."""

    landmarks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.target_type = "face"


@dataclass
class DetectionResult:
    """Result from BlurDaddy.detect()."""

    image: np.ndarray = field(repr=False)
    detections: list[Detection] = field(default_factory=list)

    @property
    def faces(self) -> list[Face]:
        return [d for d in self.detections if isinstance(d, Face)]

    def save(self, path: str | Path) -> None:
        """Save annotated image with detection boxes drawn."""
        annotated = self.image.copy()
        for det in self.detections:
            x1, y1, x2, y2 = det.box_int
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.id} ({det.confidence:.2f})"
            cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(str(path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


@dataclass
class BlurResult:
    """Result from BlurDaddy.blur()."""

    image: np.ndarray = field(repr=False)
    detections: list[Detection] = field(default_factory=list)

    @property
    def faces(self) -> list[Face]:
        return [d for d in self.detections if isinstance(d, Face)]

    def save(self, path: str | Path) -> None:
        """Save the blurred image or video frames."""
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(str(path), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
