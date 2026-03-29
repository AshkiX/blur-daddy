"""Base dataset interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ImageSample:
    """A single image with ground truth annotations."""

    path: Path
    ground_truth_boxes: list[tuple[float, float, float, float]]  # x1, y1, x2, y2
    difficulty: str = "unknown"  # easy, medium, hard, unknown


@dataclass
class VideoSample:
    """A video clip with keyframe annotations."""

    path: Path
    annotated_frames: dict[int, list[tuple[float, float, float, float]]] = field(default_factory=dict)
    # frame_idx -> list of boxes
    face_ids: dict[int, list[str]] = field(default_factory=dict)
    # frame_idx -> list of face IDs (for temporal tracking)


class ImageDataset(Protocol):
    """Protocol for image benchmark datasets."""

    name: str

    def setup(self) -> None:
        """Download / prepare the dataset."""
        ...

    def is_ready(self) -> bool:
        """Check if dataset is available locally."""
        ...

    def get_samples(self, limit: int | None = None) -> list[ImageSample]:
        """Return image samples up to limit."""
        ...


class VideoDataset(Protocol):
    """Protocol for video benchmark datasets."""

    name: str

    def setup(self) -> None:
        ...

    def is_ready(self) -> bool:
        ...

    def get_samples(self, limit: int | None = None) -> list[VideoSample]:
        ...
