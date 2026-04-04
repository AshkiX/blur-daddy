"""YouTube Faces DB loader (requires manual download due to license agreement)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.config import YTFACES_DIR
from benchmarks.datasets.base import VideoSample


class YouTubeFacesDataset:
    name = "youtube-faces"

    def __init__(self, root: Path | None = None):
        self.root = root or YTFACES_DIR

    def is_ready(self) -> bool:
        return self.root.is_dir() and any(self.root.iterdir())

    def setup(self) -> None:
        print(
            "YouTube Faces DB requires a license agreement.\n"
            "1. Visit: https://www.cs.tau.ac.il/~wolf/ytfaces/\n"
            "2. Request access and download the dataset\n"
            f"3. Extract to: {self.root}\n"
            "   Or set YTFACES_DIR env var to point to your download.\n"
        )

    def get_samples(self, limit: int | None = None) -> list[VideoSample]:
        if not self.is_ready():
            return []

        samples = []
        # YouTube Faces structure: person_name/video_num/frames/
        for person_dir in sorted(self.root.iterdir()):
            if not person_dir.is_dir():
                continue
            for video_dir in sorted(person_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                # Check for frame images or video files
                frames = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
                if frames:
                    samples.append(VideoSample(path=video_dir))
                if limit and len(samples) >= limit:
                    return samples

        return samples
