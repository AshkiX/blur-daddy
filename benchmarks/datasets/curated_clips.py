"""Loader for manually annotated curated video clips."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.config import CURATED_DIR
from benchmarks.datasets.base import VideoSample


class CuratedClipsDataset:
    name = "curated-clips"

    def __init__(self, root: Path | None = None):
        self.root = root or CURATED_DIR

    def is_ready(self) -> bool:
        return self.root.is_dir() and any(self.root.glob("*.json"))

    def setup(self) -> None:
        print(
            "Curated clips require manual setup.\n"
            f"Place video files and annotation JSONs in: {self.root}\n\n"
            "Annotation format (one JSON per clip):\n"
            "{\n"
            '  "video": "clip01.mp4",\n'
            '  "fps": 30,\n'
            '  "frames": [\n'
            '    {"frame_idx": 0, "faces": [{"box": [x1,y1,x2,y2], "id": "face_0"}]},\n'
            "    ...\n"
            "  ]\n"
            "}"
        )

    def get_samples(self, limit: int | None = None) -> list[VideoSample]:
        samples = []
        for annot_path in sorted(self.root.glob("*.json")):
            if annot_path.name == "schema.json":
                continue
            with open(annot_path) as f:
                data = json.load(f)

            if "video" not in data:
                continue

            video_path = self.root / data["video"]
            if not video_path.exists():
                print(f"Warning: video not found: {video_path}")
                continue

            annotated_frames = {}
            face_ids = {}
            for frame in data.get("frames", []):
                idx = frame["frame_idx"]
                boxes = [tuple(face["box"]) for face in frame.get("faces", [])]
                ids = [face.get("id", f"face_{i}") for i, face in enumerate(frame.get("faces", []))]
                annotated_frames[idx] = boxes
                face_ids[idx] = ids

            samples.append(
                VideoSample(path=video_path, annotated_frames=annotated_frames, face_ids=face_ids)
            )
            if limit and len(samples) >= limit:
                break

        return samples
