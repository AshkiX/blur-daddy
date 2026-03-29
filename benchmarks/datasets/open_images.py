"""Open Images V7 validation set loader — face bounding boxes.

License: CC BY 4.0 (annotations), individual Flickr licenses (images).
Safe for commercial benchmarking.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

from benchmarks.config import (
    OPEN_IMAGES_BBOX_URL,
    OPEN_IMAGES_DIR,
    OPEN_IMAGES_FACE_LABEL,
    OPEN_IMAGES_IMAGE_IDS_URL,
)
from benchmarks.datasets.base import ImageSample


class OpenImagesDataset:
    name = "open-images-v7-faces"

    def __init__(self, root: Path | None = None):
        self.root = root or OPEN_IMAGES_DIR
        self._annotations: dict[str, list[tuple]] | None = None

    def is_ready(self) -> bool:
        bbox_csv = self.root / "validation-annotations-bbox.csv"
        images_dir = self.root / "images"
        return bbox_csv.is_file() and images_dir.is_dir() and any(images_dir.iterdir())

    def setup(self) -> None:
        """Download annotations and face images from Open Images V7 validation set."""
        self.root.mkdir(parents=True, exist_ok=True)

        # Download bbox annotations
        bbox_csv = self.root / "validation-annotations-bbox.csv"
        if not bbox_csv.exists():
            print("Downloading Open Images V7 validation bbox annotations (~25MB)...")
            urllib.request.urlretrieve(OPEN_IMAGES_BBOX_URL, bbox_csv)

        # Parse face annotations to know which images we need
        face_image_ids = self._get_face_image_ids(bbox_csv)
        print(f"Found {len(face_image_ids)} images with face annotations")

        # Download image ID metadata (for URLs)
        image_ids_csv = self.root / "validation-images-with-rotation.csv"
        if not image_ids_csv.exists():
            print("Downloading image metadata...")
            urllib.request.urlretrieve(OPEN_IMAGES_IMAGE_IDS_URL, image_ids_csv)

        # Download face images
        images_dir = self.root / "images"
        images_dir.mkdir(exist_ok=True)

        # Build URL map from metadata CSV
        url_map = self._parse_image_urls(image_ids_csv)

        # Download images (only those with face annotations)
        to_download = [
            img_id for img_id in face_image_ids
            if not (images_dir / f"{img_id}.jpg").exists()
        ]

        if to_download:
            print(f"Downloading {len(to_download)} face images...")
            for i, img_id in enumerate(to_download):
                url = url_map.get(img_id)
                if not url:
                    continue
                dest = images_dir / f"{img_id}.jpg"
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as e:
                    print(f"  Failed to download {img_id}: {e}")
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i + 1}/{len(to_download)}")

        print(f"Open Images V7 ready at {self.root}")

    def _get_face_image_ids(self, bbox_csv: Path) -> set[str]:
        """Get set of image IDs that have face annotations."""
        face_ids = set()
        with open(bbox_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["LabelName"] == OPEN_IMAGES_FACE_LABEL:
                    face_ids.add(row["ImageID"])
        return face_ids

    def _parse_image_urls(self, image_ids_csv: Path) -> dict[str, str]:
        """Parse image metadata CSV to get download URLs."""
        url_map = {}
        with open(image_ids_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url_map[row["ImageID"]] = row.get("OriginalURL", "")
        return url_map

    def _parse_annotations(self) -> dict[str, list[tuple]]:
        """Parse face bounding boxes from the annotations CSV.

        Open Images format: normalized coords (0-1).
        We store them as normalized and convert to pixels when loading samples.
        """
        if self._annotations is not None:
            return self._annotations

        bbox_csv = self.root / "validation-annotations-bbox.csv"
        annotations: dict[str, list[tuple]] = {}

        with open(bbox_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["LabelName"] != OPEN_IMAGES_FACE_LABEL:
                    continue
                # Skip group-of and depiction annotations (not real faces)
                if row.get("IsGroupOf", "0") == "1" or row.get("IsDepiction", "0") == "1":
                    continue

                img_id = row["ImageID"]
                # Normalized coordinates: XMin, XMax, YMin, YMax
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
                annotations.setdefault(img_id, []).append((xmin, ymin, xmax, ymax))

        self._annotations = annotations
        return annotations

    def get_samples(self, limit: int | None = None) -> list[ImageSample]:
        annotations = self._parse_annotations()
        images_dir = self.root / "images"
        samples = []

        for img_id, norm_boxes in annotations.items():
            path = images_dir / f"{img_id}.jpg"
            if not path.exists():
                continue

            # Convert normalized coords to pixel coords
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]

            pixel_boxes = [
                (xmin * w, ymin * h, xmax * w, ymax * h)
                for xmin, ymin, xmax, ymax in norm_boxes
            ]

            samples.append(ImageSample(path=path, ground_truth_boxes=pixel_boxes))
            if limit and len(samples) >= limit:
                break

        return samples

    def get_micro_samples(self, size: int = 50) -> list[ImageSample]:
        """Get a diverse micro subset: mix of easy (few faces) and hard (many faces)."""
        annotations = self._parse_annotations()
        images_dir = self.root / "images"

        # Sort by face count for diversity
        by_count = sorted(annotations.items(), key=lambda x: len(x[1]))
        n = len(by_count)
        step = max(n // size, 1)
        selected = by_count[::step][:size]

        samples = []
        for img_id, norm_boxes in selected:
            path = images_dir / f"{img_id}.jpg"
            if not path.exists():
                continue

            import cv2
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]

            pixel_boxes = [
                (xmin * w, ymin * h, xmax * w, ymax * h)
                for xmin, ymin, xmax, ymax in norm_boxes
            ]

            difficulty = "easy" if len(norm_boxes) <= 3 else ("medium" if len(norm_boxes) <= 10 else "hard")
            samples.append(ImageSample(path=path, ground_truth_boxes=pixel_boxes, difficulty=difficulty))

        return samples
