"""Tests for benchmark dataset loaders and report generation."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from benchmarks.datasets.base import ImageSample, VideoSample
from benchmarks.datasets.curated_clips import CuratedClipsDataset
from benchmarks.report import generate_report

# ── Open Images V7 annotation parsing ────────────────────────────────────

class TestOpenImagesParser:
    def _create_fake_dataset(self, tmp_path):
        """Create a minimal fake Open Images V7 dataset for testing."""
        # Create bbox annotations CSV
        bbox_csv = tmp_path / "validation-annotations-bbox.csv"
        bbox_csv.write_text(
            "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
            "abc123,freeform,/m/0dzct,1,0.1,0.5,0.2,0.8,0,0,0,0,0\n"
            "abc123,freeform,/m/0dzct,1,0.6,0.9,0.1,0.7,0,0,0,0,0\n"
            "def456,freeform,/m/0dzct,1,0.2,0.6,0.3,0.9,0,0,0,0,0\n"
            "ghi789,freeform,/m/01g317,1,0.1,0.5,0.2,0.8,0,0,0,0,0\n"  # not a face
            "group1,freeform,/m/0dzct,1,0.1,0.5,0.2,0.8,0,0,1,0,0\n"  # IsGroupOf=1, skip
            "depict1,freeform,/m/0dzct,1,0.1,0.5,0.2,0.8,0,0,0,1,0\n"  # IsDepiction=1, skip
        )

        # Create real JPEG images (cv2.imread needs valid images)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for img_id in ["abc123", "def456"]:
            img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"{img_id}.jpg"), img)

        return tmp_path

    def test_parse_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_fake_dataset(tmp_path)

            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=tmp_path)
            annots = ds._parse_annotations()

            # abc123 has 2 faces
            assert "abc123" in annots
            assert len(annots["abc123"]) == 2

            # def456 has 1 face
            assert "def456" in annots
            assert len(annots["def456"]) == 1

            # ghi789 is not a face class — should not appear
            assert "ghi789" not in annots

            # group and depiction should be skipped
            assert "group1" not in annots
            assert "depict1" not in annots

    def test_normalized_coords(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_fake_dataset(tmp_path)

            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=tmp_path)
            annots = ds._parse_annotations()

            # First box of abc123: XMin=0.1, YMin=0.2, XMax=0.5, YMax=0.8
            box = annots["abc123"][0]
            assert box == (0.1, 0.2, 0.5, 0.8)

    def test_get_samples_converts_to_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_fake_dataset(tmp_path)

            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=tmp_path)
            samples = ds.get_samples()

            assert len(samples) >= 1
            assert all(isinstance(s, ImageSample) for s in samples)

            # Images are 200x100 (w=200, h=100)
            # First face box: (0.1*200, 0.2*100, 0.5*200, 0.8*100) = (20, 20, 100, 80)
            s = [s for s in samples if "abc123" in str(s.path)][0]
            box = s.ground_truth_boxes[0]
            assert abs(box[0] - 20.0) < 1.0
            assert abs(box[1] - 20.0) < 1.0
            assert abs(box[2] - 100.0) < 1.0
            assert abs(box[3] - 80.0) < 1.0

    def test_get_samples_with_limit(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_fake_dataset(tmp_path)

            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=tmp_path)
            samples = ds.get_samples(limit=1)
            assert len(samples) == 1

    def test_is_ready(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_fake_dataset(tmp_path)

            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=tmp_path)
            assert ds.is_ready()

    def test_not_ready_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=Path(td))
            assert not ds.is_ready()

    def test_get_micro_samples(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            self._create_fake_dataset(tmp_path)

            from benchmarks.datasets.open_images import OpenImagesDataset

            ds = OpenImagesDataset(root=tmp_path)
            samples = ds.get_micro_samples(size=10)
            assert len(samples) <= 10
            assert all(isinstance(s, ImageSample) for s in samples)
            assert all(s.difficulty in ("easy", "medium", "hard") for s in samples)


# ── Curated clips ────────────────────────────────────────────────────────

class TestCuratedClips:
    def test_not_ready_empty(self):
        with tempfile.TemporaryDirectory() as td:
            ds = CuratedClipsDataset(root=Path(td))
            assert not ds.is_ready()

    def test_skips_schema_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "schema.json").write_text("{}")
            ds = CuratedClipsDataset(root=root)
            samples = ds.get_samples()
            assert samples == []

    def test_loads_valid_annotation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "clip.mp4").write_bytes(b"\x00" * 100)
            annot = {
                "video": "clip.mp4",
                "fps": 30,
                "frames": [
                    {
                        "frame_idx": 0,
                        "faces": [{"box": [10, 10, 50, 50], "id": "face_0"}],
                    },
                    {
                        "frame_idx": 30,
                        "faces": [
                            {"box": [10, 10, 50, 50], "id": "face_0"},
                            {"box": [100, 100, 150, 150], "id": "face_1"},
                        ],
                    },
                ],
            }
            (root / "clip.json").write_text(json.dumps(annot))

            ds = CuratedClipsDataset(root=root)
            assert ds.is_ready()
            samples = ds.get_samples()
            assert len(samples) == 1
            assert isinstance(samples[0], VideoSample)
            assert 0 in samples[0].annotated_frames
            assert 30 in samples[0].annotated_frames
            assert len(samples[0].annotated_frames[30]) == 2
            assert samples[0].face_ids[0] == ["face_0"]

    def test_skips_missing_video(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            annot = {"video": "nonexistent.mp4", "frames": []}
            (root / "test.json").write_text(json.dumps(annot))

            ds = CuratedClipsDataset(root=root)
            samples = ds.get_samples()
            assert samples == []


# ── Report generation ────────────────────────────────────────────────────

class TestReportGeneration:
    def _sample_results(self):
        return {
            "timestamp": "2026-01-01T00:00:00Z",
            "tier": "micro",
            "dataset": "test",
            "num_images": 10,
            "iou_thresholds": [0.5, 0.75],
            "detection": {
                "yolo": {
                    "IoU=0.5": {"ap": 0.85, "precision": 0.9, "recall": 0.8, "f1": 0.85},
                }
            },
            "performance": {
                "yolo": {"fps": 30.0, "avg_ms_per_image": 33.3, "peak_memory_mb": 150},
            },
            "video": {
                "yolo": {
                    "temporal_consistency": {"mean_iou": 0.8, "match_rate": 0.9},
                    "flicker": {"flicker_rate": 0.05, "total_flickers": 2},
                },
            },
            "errors": {"mediapipe": "not installed"},
        }

    def test_generates_json_and_md(self):
        with tempfile.TemporaryDirectory() as td:
            results = self._sample_results()
            json_path, md_path = generate_report(results, Path(td))
            assert json_path.exists()
            assert md_path.exists()
            assert json_path.suffix == ".json"
            assert md_path.suffix == ".md"

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            results = self._sample_results()
            json_path, _ = generate_report(results, Path(td))
            loaded = json.loads(json_path.read_text())
            assert loaded["tier"] == "micro"
            assert loaded["detection"]["yolo"]["IoU=0.5"]["ap"] == 0.85

    def test_markdown_contains_tables(self):
        with tempfile.TemporaryDirectory() as td:
            results = self._sample_results()
            _, md_path = generate_report(results, Path(td))
            md = md_path.read_text()
            assert "## Detection Quality" in md
            assert "## Performance" in md
            assert "## Video Metrics" in md
            assert "## Skipped / Errors" in md
            assert "mediapipe" in md

    def test_empty_results(self):
        with tempfile.TemporaryDirectory() as td:
            results = {
                "timestamp": "2026-01-01",
                "tier": "micro",
                "dataset": "test",
                "num_images": 0,
                "detection": {},
                "performance": {},
                "video": {},
                "errors": {},
            }
            json_path, md_path = generate_report(results, Path(td))
            assert json_path.exists()
            assert md_path.exists()
