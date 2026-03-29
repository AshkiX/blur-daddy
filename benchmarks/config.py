"""Benchmark suite configuration."""

import os
from pathlib import Path

# Base paths
BENCHMARK_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_DIR.parent
CACHE_DIR = Path(os.environ.get("BLUR_DADDY_CACHE_DIR", Path.home() / ".cache" / "blur-daddy"))
DATASETS_CACHE = CACHE_DIR / "datasets"
RESULTS_DIR = BENCHMARK_DIR / "results"

# Open Images V7 dataset (CC BY 4.0 annotations, Flickr-licensed images)
OPEN_IMAGES_BBOX_URL = (
    "https://storage.googleapis.com/openimages/v7/obb/validation-annotations-bbox.csv"
)
OPEN_IMAGES_CLASS_URL = (
    "https://storage.googleapis.com/openimages/v7/obb/class-descriptions-boxable.csv"
)
OPEN_IMAGES_IMAGE_IDS_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
)
OPEN_IMAGES_FACE_LABEL = "/m/0dzct"  # "Human face"
OPEN_IMAGES_DIR = DATASETS_CACHE / "open_images_v7"

# YouTube Faces (requires manual download)
YTFACES_DIR = Path(os.environ.get("YTFACES_DIR", DATASETS_CACHE / "youtube_faces"))

# Curated clips
CURATED_DIR = BENCHMARK_DIR / "annotations"

# IoU thresholds for mAP
IOU_THRESHOLDS = [0.5, 0.75]

# Tier definitions
MICRO_SIZE = 50   # images for fast CI runs
FULL_SIZE = 1000  # images for comprehensive benchmarks

# Minimum confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Video benchmark resolutions
VIDEO_RESOLUTIONS = [(640, 360), (1280, 720), (1920, 1080)]
