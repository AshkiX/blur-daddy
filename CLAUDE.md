# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blur Daddy is an open-source face-blurring tool for images and videos. It uses YOLOv8n-face (fast, default) and MTCNN (accurate, slower) for face detection, with three blur methods: Gaussian, pixelation, and elliptical. Licensed under MIT.

The long-term vision is a dual-track product: open-source CLI/library + paid cloud API. See `BLUR-DADDY-GROWTH-SPEC.md` for the full roadmap.

## Commands

```bash
# Install dependencies (uv-managed, Python 3.12)
uv sync

# Detect faces (preview)
blur-daddy detect sample_images/sample1.jpg -o preview.jpg --json

# Blur faces in an image
blur-daddy blur sample_images/sample1.jpg -o test.jpg --method gaussian --model yolov8n-face

# Blur faces in a video
blur-daddy blur sample_videos/sample1.mp4 -o test.mp4

# Blur but keep specific faces unblurred
blur-daddy blur photo.jpg -o out.jpg --keep face-0 face-2

# Resize an image
blur-daddy-resize --input path/to/image.jpg --output resized.jpg --width 800

# Run tests
uv run pytest

# Run tests with milestone report (generates reports/M3.md)
uv run pytest --milestone-report M3

# Lint
uv run ruff check .
```

**CLI subcommands:**
- `blur-daddy detect INPUT [-o OUTPUT] [--model MODEL] [--json]` — detect faces, preview results
- `blur-daddy blur INPUT -o OUTPUT [--method METHOD] [--model MODEL] [--keep ID ...]` — blur faces, optionally keep some unblurred

## Architecture

The codebase is a simple pipeline: **input -> detect faces -> blur faces -> output**.

```
src/blur_daddy/
  __init__.py          Package root, version
  cli.py               CLI entry point (blur-daddy console script)
  resize.py            Image resize CLI (blur-daddy-resize console script)
  detection.py         Face detection (MTCNN via facenet-pytorch, YOLOv8 via ultralytics)
  blur.py              Three blur algorithms (Gaussian rect, pixelation rect, elliptical Gaussian)
  image.py             OpenCV image read/write/resize
  video.py             Frame extraction and video writing via OpenCV
  benchmark.py         timed_section context manager + memory usage tracking
models/                YOLOv8n-face weights (project root, found via _find_models_dir())
tests/                 58 tests + milestone report plugin
```

### Key design details

- **Face detection** returns `(boxes, probs, landmarks)` tuples. `boxes` is a list of `[x1, y1, x2, y2]` or `None`. Landmarks are only available from MTCNN (used for elliptical blur rotation angle).
- **YOLOv8 model** is loaded from `models/yolov8n-face.pt` via `_find_models_dir()` which walks up from the source file. Override with `BLUR_DADDY_MODELS_DIR` env var.
- **MTCNN** is initialized as a module-level singleton in `detection.py` with CUDA if available.
- **YOLO** uses a lazy singleton `_get_yolo_model()` — loaded on first use, cached thereafter.
- **Elliptical blur** creates a full-image Gaussian blur, then alpha-blends it using an elliptical mask per face.
- **Video processing** loads ALL frames into memory at once (`extract_frames`), processes each frame independently, then writes all output frames. No streaming/chunked processing yet.
- **Blur padding**: All blur methods add 15px padding around detected face bounding boxes (`PADDING` constant in `blur.py`).

### What's missing (known gaps)

- No face tracking across video frames (faces processed independently per frame)
- No audio preservation in video output
- No CI/CD or pip packaging
- No batch processing or selective blurring

## Dependencies

Core: `facenet-pytorch`, `opencv-python`, `torch`, `ultralytics`, `tqdm`, `psutil`, `omegaconf`

GPU is optional — falls back to CPU automatically via `torch.cuda.is_available()`.
