# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blur Daddy is an open-source face-blurring tool for images and videos. It uses YOLOv8n-face (fast, default) and MTCNN (accurate, slower) for face detection, with three blur methods: Gaussian, pixelation, and elliptical. Licensed under MIT.

The long-term vision is a dual-track product: open-source CLI/library + paid cloud API. See `BLUR-DADDY-GROWTH-SPEC.md` for the full roadmap.

## Commands

```bash
# Install dependencies (uv-managed, Python 3.12)
uv sync

# Blur an image (run from project root)
python main/blur_faces.py --input sample_images/sample1.jpg --output test.jpg --method gaussian --model yolov8n-face

# Blur a video
python main/blur_faces.py --input sample_videos/sample1.mp4 --output test.mp4 --method gaussian --model yolov8n-face

# Resize an image
python main/resize_image.py --input path/to/image.jpg --output resized.jpg --width 800

# Run tests
uv run pytest

# Run tests with milestone report (generates reports/M0.html)
uv run pytest --milestone-report M0

# Lint
uv run ruff check .
```

**CLI arguments for `blur_faces.py`:**
- `--input` (required): path to image or video
- `--output` (required): output filename (saved to `../output/` relative to `main/`)
- `--method`: `gaussian` (default), `elliptical`, `pixelation`
- `--model`: `yolov8n-face` (default), `mtcnn`

**Important:** The script must be run from the `main/` directory context or project root, because model paths use `../models/` relative references and output goes to `../output/`.

## Architecture

The codebase is a simple pipeline: **input -> detect faces -> blur faces -> output**.

```
main/blur_faces.py          CLI entry point, orchestrates the pipeline
  ├── utils/face_utils.py   Face detection (MTCNN via facenet-pytorch, YOLOv8 via ultralytics)
  ├── utils/blur_utils.py   Three blur algorithms (Gaussian rect, pixelation rect, elliptical Gaussian)
  ├── utils/image_utils.py  OpenCV image read/write/resize
  ├── utils/video_utils.py  Frame extraction and video writing via OpenCV
  └── utils/benchmark_utils.py  timed_section context manager + memory usage tracking
```

### Key design details

- **Face detection** returns `(boxes, probs, landmarks)` tuples. `boxes` is a list of `[x1, y1, x2, y2]` or `None`. Landmarks are only available from MTCNN (used for elliptical blur rotation angle).
- **YOLOv8 model** is loaded from `models/yolov8n-face.pt` via relative path `../models/` — this is a custom face-detection model, not standard YOLO.
- **MTCNN** is initialized as a module-level singleton in `face_utils.py` with CUDA if available.
- **Elliptical blur** creates a full-image Gaussian blur, then alpha-blends it using an elliptical mask per face. This is more natural-looking than rectangular blur.
- **Video processing** loads ALL frames into memory at once (`extract_frames`), processes each frame independently, then writes all output frames. No streaming/chunked processing yet.
- **Blur padding**: All blur methods add 15px padding around detected face bounding boxes (`PADDING` constant in `blur_utils.py`).
- **Output path**: `blur_faces.py` prepends `../output/` to the `--output` argument.
- `ffmpeg_utils.py` is an empty placeholder for future audio preservation work.

### What's missing (known gaps)

- No face tracking across video frames (faces processed independently per frame)
- No audio preservation in video output
- No CI/CD or pip packaging
- No batch processing or selective blurring

## Dependencies

Core: `facenet-pytorch`, `opencv-python`, `torch`, `ultralytics`, `tqdm`, `psutil`, `matplotlib`

GPU is optional — falls back to CPU automatically via `torch.cuda.is_available()`.
