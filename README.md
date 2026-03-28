# Blur Daddy

Fast, accurate face blurring for images and videos. Powered by YOLOv8 and MTCNN.

![Blur method comparison](docs/comparison.png)

## Features

- **Two detection models**: YOLOv8n-face (fast, default) and MTCNN (accurate, with landmarks)
- **Three blur methods**: Gaussian, pixelation, and elliptical (smooth, natural-looking)
- **Images and video**: Process single files or video streams
- **GPU optional**: Falls back to CPU automatically

## Install

```bash
pip install blur-daddy
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install blur-daddy
```

### From source

```bash
git clone https://github.com/AshkiX/blur-daddy.git
cd blur-daddy
uv sync
```

## Quick Start

```bash
# Detect faces (preview what will be blurred)
blur-daddy detect photo.jpg -o preview.jpg

# Blur all faces in an image
blur-daddy blur photo.jpg -o blurred.jpg

# Blur faces in a video
blur-daddy blur video.mp4 -o blurred.mp4

# Use pixelation instead of gaussian
blur-daddy blur photo.jpg -o blurred.jpg --method pixelation

# Use elliptical blur (smooth, follows face angle)
blur-daddy blur photo.jpg -o blurred.jpg --method elliptical

# Use MTCNN for more accurate detection
blur-daddy blur photo.jpg -o blurred.jpg --model mtcnn

# Keep specific faces unblurred (e.g. yourself)
blur-daddy blur photo.jpg -o blurred.jpg --keep face-0
```

## Python API

```python
# One-liner
import blur_daddy
blur_daddy.blur("photo.jpg", "blurred.jpg")
```

```python
# With control
from blur_daddy import BlurDaddy

bd = BlurDaddy(model="yolov8n-face", method="elliptical")

# Preview detections
preview = bd.detect("photo.jpg")
preview.faces         # [Face(id='face-0', ...), Face(id='face-1', ...)]
preview.save("preview.jpg")  # annotated image with numbered boxes

# Blur, but keep face-0 (e.g. yourself)
result = bd.blur("photo.jpg", keep=[preview.faces[0]])
result.save("blurred.jpg")
```

## Blur Methods

| Method | Description | Best for |
|--------|-------------|----------|
| `gaussian` | Rectangular Gaussian blur | Fast, general purpose |
| `pixelation` | Mosaic/pixel block effect | Classic anonymization look |
| `elliptical` | Smooth elliptical blur with feathered edges | Natural-looking, follows face angle via landmarks |

## CLI Reference

```
blur-daddy detect INPUT [-o OUTPUT] [--model MODEL] [--json]
blur-daddy blur INPUT -o OUTPUT [--method METHOD] [--model MODEL] [--keep ID ...]

Subcommands:
  detect        Detect faces and preview results
  blur          Blur detected faces in an image or video

Options (blur):
  -o, --output  Output file path (required)
  --method      gaussian (default), pixelation, elliptical
  --model       yolov8n-face (default), mtcnn
  --keep        Face IDs to protect from blurring (e.g. face-0 face-2)

Options (detect):
  -o, --output  Save annotated preview image
  --model       yolov8n-face (default), mtcnn
  --json        Print detections as JSON
```

**Supported formats:**
- Images: PNG, JPG, JPEG, BMP, WebP, TIFF
- Video: MP4, AVI, MOV, MKV, FLV, WMV, WebM

## Docker

```bash
# Build
docker build -t blur-daddy .

# Run
docker run --rm -v $(pwd):/data blur-daddy \
  blur /data/photo.jpg -o /data/blurred.jpg
```

## Performance

Benchmarks on sample image (CPU):

| Operation | Time |
|-----------|------|
| YOLO detection | ~0.3s |
| MTCNN detection | ~0.5s |
| Gaussian blur | <0.01s |
| Pixelation | <0.01s |
| Elliptical blur | ~0.04s |

GPU acceleration available via CUDA — set up PyTorch with CUDA support for faster detection.

## Development

```bash
git clone https://github.com/AshkiX/blur-daddy.git
cd blur-daddy
uv sync

# Run tests
uv run pytest

# Run tests with milestone report
uv run pytest --milestone-report M2

# Lint
uv run ruff check .
```

## License

MIT

## Acknowledgements

Sample video footage used for testing and demonstration purposes is sourced from:
- YouTube: [https://www.youtube.com/watch?v=_ofEQYA2V58](https://www.youtube.com/watch?v=_ofEQYA2V58)
