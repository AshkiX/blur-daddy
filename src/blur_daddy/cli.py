"""CLI entry point for blur-daddy.

This is a thin wrapper around BlurDaddy — all logic lives in the API.

Usage:
    blur-daddy detect photo.jpg -o preview.jpg
    blur-daddy blur photo.jpg -o blurred.jpg --keep face-0
    blur-daddy blur video.mp4 -o blurred.mp4
"""

import argparse
import json
import sys
import time

from blur_daddy.api import BlurDaddy
from blur_daddy.benchmark import get_memory_usage
from blur_daddy.video import extract_frames, get_video_metadata, write_video

SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')


def _is_image_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_IMAGE_FORMATS)


def _is_video_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_VIDEO_FORMATS)


# ---------------------------------------------------------------------------
# detect subcommand
# ---------------------------------------------------------------------------

def _cmd_detect(args):
    bd = BlurDaddy(model=args.model)
    result = bd.detect(args.input)

    # Print detections as structured output
    for det in result.detections:
        x1, y1, x2, y2 = det.box_int
        print(f"  {det.id}  box=({x1},{y1},{x2},{y2})  conf={det.confidence:.2f}")

    if not result.detections:
        print("  No detections found.")

    # Save annotated preview if output specified
    if args.output:
        result.save(args.output)
        print(f"Saved preview to {args.output}")

    # Write JSON to stdout if requested
    if args.json:
        data = [
            {"id": d.id, "box": list(d.box), "confidence": round(d.confidence, 4)}
            for d in result.detections
        ]
        print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# blur subcommand
# ---------------------------------------------------------------------------

def _cmd_blur(args):
    bd = BlurDaddy(model=args.model, method=args.method)
    t0 = time.perf_counter()

    # Resolve --keep IDs to Detection objects
    keep = None
    if args.keep:
        preview = bd.detect(args.input)
        keep_ids = set(args.keep)
        keep = [d for d in preview.detections if d.id in keep_ids]
        unknown = keep_ids - {d.id for d in keep}
        if unknown:
            print(f"Warning: unknown IDs ignored: {unknown}", file=sys.stderr)
            print("  Run 'blur-daddy detect' first to see available IDs.", file=sys.stderr)

    if _is_image_file(args.input):
        print(f"Processing image {args.input}...")
        result = bd.blur(args.input, keep=keep)
        result.save(args.output)
        print(f"Detected {len(result.detections)} face(s)")
    elif _is_video_file(args.input):
        print(f"Processing video {args.input}...")
        _process_video(bd, args.input, args.output, keep=keep)
    else:
        raise ValueError(
            f"Unsupported file type. Supported: {SUPPORTED_IMAGE_FORMATS + SUPPORTED_VIDEO_FORMATS}"
        )

    elapsed = time.perf_counter() - t0
    print(f"Saved output to {args.output}")
    print(f"Total time: {elapsed:.2f}s | Memory: {get_memory_usage()} MB")


def _process_video(bd: BlurDaddy, input_path: str, output_path: str, keep=None) -> None:
    import cv2
    from tqdm import tqdm

    frames = extract_frames(input_path)
    fps, size = get_video_metadata(input_path)

    output_frames = []
    for frame in tqdm(frames, desc="Processing frames"):
        result = bd.blur(frame, keep=keep)
        output_frames.append(cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))

    write_video(output_frames, output_path, fps, size)
    print(f"Processed {len(frames)} frames")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="blur-daddy",
        description="Fast, accurate face blurring for images and videos.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- detect ---
    p_detect = sub.add_parser("detect", help="Detect faces and preview results.")
    p_detect.add_argument("input", help="Path to input image.")
    p_detect.add_argument("-o", "--output", help="Save annotated preview image.")
    p_detect.add_argument(
        "--model", default="yolov8n-face",
        choices=["yolov8n-face", "mtcnn"], help="Detection model.",
    )
    p_detect.add_argument("--json", action="store_true", help="Print detections as JSON.")

    # --- blur ---
    p_blur = sub.add_parser("blur", help="Blur detected faces in an image or video.")
    p_blur.add_argument("input", help="Path to input image or video.")
    p_blur.add_argument("-o", "--output", required=True, help="Output file path.")
    p_blur.add_argument(
        "--method", default="gaussian",
        choices=["gaussian", "elliptical", "pixelation"], help="Blurring method.",
    )
    p_blur.add_argument(
        "--model", default="yolov8n-face",
        choices=["yolov8n-face", "mtcnn"], help="Detection model.",
    )
    p_blur.add_argument(
        "--keep", nargs="+", metavar="ID",
        help="Detection IDs to protect from blurring (e.g. face-0 face-2).",
    )

    return parser


def main_cli():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "detect":
        _cmd_detect(args)
    elif args.command == "blur":
        _cmd_blur(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
