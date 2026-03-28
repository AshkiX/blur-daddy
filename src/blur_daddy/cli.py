"""CLI entry point for blur-daddy.

This is a thin wrapper around BlurDaddy — all logic lives in the API.
"""

import argparse
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


def _process_image(bd: BlurDaddy, input_path: str, output_path: str) -> None:
    """Process a single image through the API."""
    result = bd.blur(input_path)
    result.save(output_path)
    print(f"Detected {len(result.detections)} face(s)")


def _process_video(bd: BlurDaddy, input_path: str, output_path: str) -> None:
    """Process a video frame-by-frame through the API."""
    from tqdm import tqdm

    frames = extract_frames(input_path)
    fps, size = get_video_metadata(input_path)

    output_frames = []
    for frame in tqdm(frames, desc="Processing frames"):
        result = bd.blur(frame)
        # Convert RGB result back to BGR for video writing
        import cv2
        output_frames.append(cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))

    write_video(output_frames, output_path, fps, size)
    print(f"Processed {len(frames)} frames")


def main(args):
    bd = BlurDaddy(model=args.model, method=args.method)

    t0 = time.perf_counter()

    if _is_image_file(args.input):
        print(f"Processing image {args.input}...")
        _process_image(bd, args.input, args.output)
    elif _is_video_file(args.input):
        print(f"Processing video {args.input}...")
        _process_video(bd, args.input, args.output)
    else:
        raise ValueError(
            f"Unsupported file type. Supported image formats: {SUPPORTED_IMAGE_FORMATS}. "
            f"Supported video formats: {SUPPORTED_VIDEO_FORMATS}."
        )

    elapsed = time.perf_counter() - t0
    print(f"Saved output to {args.output}")
    print(f"Total time: {elapsed:.2f}s | Memory: {get_memory_usage()} MB")


def _build_parser():
    parser = argparse.ArgumentParser(description="Blur faces in an image or video.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output file.")
    parser.add_argument(
        "--method", type=str, default="gaussian",
        choices=["gaussian", "elliptical", "pixelation"], help="Blurring method.",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n-face",
        choices=["yolov8n-face", "mtcnn"], help="Model to use for face detection.",
    )
    return parser


def main_cli():
    args = _build_parser().parse_args()
    main(args)


if __name__ == "__main__":
    main_cli()
