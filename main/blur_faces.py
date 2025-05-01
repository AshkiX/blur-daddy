import cv2
import argparse
from utils.benchmark_utils import timed_section, get_memory_usage
from utils.face_utils import detect_faces, get_face_angle
from utils.blur_utils import apply_rect_gaussian_blur, apply_rect_pixelation, apply_elliptical_gaussian_blur
from utils.image_utils import read_image, save_image
from utils.video_utils import extract_frames, get_video_metadata, write_video
from tqdm import tqdm


DEBUG = False
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

def is_image_file(path: str):
    """
    Check if a file is an image.
    """
    return path.endswith(SUPPORTED_IMAGE_FORMATS)

def is_video_file(path: str):
    """
    Check if a file is a video.
    """
    return path.endswith(SUPPORTED_VIDEO_FORMATS)


def process_input(file_path: str, logger: dict):
    frames = []

    if is_image_file(args.input):
        print(f"Processing image {file_path}...")
        with timed_section("Image load time", logger):
            frames.append(read_image(file_path))
    elif is_video_file(file_path):
        print(f"Processing video {file_path}...")
        with timed_section("Video load time", logger):
            frames = extract_frames(file_path)
        with timed_section("Video metadata extraction time", logger):
            fps, size = get_video_metadata(file_path)
    else:
        raise ValueError(f"Unsupported file type. Supported image formats: {SUPPORTED_IMAGE_FORMATS}. Supported video formats: {SUPPORTED_VIDEO_FORMATS}.")

    output_frames = []
    for i, frame in enumerate(tqdm(frames, desc="Processing frames", total=len(frames))):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with timed_section("Face detection time", logger):
            boxes, _, landmarks = detect_faces(image_rgb)
        if DEBUG:
            print(f"Detected {len(boxes) if boxes is not None else 0} faces in the frame {i+1}.")

        with timed_section("Blurring time", logger):
            if boxes is not None:
                if args.method == 'gaussian':
                    frame = apply_rect_gaussian_blur(frame, boxes)
                elif args.method == 'elliptical':
                    frame = apply_elliptical_gaussian_blur(frame, boxes, landmarks)
                elif args.method == 'pixelation':
                    frame = apply_rect_pixelation(frame, boxes)
                else:
                    raise ValueError(f"Unsupported method. Supported methods: gaussian, elliptical, pixelation.")
        
        output_frames.append(frame)

    with timed_section("Output time", logger):
        if len(output_frames) == 1:
            save_image(output_frames[0], f"../output/{args.output}")
        else:
            write_video(output_frames, f"../output/{args.output}", fps, size)
    
    return logger

def main(args):
    logger = {}

    with timed_section("Total processing time", logger):
        logger = process_input(args.input, logger)
    
    print(f"Saved output to {args.output}")

    print("Performance Metrics:")
    print(f"Memory usage: {get_memory_usage()} MB")
    print(f"Total processing time: {logger['Total processing time']:.2f} seconds")
    for metric, time_value in logger.items():
        print(f"{metric}: {time_value:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur faces in an image or video.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output file.")
    parser.add_argument("--method", type=str, default="gaussian", choices=["gaussian", "elliptical", "pixelation"], help="Blurring method.")
    args = parser.parse_args()
    main(args)
    