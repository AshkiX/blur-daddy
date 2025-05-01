import cv2
import argparse
from utils.image_utils import resize_image

def main(args):
    image = cv2.imread(args.input)
    resized_image = resize_image(image, width=args.width, height=args.height)
    cv2.imwrite(args.output, resized_image)
    print(f"Saved resized image to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an image. Only one of width or height should be provided.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default="resized.jpg", help="Path to save the resized image.")
    parser.add_argument("--width", type=int, help="Width to resize the image to.")
    parser.add_argument("--height", type=int, help="Height to resize the image to.")
    args = parser.parse_args()
    main(args)
    