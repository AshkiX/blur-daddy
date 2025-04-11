import cv2
import argparse
from utils.face_utils import detect_faces
from utils.blur_utils import apply_gaussian_blur, apply_pixelation

def main(args):
    image = cv2.imread(args.input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, probs = detect_faces(image_rgb)
    print(f"Detected {len(boxes) if boxes is not None else 0} faces in the image.")

    if boxes is not None:
        # print(boxes)
        for box in boxes:
            if args.method == 'gaussian':
                image = apply_gaussian_blur(image, box)
            else:
                image = apply_pixelation(image, box)

    cv2.imwrite(f"../output/{args.output}", image)
    print(f"Saved blurred image to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur faces in an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default="blurred.jpg", help="Path to save the blurred image.")
    parser.add_argument("--method", type=str, default="gaussian", choices=["gaussian", "pixelation"], help="Blurring method.")
    args = parser.parse_args()
    main(args)
    