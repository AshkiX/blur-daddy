import cv2
import argparse
from utils.face_utils import detect_faces, get_face_angle
from utils.blur_utils import apply_rect_gaussian_blur, apply_rect_pixelation, apply_elliptical_gaussian_blur

def main(args):
    image = cv2.imread(args.input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, probs, landmarks = detect_faces(image_rgb)
    print(f"Detected {len(boxes) if boxes is not None else 0} faces in the image.")

    if boxes is not None:
        for box, landmark in zip(boxes, landmarks):
            if args.method == 'gaussian':
                image = apply_rect_gaussian_blur(image, box)
            elif args.method == 'elliptical':
                angle = get_face_angle(landmark)
                image = apply_elliptical_gaussian_blur(image, box, angle)
            else:
                image = apply_rect_pixelation(image, box)

    cv2.imwrite(f"../output/{args.output}", image)
    print(f"Saved blurred image to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur faces in an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default="blurred.jpg", help="Path to save the blurred image.")
    parser.add_argument("--method", type=str, default="gaussian", choices=["gaussian", "elliptical", "pixelation"], help="Blurring method.")
    args = parser.parse_args()
    main(args)
    