import cv2
import numpy as np

PADDING = 15

def get_padded_clamped_box(image_shape, box):
    """
    Get a padded and clamped box for a given image shape and box.
    """
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(img_w, x2 + PADDING)
    y2 = min(img_h, y2 + PADDING)
    return (x1, y1, x2, y2)

def apply_rect_gaussian_blur(image, box):
    """
    Apply a Gaussian blur to a box in an image.
    """
    x1, y1, x2, y2 = get_padded_clamped_box(image.shape, box)
    region = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, (21, 21), 15)
    image[y1:y2, x1:x2] = blurred
    return image

def apply_rect_pixelation(image, box, blocks=10):
    """
    Pixelate a box in an image.
    """
    x1, y1, x2, y2 = get_padded_clamped_box(image.shape, box)
    region = image[y1:y2, x1:x2]
    h, w = region.shape[:2]
    temp = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated
    return image

def apply_elliptical_gaussian_blur(image, box, angle):
    """
    Apply an elliptical Gaussian blur to a box in an image.
    """
    x1, y1, x2, y2 = get_padded_clamped_box(image.shape, box)
    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1

    mask = create_soft_ellipse_mask(image.shape, (cx, cy), (w // 2, h // 2), angle)
    # Save the mask as an image
    # cv2.imwrite(f"../output/mask_{cx}_{cy}.png", mask)

    blurred = cv2.GaussianBlur(image, (21, 21), 15)

    final = blend_images(image, blurred, mask)
    return final

def create_soft_ellipse_mask(image_shape, center, axes, angle, feather_amount=41):
    """
    Create a soft elliptical mask for a given image shape, center, axes, and angle.
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, (255), -1)

    mask = cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0)
    return mask

def blend_images(original, blurred, mask):
    """
    Blend two images using a mask.
    Mask must be single channel, values between 0 and 255.
    """
    mask = mask.astype(float) / 255.0
    mask = np.expand_dims(mask, axis=2)
    blended = (original * (1 - mask) + blurred * mask).astype(np.uint8)
    return blended
