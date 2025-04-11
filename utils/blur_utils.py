import cv2

PADDING = 10

def apply_gaussian_blur(image, box):
    """
    Apply a Gaussian blur to a box in an image.
    """
    # [print(coord) for coord in box]
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 -= PADDING
    y1 -= PADDING
    x2 += PADDING
    y2 += PADDING
    region = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, (51, 51), 30)
    image[y1:y2, x1:x2] = blurred
    return image

def apply_pixelation(image, box, blocks=10):
    """
    Pixelate a box in an image.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 -= PADDING
    y1 -= PADDING
    x2 += PADDING
    y2 += PADDING
    region = image[y1:y2, x1:x2]
    h, w = region.shape[:2]
    temp = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated
    return image
