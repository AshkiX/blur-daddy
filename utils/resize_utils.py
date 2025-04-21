import cv2

def resize_image(image, width=None, height=None):
    """
    Resize an image to a specific width or height. Only one of width or height should be provided.
    Args:
        image: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
    Returns:
        The resized image.
    """
    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        raise ValueError("Only one of width or height should be provided.")
    elif width is not None:
        height = int(width / image.shape[1] * image.shape[0])
    elif height is not None:
        width = int(height / image.shape[0] * image.shape[1])
    return cv2.resize(image, (width, height))
