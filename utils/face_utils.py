from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(image):
    """
    Detect faces in an image using MTCNN.

    Args:
        image (PIL.Image.Image): The image to detect faces in.

    Returns:
        tuple: A tuple containing (boxes, probs) where:
            - boxes: A list of bounding boxes for the detected faces [x1, y1, x2, y2] or None if no faces detected
            - probs: A list of detection probabilities for each face or None if no faces detected
    """
    boxes, probs = mtcnn.detect(image)
    return boxes, probs
