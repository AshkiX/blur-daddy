from facenet_pytorch import MTCNN
import torch
import numpy as np
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
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    return boxes, probs, landmarks

def get_face_angle(landmarks):
    """
    Get the angle of a face based on its landmarks.
    """
    left_eye, right_eye = landmarks[0], landmarks[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle
