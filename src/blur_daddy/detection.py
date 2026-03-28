import os

import numpy as np
import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO

_mtcnn_model = None


def _get_mtcnn_model():
    global _mtcnn_model
    if _mtcnn_model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _mtcnn_model = MTCNN(keep_all=True, device=device)
    return _mtcnn_model

def _find_models_dir():
    """Find models/ dir by walking up from this file until we find it."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        candidate = os.path.join(d, "models")
        if os.path.isdir(candidate):
            return candidate
        d = os.path.dirname(d)
    if os.path.isdir("models"):
        return os.path.abspath("models")
    raise FileNotFoundError("Cannot locate models/ directory")


_MODELS_DIR = os.environ.get("BLUR_DADDY_MODELS_DIR") or _find_models_dir()
_yolo_model = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(os.path.join(_MODELS_DIR, "yolov8n-face.pt"))
    return _yolo_model


def detect_faces_mtcnn(image):
    """
    Detect faces in an image using MTCNN.

    Args:
        image (PIL.Image.Image): The image to detect faces in.

    Returns:
        tuple: A tuple containing (boxes, probs) where:
            - boxes: A list of bounding boxes for the detected faces [x1, y1, x2, y2] or None if no faces detected
            - probs: A list of detection probabilities for each face or None if no faces detected
    """
    boxes, probs, landmarks = _get_mtcnn_model().detect(image, landmarks=True)
    return boxes, probs, landmarks

def detect_faces_yolo(image):
    """
    Detect faces in an image using YOLO.
    """
    model = _get_yolo_model()
    results = model(image, verbose=False)[0]

    boxes = []
    confs = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        boxes.append([x1, y1, x2, y2])
        confs.append(conf)

    return boxes if boxes else None, confs if confs else None, None

def get_face_angle(landmarks):
    """
    Get the angle of a face based on its landmarks.
    """
    left_eye, right_eye = landmarks[0], landmarks[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle
