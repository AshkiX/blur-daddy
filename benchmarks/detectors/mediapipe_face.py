"""Wrapper for MediaPipe Face Detection."""

from __future__ import annotations

import numpy as np

from benchmarks.detectors.base import Detection, DetectorBase


class MediaPipeFaceDetector(DetectorBase):
    name = "mediapipe"

    def __init__(self, model_selection: int = 1):
        self._detector = None
        self._model_selection = model_selection  # 0=short-range, 1=full-range

    def _get_detector(self):
        if self._detector is None:
            try:
                import mediapipe as mp
                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=self._model_selection,
                    min_detection_confidence=0.5,
                )
            except ImportError:
                raise ImportError(
                    "mediapipe not installed. Install with: pip install mediapipe"
                )
        return self._detector

    def detect(self, image: np.ndarray) -> list[Detection]:
        import cv2

        detector = self._get_detector()
        h, w = image.shape[:2]
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            return []

        detections = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = bbox.xmin * w
            y1 = bbox.ymin * h
            x2 = (bbox.xmin + bbox.width) * w
            y2 = (bbox.ymin + bbox.height) * h
            conf = det.score[0]
            detections.append(Detection(box=(x1, y1, x2, y2), confidence=float(conf)))

        return detections

