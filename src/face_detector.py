"""MediaPipe face detection wrapper with MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np

from src.config import CROP_PADDING, FACE_DETECTION_CONFIDENCE


class FaceDetector:
    """Detects faces in a frame and returns face region crops."""

    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self._detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
        )

    def close(self):
        self._detector.close()

    def detect(self, frame: np.ndarray):
        """Detect faces and return list of FaceRegion namedtuples."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                face = self._extract_face(frame, detection)
                if face is not None:
                    faces.append(face)
        return faces

    def _extract_face(self, frame, detection):
        h, w, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box

        cx = int(bbox.xmin * w) + int(bbox.width * w) // 2
        cy = int(bbox.ymin * h) + int(bbox.height * h) // 2
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        crop_w = int(bw * (1 + 2 * CROP_PADDING))
        crop_h = int(bh * (1 + 2 * CROP_PADDING))
        crop_w = max(crop_w, 10)
        crop_h = max(crop_h, 10)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, cx + crop_w // 2)
        y2 = min(h, cy + crop_h // 2)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        return FaceRegion(
            x1=x1, y1=y1, x2=x2, y2=y2,
            crop=face_roi,
        )


class FaceRegion:
    """Bounding box and cropped face region."""
    __slots__ = ("x1", "y1", "x2", "y2", "crop")

    def __init__(self, x1, y1, x2, y2, crop):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.crop = crop
