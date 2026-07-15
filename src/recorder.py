"""Video and CSV recording for the demo session."""

import csv
import os
from datetime import datetime

import cv2
import numpy as np

from src.config import EMOTIONS, RECORD_DIR


class Recorder:
    """Records session video + emotion timeline CSV."""

    def __init__(self, fps=30.0):
        self.fps = fps
        self._video_writer = None
        self._csv_file = None
        self._csv_writer = None
        self._is_recording = False
        self._start_time = None
        os.makedirs(RECORD_DIR, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start_recording(self, frame_size=(640, 480)):
        """Begin a new recording session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Try multiple codecs, fall back to any that works
        video_path = os.path.join(RECORD_DIR, f"emotion_{timestamp}.mp4")
        codecs = [cv2.VideoWriter_fourcc(*"mp4v"),
                  cv2.VideoWriter_fourcc(*"XVID"),
                  cv2.VideoWriter_fourcc(*"MJPG"),
                  cv2.VideoWriter_fourcc(*"DIVX")]
        self._video_writer = None
        for codec in codecs:
            self._video_writer = cv2.VideoWriter(
                video_path, codec, self.fps, frame_size
            )
            if self._video_writer.isOpened():
                break
            self._video_writer.release()

        if self._video_writer is None or not self._video_writer.isOpened():
            print("Warning: No video codec available, recording video only (CSV will still work).")
            self._video_writer = None

        # CSV
        csv_path = os.path.join(RECORD_DIR, f"emotion_{timestamp}.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["timestamp_sec", "dominant_emotion", "confidence"] + EMOTIONS
        )

        self._is_recording = True
        self._start_time = None
        print(f"Recording: {video_path} + CSV")
        return video_path, csv_path

    def stop_recording(self):
        """Stop recording and close all files."""
        self._is_recording = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        print("Recording stopped.")

    def write_frame(self, frame: np.ndarray, predictions: np.ndarray,
                    elapsed_sec: float = None):
        """Write a video frame + CSV row."""
        if not self._is_recording:
            return

        # Video frame
        if self._video_writer:
            self._video_writer.write(frame)

        # CSV row
        if self._csv_writer:
            if self._start_time is None:
                self._start_time = elapsed_sec if elapsed_sec else 0.0
            ts = (elapsed_sec - self._start_time) if elapsed_sec else 0.0
            emotion_idx = int(np.argmax(predictions))
            confidence = float(predictions[emotion_idx])
            self._csv_writer.writerow(
                [f"{ts:.2f}", EMOTIONS[emotion_idx], f"{confidence:.4f}"]
                + [f"{p:.4f}" for p in predictions]
            )

    def close(self):
        """Cleanup."""
        if self._is_recording:
            self.stop_recording()
