"""Prediction smoothing via rolling average and history tracking."""

from collections import deque

import numpy as np

from src.config import SMOOTHING_WINDOW, HISTORY_SECONDS


class EmotionTracker:
    """Rolling prediction buffer for smoothing + full history for charts."""

    def __init__(self):
        self._buffer = deque(maxlen=SMOOTHING_WINDOW)
        # History stores (timestamp_sec, predictions_array) for charting
        self._history = deque()
        self._start_time = None
        self._fps = 30.0  # will be updated from main loop

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, value: float):
        self._fps = value

    def reset(self):
        self._buffer.clear()
        self._history.clear()
        self._start_time = None

    def update(self, raw_preds: np.ndarray, timestamp_sec: float = None):
        """Add raw prediction, return smoothed prediction."""
        if self._start_time is None:
            self._start_time = timestamp_sec if timestamp_sec else 0.0

        self._buffer.append(raw_preds)
        smoothed = np.mean(self._buffer, axis=0)

        # Store in history
        elapsed = timestamp_sec if timestamp_sec else 0.0
        if self._start_time is not None:
            elapsed = elapsed - self._start_time

        self._history.append((elapsed, smoothed.copy()))

        # Trim history to HISTORY_SECONDS
        cutoff = elapsed - HISTORY_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        return smoothed

    @property
    def smoothed(self) -> np.ndarray:
        """Current smoothed prediction, or zeros if buffer empty."""
        if not self._buffer:
            return np.zeros(7, dtype=np.float32)
        return np.mean(self._buffer, axis=0)

    def get_history(self):
        """Return list of (elapsed_sec, predictions_array)."""
        return list(self._history)

    def clear_buffer(self):
        """Clear the smoothing buffer (e.g. when face is lost)."""
        self._buffer.clear()

    @property
    def elapsed_seconds(self) -> float:
        """Session elapsed time in seconds."""
        if not self._history:
            return 0.0
        return self._history[-1][0]
