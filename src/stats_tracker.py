"""Track session statistics: dominant emotion, transitions, time distribution."""

from collections import Counter

import numpy as np

from src.config import EMOTIONS, STATS_WINDOW_SECONDS


class StatsTracker:
    """Tracks per-session emotion statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._emotion_counter = Counter()
        self._total_frames = 0
        self._current_emotion = None
        self._transition_count = 0
        self._last_emotion = None
        self._predictions_window = []
        self._time_in_emotion = {e: 0.0 for e in EMOTIONS}

    def update(self, predictions: np.ndarray, elapsed_sec: float):
        """Update stats with a smoothed prediction at given elapsed time."""
        idx = int(np.argmax(predictions))
        emotion = EMOTIONS[idx]
        self._emotion_counter[emotion] += 1
        self._total_frames += 1

        # Track transitions
        self._current_emotion = emotion
        if self._last_emotion is not None and emotion != self._last_emotion:
            self._transition_count += 1
        self._last_emotion = emotion

        # Track time in each emotion (approximate)
        self._predictions_window.append((elapsed_sec, emotion))
        # Prune old entries beyond STATS_WINDOW_SECONDS
        cutoff = elapsed_sec - STATS_WINDOW_SECONDS
        self._predictions_window = [
            (t, e) for t, e in self._predictions_window if t >= cutoff
        ]

        # Compute time distribution over the window
        recent_counter = Counter(e for _, e in self._predictions_window)
        total_recent = sum(recent_counter.values())
        if total_recent > 0:
            for e in EMOTIONS:
                self._time_in_emotion[e] = recent_counter.get(e, 0) / total_recent

    @property
    def dominant_emotion(self) -> str:
        """Overall dominant emotion."""
        if not self._total_frames:
            return "none"
        return self._emotion_counter.most_common(1)[0][0]

    @property
    def dominant_pct(self) -> float:
        """Overall dominant emotion percentage."""
        if not self._total_frames:
            return 0.0
        return self._emotion_counter.most_common(1)[0][1] / self._total_frames * 100

    @property
    def transition_count(self) -> int:
        return max(0, self._transition_count)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def recent_time_distribution(self):
        """Return dict of emotion -> fraction (0..1) over last STATS_WINDOW_SECONDS."""
        return self._time_in_emotion

    @property
    def current_emotion(self):
        return self._current_emotion

    def get_full_stats(self):
        """Return all stats as a dict for display."""
        return {
            "dominant": self.dominant_emotion,
            "dominant_pct": self.dominant_pct,
            "transitions": self.transition_count,
            "total_frames": self.total_frames,
            "current": self.current_emotion,
            "distribution": self.recent_time_distribution,
        }
