import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "emotiefflib_mobilenet_7.tflite")
FACE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models",
                               "face_detection_short_range.tflite")

IMG_SIZE = (224, 224)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOJI_MAP = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😊",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

EMOTION_COLORS = {
    "angry": (0, 0, 200),
    "disgust": (0, 180, 0),
    "fear": (180, 0, 180),
    "happy": (0, 220, 220),
    "neutral": (140, 140, 140),
    "sad": (200, 100, 0),
    "surprise": (220, 220, 0),
}

# Per-class confidence thresholds (lower for sad/surprise due to class bias)
PER_CLASS_THRESHOLD = {
    "angry": 0.12,
    "disgust": 0.12,
    "fear": 0.12,
    "happy": 0.12,
    "neutral": 0.12,
    "sad": 0.08,
    "surprise": 0.08,
}

# Always show the argmax emotion even below threshold
ALWAYS_SHOW_ARGMAX = True

CROP_PADDING = -0.1
FACE_DETECTION_CONFIDENCE = 0.5

# Smoothing: rolling average over N frames (low=responsive, high=stable)
SMOOTHING_WINDOW = 3

# History chart: how many seconds of data to show
HISTORY_SECONDS = 30

# Stats: moving window for dominant emotion calculation
STATS_WINDOW_SECONDS = 10

# Side-by-side mode: show raw camera next to annotated
SIDE_BY_SIDE_ENABLED = True

# Recording
RECORD_VIDEO = False
RECORD_CSV = False
RECORD_DIR = os.path.join(PROJECT_ROOT, "recordings")
