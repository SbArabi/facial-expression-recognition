"""TFLite emotion classifier wrapper using ai-edge-litert."""

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from src.config import MODEL_PATH, IMG_SIZE, EMOTIONS


class EmotionClassifier:
    """Loads the TFLite model and runs inference on face crops."""

    def __init__(self):
        self.interpreter = Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self._input_idx = self._input_details[0]["index"]
        self._output_idx = self._output_details[0]["index"]

        shape = self._input_details[0]["shape"]
        dtype = self._input_details[0]["dtype"]
        out_shape = self._output_details[0]["shape"]
        print(f"Model: {shape} {dtype} -> {out_shape}")
        print(f"Emotions: {EMOTIONS}")

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run inference on a BGR face crop. Returns float32 array of shape (7,)."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        input_tensor = resized.astype(np.float32).reshape(
            1, IMG_SIZE[0], IMG_SIZE[1], 3
        )
        self.interpreter.set_tensor(self._input_idx, input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._output_idx)[0]
