#!/usr/bin/env python3
"""Enhanced Facial Expression Recognition Demo.

Orchestrates face detection, emotion classification, prediction smoothing,
statistics tracking, rich UI dashboard, and recording.
"""

import os
import pickle
import sys
import time

import cv2
import numpy as np

from src.config import (
    EMOTIONS, EMOJI_MAP, EMOTION_COLORS,
    SIDE_BY_SIDE_ENABLED, RECORD_VIDEO, RECORD_CSV, PROJECT_ROOT,
)
from src.face_detector import FaceDetector
from src.emotion_classifier import EmotionClassifier
from src.emotion_tracker import EmotionTracker
from src.stats_tracker import StatsTracker
from src.ui_dashboard import UIDashboard
from src.recorder import Recorder


def load_calibrator():
    """Load post-hoc calibration (scale+bias per class). Returns None if unavailable."""
    cal_path = os.path.join(PROJECT_ROOT, "models", "calibrator.pkl")
    cal_path2 = os.path.join(PROJECT_ROOT, "calibrator.pkl")
    for p in [cal_path, cal_path2]:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                cal = pickle.load(f)
            print(f"Calibrator loaded: {p}")
            return cal
    return None


def apply_calibration(raw_preds, calibrator):
    """Apply calibration. Supports softmax_regression (new) or scale+bias (legacy)."""
    if calibrator is None:
        return raw_preds
    # Ensure 2D (batch dimension) for consistent processing
    is_1d = raw_preds.ndim == 1
    X = raw_preds.reshape(1, -1) if is_1d else raw_preds.copy()
    cal_type = calibrator.get("type", "legacy")
    if cal_type == "softmax_regression":
        X_norm = (X - calibrator["mean"]) / calibrator["std"]
        bias_col = np.ones((X.shape[0], 1), dtype=np.float32)
        X_aug = np.concatenate([X_norm, bias_col], axis=1)
        logits = X_aug @ calibrator["W"]
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        result = exp / np.sum(exp, axis=-1, keepdims=True)
    else:
        eps = 1e-10
        log_X = np.log(X + eps)
        logits = log_X * calibrator["scales"] + calibrator["biases"]
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        result = exp / np.sum(exp, axis=-1, keepdims=True)
    return result.flatten() if is_1d else result


def main():
    print("=" * 60)
    print("  Enhanced Facial Expression Recognition Demo")
    print("=" * 60)

    # ── Init components ───────────────────────────────────────────
    print("\n[1/7] Loading emotion model ...")
    classifier = EmotionClassifier()

    print("\n[2/7] Loading calibrator ...")
    calibrator = load_calibrator()

    print("\n[3/7] Initializing face detector ...")
    detector = FaceDetector()

    print("\n[4/7] Setting up camera ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {frame_width}x{frame_height}")

    print("\n[5/7] Initializing tracker and stats ...")
    tracker = EmotionTracker()
    stats_tracker = StatsTracker()

    print("\n[6/7] Building dashboard ...")
    # Use fixed display size for dashboard (fits all screens)
    DISPLAY_W, DISPLAY_H = 480, 360
    dashboard = UIDashboard(DISPLAY_W, DISPLAY_H)

    print("\n[7/7] Preparing recorder ...")
    recorder = Recorder(fps=30.0)

    if RECORD_VIDEO or RECORD_CSV:
        recorder.start_recording((dashboard.dashboard_w, dashboard.dashboard_h))

    cv2.namedWindow("Facial Expression Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Facial Expression Recognition", dashboard.dashboard_w, dashboard.dashboard_h)

    # ── Main loop ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Demo running! Press:")
    print("    q  - Quit")
    print("    r  - Toggle recording")
    print("    s  - Toggle side-by-side mode")
    print("=" * 60 + "\n")

    side_by_side = SIDE_BY_SIDE_ENABLED
    frame_times = []
    fps_display = 30.0

    while True:
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            continue

        # -- Face detection --
        faces = detector.detect(frame)
        predictions = np.zeros(7, dtype=np.float32)

        if faces:
            face = faces[0]  # Process the first (largest) face
            raw_preds = classifier.predict(face.crop)
            # Apply calibration to correct class confusion
            cal_preds = apply_calibration(raw_preds, calibrator)
            smoothed = tracker.update(cal_preds, time.time())
            predictions = smoothed  # smoothed for chart/stats

            stats_tracker.update(smoothed, tracker.elapsed_seconds)

            # Use calibrated RAW prediction for face overlay (instant, responsive)
            cal_idx = int(np.argmax(cal_preds))
            cal_emotion = EMOTIONS[cal_idx]
            cal_confidence = float(cal_preds[cal_idx])
            overlay_label = f"{EMOJI_MAP.get(cal_emotion, '')} {cal_emotion} {cal_confidence*100:.1f}%"
            overlay_color = EMOTION_COLORS.get(cal_emotion, (0, 255, 0))

            dashboard.set_face_bbox(
                face.x1, face.y1, face.x2, face.y2,
                overlay_label, overlay_color,
            )
        else:
            tracker.clear_buffer()

        # -- Build dashboard --
        stats = stats_tracker.get_full_stats()
        elapsed = tracker.elapsed_seconds
        history = tracker.get_history()

        dashboard_frame = dashboard.build(
            camera_frame=frame,
            predictions=predictions,
            history=history,
            stats=stats,
            elapsed_sec=elapsed,
            fps=fps_display,
            is_recording=recorder.is_recording,
            side_by_side=side_by_side,
        )

        # -- Recording --
        if recorder.is_recording:
            recorder.write_frame(dashboard_frame, predictions, elapsed)

        # -- Show --
        cv2.imshow("Facial Expression Recognition", dashboard_frame)

        # -- FPS calculation --
        dt = time.perf_counter() - loop_start
        frame_times.append(dt)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_dt = np.mean(frame_times) if frame_times else 0.016
        fps_display = 1.0 / max(avg_dt, 0.001)
        tracker.fps = fps_display

        # -- Key handling --
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if recorder.is_recording:
                recorder.stop_recording()
                print("Recording stopped.")
            else:
                paths = recorder.start_recording(
                    (dashboard.dashboard_w, dashboard.dashboard_h)
                )
                print(f"Recording started: {paths[0]}")
        elif key == ord("s"):
            side_by_side = not side_by_side
            print(f"Side-by-side mode: {'ON' if side_by_side else 'OFF'}")

    # ── Cleanup ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    recorder.close()

    # Print session summary
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    stats = stats_tracker.get_full_stats()
    print(f"  Duration:        {elapsed:.1f}s")
    print(f"  Frames processed: {stats['total_frames']}")
    print(f"  Dominant emotion: {stats['dominant']} ({stats['dominant_pct']:.0f}%)")
    print(f"  Transitions:      {stats['transitions']}")
    print(f"  Avg FPS:          {fps_display:.0f}")
    print("=" * 60)
    print("Demo exited cleanly.")


if __name__ == "__main__":
    main()
