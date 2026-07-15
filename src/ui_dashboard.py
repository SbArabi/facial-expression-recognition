"""Full-featured UI dashboard rendered via OpenCV."""

import cv2
import numpy as np

from src.config import (
    EMOTIONS, EMOJI_MAP, EMOTION_COLORS,
    PER_CLASS_THRESHOLD, HISTORY_SECONDS,
)

# ── Layout constants ──────────────────────────────────────────────
PANEL_BG = (30, 30, 30)
PANEL_BORDER = (60, 60, 60)
TEXT_FG = (240, 240, 240)
TEXT_DIM = (160, 160, 160)
ACCENT = (100, 200, 255)

TITLE_HEIGHT = 36
STATS_BAR_HEIGHT = 28
HISTORY_HEIGHT = 140
BARS_WIDTH = 220
PADDING = 8

TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(img, text, x, y, font=FONT, scale=0.5, color=TEXT_FG,
              thickness=1, shadow=True):
    """Draw text with optional drop shadow."""
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), font, scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


def _draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=-1, radius=6):
    """Draw a filled rounded rectangle."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


class UIDashboard:
    """Builds the enhanced UI frame with all panels."""

    def __init__(self, display_w=480, display_h=360):
        self.frame_w = display_w
        self.frame_h = display_h

        # Compute base layouts
        self._bars_width = BARS_WIDTH + PADDING * 3
        self._base_h = (TITLE_HEIGHT + display_h + HISTORY_HEIGHT
                        + STATS_BAR_HEIGHT + PADDING * 5)

        # Dashboard dimensions per mode
        self._single_total = display_w + PADDING
        self._side_total = display_w * 2 + PADDING * 2
        self.single_w = self._single_total + self._bars_width + PADDING * 2
        self.side_w = self._side_total + self._bars_width + PADDING * 2

        self.dashboard_w = self.single_w
        self.dashboard_h = self._base_h

        self._face_bbox = None
        self._face_label = ""
        self._face_color = (0, 255, 0)

    def set_face_bbox(self, x1, y1, x2, y2, label="", color=(0, 255, 0)):
        self._face_bbox = (x1, y1, x2, y2)
        self._face_label = label
        self._face_color = color

    @staticmethod
    def draw_face_overlay(frame, x1, y1, x2, y2, label: str, color=(0, 255, 0)):
        """Draw bounding box + label on a camera frame copy."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 2)
        label_y = max(y1 - 10, 5)
        cv2.rectangle(
            frame,
            (x1 - 2, label_y - th - 4),
            (x1 + tw + 6, label_y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, label, (x1, label_y), FONT, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def build(self, camera_frame, predictions, history, stats,
              elapsed_sec, fps, is_recording, side_by_side=False):
        """Assemble the full dashboard canvas."""
        # Pick dashboard dimensions for this mode
        self.dashboard_w = self.side_w if side_by_side else self.single_w
        self.dashboard_h = self._base_h
        dash = np.full((self.dashboard_h, self.dashboard_w, 3), 20, dtype=np.uint8)

        H, W = PADDING, PADDING

        # Title bar
        title_text = "Facial Expression Recognition"
        _put_text(dash, title_text, W + 10, H + 24,
                  font=TITLE_FONT, scale=0.6, color=ACCENT)

        record_tag = " REC" if is_recording else ""
        fps_text = f"{fps:.0f} FPS"
        _put_text(dash, fps_text, self.dashboard_w - PADDING - 80, H + 24,
                  scale=0.5, color=TEXT_DIM)
        rec_color = (0, 0, 200) if is_recording else TEXT_DIM
        _put_text(dash, record_tag, self.dashboard_w - PADDING - 140, H + 24,
                  scale=0.5, color=rec_color)

        H += TITLE_HEIGHT + PADDING

        # Camera frame (left) — resize first, then annotate
        frame_resized = cv2.resize(camera_frame.copy(), (self.frame_w, self.frame_h))
        if self._face_bbox:
            x1, y1, x2, y2 = self._face_bbox
            # Scale bbox from original camera size to display size
            orig_h, orig_w = camera_frame.shape[:2]
            sx = self.frame_w / orig_w
            sy = self.frame_h / orig_h
            dx1, dy1 = int(x1 * sx), int(y1 * sy)
            dx2, dy2 = int(x2 * sx), int(y2 * sy)
            self.draw_face_overlay(
                frame_resized, dx1, dy1, dx2, dy2,
                self._face_label, self._face_color,
            )
        dash[H:H + self.frame_h, W:W + self.frame_w] = frame_resized

        cam_right_edge = W + self.frame_w

        # Side-by-side: raw unannotated feed
        if side_by_side and self._face_bbox:
            sx = cam_right_edge + PADDING
            frame_raw = cv2.resize(camera_frame.copy(), (self.frame_w, self.frame_h))
            if (H + self.frame_h <= dash.shape[0]
                    and sx + self.frame_w <= dash.shape[1]):
                dash[H:H + self.frame_h, sx:sx + self.frame_w] = frame_raw
            cv2.rectangle(dash, (sx - 2, H - 2),
                          (sx + self.frame_w + 2, H + self.frame_h + 2),
                          PANEL_BORDER, 1)
            _put_text(dash, "Raw input", sx + 4, H + 18,
                      scale=0.4, color=TEXT_DIM)
            cam_right_edge = sx + self.frame_w + PADDING

        # Emotion distribution bars (right panel)
        bars_x = cam_right_edge + PADDING
        bars_y = H + 10
        self._draw_bars(dash, predictions, bars_x, bars_y)

        H += self.frame_h + PADDING

        # Emotion history chart (bottom)
        self._draw_history_chart(dash, history, H, elapsed_sec)

        H += HISTORY_HEIGHT + PADDING

        # Stats bar (bottom)
        self._draw_stats_bar(dash, stats, elapsed_sec, H)

        return dash

    def _draw_bars(self, dash, predictions, x, y):
        """Draw the probability distribution bars."""
        bar_width = 200
        bar_height = 16
        gap = 3
        num_bars = len(EMOTIONS)
        panel_h = num_bars * (bar_height + gap) + 20

        # Panel background
        if y + panel_h <= dash.shape[0] and x + bar_width + 80 <= dash.shape[1]:
            _draw_rounded_rect(dash, x, y, x + bar_width + 80, y + panel_h,
                               (25, 25, 25))

        _put_text(dash, "EMOTION PROBABILITIES", x + 10, y + 16,
                  scale=0.4, color=ACCENT)

        for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
            bx = x + 10
            by = y + 26 + i * (bar_height + gap)
            color = EMOTION_COLORS[emotion]
            threshold = PER_CLASS_THRESHOLD.get(emotion, 0.12)
            filled = int(bar_width * min(prob, 1.0))

            # Bar background
            cv2.rectangle(dash, (bx, by), (bx + bar_width, by + bar_height),
                          (45, 45, 45), -1)

            # Show threshold line
            th_x = bx + int(bar_width * threshold)
            cv2.line(dash, (th_x, by), (th_x, by + bar_height),
                     (100, 100, 100), 1)

            # Filled bar
            if filled > 0:
                cv2.rectangle(dash, (bx, by), (bx + filled, by + bar_height),
                              color, -1)

            # Border
            cv2.rectangle(dash, (bx, by), (bx + bar_width, by + bar_height),
                          (70, 70, 70), 1)

            # Label
            emoji = EMOJI_MAP.get(emotion, "")
            label = f"{emoji} {emotion} {prob * 100:.0f}%"
            label_x = bx + bar_width + 8
            label_y = by + bar_height - 3
            threshold_check = " ✓" if prob >= threshold else ""
            _put_text(dash, label + threshold_check, label_x, label_y,
                      scale=0.4, color=TEXT_FG)

    def _draw_history_chart(self, dash, history, y_start, current_elapsed):
        """Draw a multi-line emotion history chart."""
        chart_x = PADDING
        chart_y = y_start + 8
        chart_w = self.dashboard_w - PADDING * 2
        chart_h = HISTORY_HEIGHT - 16

        if chart_h <= 0 or chart_w <= 0:
            return

        # Panel background
        _draw_rounded_rect(dash, chart_x - 2, chart_y - 2,
                           chart_x + chart_w + 2, chart_y + chart_h + 2,
                           (25, 25, 25))

        # Title
        _put_text(dash, "EMOTION HISTORY", chart_x + 8, chart_y + 16,
                  scale=0.4, color=ACCENT)

        # Chart area (leave room for title + labels)
        plot_x = chart_x + 50
        plot_y = chart_y + 26
        plot_w = chart_w - 60
        plot_h = chart_h - 46

        if plot_w < 10 or plot_h < 10:
            return

        # Grid lines
        grid_color = (45, 45, 45)
        for gi in range(1, 5):
            gy = plot_y + (plot_h // 5) * gi
            cv2.line(dash, (plot_x, gy), (plot_x + plot_w, gy), grid_color, 1)

        if not history or len(history) < 2:
            _put_text(dash, "Waiting for data...", plot_x + plot_w // 2 - 60,
                      plot_y + plot_h // 2, scale=0.4, color=TEXT_DIM)
            return

        times = np.array([t for t, _ in history])
        preds = np.array([p for _, p in history])

        if times[-1] - times[0] < 0.01:
            return

        # Time axis: show last HISTORY_SECONDS
        t_min = max(0, times[-1] - HISTORY_SECONDS)
        t_max = times[-1]
        t_range = max(t_max - t_min, 1.0)

        # Draw per-emotion lines
        line_points = []
        for ei, emotion in enumerate(EMOTIONS):
            color = EMOTION_COLORS[emotion]
            pts = []
            for ti, pi in zip(times, preds[:, ei]):
                if ti < t_min:
                    continue
                px = int(plot_x + (ti - t_min) / t_range * plot_w)
                py = int(plot_y + plot_h - (pi / 1.0) * plot_h)
                py = max(plot_y, min(plot_y + plot_h, py))
                pts.append((px, py))
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(dash, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

        # Y-axis labels
        for yi, val in enumerate([0, 25, 50, 75, 100]):
            ly = plot_y + plot_h - int((val / 100) * plot_h)
            _put_text(dash, f"{val}%", plot_x - 40, ly + 4,
                      scale=0.35, color=TEXT_DIM)

        # X-axis labels
        for ti in range(0, int(HISTORY_SECONDS) + 1, 5):
            tx = int(plot_x + (t_max - max(t_min, t_max - ti)) / t_range * plot_w)
            _put_text(dash, f"-{ti}s", tx - 10, plot_y + plot_h + 14,
                      scale=0.35, color=TEXT_DIM)

        # Legend (compact, bottom-right of chart)
        legend_x = plot_x + plot_w - 120
        legend_y = plot_y + 2
        for li, emotion in enumerate(EMOTIONS):
            lx = legend_x + (li % 4) * 30
            ly = legend_y + (li // 4) * 14
            cv2.rectangle(dash, (lx, ly), (lx + 8, ly + 8),
                          EMOTION_COLORS[emotion], -1)
            _put_text(dash, emotion[:4], lx + 10, ly + 8,
                      scale=0.3, color=TEXT_DIM)

    def _draw_stats_bar(self, dash, stats, elapsed_sec, y):
        """Draw the statistics bar at the bottom."""
        bar_y = y
        bar_h = STATS_BAR_HEIGHT
        bar_w = self.dashboard_w - PADDING * 2

        if bar_y + bar_h > dash.shape[0]:
            return

        _draw_rounded_rect(dash, PADDING, bar_y,
                           PADDING + bar_w, bar_y + bar_h,
                           (25, 25, 25))

        dominant = stats.get("dominant", "?")
        dom_pct = stats.get("dominant_pct", 0)
        transitions = stats.get("transitions", 0)
        frames = stats.get("total_frames", 0)
        current = stats.get("current", "?")
        dist = stats.get("distribution", {})

        dist_text = "  |  ".join(
            f"{e[:3]}: {dist.get(e, 0) * 100:.0f}%"
            for e in EMOTIONS[:4]
        )
        dist_text2 = "  |  ".join(
            f"{e[:3]}: {dist.get(e, 0) * 100:.0f}%"
            for e in EMOTIONS[4:]
        )

        mins = int(elapsed_sec // 60)
        secs = int(elapsed_sec % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        stats_line = (
            f"Session: {time_str}  |  "
            f"Frames: {frames}  |  "
            f"Dominant: {dominant} ({dom_pct:.0f}%)  |  "
            f"Transitions: {transitions}  |  "
            f"Current: {current}  |  "
            f"Recent: {dist_text}"
        )
        _put_text(dash, stats_line, PADDING + 10, bar_y + 18,
                  scale=0.38, color=TEXT_FG)
