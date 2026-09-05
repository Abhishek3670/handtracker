"""Touchless Media Controller HUD overlay with radial volume dial, wake ring, and status badges."""
from __future__ import annotations
import math
import time
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

from ..controllers.state_machine import ControllerState


class MediaHUDOverlay:
    """Renders floating state badge, radial volume dial, circular wake progress ring, and action toasts."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def draw(self, frame, controller: Any, timestamp: float | None = None):
        """Draw all media HUD components in-place on frame."""
        if not self.enabled or controller is None:
            return frame

        ts = time.time() if timestamp is None else float(timestamp)
        h, w = frame.shape[:2]

        self._draw_status_badge(frame, controller, ts, w, h)
        self._draw_wake_progress_ring(frame, controller, ts, w, h)
        self._draw_radial_volume_dial(frame, controller, ts, w, h)
        self._draw_action_toast(frame, controller, ts, w, h)
        return frame

    render = draw

    def _draw_status_badge(self, frame, controller, ts: float, width: int, height: int) -> None:
        state = controller.state_machine.state
        if state == ControllerState.SLEEPING:
            text = "SLEEPING (Hold Open Palm 1s to Wake)"
            bg_color = (40, 40, 40)
            text_color = (180, 180, 180)
            border_color = (80, 80, 80)
        elif state == ControllerState.WAKING:
            pct = int(controller.state_machine.hold_progress * 100)
            text = f"WAKING... {pct}%"
            bg_color = (20, 60, 90)
            text_color = (0, 220, 255)
            border_color = (0, 220, 255)
        else:  # ACTIVE
            time_left = controller.state_machine.time_until_sleep(ts)
            text = f"ACTIVE (Idle in {time_left:.1f}s)"
            bg_color = (20, 70, 20)
            text_color = (50, 255, 100)
            border_color = (50, 255, 100)

        badge_x = width - 290
        badge_y = 15
        badge_w = 275
        badge_h = 32

        if cv2 is not None:
            # Draw rounded box
            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), bg_color, -1)
            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), border_color, 1, cv2.LINE_AA)
            cv2.putText(frame, text, (badge_x + 10, badge_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)
        else:
            # Fallback for numpy
            y1 = max(0, badge_y)
            y2 = min(height, badge_y + badge_h)
            x1 = max(0, badge_x)
            x2 = min(width, badge_x + badge_w)
            frame[y1:y2, x1:x2] = text_color

    def _draw_wake_progress_ring(self, frame, controller, ts: float, width: int, height: int) -> None:
        if not controller.state_machine.is_waking:
            return
        progress = controller.state_machine.hold_progress
        cx, cy = width // 2, height // 2
        radius = 45

        if cv2 is not None:
            # Background track
            cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 4, cv2.LINE_AA)
            # Progress arc
            end_angle = -90 + (progress * 360)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, end_angle, (0, 220, 255), 4, cv2.LINE_AA)
            pct_text = f"{int(progress * 100)}%"
            (tw, th), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, pct_text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        else:
            if 0 <= cy < height and 0 <= cx < width:
                frame[max(0, cy - radius):min(height, cy + radius), max(0, cx - radius):min(width, cx + radius)] = (0, 220, 255)

    def _draw_radial_volume_dial(self, frame, controller, ts: float, width: int, height: int) -> None:
        # Radial volume dial in top right area below badge
        cx = width - 70
        cy = 95
        radius = 32
        vol = controller.volume
        muted = controller.is_muted

        if cv2 is not None:
            # Dial background
            cv2.circle(frame, (cx, cy), radius + 6, (25, 25, 25), -1)
            cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 3, cv2.LINE_AA)

            # Arc from 135 deg to 405 deg (270 deg range)
            start_deg = 135
            total_deg = 270
            fill_deg = start_deg + (vol / 100.0) * total_deg
            color = (80, 80, 240) if muted else (0, 220, 255) if vol > 70 else (50, 255, 100)

            if vol > 0 and not muted:
                cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_deg, fill_deg, color, 4, cv2.LINE_AA)

            vol_text = "MUTE" if muted else f"{vol}%"
            (tw, th), _ = cv2.getTextSize(vol_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, vol_text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, "VOL", (cx - 12, cy - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            if 0 <= cy < height and 0 <= cx < width:
                frame[max(0, cy - radius):min(height, cy + radius), max(0, cx - radius):min(width, cx + radius)] = (0, 220, 255)

    def _draw_action_toast(self, frame, controller, ts: float, width: int, height: int) -> None:
        toast = controller.get_active_toast(ts)
        if not toast:
            return

        toast_text = f"🎵 {toast}"
        y_pos = height - 45
        if cv2 is not None:
            (tw, th), _ = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx = max(10, (width - tw) // 2)
            cv2.rectangle(frame, (tx - 12, y_pos - th - 8), (tx + tw + 12, y_pos + 8), (20, 20, 20), -1)
            cv2.rectangle(frame, (tx - 12, y_pos - th - 8), (tx + tw + 12, y_pos + 8), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, toast_text, (tx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            y_idx = min(max(0, y_pos), height - 1)
            x_start = max(0, width // 4)
            x_end = min(width, (3 * width) // 4)
            frame[y_idx, x_start:x_end] = (0, 255, 255)
