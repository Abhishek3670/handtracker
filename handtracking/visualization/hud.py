"""In-place OpenCV HUD renderer."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Iterable
from ..inference.models import HAND_CONNECTIONS, HandLandmarks, HandednessLabel
try:
    import cv2
except ImportError:
    cv2 = None

@dataclass
class HUDNotification:
    text: str
    expires_at: float
    color: tuple[int, int, int] = (0, 255, 255)

class HUDOverlay:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.show_telemetry = True
        self.notifications: list[HUDNotification] = []

    def notify(self, text: str, duration: float = 1.5, timestamp: float | None = None, color: tuple[int, int, int] = (0, 255, 255)):
        now = time.time() if timestamp is None else float(timestamp)
        self.notifications.append(HUDNotification(text=str(text), expires_at=now + duration, color=color))

    add_notification = notify

    def draw(self, frame, hands: Iterable[HandLandmarks] = (), results=(), telemetry=None, temporal_gestures: Iterable[str] = (), timestamp: float | None = None):
        if not self.enabled:
            return frame
        now = time.time() if timestamp is None else float(timestamp)
        for tg in temporal_gestures:
            if tg:
                self.notify(str(tg), timestamp=now)
        hands = tuple(hands)
        results = tuple(results)
        for index, hand in enumerate(hands):
            height, width = frame.shape[:2]
            points = [(round(p.x * (width - 1)), round(p.y * (height - 1))) for p in hand.landmarks]
            left = hand.handedness.label == HandednessLabel.LEFT
            color = (255, 180, 40) if left else (40, 220, 80)
            if cv2 is None:
                for x, y in points:
                    if 0 <= y < height and 0 <= x < width:
                        frame[y, x] = color
                for a, b in HAND_CONNECTIONS:
                    for fraction in (.25, .5, .75):
                        x = round(points[a][0] + fraction * (points[b][0] - points[a][0]))
                        y = round(points[a][1] + fraction * (points[b][1] - points[a][1]))
                        if 0 <= y < height and 0 <= x < width:
                            frame[y, x] = color
            else:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, points[a], points[b], color, 1, cv2.LINE_AA)
                for i, point in enumerate(points):
                    cv2.circle(frame, point, 5 if i in (4, 8, 12, 16, 20) else 3, color, -1, cv2.LINE_AA)
            gesture_result = results[index] if index < len(results) else None
            gesture_name = getattr(getattr(gesture_result, "gesture", ""), "value", getattr(gesture_result, "gesture", ""))
            pinch_distance = getattr(gesture_result, "pinch_distance", 999.0)
            if bool(getattr(gesture_result, "is_pinch", False)) or str(gesture_name).lower() == "pinch" or pinch_distance < .35:
                if cv2 is None:
                    for fraction in tuple(i / 10 for i in range(11)):
                        x = round(points[4][0] + fraction * (points[8][0] - points[4][0]))
                        y = round(points[4][1] + fraction * (points[8][1] - points[4][1]))
                        if 0 <= y < height and 0 <= x < width:
                            frame[y, x] = (0, 220, 255)
                else:
                    cv2.line(frame, points[4], points[8], (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, points[4], 7, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, points[8], 7, (0, 220, 255), 2, cv2.LINE_AA)
            x1, y1, x2, y2 = hand.bounding_box.pixel_coordinates(width, height)
            if cv2 is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = getattr(gesture_result, "gesture", "") if gesture_result is not None else ""
            if cv2 is not None:
                cv2.putText(frame, str(getattr(label, "value", label)), (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        if telemetry is not None and self.show_telemetry:
            self.draw_telemetry(frame, telemetry)
        self.draw_notifications(frame, timestamp=now)
        return frame

    render = draw

    def draw_telemetry(self, frame, telemetry):
        """Draw the compact per-stage latency breakdown in-place."""
        if cv2 is None:
            return frame
        metrics = telemetry.averages() if getattr(telemetry, "samples", ()) else telemetry.latency
        text = (f"FPS {telemetry.smoothed_fps:.1f} | Inf {metrics.inference_ms:.2f}ms | "
                f"Smooth {metrics.smoothing_ms:.2f}ms | Gest {metrics.gestures_ms:.2f}ms | "
                f"Render {metrics.render_ms:.2f}ms | Total {metrics.total_ms:.2f}ms")
        cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1, cv2.LINE_AA)
        return frame

    def draw_notifications(self, frame, timestamp: float | None = None):
        """Render transient on-screen text banners / toasts for dynamic temporal gestures."""
        if not self.notifications:
            return frame
        now = time.time() if timestamp is None else float(timestamp)
        self.notifications = [n for n in self.notifications if n.expires_at >= now]
        if not self.notifications:
            return frame
        height, width = frame.shape[:2]
        for i, notif in enumerate(self.notifications[-3:]):
            text = f">> GESTURE: {notif.text} <<"
            y_pos = 50 + (i * 28)
            if cv2 is not None:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx = max(10, (width - tw) // 2)
                cv2.rectangle(frame, (tx - 6, y_pos - th - 4), (tx + tw + 6, y_pos + 4), (20, 20, 20), -1)
                cv2.rectangle(frame, (tx - 6, y_pos - th - 4), (tx + tw + 6, y_pos + 4), notif.color, 1)
                cv2.putText(frame, text, (tx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, notif.color, 2, cv2.LINE_AA)
            else:
                y_idx = min(max(0, y_pos), height - 1)
                x_start = max(0, width // 4)
                x_end = min(width, (3 * width) // 4)
                frame[y_idx, x_start:x_end] = notif.color
        return frame
