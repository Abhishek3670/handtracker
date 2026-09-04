"""In-place OpenCV HUD renderer."""
from __future__ import annotations
from typing import Iterable
from ..inference.models import HAND_CONNECTIONS, HandLandmarks, HandednessLabel
try:
    import cv2
except ImportError: cv2 = None

class HUDOverlay:
    def __init__(self, enabled: bool = True): self.enabled = enabled; self.show_telemetry = True
    def draw(self, frame, hands: Iterable[HandLandmarks] = (), results=(), telemetry=None):
        if not self.enabled: return frame
        hands = tuple(hands); results = tuple(results)
        for index, hand in enumerate(hands):
            height, width = frame.shape[:2]; points = [(round(p.x*(width-1)), round(p.y*(height-1))) for p in hand.landmarks]
            left = hand.handedness.label == HandednessLabel.LEFT; color = (255, 180, 40) if left else (40, 220, 80)
            if cv2 is None:
                for x, y in points:
                    if 0 <= y < height and 0 <= x < width: frame[y, x] = color
                for a, b in HAND_CONNECTIONS:
                    for fraction in (.25, .5, .75):
                        x = round(points[a][0] + fraction*(points[b][0]-points[a][0])); y = round(points[a][1] + fraction*(points[b][1]-points[a][1]))
                        if 0 <= y < height and 0 <= x < width: frame[y, x] = color
            else:
                for a,b in HAND_CONNECTIONS: cv2.line(frame, points[a], points[b], color, 1, cv2.LINE_AA)
                for i, point in enumerate(points): cv2.circle(frame, point, 5 if i in (4,8,12,16,20) else 3, color, -1, cv2.LINE_AA)
            gesture_result = results[index] if index < len(results) else None
            gesture_name = getattr(getattr(gesture_result, "gesture", ""), "value", getattr(gesture_result, "gesture", ""))
            pinch_distance = getattr(gesture_result, "pinch_distance", 999.0)
            if bool(getattr(gesture_result, "is_pinch", False)) or str(gesture_name).lower() == "pinch" or pinch_distance < .35:
                if cv2 is None:
                    for fraction in tuple(i / 10 for i in range(11)):
                        x = round(points[4][0] + fraction*(points[8][0]-points[4][0])); y = round(points[4][1] + fraction*(points[8][1]-points[4][1]))
                        if 0 <= y < height and 0 <= x < width: frame[y, x] = (0, 220, 255)
                else:
                    cv2.line(frame, points[4], points[8], (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, points[4], 7, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, points[8], 7, (0, 220, 255), 2, cv2.LINE_AA)
            x1,y1,x2,y2 = hand.bounding_box.pixel_coordinates(width, height)
            if cv2 is not None: cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
            label = getattr(gesture_result, "gesture", "") if gesture_result is not None else ""
            if cv2 is not None: cv2.putText(frame, str(getattr(label, "value", label)), (x1, max(14,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        if telemetry is not None and self.show_telemetry: self.draw_telemetry(frame, telemetry)
        return frame
    render = draw

    def draw_telemetry(self, frame, telemetry):
        """Draw the compact per-stage latency breakdown in-place."""
        if cv2 is None: return frame
        metrics = telemetry.averages() if getattr(telemetry, "samples", ()) else telemetry.latency
        text = (f"FPS {telemetry.smoothed_fps:.1f} | Inf {metrics.inference_ms:.2f}ms | "
                f"Smooth {metrics.smoothing_ms:.2f}ms | Gest {metrics.gestures_ms:.2f}ms | "
                f"Render {metrics.render_ms:.2f}ms | Total {metrics.total_ms:.2f}ms")
        cv2.putText(frame, text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, .42, (230,230,230), 1, cv2.LINE_AA)
        return frame
