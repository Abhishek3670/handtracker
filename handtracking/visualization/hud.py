"""In-place OpenCV HUD renderer with telemetry, notifications, instructions bar, and cheat sheet modal."""
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
        self.show_instructions = True
        self.show_help = False
        self.notifications: list[HUDNotification] = []

    def notify(self, text: str, duration: float = 1.5, timestamp: float | None = None, color: tuple[int, int, int] = (0, 255, 255)):
        now = time.time() if timestamp is None else float(timestamp)
        self.notifications.append(HUDNotification(text=str(text), expires_at=now + duration, color=color))

    add_notification = notify

    def draw(
        self,
        frame,
        hands: Iterable[HandLandmarks] = (),
        results=(),
        telemetry=None,
        temporal_gestures: Iterable[str] = (),
        timestamp: float | None = None,
        mode: str | None = None,
        ar_active: bool = False,
        media_active: bool = False,
        canvas_active: bool = False,
    ):
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
        
        if self.show_instructions:
            self.draw_instructions(frame, mode=mode, ar_active=ar_active, media_active=media_active, canvas_active=canvas_active)
            
        if self.show_help:
            self.draw_help_modal(frame)
            
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

    def draw_instructions(
        self,
        frame,
        mode: str | None = None,
        ar_active: bool = False,
        media_active: bool = False,
        canvas_active: bool = False,
    ):
        """Render context-aware interactive instruction bar at the bottom of the screen."""
        if ar_active or mode == "ar":
            text = "AR Ball: Palm=Bounce | Pinch=Grab/Throw | Tip=Volley | b=Reset | s=Skin | g=Gravity  [h: Help | q: Exit]"
        elif media_active or mode == "media":
            text = "Media: Hold 1s=Wake | Circle=Vol +/- | Peace=Play/Pause | Swipe=Track | w=Wake  [h: Help | q: Exit]"
        elif canvas_active or mode == "canvas":
            text = "Canvas: Pinch=Draw | Open=Lift | c=Clear | 1-4=Colors  [h: Help | q: Exit]"
        else:
            text = "Gestures: Fist | Palm | Pinch | Peace | Point | Thumbs | OK  [h: Help | q: Exit]"

        height, width = frame.shape[:2]
        bar_h = 28
        bar_y1 = height - bar_h - 4
        bar_y2 = height - 4

        if cv2 is not None:
            # Semi-transparent background pill
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, bar_y1), (width - 8, bar_y2), (18, 18, 18), -1)
            cv2.rectangle(overlay, (8, bar_y1), (width - 8, bar_y2), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Draw instruction text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            tx = max(16, (width - tw) // 2)
            cv2.putText(frame, text, (tx, bar_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        else:
            y1 = max(0, bar_y1)
            y2 = min(height, bar_y2)
            frame[y1:y2, 8:width - 8] = (40, 40, 40)
        return frame

    def draw_help_modal(self, frame):
        """Render centered semi-transparent cheat sheet modal card detailing all gestures, actions, and hotkeys."""
        height, width = frame.shape[:2]
        modal_w = min(680, width - 40)
        modal_h = min(420, height - 40)
        x1 = (width - modal_w) // 2
        y1 = (height - modal_h) // 2
        x2 = x1 + modal_w
        y2 = y1 + modal_h

        if cv2 is not None:
            # Dark blurred modal card background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 18, 24), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

            # Header
            header = "HAND TRACKING SYSTEM -- HELP & CHEAT SHEET"
            (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, header, (x1 + (modal_w - tw) // 2, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x1 + 20, y1 + 42), (x2 - 20, y1 + 42), (60, 70, 85), 1, cv2.LINE_AA)

            lines = [
                ("[ STATIC GESTURES ]", (255, 200, 50)),
                ("  - Open Palm    : Disengage / Wake Hold", (220, 220, 220)),
                ("  - Fist         : Mute / Grip", (220, 220, 220)),
                ("  - Pinch (Tip)  : Draw (Canvas) / Grab & Throw (AR)", (220, 220, 220)),
                ("  - Peace Sign   : Play / Pause Media", (220, 220, 220)),
                ("  - Pointing     : Direct / Select", (220, 220, 220)),
                ("  - Thumbs Up/Dn : Approve / Reject", (220, 220, 220)),
                ("", (0, 0, 0)),
                ("[ DYNAMIC MOTIONS & MEDIA ]", (50, 255, 150)),
                ("  - Swipe L/R    : Previous / Next Track", (220, 220, 220)),
                ("  - Circle CW/CCW: Volume Up / Down", (220, 220, 220)),
                ("  - Wave Motion  : Wave / Shake", (220, 220, 220)),
                ("", (0, 0, 0)),
                ("[ KEYBOARD SHORTCUTS ]", (255, 120, 220)),
                ("  - 'h' : Toggle this Help Cheat Sheet", (220, 220, 220)),
                ("  - 'q' / ESC : Exit Application", (220, 220, 220)),
                ("  - 'w' : Force Media Wake / Sleep  |  'm' : Toggle Media HUD", (220, 220, 220)),
                ("  - 'b' : Reset AR Ball  |  's' : Cycle Skins  |  'g' : Gravity", (220, 220, 220)),
                ("  - 'c' : Clear Canvas   |  '1'-'4' : Canvas Colors", (220, 220, 220)),
            ]

            curr_y = y1 + 65
            for text, col in lines:
                if text:
                    font_scale = 0.46 if text.startswith("[") else 0.40
                    thickness = 2 if text.startswith("[") else 1
                    cv2.putText(frame, text, (x1 + 25, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness, cv2.LINE_AA)
                curr_y += 18

            # Footer
            close_hint = "Press 'h' to close"
            (ctw, cth), _ = cv2.getTextSize(close_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(frame, close_hint, (x1 + (modal_w - ctw) // 2, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 175), 1, cv2.LINE_AA)
        else:
            frame[y1:y2, x1:x2] = (30, 30, 30)
        return frame
