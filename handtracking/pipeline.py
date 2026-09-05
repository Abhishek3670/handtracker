"""Unified capture-to-render hand tracking pipeline."""
from __future__ import annotations
import time
import numpy as np
from .ar import ARPhysicsEngine, BallRenderer, Virtual3DRoomRenderer
from .capture import AsyncWebcamCapture
from .filtering import HandSmoother
from .gestures import AirCanvas, GestureEventDispatcher, GestureRecognizer, TemporalGestureRecognizer
from .inference import DetectionResult, create_detector
from .visualization import HUDOverlay, MediaHUDOverlay, PipelineTelemetry, StageLatency


class HandTrackingPipeline:
    def __init__(
        self,
        capture=None,
        detector=None,
        smoother=None,
        recognizer=None,
        dispatcher=None,
        hud=None,
        smoothing=True,
        temporal=None,
        canvas=None,
        media_controller=None,
        media_hud=None,
        ar_physics=None,
        ar_renderer=None,
        room_renderer=None,
        virtual_room: bool = False,
    ):
        self.capture = capture
        self.detector = detector
        self.smoother = smoother or HandSmoother()
        self.recognizer = recognizer or GestureRecognizer()
        self.dispatcher = dispatcher or GestureEventDispatcher()
        self.hud = hud or HUDOverlay()
        self.smoothing = smoothing
        self.telemetry = PipelineTelemetry()
        self.temporal = temporal
        self.canvas = canvas
        self.media_controller = media_controller
        self.media_hud = media_hud or (MediaHUDOverlay() if media_controller is not None else None)
        self.ar_physics = ar_physics
        self.ar_renderer = ar_renderer or (BallRenderer() if ar_physics is not None else None)
        self.room_renderer = room_renderer or (Virtual3DRoomRenderer() if ar_physics is not None else None)
        self.virtual_room = virtual_room

    def toggle_virtual_room(self) -> bool:
        """Toggle 3D digital cyber room rendering mode."""
        self.virtual_room = not self.virtual_room
        return self.virtual_room

    def process_frame(self, frame):
        started = time.perf_counter()
        result = self.detector.detect(frame)
        infer = getattr(result, "inference_latency_ms", 0.0)
        hands = result.hands
        timestamp = getattr(result, "timestamp", time.time()) or time.time()

        smooth_start = time.perf_counter()
        smoothed = self.smoother.smooth(hands, timestamp) if self.smoothing else hands
        smooth_ms = (time.perf_counter() - smooth_start) * 1000

        gesture_start = time.perf_counter()
        gestures = [self.recognizer.recognize(h) for h in smoothed]
        temporal_gestures = []
        for hand, gesture in zip(smoothed, gestures):
            hand_id = hand.handedness.label
            self.dispatcher.update(hand_id, gesture, timestamp)
            if self.temporal:
                tg = self.temporal.update(hand_id, hand.landmarks[8], timestamp)
                if tg:
                    temporal_gestures.append(tg)
            if self.canvas:
                if gesture.is_pinch:
                    self.canvas.update((hand.landmarks[8].x, hand.landmarks[8].y), drawing=True)
                else:
                    self.canvas.end_stroke()

        if self.media_controller is not None:
            static_names = [getattr(getattr(g, "gesture", g), "value", getattr(g, "gesture", g)) for g in gestures]
            self.media_controller.process_gestures(static_names, temporal_gestures, timestamp=timestamp)

        if self.ar_physics is not None:
            self.ar_physics.step(smoothed, gestures, timestamp=timestamp)

        gesture_ms = (time.perf_counter() - gesture_start) * 1000

        render_start = time.perf_counter()
        if self.virtual_room and self.room_renderer is not None and self.ar_physics is not None:
            canvas_frame = np.empty_like(frame)
            self.room_renderer.render_room(
                canvas_frame,
                self.ar_physics,
                smoothed,
                raw_webcam=frame,
                timestamp=timestamp,
            )
        else:
            canvas_frame = frame

        output = self.hud.draw(
            canvas_frame,
            smoothed,
            gestures,
            self.telemetry,
            temporal_gestures=temporal_gestures,
            timestamp=timestamp,
            ar_active=(self.ar_physics is not None),
            media_active=(self.media_controller is not None),
            canvas_active=(self.canvas is not None),
        )
        if self.canvas:
            self.canvas.render(output)
        if self.ar_physics is not None and self.ar_renderer is not None:
            proj_fn = self.room_renderer.project_3d if self.room_renderer is not None else None
            focal_depth = getattr(self.room_renderer, "focal_depth", 0.85) if self.room_renderer is not None else 0.85
            self.ar_renderer.draw(
                output,
                self.ar_physics,
                smoothed,
                timestamp=timestamp,
                virtual_room=self.virtual_room,
                projection_fn=proj_fn,
                focal_depth=focal_depth,
            )
        if self.media_hud and self.media_controller is not None:
            self.media_hud.draw(output, self.media_controller, timestamp=timestamp)
        render_ms = (time.perf_counter() - render_start) * 1000

        latency = StageLatency(
            inference_ms=infer,
            smoothing_ms=smooth_ms,
            gestures_ms=gesture_ms,
            render_ms=render_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )
        self.telemetry.record(latency, timestamp, getattr(self.capture, "dropped_frames", 0) if self.capture else 0)
        return output, gestures, self.telemetry

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for owner in (self.capture, self.detector):
            if owner is not None and hasattr(owner, "stop"):
                owner.stop()
            elif owner is not None and hasattr(owner, "close"):
                owner.close()
