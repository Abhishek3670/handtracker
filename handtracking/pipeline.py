"""Unified capture-to-render hand tracking pipeline."""
from __future__ import annotations
import time
from .capture import AsyncWebcamCapture
from .inference import create_detector, DetectionResult
from .filtering import HandSmoother
from .gestures import GestureRecognizer, GestureEventDispatcher, TemporalGestureRecognizer, AirCanvas
from .visualization import HUDOverlay, PipelineTelemetry, StageLatency

class HandTrackingPipeline:
    def __init__(self, capture=None, detector=None, smoother=None, recognizer=None, dispatcher=None, hud=None, smoothing=True, temporal=None, canvas=None):
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
        gesture_ms = (time.perf_counter() - gesture_start) * 1000
        
        render_start = time.perf_counter()
        output = self.hud.draw(frame, smoothed, gestures, self.telemetry, temporal_gestures=temporal_gestures, timestamp=timestamp)
        render_ms = (time.perf_counter() - render_start) * 1000
        
        if self.canvas:
            self.canvas.render(output)
            
        latency = StageLatency(inference_ms=infer, smoothing_ms=smooth_ms, gestures_ms=gesture_ms, render_ms=render_ms, total_ms=(time.perf_counter() - started) * 1000)
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
