"""MediaPipe-backed hand detection with a small backend-neutral interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any

from .models import BoundingBox, DetectionResult, Handedness, HandLandmarks, Landmark3D

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]


class BaseHandDetector(ABC):
    """Interface implemented by hand landmark detection backends."""

    @abstractmethod
    def detect(self, frame: Any) -> DetectionResult:
        raise NotImplementedError

    def process(self, frame: Any) -> DetectionResult:
        return self.detect(frame)

    def close(self) -> None:
        """Release backend resources; backends without resources need no-op."""

    def __enter__(self) -> "BaseHandDetector":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class MediaPipeHandDetector(BaseHandDetector):
    """Live-stream MediaPipe Hands detector.

    MediaPipe is imported lazily at module load and instantiated only when the
    detector is constructed, allowing model/data classes to work in minimal CI.
    """

    def __init__(self, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 *, hands_solution: Any = None) -> None:
        if hands_solution is None:
            if mp is None:
                raise RuntimeError("mediapipe is required for MediaPipeHandDetector")
            hands_solution = mp.solutions.hands
        self._hands = hands_solution.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self.last_error: str | None = None

    def detect(self, frame: Any) -> DetectionResult:
        started = time.perf_counter()
        timestamp = time.time()
        if frame is None or getattr(frame, "size", 1) == 0:
            return DetectionResult(timestamp=timestamp, error="empty frame")
        try:
            rgb = self._to_rgb(frame)
            result = self._hands.process(rgb)
            hands = tuple(self._convert_hand(result, i)
                          for i in range(len(getattr(result, "multi_hand_landmarks", None) or [])))
            self.last_error = None
            return DetectionResult(hands=hands, timestamp=timestamp,
                                   inference_latency_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:
            self.last_error = str(exc)
            return DetectionResult(timestamp=timestamp,
                                   inference_latency_ms=(time.perf_counter() - started) * 1000,
                                   error=str(exc))

    def _to_rgb(self, frame: Any) -> Any:
        if cv2 is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame[:, :, ::-1]

    def _convert_hand(self, result: Any, index: int) -> HandLandmarks:
        raw = result.multi_hand_landmarks[index].landmark
        landmarks = tuple(Landmark3D(float(p.x), float(p.y), float(p.z),
                                     float(getattr(p, "visibility", 1.0))) for p in raw)
        raw_handed = (getattr(result, "multi_handedness", None) or [None] * len(landmarks))[index]
        classification = getattr(raw_handed, "classification", [None])[0]
        label = getattr(classification, "label", "Right")
        confidence = float(getattr(classification, "score", 0.0))
        world = getattr(result, "multi_hand_world_landmarks", None)
        world_points = None
        if world and index < len(world):
            world_points = tuple(Landmark3D(float(p.x), float(p.y), float(p.z),
                                             float(getattr(p, "visibility", 1.0)))
                                 for p in world[index].landmark)
        return HandLandmarks(landmarks, Handedness(label, confidence),
                             BoundingBox.from_landmarks(landmarks), world_points)

    def close(self) -> None:
        close = getattr(self._hands, "close", None)
        if callable(close):
            close()


def create_detector(backend: str = "mediapipe", **kwargs: Any) -> BaseHandDetector:
    """Create a configured detector by backend name."""
    if backend.lower() in ("mediapipe", "mp", "blazehand"):
        return MediaPipeHandDetector(**kwargs)
    raise ValueError(f"unsupported detector backend: {backend}")
