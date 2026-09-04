"""Typed data models for 21-point hand detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from typing import Iterable


WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP = 5, 6, 7, 8
MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP = 9, 10, 11, 12
RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP), (WRIST, INDEX_FINGER_MCP),
    (INDEX_FINGER_MCP, INDEX_FINGER_PIP), (INDEX_FINGER_PIP, INDEX_FINGER_DIP),
    (INDEX_FINGER_DIP, INDEX_FINGER_TIP), (INDEX_FINGER_MCP, MIDDLE_FINGER_MCP),
    (WRIST, MIDDLE_FINGER_MCP), (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP),
    (MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP), (MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP),
    (MIDDLE_FINGER_MCP, RING_FINGER_MCP), (WRIST, RING_FINGER_MCP),
    (RING_FINGER_MCP, RING_FINGER_PIP), (RING_FINGER_PIP, RING_FINGER_DIP),
    (RING_FINGER_DIP, RING_FINGER_TIP), (RING_FINGER_MCP, PINKY_MCP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),
)


class HandednessLabel(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass(frozen=True)
class Landmark3D:
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def pixel_coordinate(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized x/y coordinates to a clamped pixel position."""
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        return (
            min(width - 1, max(0, round(self.x * (width - 1)))),
            min(height - 1, max(0, round(self.y * (height - 1)))),
        )

    # Convenient spelling used by some renderers.
    to_pixel = pixel_coordinate


@dataclass(frozen=True)
class Handedness:
    label: HandednessLabel | str
    confidence: float

    LEFT = HandednessLabel.LEFT
    RIGHT = HandednessLabel.RIGHT

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("handedness confidence must be between 0 and 1")
        if isinstance(self.label, str) and self.label.lower() in ("left", "right"):
            object.__setattr__(self, "label", HandednessLabel(self.label.title()))


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("bounding box minimum must not exceed maximum")

    @classmethod
    def from_landmarks(cls, landmarks: Iterable[Landmark3D]) -> "BoundingBox":
        points = tuple(landmarks)
        if not points:
            raise ValueError("at least one landmark is required")
        return cls(min(p.x for p in points), min(p.y for p in points),
                   max(p.x for p in points), max(p.y for p in points))

    def pixel_coordinates(self, width: int, height: int) -> tuple[int, int, int, int]:
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        return (
            max(0, min(width - 1, floor(self.x_min * width))),
            max(0, min(height - 1, floor(self.y_min * height))),
            max(0, min(width - 1, ceil(self.x_max * width))),
            max(0, min(height - 1, ceil(self.y_max * height))),
        )

    to_pixels = pixel_coordinates


@dataclass(frozen=True)
class HandLandmarks:
    landmarks: tuple[Landmark3D, ...]
    handedness: Handedness
    bounding_box: BoundingBox
    world_landmarks: tuple[Landmark3D, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError("a hand must contain exactly 21 landmarks")

    @property
    def wrist_position(self) -> Landmark3D:
        return self.landmarks[WRIST]

    @property
    def palm_center(self) -> Landmark3D:
        points = [self.landmarks[i] for i in (WRIST, INDEX_FINGER_MCP,
                                                MIDDLE_FINGER_MCP, RING_FINGER_MCP,
                                                PINKY_MCP)]
        return Landmark3D(*(sum(getattr(p, axis) for p in points) / len(points)
                            for axis in ("x", "y", "z")))


@dataclass(frozen=True)
class DetectionResult:
    hands: tuple[HandLandmarks, ...] = field(default_factory=tuple)
    timestamp: float = 0.0
    inference_latency_ms: float = 0.0
    error: str | None = None

    @property
    def detected(self) -> bool:
        return bool(self.hands)

    @property
    def hand_landmarks(self) -> tuple[HandLandmarks, ...]:
        return self.hands
