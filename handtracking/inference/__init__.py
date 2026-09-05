"""Hand landmark inference APIs."""

from .depth import estimate_hand_depth
from .detector import BaseHandDetector, MediaPipeHandDetector, create_detector
from .models import (
    BoundingBox,
    DetectionResult,
    Handedness,
    HandLandmarks,
    Landmark3D,
)

__all__ = [
    "BaseHandDetector",
    "BoundingBox",
    "DetectionResult",
    "Handedness",
    "HandLandmarks",
    "Landmark3D",
    "MediaPipeHandDetector",
    "create_detector",
    "estimate_hand_depth",
]
