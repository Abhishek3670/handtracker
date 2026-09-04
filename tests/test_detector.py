import time
from types import SimpleNamespace

import numpy as np

from handtracking.inference.detector import MediaPipeHandDetector
from handtracking.inference.models import (
    BoundingBox, Handedness, HandLandmarks, Landmark3D, WRIST,
)


def landmarks():
    return tuple(Landmark3D(i / 20, i / 20, -i / 100) for i in range(21))


def test_landmark_and_box_pixel_helpers():
    point = Landmark3D(0.5, 0.25, 0)
    assert point.pixel_coordinate(100, 80) == (50, 20)
    box = BoundingBox.from_landmarks(landmarks())
    assert box.pixel_coordinates(100, 100) == (0, 0, 100 - 1, 100 - 1)


def test_hand_calculations():
    points = landmarks()
    hand = HandLandmarks(points, Handedness(Handedness.LEFT, 0.9),
                         BoundingBox.from_landmarks(points))
    assert hand.wrist_position is points[WRIST]
    assert hand.palm_center.x > 0


class FakeHands:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def process(self, frame):
        pts = [SimpleNamespace(x=i / 20, y=i / 20, z=0) for i in range(21)]
        hand = SimpleNamespace(landmark=pts)
        handed = SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.95)])
        return SimpleNamespace(multi_hand_landmarks=[hand], multi_handedness=[handed],
                               multi_hand_world_landmarks=None)

    def close(self):
        self.closed = True


class FakeSolution:
    Hands = FakeHands


def test_detector_converts_mock_result_and_reports_latency():
    detector = MediaPipeHandDetector(hands_solution=FakeSolution)
    result = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8))
    assert len(result.hands) == 1
    assert result.hands[0].handedness.label == Handedness.LEFT
    assert result.inference_latency_ms >= 0
    assert result.error is None
    detector.close()


def test_detector_handles_empty_frame():
    detector = MediaPipeHandDetector(hands_solution=FakeSolution)
    result = detector.detect(None)
    assert not result.detected
    assert result.error == "empty frame"
