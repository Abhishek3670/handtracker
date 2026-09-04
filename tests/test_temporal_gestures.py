import math
from handtracking.gestures import CircleDetector, TemporalGestureRecognizer, TrajectoryBuffer, WaveDetector
from handtracking.inference.models import Landmark3D
from handtracking.inference.detector import MediaPipeHandDetector


def test_trajectory_math_and_swipe():
    buffer = TrajectoryBuffer(10)
    buffer.add(Landmark3D(0, 0, 0), 0)
    buffer.add(Landmark3D(0.2, 0, 0), 0.1)
    assert buffer.displacement() == (0.2, 0, 0) and buffer.velocity()[0] == 2
    recognizer = TemporalGestureRecognizer(window_size=10, swipe_threshold=0.1, min_duration=0.01)
    assert recognizer.update("left", Landmark3D(0, 0, 0), 0) is None
    assert recognizer.update("left", Landmark3D(0.2, 0, 0), 0.1) is None
    assert recognizer.update("left", Landmark3D(0.3, 0, 0), 0.2) == "Swipe Right"


def test_swipe_directions():
    recognizer = TemporalGestureRecognizer(window_size=10, swipe_threshold=0.1, min_duration=0.01)
    # Swipe Left
    assert recognizer.update("h1", Landmark3D(0.5, 0.5, 0), 0.0) is None
    assert recognizer.update("h1", Landmark3D(0.3, 0.5, 0), 0.1) is None
    assert recognizer.update("h1", Landmark3D(0.1, 0.5, 0), 0.2) == "Swipe Left"

    # Swipe Up (y decreasing in screen coords)
    assert recognizer.update("h2", Landmark3D(0.5, 0.5, 0), 0.0) is None
    assert recognizer.update("h2", Landmark3D(0.5, 0.3, 0), 0.1) is None
    assert recognizer.update("h2", Landmark3D(0.5, 0.1, 0), 0.2) == "Swipe Up"

    # Swipe Down (y increasing)
    assert recognizer.update("h3", Landmark3D(0.5, 0.1, 0), 0.0) is None
    assert recognizer.update("h3", Landmark3D(0.5, 0.3, 0), 0.1) is None
    assert recognizer.update("h3", Landmark3D(0.5, 0.5, 0), 0.2) == "Swipe Down"


def test_circle_detector_cw_and_ccw():
    cw_detector = CircleDetector(window_size=20, min_radius=0.03, min_angle=4.5)
    ccw_detector = CircleDetector(window_size=20, min_radius=0.03, min_angle=4.5)

    # 16-point circle Clockwise: x = 0.5 + 0.1*cos(t), y = 0.5 + 0.1*sin(t)
    detected_cw = None
    for i in range(16):
        t = (2 * math.pi * i) / 15
        x = 0.5 + 0.1 * math.cos(t)
        y = 0.5 + 0.1 * math.sin(t)
        res = cw_detector.update(Landmark3D(x, y, 0), i * 0.05)
        if res:
            detected_cw = res
    assert detected_cw == "Circle CW"

    # 16-point circle Counter-Clockwise: x = 0.5 + 0.1*cos(t), y = 0.5 - 0.1*sin(t)
    detected_ccw = None
    for i in range(16):
        t = (2 * math.pi * i) / 15
        x = 0.5 + 0.1 * math.cos(t)
        y = 0.5 - 0.1 * math.sin(t)
        res = ccw_detector.update(Landmark3D(x, y, 0), i * 0.05)
        if res:
            detected_ccw = res
    assert detected_ccw == "Circle CCW"


def test_wave_detector():
    detector = WaveDetector(window_size=15, min_reversals=2, min_span=0.04)
    # Wave motion: oscillating x coordinates
    xs = [0.2, 0.25, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15, 0.2, 0.25, 0.3, 0.35]
    detected = False
    for i, x in enumerate(xs):
        if detector.update(Landmark3D(x, 0.5, 0), i * 0.05):
            detected = True
    assert detected is True


def test_temporal_gesture_recognizer_integrates_circles_and_waves():
    recognizer = TemporalGestureRecognizer(window_size=20)

    # Circle CW through TemporalGestureRecognizer
    circle_res = None
    for i in range(16):
        t = (2 * math.pi * i) / 15
        x = 0.5 + 0.1 * math.cos(t)
        y = 0.5 + 0.1 * math.sin(t)
        res = recognizer.update("hand_cw", Landmark3D(x, y, 0), i * 0.05)
        if res:
            circle_res = res
    assert circle_res == "Circle CW"

    # Wave through TemporalGestureRecognizer
    wave_res = None
    xs = [0.2, 0.25, 0.3, 0.35, 0.3, 0.25, 0.2, 0.15, 0.2, 0.25, 0.3, 0.35]
    for i, x in enumerate(xs):
        res = recognizer.update("hand_wave", Landmark3D(x, 0.5, 0), i * 0.05)
        if res:
            wave_res = res
    assert wave_res == "Wave"


def test_temporal_buffers_are_isolated():
    recognizer = TemporalGestureRecognizer()
    recognizer.update("a", Landmark3D(0, 0, 0), 0)
    recognizer.update("b", Landmark3D(1, 0, 0), 0)
    assert set(recognizer.buffers) == {"a", "b"}
    recognizer.reset("a")
    assert "a" not in recognizer.buffers and "b" in recognizer.buffers
    recognizer.reset()
    assert len(recognizer.buffers) == 0


def test_model_complexity_is_forwarded():
    received = []
    class Hands:
        def __init__(self, **kwargs):
            received.append(kwargs)
    Solution = type("Solution", (), {"Hands": Hands})
    MediaPipeHandDetector(hands_solution=Solution, model_complexity=0)
    assert received[0]["model_complexity"] == 0
