"""Comprehensive Unit Tests for Digital AR Baby-Pink Heart on Palm."""
import math
import numpy as np
import pytest

from handtracking.ar.heart import (
    ARHeartEngine,
    HeartState,
    PalmOpennessEstimator,
    generate_heart_mesh_2d,
)
from handtracking.demo import build_parser
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D
from handtracking.pipeline import HandTrackingPipeline


def make_synthetic_hand(
    is_open: bool = True,
    facing_palm: bool = True,
    handedness: str = "Right",
    center_x: float = 0.5,
    center_y: float = 0.5,
    z: float = 0.0,
) -> HandLandmarks:
    """Create synthetic open palm or closed fist landmarks with front/back orientation."""
    points = [Landmark3D(center_x, center_y, z)] * 21

    # Wrist
    points[0] = Landmark3D(center_x, center_y + 0.12, z)

    # Determine lateral orientation (Index vs Pinky) based on handedness and palmar/dorsal side
    is_index_right = (handedness == "Right" and facing_palm) or (handedness == "Left" and not facing_palm)

    if is_index_right:
        points[1] = Landmark3D(center_x + 0.04, center_y + 0.06, z)
        points[2] = Landmark3D(center_x + 0.06, center_y + 0.03, z)
        points[5] = Landmark3D(center_x + 0.04, center_y - 0.02, z)  # Index MCP
        points[9] = Landmark3D(center_x, center_y - 0.03, z)         # Middle MCP
        points[13] = Landmark3D(center_x - 0.03, center_y - 0.02, z) # Ring MCP
        points[17] = Landmark3D(center_x - 0.05, center_y + 0.01, z) # Pinky MCP
        if is_open:
            points[4] = Landmark3D(center_x + 0.12, center_y - 0.02, z)   # Thumb tip
            points[8] = Landmark3D(center_x + 0.05, center_y - 0.18, z)   # Index tip
            points[12] = Landmark3D(center_x, center_y - 0.20, z)         # Middle tip
            points[16] = Landmark3D(center_x - 0.04, center_y - 0.18, z)  # Ring tip
            points[20] = Landmark3D(center_x - 0.08, center_y - 0.15, z)  # Pinky tip
        else:
            points[4] = Landmark3D(center_x + 0.02, center_y + 0.02, z)
            points[8] = Landmark3D(center_x + 0.03, center_y + 0.02, z)
            points[12] = Landmark3D(center_x, center_y + 0.02, z)
            points[16] = Landmark3D(center_x - 0.02, center_y + 0.02, z)
            points[20] = Landmark3D(center_x - 0.04, center_y + 0.03, z)
    else:
        points[1] = Landmark3D(center_x - 0.04, center_y + 0.06, z)
        points[2] = Landmark3D(center_x - 0.06, center_y + 0.03, z)
        points[5] = Landmark3D(center_x - 0.04, center_y - 0.02, z)  # Index MCP
        points[9] = Landmark3D(center_x, center_y - 0.03, z)         # Middle MCP
        points[13] = Landmark3D(center_x + 0.03, center_y - 0.02, z) # Ring MCP
        points[17] = Landmark3D(center_x + 0.05, center_y + 0.01, z) # Pinky MCP
        if is_open:
            points[4] = Landmark3D(center_x - 0.12, center_y - 0.02, z)   # Thumb tip
            points[8] = Landmark3D(center_x - 0.05, center_y - 0.18, z)   # Index tip
            points[12] = Landmark3D(center_x, center_y - 0.20, z)         # Middle tip
            points[16] = Landmark3D(center_x + 0.04, center_y - 0.18, z)  # Ring tip
            points[20] = Landmark3D(center_x + 0.08, center_y - 0.15, z)  # Pinky tip
        else:
            points[4] = Landmark3D(center_x - 0.02, center_y + 0.02, z)
            points[8] = Landmark3D(center_x - 0.03, center_y + 0.02, z)
            points[12] = Landmark3D(center_x, center_y + 0.02, z)
            points[16] = Landmark3D(center_x + 0.02, center_y + 0.02, z)
            points[20] = Landmark3D(center_x + 0.04, center_y + 0.03, z)

    return HandLandmarks(tuple(points), Handedness(handedness, 0.98), BoundingBox.from_landmarks(points))



def test_heart_mesh_generation():
    """Verify parametric 2D heart contour generation and bounds."""
    mesh = generate_heart_mesh_2d(num_points=48)
    assert len(mesh) == 48

    xs = [pt[0] for pt in mesh]
    ys = [pt[1] for pt in mesh]

    # Verify normalization bounds roughly in [-1.0, 1.0]
    assert min(xs) >= -1.05 and max(xs) <= 1.05
    assert min(ys) >= -1.05 and max(ys) <= 1.05

    # Verify symmetry about x axis (x ~ -x for symmetric t)
    for i in range(1, 24):
        p1 = mesh[i]
        p2 = mesh[48 - i]
        assert abs(p1[0] + p2[0]) < 1e-4
        assert abs(p1[1] - p2[1]) < 1e-4


def test_heart_state_defaults():
    """Verify default values and configuration in HeartState."""
    state = HeartState()
    assert state.min_scale == 0.075
    assert state.max_scale == 1.0
    assert state.base_radius == 0.140
    assert state.color_bgr == (193, 182, 255)  # Baby Pink #FFB6C1
    assert state.is_visible is False
    assert state.scale == 1.0
    assert state.openness == 1.0


def test_palm_openness_estimator():
    """Verify continuous openness metric for open palm vs closed fist."""
    open_hand = make_synthetic_hand(is_open=True)
    closed_hand = make_synthetic_hand(is_open=False)

    openness_high = PalmOpennessEstimator.compute_openness(open_hand)
    openness_low = PalmOpennessEstimator.compute_openness(closed_hand)

    assert openness_high >= 0.85
    assert openness_low <= 0.15
    assert openness_high > openness_low


def test_heart_engine_step_and_scaling():
    """Verify heart engine updates position, normal, openness, and scale dynamically."""
    engine = ARHeartEngine(enabled=True)
    open_hand = make_synthetic_hand(is_open=True, center_x=0.4, center_y=0.6)

    # 1. Step with open palm
    for step_i in range(15):
        t = step_i * 0.033
        state = engine.step([open_hand], timestamp=t)

    assert state.is_visible is True
    assert state.openness > 0.70
    assert state.scale > 0.65  # Expanding towards max scale
    assert abs(state.position[0] - 0.4) < 0.1
    assert abs(state.position[1] - 0.6) < 0.1

    # 2. Step with closed fist (shrinks heart towards seed scale)
    closed_hand = make_synthetic_hand(is_open=False, center_x=0.4, center_y=0.6)
    for step_i in range(30):
        t = 0.5 + step_i * 0.033
        state = engine.step([closed_hand], timestamp=t)

    assert state.openness < 0.30
    assert state.scale < 0.40  # Shrinking towards min_scale 0.15
    assert state.scale >= state.min_scale - 0.05

    # 3. Step with no hands (fades out)
    for step_i in range(25):
        t = 1.5 + step_i * 0.033
        state = engine.step([], timestamp=t)

    assert state.is_visible is False or state.alpha <= 0.05


def test_heart_activation_on_open_palm_only():
    """Verify digital heart only activates and appears when an open palm is shown."""
    engine = ARHeartEngine(enabled=True)
    closed_hand = make_synthetic_hand(is_open=False)
    open_hand = make_synthetic_hand(is_open=True)

    # 1. Initially showing closed fist -> heart must NOT appear
    state_closed = engine.step([closed_hand], timestamp=0.0)
    assert state_closed.is_visible is False
    assert state_closed.is_activated is False

    # 2. Opening palm -> heart activates and becomes visible
    for i in range(5):
        state_open = engine.step([open_hand], timestamp=0.1 + i * 0.033)
    assert state_open.is_visible is True
    assert state_open.is_activated is True

    # 3. Closing palm after activation -> stays tracked and shrinks
    for i in range(15):
        state_shrunk = engine.step([closed_hand], timestamp=0.3 + i * 0.033)
    assert state_shrunk.is_visible is True
    assert state_shrunk.scale < 0.40


def test_heart_engine_toggle_and_reset():
    """Verify toggle and reset behavior."""
    engine = ARHeartEngine(enabled=True)
    assert engine.enabled is True

    toggled = engine.toggle()
    assert toggled is False
    assert engine.enabled is False

    engine.toggle()
    assert engine.enabled is True

    engine.state.is_visible = True
    engine.reset()
    assert engine.state.is_visible is False
    assert engine.state.is_activated is False
    assert engine.state.scale == engine.state.max_scale



def test_heart_engine_heartbeat_pulse():
    """Verify heartbeat pulse advances with time."""
    engine = ARHeartEngine(enabled=True, pulse_bpm=80.0)
    open_hand = make_synthetic_hand(is_open=True)

    engine.step([open_hand], timestamp=0.0)
    phase0 = engine.state.pulse_phase

    engine.step([open_hand], timestamp=0.5)
    phase1 = engine.state.pulse_phase

    expected_delta = 2.0 * math.pi * (80.0 / 60.0) * 0.5
    assert abs((phase1 - phase0) - expected_delta) < 0.1


def test_heart_engine_rendering():
    """Verify drawing layered baby-pink heart on an image canvas."""
    engine = ARHeartEngine(enabled=True)
    open_hand = make_synthetic_hand(is_open=True)

    for i in range(10):
        engine.step([open_hand], timestamp=i * 0.033)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = engine.draw(frame, timestamp=0.33)

    assert out is frame
    assert np.any(frame > 0)  # Has rendered colored pixels

    # Test with custom 3D projection function
    def mock_proj(x, y, z, w, h):
        return int(x * w), int(y * h)

    frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
    out2 = engine.draw(frame2, timestamp=0.5, projection_fn=mock_proj)
    assert out2 is frame2
    assert np.any(frame2 > 0)


def test_pipeline_heart_integration():
    """Verify HandTrackingPipeline executes heart step, draw, and toggle."""
    class MockDetector:
        def detect(self, frame):
            from handtracking.inference.models import DetectionResult
            return DetectionResult(hands=(make_synthetic_hand(is_open=True),), timestamp=1.0)

    heart = ARHeartEngine(enabled=True)
    pipe = HandTrackingPipeline(
        detector=MockDetector(),
        heart_engine=heart,
    )

    assert pipe.heart_engine is heart

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out, gestures, telemetry = pipe.process_frame(frame)

    assert out.shape == (240, 320, 3)
    assert np.any(out > 0)
    assert heart.state.is_visible is True

    # Test toggle
    toggled = pipe.toggle_heart()
    assert toggled is False
    assert pipe.heart_engine.enabled is False

    pipe.close()


def test_demo_parser_heart_flags():
    """Verify demo CLI argument parser handles --heart and -ht."""
    parser = build_parser()
    args1 = parser.parse_args(["--heart"])
    assert args1.heart is True

    args2 = parser.parse_args(["-ht"])
    assert args2.heart is True

    args_default = parser.parse_args([])
    assert args_default.heart is False


def test_palm_facing_camera_check():
    """Verify PalmOpennessEstimator correctly identifies palm vs back of hand."""
    r_palm = make_synthetic_hand(facing_palm=True, handedness="Right")
    r_back = make_synthetic_hand(facing_palm=False, handedness="Right")
    l_palm = make_synthetic_hand(facing_palm=True, handedness="Left")
    l_back = make_synthetic_hand(facing_palm=False, handedness="Left")

    assert PalmOpennessEstimator.is_palm_facing_camera(r_palm) is True
    assert PalmOpennessEstimator.is_palm_facing_camera(r_back) is False
    assert PalmOpennessEstimator.is_palm_facing_camera(l_palm) is True
    assert PalmOpennessEstimator.is_palm_facing_camera(l_back) is False


def test_heart_suppression_on_back_of_hand():
    """Verify digital heart is suppressed and does not appear when the back of hand is shown."""
    engine = ARHeartEngine(enabled=True)
    back_hand = make_synthetic_hand(is_open=True, facing_palm=False, handedness="Right")
    palm_hand = make_synthetic_hand(is_open=True, facing_palm=True, handedness="Right")

    # 1. Back of hand -> should not activate/appear
    for i in range(10):
        state = engine.step([back_hand], timestamp=i * 0.033)
    assert state.is_visible is False
    assert state.is_activated is False

    # 2. Turn to palm -> activates and appears
    for i in range(10):
        state = engine.step([palm_hand], timestamp=0.5 + i * 0.033)
    assert state.is_visible is True
    assert state.is_activated is True

    # 3. Turn back to dorsal side -> immediately suppresses/fades out
    for i in range(10):
        state = engine.step([back_hand], timestamp=1.0 + i * 0.033)
    assert state.is_visible is False

