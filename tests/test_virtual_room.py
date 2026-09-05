import numpy as np
import pytest

from handtracking.ar.colliders import PalmCollider
from handtracking.ar.physics import ARPhysicsEngine, BallState
from handtracking.ar.room import Virtual3DRoomRenderer
from handtracking.demo import build_parser
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D
from handtracking.pipeline import HandTrackingPipeline
from handtracking.visualization import HUDOverlay


def make_test_hand(x=0.5, y=0.5, z=0.0):
    points = [Landmark3D(x, y, z)] * 21
    points[0] = Landmark3D(x, y + 0.1, z)
    points[5] = Landmark3D(x - 0.05, y, z)
    points[17] = Landmark3D(x + 0.05, y, z)
    # Tips
    points[4] = Landmark3D(x - 0.08, y - 0.05, z)
    points[8] = Landmark3D(x - 0.04, y - 0.1, z)
    points[12] = Landmark3D(x, y - 0.12, z)
    points[16] = Landmark3D(x + 0.04, y - 0.1, z)
    points[20] = Landmark3D(x + 0.08, y - 0.05, z)
    return HandLandmarks(tuple(points), Handedness("Right", 0.95), BoundingBox.from_landmarks(points))


def test_virtual_room_perspective_projection():
    renderer = Virtual3DRoomRenderer(focal_depth=0.85)
    width, height = 640, 480

    # Center point at z=0 should map to screen center (320, 240)
    u0, v0 = renderer.project_3d(0.5, 0.5, 0.0, width, height)
    assert abs(u0 - 320) <= 2
    assert abs(v0 - 240) <= 2

    # Point deeper into the screen (z=0.5) should contract towards center
    u_deep, v_deep = renderer.project_3d(0.8, 0.5, 0.5, width, height)
    u_front, v_front = renderer.project_3d(0.8, 0.5, 0.0, width, height)
    assert u_deep < u_front  # contracted towards center


def test_virtual_room_rendering_components():
    renderer = Virtual3DRoomRenderer(show_pip=True, pip_scale=0.25)
    engine = ARPhysicsEngine()
    engine.ball.position = (0.5, 0.4, 0.1)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_cam = np.ones((480, 640, 3), dtype=np.uint8) * 128
    hand = make_test_hand()

    renderer.render_room(frame, engine, hands=[hand], raw_webcam=raw_cam, timestamp=1.0)

    # Frame should no longer be all black (cyber background and grids drawn)
    assert np.any(frame > 0)

    # Test wall pulse trigger
    renderer.trigger_wall_pulse(timestamp=1.0, color=(0, 255, 200))
    assert renderer.wall_glow_time == 1.0
    assert renderer.wall_glow_color == (0, 255, 200)


def test_pipeline_virtual_room_toggle_and_rendering():
    class MockDetector:
        def detect(self, frame):
            from handtracking.inference.models import DetectionResult
            return DetectionResult(hands=(make_test_hand(),), timestamp=1.0)

    engine = ARPhysicsEngine()
    pipe = HandTrackingPipeline(
        detector=MockDetector(),
        ar_physics=engine,
        virtual_room=False,
    )

    assert pipe.virtual_room is False
    pipe.toggle_virtual_room()
    assert pipe.virtual_room is True

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out, gestures, telemetry = pipe.process_frame(frame)

    assert out.shape == frame.shape
    assert np.any(out > 0)
    pipe.close()


def test_demo_parser_virtual_room_flags():
    parser = build_parser()

    args1 = parser.parse_args(["--virtual-room"])
    assert args1.virtual_room is True

    args2 = parser.parse_args(["--virtual-space"])
    assert args2.virtual_room is True

    args3 = parser.parse_args(["-vr"])
    assert args3.virtual_room is True


def test_pipeline_forwards_projection_fn_to_ar_renderer():
    captured_kwargs = []

    class MockRenderer:
        def draw(self, frame, engine, hands=(), timestamp=None, **kwargs):
            captured_kwargs.append(kwargs)
            return frame

    class MockDetector:
        def detect(self, frame):
            from handtracking.inference.models import DetectionResult
            return DetectionResult(hands=(make_test_hand(),), timestamp=1.0)

    pipe = HandTrackingPipeline(
        detector=MockDetector(),
        ar_physics=ARPhysicsEngine(),
        ar_renderer=MockRenderer(),
        virtual_room=True,
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pipe.process_frame(frame)

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["virtual_room"] is True
    assert captured_kwargs[0]["projection_fn"] is not None
    assert callable(captured_kwargs[0]["projection_fn"])
    pipe.close()

