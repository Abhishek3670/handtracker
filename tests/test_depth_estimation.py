import math
import numpy as np
import pytest

from handtracking.ar.colliders import PalmCollider, FingertipCollider
from handtracking.ar.physics import ARPhysicsEngine, BallInteractionState
from handtracking.ar.room import Virtual3DRoomRenderer
from handtracking.inference.depth import estimate_hand_depth
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D


def make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.32, 0.0), thumb_tip=None, index_tip=None):
    """Helper to build a 21-landmark hand with configurable wrist and middle MCP."""
    pts = [Landmark3D(0.5, 0.5, 0.0)] * 21
    pts[0] = Landmark3D(*p0)
    pts[5] = Landmark3D(p0[0] - 0.05, p0[1] - 0.15, p0[2])
    pts[9] = Landmark3D(*p9)
    pts[17] = Landmark3D(p0[0] + 0.05, p0[1] - 0.15, p0[2])

    t_tip = thumb_tip if thumb_tip is not None else (p0[0] - 0.06, p0[1] - 0.2, p0[2])
    i_tip = index_tip if index_tip is not None else (p0[0] - 0.02, p0[1] - 0.25, p0[2])
    pts[4] = Landmark3D(*t_tip)
    pts[8] = Landmark3D(*i_tip)
    pts[12] = Landmark3D(p0[0], p0[1] - 0.28, p0[2])
    pts[16] = Landmark3D(p0[0] + 0.03, p0[1] - 0.25, p0[2])
    pts[20] = Landmark3D(p0[0] + 0.06, p0[1] - 0.2, p0[2])

    return HandLandmarks(tuple(pts), Handedness("Right", 0.95), BoundingBox.from_landmarks(pts))


def test_estimate_hand_depth_baseline():
    # Wrist at (0.5, 0.5), Middle MCP at (0.5, 0.32) -> dy = 0.18 = ref_span
    hand = make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.32, 0.0))
    z = estimate_hand_depth(hand, ref_span=0.18, gain=0.85)
    assert pytest.approx(z, abs=1e-4) == 0.0


def test_estimate_hand_depth_scaling_and_clamping():
    # 1. Close hand (large palm span: 0.36) -> Z < 0
    close_hand = make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.14, 0.0))
    z_close = estimate_hand_depth(close_hand, ref_span=0.18, gain=0.85)
    # ((0.18 / 0.36) - 1.0) * 0.85 = -0.5 * 0.85 = -0.425
    assert pytest.approx(z_close, abs=1e-3) == -0.425

    # 2. Far hand (small palm span: 0.09) -> Z > 0 (clamped to max_depth 0.55)
    far_hand = make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.41, 0.0))
    z_far = estimate_hand_depth(far_hand, ref_span=0.18, gain=0.85, max_depth=0.55)
    assert z_far == 0.55

    # 3. Very close hand (huge palm span: 0.60) -> clamped to min_depth -0.55
    giant_hand = make_custom_hand(p0=(0.5, 0.7, 0.0), p9=(0.5, 0.1, 0.0))
    z_giant = estimate_hand_depth(giant_hand, ref_span=0.18, gain=0.85, min_depth=-0.55)
    assert z_giant == -0.55


def test_estimate_hand_depth_edge_cases():
    # Degenerate zero distance
    zero_hand = make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.5, 0.0))
    assert estimate_hand_depth(zero_hand) == 0.0

    # Empty / incomplete landmarks
    assert estimate_hand_depth([]) == 0.0
    assert estimate_hand_depth([Landmark3D(0.5, 0.5, 0.0)] * 5) == 0.0


def test_physics_engine_3d_pinch_depth_tracking():
    engine = ARPhysicsEngine()
    engine.ball.position = (0.5, 0.35, 0.0)

    # Hand at baseline depth (Z=0.0) pinching at (0.5, 0.35, 0.0)
    hand_z0 = make_custom_hand(
        p0=(0.5, 0.5, 0.0),
        p9=(0.5, 0.32, 0.0),
        thumb_tip=(0.5, 0.35, 0.0),
        index_tip=(0.5, 0.35, 0.0),
    )
    engine.step(hands=[hand_z0], timestamp=1.0)
    assert engine.ball.state == BallInteractionState.GRABBED
    assert pytest.approx(engine.ball.position[2], abs=0.01) == 0.0

    # Move pinching hand far (span = 0.12 -> Z_hand ~ 0.425)
    # dy = 0.12 -> p9 = (0.5, 0.38)
    # Z_hand = ((0.18/0.12) - 1.0) * 0.85 = 0.5 * 0.85 = 0.425
    hand_far = make_custom_hand(
        p0=(0.5, 0.5, 0.0),
        p9=(0.5, 0.38, 0.0),
        thumb_tip=(0.5, 0.35, 0.0),
        index_tip=(0.5, 0.35, 0.0),
    )
    engine.step(hands=[hand_far], timestamp=1.05)
    assert engine.ball.state == BallInteractionState.GRABBED
    assert engine.ball.position[2] > 0.35


def test_physics_engine_forward_throw_and_wall_bounce():
    engine = ARPhysicsEngine()
    engine.ball.position = (0.5, 0.35, 0.0)

    # Frame 1: Pinch at Z = -0.2
    # L_palm = 0.235 -> Z_hand ~ -0.2
    hand_t1 = make_custom_hand(
        p0=(0.5, 0.5, 0.0),
        p9=(0.5, 0.265, 0.0),
        thumb_tip=(0.5, 0.35, 0.0),
        index_tip=(0.5, 0.35, 0.0),
    )
    engine.step(hands=[hand_t1], timestamp=1.0)
    assert engine.ball.state == BallInteractionState.GRABBED

    # Frame 2: Move hand deep into room (throw motion forward, Z = +0.2)
    # L_palm = 0.145 -> Z_hand ~ +0.2
    hand_t2 = make_custom_hand(
        p0=(0.5, 0.5, 0.0),
        p9=(0.5, 0.355, 0.0),
        thumb_tip=(0.5, 0.35, 0.0),
        index_tip=(0.5, 0.35, 0.0),
    )
    engine.step(hands=[hand_t2], timestamp=1.1)
    assert engine.ball.state == BallInteractionState.GRABBED

    # Frame 3: Release pinch (open hand)
    hand_t3 = make_custom_hand(
        p0=(0.5, 0.5, 0.0),
        p9=(0.5, 0.355, 0.0),
        thumb_tip=(0.2, 0.35, 0.0),
        index_tip=(0.8, 0.35, 0.0),
    )
    engine.step(hands=[hand_t3], timestamp=1.2)
    assert engine.ball.state == BallInteractionState.FREE_FLIGHT
    # Positive forward depth velocity (moving towards back wall)
    assert engine.ball.velocity[2] > 0.5

    # Step physics until ball reaches back wall (bounds_max[2] = 0.6)
    for step_idx in range(50):
        engine.step(dt=0.016, timestamp=1.2 + step_idx * 0.016)
        if engine.last_wall_impact_time is not None:
            break

    assert engine.last_wall_impact_time is not None


def test_room_renderer_holographic_hand_depth():
    room = Virtual3DRoomRenderer(focal_depth=0.85)
    width, height = 640, 480

    close_hand = make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.14, 0.0))  # Z < 0
    far_hand = make_custom_hand(p0=(0.5, 0.5, 0.0), p9=(0.5, 0.41, 0.0))    # Z > 0

    frame_close = np.zeros((height, width, 3), dtype=np.uint8)
    frame_far = np.zeros((height, width, 3), dtype=np.uint8)
    engine = ARPhysicsEngine()

    room.render_room(frame_close, engine, hands=[close_hand], timestamp=1.0)
    room.render_room(frame_far, engine, hands=[far_hand], timestamp=1.0)

    assert np.any(frame_close > 0)
    assert np.any(frame_far > 0)
