import pytest
from handtracking.ar.colliders import FingertipCollider, HandVelocityTracker, PalmCollider, PointCollider
from handtracking.ar.physics import ARPhysicsEngine, BallInteractionState, BallState, ImpactRipple
from handtracking.inference.models import BoundingBox, Handedness, HandLandmarks, Landmark3D


def make_test_hand(wrist=(0.5, 0.5, 0.0), index_mcp=(0.45, 0.4, 0.0), pinky_mcp=(0.55, 0.4, 0.0)):
    points = [Landmark3D(0.5, 0.5, 0.0)] * 21
    points[0] = Landmark3D(*wrist)
    points[5] = Landmark3D(*index_mcp)
    points[9] = Landmark3D(wrist[0], wrist[1] - 0.18, wrist[2])
    points[17] = Landmark3D(*pinky_mcp)
    # Tips
    points[4] = Landmark3D(0.4, 0.35, 0.0)  # Thumb tip
    points[8] = Landmark3D(0.45, 0.3, 0.0)  # Index tip
    points[12] = Landmark3D(0.5, 0.28, 0.0)  # Middle tip
    points[16] = Landmark3D(0.55, 0.3, 0.0)  # Ring tip
    points[20] = Landmark3D(0.6, 0.35, 0.0)  # Pinky tip
    return HandLandmarks(tuple(points), Handedness("Right", 0.95), BoundingBox.from_landmarks(points))


def test_palm_collider_math_and_collision():
    hand = make_test_hand(wrist=(0.5, 0.5, 0.0), index_mcp=(0.45, 0.4, 0.0), pinky_mcp=(0.55, 0.4, 0.0))
    palm = PalmCollider.from_hand(hand)

    # Origin should be near centroid of (0, 5, 17)
    assert pytest.approx(palm.origin[0], 0.01) == 0.5
    assert pytest.approx(palm.origin[1], 0.01) == 0.433
    assert palm.radius > 0.05

    # Point directly above palm center
    test_pt = (palm.origin[0], palm.origin[1], palm.origin[2] - 0.04)
    dist = palm.distance_to_plane(test_pt)
    assert abs(dist) <= 0.05

    # Check collision with ball of radius 0.05
    is_col, col_dist, normal = palm.check_collision(test_pt, ball_radius=0.05)
    assert is_col is True

    # Ball far away should not collide
    far_pt = (0.1, 0.1, 0.5)
    is_col_far, _, _ = palm.check_collision(far_pt, ball_radius=0.05)
    assert is_col_far is False


def test_fingertip_collider():
    hand = make_test_hand()
    fingertips = FingertipCollider.from_hand(hand, tip_radius=0.03)

    # Ball right on index fingertip (8)
    tip8_pos = (hand.landmarks[8].x, hand.landmarks[8].y, hand.landmarks[8].z)
    hit, tip_idx, normal = fingertips.check_collision(tip8_pos, ball_radius=0.05)
    assert hit is True
    assert tip_idx == 8

    # Ball far from all fingertips
    hit_far, tip_idx_far, _ = fingertips.check_collision((0.9, 0.9, 0.0), ball_radius=0.05)
    assert hit_far is False
    assert tip_idx_far == -1


def test_hand_velocity_tracker():
    tracker = HandVelocityTracker()
    v0 = tracker.update("hand1", (0.5, 0.5, 0.0), timestamp=1.0)
    assert v0 == (0.0, 0.0, 0.0)

    # Move right by 0.1 in 0.1s -> velocity ~ 1.0 in x
    v1 = tracker.update("hand1", (0.6, 0.5, 0.0), timestamp=1.1)
    assert v1[0] > 0.5
    assert tracker.get_velocity("hand1") == v1

    tracker.reset("hand1")
    assert tracker.get_velocity("hand1") == (0.0, 0.0, 0.0)


def test_physics_engine_gravity_and_floor_bounce():
    engine = ARPhysicsEngine(gravity=(0.0, 2.0, 0.0), restitution=0.8)
    engine.ball.position = (0.5, 0.5, 0.0)
    engine.ball.velocity = (0.0, 0.0, 0.0)

    # Step physics for several frames
    for i in range(10):
        engine.step(dt=0.016, timestamp=i * 0.016)

    # Ball should have fallen downwards (y increased)
    assert engine.ball.position[1] > 0.5
    assert engine.ball.velocity[1] > 0

    # Force ball to bottom floor to test elastic bounce
    engine.ball.position = (0.5, 0.94, 0.0)
    engine.ball.velocity = (0.0, 1.5, 0.0)
    engine.step(dt=0.016, timestamp=1.0)

    # Velocity in y should be inverted (negative / moving upward)
    assert engine.ball.velocity[1] < 0


def test_physics_engine_pinch_grab_and_throw():
    engine = ARPhysicsEngine()
    engine.ball.position = (0.42, 0.32, 0.0)
    engine.ball.velocity = (0.0, 0.0, 0.0)

    # Hand with pinch at thumb (4) and index (8) close to ball
    hand = make_test_hand()
    pts = list(hand.landmarks)
    # Put pinch at (0.42, 0.32, 0.0)
    pts[4] = Landmark3D(0.42, 0.32, 0.0)
    pts[8] = Landmark3D(0.42, 0.32, 0.0)
    pinching_hand = HandLandmarks(tuple(pts), hand.handedness, hand.bounding_box)

    # Step 1: Grab ball
    engine.step(hands=[pinching_hand], timestamp=10.0)
    assert engine.ball.state == BallInteractionState.GRABBED
    assert engine.ball.position == (0.42, 0.32, 0.0)

    # Step 2: Move pinch point
    pts[4] = Landmark3D(0.60, 0.20, 0.0)
    pts[8] = Landmark3D(0.60, 0.20, 0.0)
    moved_hand = HandLandmarks(tuple(pts), hand.handedness, hand.bounding_box)
    engine.step(hands=[moved_hand], timestamp=10.1)
    assert engine.ball.state == BallInteractionState.GRABBED
    assert engine.ball.position == (0.60, 0.20, 0.0)

    # Step 3: Release pinch (open hand) -> Thrown
    pts[4] = Landmark3D(0.2, 0.4, 0.0)
    pts[8] = Landmark3D(0.8, 0.1, 0.0)
    open_hand = HandLandmarks(tuple(pts), hand.handedness, hand.bounding_box)
    engine.step(hands=[open_hand], timestamp=10.2)
    assert engine.ball.state == BallInteractionState.FREE_FLIGHT


def test_impact_ripple_animation():
    ripple = ImpactRipple(center=(0.5, 0.5, 0.0), birth_time=10.0, lifetime=0.3)
    assert ripple.alpha == 1.0

    # Halfway
    still_alive = ripple.update(10.15)
    assert still_alive is True
    assert 0.0 < ripple.alpha < 1.0
    assert ripple.radius > 0.015

    # Expired
    expired = ripple.update(10.35)
    assert expired is False
    assert ripple.alpha == 0.0


def test_point_collider_and_25d_collisions():
    # 1. PointCollider 3D and 2.5D collision checks
    collider = PointCollider(position=(0.5, 0.5, 0.0), radius=0.04, z_threshold=0.20)

    # A. 3D hit
    hit_3d, dist_3d, normal_3d = collider.check_collision((0.5, 0.52, 0.0), ball_radius=0.05)
    assert hit_3d is True
    assert dist_3d < 0.09

    # B. 2.5D hit with Z offset within threshold (e.g. z = 0.12)
    hit_25d, dist_25d, normal_25d = collider.check_collision((0.5, 0.52, 0.12), ball_radius=0.05)
    assert hit_25d is True

    # C. Miss when screen-space 2D is far
    miss, _, _ = collider.check_collision((0.8, 0.8, 0.0), ball_radius=0.05)
    assert miss is False

    # 2. 2.5D Pinch Grab in Physics Engine with depth variance
    engine = ARPhysicsEngine()
    engine.ball.position = (0.5, 0.5, 0.0)

    hand = make_test_hand()
    pts = list(hand.landmarks)
    # Pinch centered at (0.5, 0.5) with z = 0.15
    pts[4] = Landmark3D(0.49, 0.5, 0.15)
    pts[8] = Landmark3D(0.51, 0.5, 0.15)
    depth_pinching_hand = HandLandmarks(tuple(pts), hand.handedness, hand.bounding_box)

    engine.step(hands=[depth_pinching_hand], timestamp=1.0)
    assert engine.ball.state == BallInteractionState.GRABBED

