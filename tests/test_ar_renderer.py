import numpy as np
import pytest
from handtracking.ar.physics import ARPhysicsEngine
from handtracking.ar.renderer import BallRenderer, BallSkin, SKIN_CYCLE
from handtracking.inference.models import BoundingBox, Handedness, HandLandmarks, Landmark3D


def make_test_hand():
    points = [Landmark3D(0.5, 0.5, 0.0)] * 21
    points[0] = Landmark3D(0.5, 0.6, 0.0)
    points[5] = Landmark3D(0.45, 0.45, 0.0)
    points[17] = Landmark3D(0.55, 0.45, 0.0)
    return HandLandmarks(tuple(points), Handedness("Right", 0.95), BoundingBox.from_landmarks(points))


def test_ball_renderer_skins_and_cycling():
    renderer = BallRenderer(skin=BallSkin.BASKETBALL)
    assert renderer.skin == BallSkin.BASKETBALL

    s1 = renderer.cycle_skin()
    assert s1 == BallSkin.CHROME
    s2 = renderer.cycle_skin()
    assert s2 == BallSkin.TENNIS
    s3 = renderer.cycle_skin()
    assert s3 == BallSkin.NEON
    s4 = renderer.cycle_skin()
    assert s4 == BallSkin.BASKETBALL

    renderer.set_skin("chrome")
    assert renderer.skin == BallSkin.CHROME
    renderer.set_skin(BallSkin.TENNIS)
    assert renderer.skin == BallSkin.TENNIS


def test_ball_renderer_sphere_sprite_generation():
    renderer = BallRenderer()
    for skin in SKIN_CYCLE:
        sprite = renderer._generate_shaded_sphere(radius_px=24, skin=skin)
        assert sprite.shape == (49, 49, 4)
        assert sprite.dtype == np.uint8
        # Center should be non-transparent
        assert sprite[24, 24, 3] == 255
        # Corner outside circle should be transparent
        assert sprite[0, 0, 3] == 0


def test_ball_renderer_draws_on_synthetic_frame():
    renderer = BallRenderer(skin=BallSkin.BASKETBALL)
    engine = ARPhysicsEngine()
    engine.ball.position = (0.5, 0.4, 0.0)
    engine.spawn_ripple((0.5, 0.5, 0.0), timestamp=10.0)

    hand = make_test_hand()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    out = renderer.draw(frame, engine, hands=[hand], timestamp=10.1)
    assert out is frame
    assert int(frame.sum()) > 0
