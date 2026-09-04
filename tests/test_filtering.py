from statistics import pstdev

import pytest

from handtracking.filtering import HandSmoother, LandmarkSmoother3D, LowPassFilter, OneEuroFilter
from handtracking.inference.models import BoundingBox, Handedness, HandednessLabel, HandLandmarks, Landmark3D


def make_hand(x: float, label: HandednessLabel = HandednessLabel.LEFT) -> HandLandmarks:
    points = tuple(Landmark3D(x + i * .001, .2 + i * .001, -.1) for i in range(21))
    return HandLandmarks(points, Handedness(label, .9), BoundingBox.from_landmarks(points))


def test_low_pass_math_and_step_response():
    filt = LowPassFilter(.25)
    assert filt.filter(0) == 0
    assert filt.filter(1) == pytest.approx(.25)
    assert filt.filter(1) == pytest.approx(.4375)


def test_one_euro_suppresses_stationary_noise_and_has_no_step_overshoot():
    filt = OneEuroFilter(min_cutoff=1, beta=0, d_cutoff=1)
    values = [0.0, .05, -.05, .04, -.04] * 20
    output = [filt.filter(value, i / 60) for i, value in enumerate(values)]
    assert pstdev(output[20:]) < .2 * pstdev(values)

    step = OneEuroFilter(beta=0)
    response = [step.filter(value, i / 60) for i, value in enumerate([0, 1, 1, 1])]
    assert max(response) <= 1


def test_landmark_smoother_coordinates_and_reset():
    smoother = LandmarkSmoother3D(beta=0)
    assert smoother.filter(Landmark3D(1, 2, 3, .7), 0) == Landmark3D(1, 2, 3, .7)
    assert smoother.filter(Landmark3D(3, 4, 5), 1 / 60).x < 3
    smoother.reset()
    assert smoother.filter(Landmark3D(9, 8, 7), 0) == Landmark3D(9, 8, 7)


def test_hand_smoother_isolates_hands_and_rebuilds_box():
    smoother = HandSmoother(beta=0)
    left = smoother.process(make_hand(0), 0)
    right = smoother.process(make_hand(10, HandednessLabel.RIGHT), 0)
    assert left.landmarks[0].x == 0
    assert right.landmarks[0].x == 10
    updated = smoother.process(make_hand(2), 1 / 60)
    assert updated.landmarks[0].x < 2
    assert updated.bounding_box == BoundingBox.from_landmarks(updated.landmarks)
    assert updated.palm_center == HandLandmarks(updated.landmarks, updated.handedness,
                                                  updated.bounding_box).palm_center


def test_hand_smoother_timeout_resets_state():
    smoother = HandSmoother(beta=0, timeout=.5)
    smoother.process(make_hand(0), 0)
    assert smoother.process(make_hand(10), .1).landmarks[0].x < 10
    assert smoother.process(make_hand(20), 1.0).landmarks[0].x == 20

