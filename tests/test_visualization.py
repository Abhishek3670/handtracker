import numpy as np
from types import SimpleNamespace
from handtracking.visualization import HUDOverlay, PipelineTelemetry, StageLatency, StageTimer
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D

def make_hand():
    points = tuple(Landmark3D(.2 + i*.01, .2 + i*.005, 0) for i in range(21))
    return HandLandmarks(points, Handedness("Left", .9), BoundingBox.from_landmarks(points))

def test_stage_timer_and_telemetry_metrics():
    with StageTimer() as timer: pass
    assert timer.elapsed_us >= 0
    telemetry = PipelineTelemetry()
    telemetry.record(StageLatency(total_ms=2), 0)
    telemetry.record(StageLatency(total_ms=4), .1)
    assert telemetry.instant_fps == 10
    assert telemetry.averages().total_ms == 3

def test_hud_draws_in_place_on_synthetic_frame():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert HUDOverlay().draw(frame) is frame

def test_hud_draws_pinch_and_stage_telemetry():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    telemetry = PipelineTelemetry(); telemetry.record(StageLatency(inference_ms=4.2, smoothing_ms=.1, gestures_ms=.1, render_ms=.8, total_ms=5.2), 0)
    result = SimpleNamespace(gesture="PINCH", pinch_distance=.1, is_pinch=True)
    output = HUDOverlay().draw(frame, [make_hand()], [result], telemetry)
    assert output is frame and int(frame.sum()) > 0
