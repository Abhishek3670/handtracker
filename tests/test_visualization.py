import numpy as np
from types import SimpleNamespace
from handtracking.visualization import HUDOverlay, MediaHUDOverlay, PipelineTelemetry, StageLatency, StageTimer
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D
from handtracking.controllers import ControllerStateMachine, KeySynthesizer, MediaController

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
    telemetry = PipelineTelemetry()
    telemetry.record(StageLatency(inference_ms=4.2, smoothing_ms=0.1, gestures_ms=0.1, render_ms=0.8, total_ms=5.2), 0)
    result = SimpleNamespace(gesture="PINCH", pinch_distance=0.1, is_pinch=True)
    output = HUDOverlay().draw(frame, [make_hand()], [result], telemetry)
    assert output is frame and int(frame.sum()) > 0


def test_hud_temporal_gesture_notifications():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hud = HUDOverlay()
    hud.notify("Swipe Left", duration=1.0, timestamp=10.0)
    assert len(hud.notifications) == 1
    # Drawing within active duration renders notification banner
    output = hud.draw(frame, temporal_gestures=["Circle CW"], timestamp=10.2)
    assert output is frame and int(frame.sum()) > 0
    assert len(hud.notifications) == 2
    # Drawing after expiration clears past notifications
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    hud.draw(frame2, timestamp=20.0)
    assert len(hud.notifications) == 0


def test_media_hud_overlay_renders_all_states():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    media_hud = MediaHUDOverlay()
    ctrl = MediaController(synthesizer=KeySynthesizer(dry_run=True), initial_volume=60)

    # 1. Sleeping state render
    out1 = media_hud.draw(frame, ctrl, timestamp=10.0)
    assert out1 is frame and int(frame.sum()) > 0

    # 2. Waking state render (with progress ring)
    frame.fill(0)
    ctrl.state_machine.update(["open_palm"], timestamp=10.0)
    ctrl.state_machine.update(["open_palm"], timestamp=10.5)
    out2 = media_hud.draw(frame, ctrl, timestamp=10.5)
    assert out2 is frame and int(frame.sum()) > 0

    # 3. Active state render with radial volume dial & toast
    frame.fill(0)
    ctrl.state_machine.wake(timestamp=11.0)
    ctrl.set_toast("Volume Up 🔊 65%", duration=1.5, timestamp=11.0)
    out3 = media_hud.draw(frame, ctrl, timestamp=11.0)
    assert out3 is frame and int(frame.sum()) > 0
