import numpy as np
from types import SimpleNamespace
from handtracking.ar import ARPhysicsEngine, BallRenderer, BallSkin
from handtracking.config import MediaConfig
from handtracking.controllers import KeySynthesizer, MediaController
from handtracking.inference import DetectionResult
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D
from handtracking.pipeline import HandTrackingPipeline
from handtracking.gestures import AirCanvas, GestureRecognizer, GestureResult, GestureType, TemporalGestureRecognizer
from handtracking.visualization import HUDOverlay, MediaHUDOverlay


class Detector:
    def detect(self, frame):
        return DetectionResult(timestamp=1.0)


def make_test_hand(x=0.5, y=0.5):
    points = tuple(Landmark3D(x, y, 0) for _ in range(21))
    return HandLandmarks(points, Handedness("Right", 0.95), BoundingBox.from_landmarks(points))


def test_pipeline_mock_flow():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    output, gestures, telemetry = HandTrackingPipeline(detector=Detector()).process_frame(frame)
    assert output is frame and gestures == [] and telemetry.latency.total_ms >= 0


def test_pipeline_with_temporal_and_canvas():
    class MovingHandDetector:
        def __init__(self):
            self.frame_idx = 0

        def detect(self, frame):
            x = 0.2 + (self.frame_idx * 0.1)
            self.frame_idx += 1
            return DetectionResult(
                hands=(make_test_hand(x=x, y=0.5),),
                timestamp=self.frame_idx * 0.1,
                inference_latency_ms=2.0,
            )

    canvas = AirCanvas()
    temporal = TemporalGestureRecognizer(window_size=10, swipe_threshold=0.15, min_duration=0.05)
    hud = HUDOverlay()

    pipe = HandTrackingPipeline(
        detector=MovingHandDetector(),
        temporal=temporal,
        canvas=canvas,
        hud=hud,
        smoothing=False,
    )

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # Process several frames to trigger swipe
    for _ in range(4):
        out_frame, gestures, telemetry = pipe.process_frame(frame)

    assert out_frame is frame
    # Verify temporal recognition populated HUD notifications
    assert any("Swipe Right" in n.text for n in hud.notifications)
    pipe.close()


def test_pipeline_with_media_controller():
    class FakeDetector:
        def detect(self, frame):
            return DetectionResult(
                hands=(make_test_hand(x=0.5, y=0.5),),
                timestamp=1.0,
            )

    synthesizer = KeySynthesizer(dry_run=True)
    media_ctrl = MediaController(synthesizer=synthesizer, initial_volume=70)
    media_ctrl.state_machine.wake(timestamp=0.5)

    pipe = HandTrackingPipeline(
        detector=FakeDetector(),
        media_controller=media_ctrl,
        smoothing=False,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out_frame, gestures, telemetry = pipe.process_frame(frame)
    assert out_frame is frame
    assert pipe.media_hud is not None
    pipe.close()


def test_pipeline_with_ar_ball():
    class FakeDetector:
        def detect(self, frame):
            return DetectionResult(
                hands=(make_test_hand(x=0.5, y=0.5),),
                timestamp=1.0,
            )

    ar_physics = ARPhysicsEngine()
    ar_renderer = BallRenderer(skin=BallSkin.CHROME)

    pipe = HandTrackingPipeline(
        detector=FakeDetector(),
        ar_physics=ar_physics,
        ar_renderer=ar_renderer,
        smoothing=False,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out_frame, gestures, telemetry = pipe.process_frame(frame)
    assert out_frame is frame
    assert pipe.ar_physics.ball.position is not None
    pipe.close()
