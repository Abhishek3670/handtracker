import numpy as np
from handtracking.inference import DetectionResult
from handtracking.pipeline import HandTrackingPipeline

class Detector:
    def detect(self, frame): return DetectionResult(timestamp=1.0)

def test_pipeline_mock_flow():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    output, gestures, telemetry = HandTrackingPipeline(detector=Detector()).process_frame(frame)
    assert output is frame and gestures == [] and telemetry.latency.total_ms >= 0
