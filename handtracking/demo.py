"""Command-line live demo and synthetic benchmark."""
from __future__ import annotations

import argparse
import time

try:
    import cv2
except ImportError:
    cv2 = None

from .capture import AsyncWebcamCapture
from .inference import create_detector, DetectionResult
from .pipeline import HandTrackingPipeline


def _parse_camera(val: str) -> int | str:
    return int(val) if str(val).isdigit() else val


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HandTracking live demo")
    p.add_argument("--camera", default=0, type=_parse_camera, help="Camera device index (e.g. 0) or video file path")
    p.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    p.add_argument("--height", type=int, default=720, help="Webcam capture height")
    p.add_argument("--no-smoothing", action="store_true", help="Disable 1 Euro adaptive smoothing")
    p.add_argument("--mirror", action="store_true", default=True, help="Mirror webcam display horizontally")
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror mode")
    p.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    p.add_argument("--benchmark", type=int, metavar="N", help="Run synthetic benchmark of N frames")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.benchmark is not None:
        import numpy as np

        class BenchmarkDetector:
            def detect(self, frame: np.ndarray) -> DetectionResult:
                return DetectionResult()

        pipe = HandTrackingPipeline(detector=BenchmarkDetector(), smoothing=not args.no_smoothing)
        sample_frame = np.zeros((args.height or 480, args.width or 640, 3), dtype=np.uint8)
        for _ in range(max(0, args.benchmark)):
            pipe.process_frame(sample_frame)
        print(f"frames={args.benchmark} avg_ms={pipe.telemetry.averages().total_ms:.3f} fps={pipe.telemetry.smoothed_fps:.1f}")
        return 0

    if cv2 is None and not args.headless:
        print("OpenCV (cv2) is required for live GUI display. Please install opencv-python.")
        return 1

    with (
        AsyncWebcamCapture(args.camera, width=args.width, height=args.height) as capture,
        HandTrackingPipeline(capture=capture, detector=create_detector(), smoothing=not args.no_smoothing) as pipe,
    ):
        print(f"HandTracking live feed started (Camera: {args.camera}, {args.width}x{args.height}). Press 'q' to exit.")
        while True:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.001)
                continue
            if args.mirror and cv2 is not None:
                frame = cv2.flip(frame, 1)
            output, _, _ = pipe.process_frame(frame)
            if not args.headless and cv2 is not None:
                cv2.imshow("HandTracking (Press 'q' to exit)", output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

