"""Command-line live demo and synthetic benchmark."""
from __future__ import annotations
import argparse, time
def build_parser():
    p=argparse.ArgumentParser(description="HandTracking live demo"); p.add_argument("--camera", default=0); p.add_argument("--width", type=int); p.add_argument("--height", type=int); p.add_argument("--no-smoothing", action="store_true"); p.add_argument("--mirror", action="store_true"); p.add_argument("--headless", action="store_true"); p.add_argument("--benchmark", type=int, metavar="N"); return p
def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.benchmark is not None:
        from .pipeline import HandTrackingPipeline
        import numpy as np
        from .inference import DetectionResult
        class Detector:
            def detect(self, frame): return DetectionResult()
        pipe=HandTrackingPipeline(detector=Detector(), smoothing=not args.no_smoothing)
        for _ in range(max(0,args.benchmark)): pipe.process_frame(np.zeros((args.height or 480,args.width or 640,3),dtype=np.uint8))
        print(f"frames={args.benchmark} avg_ms={pipe.telemetry.averages().total_ms:.3f} fps={pipe.telemetry.smoothed_fps:.1f}"); return 0
    from .capture import AsyncWebcamCapture
    from .inference import create_detector
    import cv2
    with AsyncWebcamCapture(args.camera, width=args.width, height=args.height) as capture, HandTrackingPipeline(capture=capture, detector=create_detector(), smoothing=not args.no_smoothing) as pipe:
        while True:
            ok, frame=capture.read()
            if not ok: time.sleep(.001); continue
            if args.mirror: frame=cv2.flip(frame,1)
            output,_,_=pipe.process_frame(frame)
            if not args.headless: cv2.imshow("HandTracking", output)
            if cv2.waitKey(1) & 0xff == ord("q"): break
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
