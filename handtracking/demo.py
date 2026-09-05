"""Command-line live demo and synthetic benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

try:
    import cv2
except ImportError:
    cv2 = None

from .ar import ARPhysicsEngine, BallRenderer, BallSkin
from .capture import AsyncWebcamCapture
from .config import MediaConfig
from .controllers import MediaController
from .gestures import AirCanvas, TemporalGestureRecognizer
from .inference import DetectionResult, create_detector
from .pipeline import HandTrackingPipeline


def _parse_camera(val: str) -> int | str:
    return int(val) if str(val).isdigit() else val


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HandTracking live demo")
    p.add_argument("--camera", default=0, type=_parse_camera, help="Camera device index (e.g. 0) or video file path")
    p.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    p.add_argument("--height", type=int, default=720, help="Webcam capture height")
    p.add_argument("--model-complexity", type=int, choices=(0, 1), default=1)
    p.add_argument("--canvas", action="store_true", help="Enable air canvas")
    p.add_argument("--temporal", action="store_true", default=True, help="Enable dynamic temporal gesture engine")
    p.add_argument("--no-temporal", dest="temporal", action="store_false", help="Disable dynamic temporal gestures")
    p.add_argument("--media", action="store_true", help="Enable touchless media and entertainment controller")
    p.add_argument("--config", default="config.yaml", help="Path to media controller configuration YAML/JSON file")
    p.add_argument("--ar-ball", "--ar", dest="ar_ball", action="store_true", help="Enable AR 3D interactive physics ball")
    p.add_argument(
        "--virtual-room",
        "--virtual-space",
        "-vr",
        dest="virtual_room",
        action="store_true",
        help="Render digital 3D cyber-space environment instead of webcam feed",
    )
    p.add_argument(
        "--gpu-render",
        "--gpu",
        dest="gpu_render",
        action="store_true",
        help="Enable ModernGL hardware-accelerated GPU shader rendering for 3D Cyber Room & ball shading",
    )
    p.add_argument(
        "--ar-skin",
        default="basketball",
        choices=("basketball", "chrome", "tennis", "neon"),
        help="AR Ball material skin",
    )
    p.add_argument("--no-smoothing", action="store_true", help="Disable 1 Euro adaptive smoothing")
    p.add_argument("--mirror", action="store_true", default=True, help="Mirror webcam display horizontally")
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror mode")
    p.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    p.add_argument("--benchmark", type=int, metavar="N", help="Run synthetic benchmark of N frames")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.virtual_room or args.gpu_render:
        args.ar_ball = True
        args.virtual_room = True

    if args.benchmark is not None:
        import numpy as np

        class BenchmarkDetector:
            def detect(self, frame: np.ndarray) -> DetectionResult:
                return DetectionResult()

        media_ctrl = None
        if args.media:
            media_cfg = MediaConfig.load(args.config)
            media_ctrl = MediaController(config=media_cfg)

        ar_engine = ARPhysicsEngine() if args.ar_ball else None

        pipe = HandTrackingPipeline(
            detector=BenchmarkDetector(),
            smoothing=not args.no_smoothing,
            temporal=TemporalGestureRecognizer() if args.temporal else None,
            media_controller=media_ctrl,
            ar_physics=ar_engine,
            virtual_room=args.virtual_room,
            use_gpu_render=args.gpu_render,
        )
        sample_frame = np.zeros((args.height or 480, args.width or 640, 3), dtype=np.uint8)
        for _ in range(max(0, args.benchmark)):
            pipe.process_frame(sample_frame)
        print(f"frames={args.benchmark} avg_ms={pipe.telemetry.averages().total_ms:.3f} fps={pipe.telemetry.smoothed_fps:.1f}")
        return 0

    if cv2 is None and not args.headless:
        print("OpenCV (cv2) is required for live GUI display. Please install opencv-python.")
        return 1

    media_controller = None
    if args.media:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Generating default media controller configuration at: {config_path}")
            MediaConfig.create_default(config_path)
        media_cfg = MediaConfig.load(config_path)
        media_controller = MediaController(config=media_cfg)
        print(f"Touchless Media Controller ENABLED. Wake gesture: '{media_cfg.wake_gesture}' (Hold 1.0s to wake).")

    ar_physics = ARPhysicsEngine() if args.ar_ball else None
    ar_renderer = BallRenderer(skin=BallSkin(args.ar_skin)) if args.ar_ball else None
    if args.ar_ball:
        print(f"AR 3D Interactive Ball ENABLED (Skin: {args.ar_skin.title()}). Controls: 'v' 3D Space, 'u' GPU Shaders, 'b' Reset Ball, 's' Skins, 'g' Gravity.")

    with (
        AsyncWebcamCapture(args.camera, width=args.width, height=args.height) as capture,
        HandTrackingPipeline(
            capture=capture,
            detector=create_detector(model_complexity=args.model_complexity),
            smoothing=not args.no_smoothing,
            temporal=TemporalGestureRecognizer() if args.temporal else None,
            canvas=AirCanvas() if args.canvas else None,
            media_controller=media_controller,
            ar_physics=ar_physics,
            ar_renderer=ar_renderer,
            virtual_room=args.virtual_room,
            use_gpu_render=args.gpu_render,
        ) as pipe,
    ):
        if pipe.gpu_renderer is not None and pipe.gpu_renderer.is_gpu_available:
            renderer_info = pipe.gpu_renderer.ctx.info.get("GL_RENDERER", "OpenGL GPU")
            print(f"ModernGL Hardware GPU Acceleration ENABLED: {renderer_info}")

        window_name = "HandTracking (Press 'q' to exit)"
        if not args.headless and cv2 is not None:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"HandTracking live feed started (Camera: {args.camera}, {args.width}x{args.height}). Press 'q' to exit.")
        if args.canvas:
            print("Air Canvas Controls: Pinch index+thumb to draw | Press 'c' to clear canvas | Press '1'-'4' for colors")
        if args.media:
            print("Media Controls: Press 'w' to toggle wake/sleep | Press 'm' to toggle media HUD")
        if args.ar_ball:
            print("AR Ball Controls: Bounce with palm or fingertips | Pinch to grab & throw | 'v' 3D Space | 'u' GPU Shaders | 'b' Reset | 's' Skins | 'g' Gravity")
        start_wait = time.time()
        first_frame_shown = False
        while True:
            ok, frame = capture.read()
            if not ok:
                if not first_frame_shown and (time.time() - start_wait > 3.0):
                    print("Waiting for camera frames... If the webcam is in use by another app, please close it.")
                    start_wait = time.time()
                time.sleep(0.001)
                continue
            first_frame_shown = True
            if args.mirror and cv2 is not None:
                frame = cv2.flip(frame, 1)
            output, _, _ = pipe.process_frame(frame)
            if not args.headless and cv2 is not None:
                cv2.imshow(window_name, output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    break
                elif key == ord("c") and pipe.canvas is not None:
                    pipe.canvas.clear()
                    print("Canvas cleared!")
                elif key == ord("v") and pipe.ar_physics is not None:
                    pipe.toggle_virtual_room()
                    print(f"3D Cyber-Space Environment: {pipe.virtual_room}")
                elif key == ord("u") and pipe.gpu_renderer is not None:
                    pipe.toggle_gpu_render()
                    status = "ENABLED" if pipe.use_gpu_render else "DISABLED"
                    print(f"Hardware GPU Shader Rendering: {status}")
                elif key == ord("b") and pipe.ar_physics is not None:
                    pipe.ar_physics.ball.reset(0.5, 0.25, 0.0)
                    print("AR Ball reset to center position.")
                elif key == ord("s") and pipe.ar_renderer is not None:
                    new_skin = pipe.ar_renderer.cycle_skin()
                    print(f"AR Ball Skin switched to: {new_skin.value.title()}")
                elif key == ord("g") and pipe.ar_physics is not None:
                    pipe.ar_physics.enable_gravity = not pipe.ar_physics.enable_gravity
                    print(f"AR Gravity enabled: {pipe.ar_physics.enable_gravity}")
                elif key == ord("w") and pipe.media_controller is not None:
                    if pipe.media_controller.state_machine.is_active:
                        pipe.media_controller.state_machine.sleep()
                        print("Media controller forced to SLEEPING.")
                    else:
                        pipe.media_controller.state_machine.wake()
                        print("Media controller forced to ACTIVE.")
                elif key == ord("h") and pipe.hud is not None:
                    pipe.hud.show_help = not pipe.hud.show_help
                    print(f"Help cheat sheet visible: {pipe.hud.show_help}")
                elif key == ord("m") and pipe.media_hud is not None:
                    pipe.media_hud.enabled = not pipe.media_hud.enabled
                    print(f"Media HUD overlay enabled: {pipe.media_hud.enabled}")
                elif pipe.canvas is not None:
                    if key == ord("1"):
                        pipe.canvas.set_color((0, 255, 0))  # Green
                    elif key == ord("2"):
                        pipe.canvas.set_color((255, 0, 0))  # Blue
                    elif key == ord("3"):
                        pipe.canvas.set_color((0, 0, 255))  # Red
                    elif key == ord("4"):
                        pipe.canvas.set_color((0, 255, 255))  # Yellow
        if not args.headless and cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
