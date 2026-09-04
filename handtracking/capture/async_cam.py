"""Asynchronous, latest-frame-only webcam capture.

The capture worker drains the driver's queue with ``grab`` and then retrieves
the newest frame.  A single Python reference is swapped for each frame; under
CPython the assignment/read is atomic, so consumers never wait on a lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional

try:  # Keep imports usable in test environments without OpenCV installed.
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal installs
    cv2 = None  # type: ignore[assignment]


class AsyncWebcamCapture:
    """Continuously capture a webcam while exposing only its freshest frame.

    ``camera`` is an optional OpenCV-compatible object, useful for tests and
    applications that already own a configured capture device.  Otherwise a
    ``cv2.VideoCapture`` is created from ``source``.
    """

    def __init__(
        self,
        source: int | str = 0,
        *,
        camera: Any = None,
        camera_factory: Optional[Callable[[int | str], Any]] = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        backend: int | None = None,
        auto_start: bool = True,
    ) -> None:
        self.source = source
        self._camera = camera
        self._camera_factory = camera_factory
        self._width, self._height, self._requested_fps = width, height, fps
        self._backend = backend
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

        self._latest_frame: Any = None
        self._latest_sequence = 0
        self._last_read_sequence = 0
        self._frames_captured = 0
        self._frames_dropped = 0
        self._timestamps: deque[float] = deque(maxlen=120)

        if auto_start:
            self.start()

    def _open_camera(self) -> bool:
        source = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        if self._camera is None:
            if self._camera_factory is not None:
                self._camera = self._camera_factory(source)
            elif cv2 is not None:
                args = (source,) if self._backend is None else (source, self._backend)
                self._camera = cv2.VideoCapture(*args)
                if (not self._camera.isOpened() and isinstance(source, int) and
                        self._backend is None and hasattr(cv2, "CAP_DSHOW")):
                    self._camera.release()
                    self._camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                return False
        if self._width is not None and cv2 is not None:
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None and cv2 is not None:
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._requested_fps is not None and cv2 is not None:
            self._camera.set(cv2.CAP_PROP_FPS, self._requested_fps)
        is_opened = getattr(self._camera, "isOpened", None)
        return bool(is_opened()) if callable(is_opened) else True

    def start(self) -> bool:
        """Open the camera and start the daemon capture worker."""
        if self._running:
            return True
        if not self._open_camera():
            return False
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="async-webcam", daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self._camera.grab():
                    self._stop_event.wait(0.001)
                    continue
                result = self._camera.retrieve()
                if isinstance(result, tuple):
                    ok, frame = result
                else:
                    ok, frame = True, result
                if ok:
                    now = time.monotonic()
                    # The GIL makes these reference/counter swaps atomic.
                    if self._latest_frame is not None and self._latest_sequence > self._last_read_sequence:
                        self._frames_dropped += 1
                    self._latest_frame = frame
                    self._latest_sequence += 1
                    self._frames_captured += 1
                    self._timestamps.append(now)
            except Exception:
                # A transient driver error must not kill the worker silently.
                self._stop_event.wait(0.001)
        self._running = False

    def read(self) -> tuple[bool, Any]:
        """Return the newest frame immediately, or ``(False, None)``."""
        frame = self._latest_frame
        sequence = self._latest_sequence
        if frame is None or sequence == self._last_read_sequence:
            return False, None
        self._last_read_sequence = sequence
        return True, frame

    def stop(self) -> None:
        """Stop capture, join the worker, and release the camera."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._running = False
        if self._camera is not None:
            release = getattr(self._camera, "release", None)
            if callable(release):
                release()

    def __enter__(self) -> "AsyncWebcamCapture":
        if not self._running and not self.start():
            raise RuntimeError("Unable to open webcam")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frames_captured

    @property
    def dropped_frames(self) -> int:
        return self._frames_dropped

    @property
    def fps(self) -> float:
        """Estimate capture FPS over the retained recent timestamp window."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def is_opened(self) -> bool:
        if self._camera is None:
            return False
        check = getattr(self._camera, "isOpened", None)
        return bool(check()) if callable(check) else self._running
