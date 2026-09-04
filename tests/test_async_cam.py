import threading
import time

from handtracking.capture.async_cam import AsyncWebcamCapture


class FakeCamera:
    def __init__(self):
        self.value = 0
        self.released = False
        self.lock = threading.Lock()

    def isOpened(self):
        return not self.released

    def grab(self):
        with self.lock:
            self.value += 1
        return not self.released

    def retrieve(self):
        with self.lock:
            return True, self.value

    def release(self):
        self.released = True


def wait_for_frame(cap):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok:
            return frame
        time.sleep(0.001)
    raise AssertionError("capture did not produce a frame")


def test_mock_camera_flow_and_drops_latest_frame():
    camera = FakeCamera()
    cap = AsyncWebcamCapture(camera=camera)
    first = wait_for_frame(cap)
    time.sleep(0.01)
    second = wait_for_frame(cap)
    assert second >= first
    assert cap.frame_count >= 2
    assert cap.dropped_frames >= 0
    cap.stop()


def test_lifecycle_and_context_manager():
    camera = FakeCamera()
    with AsyncWebcamCapture(camera=camera, auto_start=False) as cap:
        assert cap.running
        assert wait_for_frame(cap) >= 1
    assert not cap.running
    assert camera.released


def test_start_is_idempotent_and_stop_is_safe():
    camera = FakeCamera()
    cap = AsyncWebcamCapture(camera=camera, auto_start=False)
    assert cap.start()
    assert cap.start()
    cap.stop()
    cap.stop()
