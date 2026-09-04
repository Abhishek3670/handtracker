import threading
import time

from handtracking.capture import async_cam
from handtracking.capture.async_cam import AsyncWebcamCapture
from handtracking.demo import _parse_camera, build_parser

class FakeCamera:
    def __init__(self): self.value=0; self.released=False; self.lock=threading.Lock()
    def isOpened(self): return not self.released
    def grab(self):
        with self.lock: self.value += 1
        return not self.released
    def retrieve(self):
        with self.lock: return True, self.value
    def release(self): self.released=True

def wait_for_frame(cap):
    deadline=time.monotonic()+1
    while time.monotonic()<deadline:
        ok, frame=cap.read()
        if ok: return frame
        time.sleep(.001)
    raise AssertionError("capture did not produce a frame")

def test_mock_camera_flow_and_drops_latest_frame():
    camera=FakeCamera(); cap=AsyncWebcamCapture(camera=camera)
    first=wait_for_frame(cap); time.sleep(.01); second=wait_for_frame(cap)
    assert second>=first and cap.frame_count>=2 and cap.dropped_frames>=0; cap.stop()

def test_lifecycle_and_context_manager():
    camera=FakeCamera()
    with AsyncWebcamCapture(camera=camera, auto_start=False) as cap: assert cap.running and wait_for_frame(cap)>=1
    assert not cap.running and camera.released

def test_start_is_idempotent_and_stop_is_safe():
    camera=FakeCamera(); cap=AsyncWebcamCapture(camera=camera, auto_start=False)
    assert cap.start() and cap.start(); cap.stop(); cap.stop()

class Camera:
    def isOpened(self): return True

def test_numeric_camera_source_is_normalized_for_factory():
    received=[]; camera=Camera()
    capture=AsyncWebcamCapture("1", camera_factory=lambda source: received.append(source) or camera, auto_start=False)
    assert capture.start() is True; capture.stop(); assert received==[1]

def test_demo_camera_parser_accepts_indices_and_paths():
    assert _parse_camera("0")==0 and _parse_camera("camera.mp4")=="camera.mp4"
    assert build_parser().parse_args(["--camera","1"]).camera==1

def test_dshow_fallback_retries_integer_device(monkeypatch):
    calls=[]
    class OpenState:
        def __init__(self, opened): self.opened=opened
        def isOpened(self): return self.opened
        def release(self): pass
    class FakeCV2:
        CAP_DSHOW=700
        def VideoCapture(self, source, *backend): calls.append((source,backend)); return OpenState(bool(backend))
    monkeypatch.setattr(async_cam, "cv2", FakeCV2())
    capture=AsyncWebcamCapture("0", auto_start=False)
    assert capture.start() is True; capture.stop(); assert calls==[(0,()),(0,(700,))]
