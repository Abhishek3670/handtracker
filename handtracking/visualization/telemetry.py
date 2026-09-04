"""Low-overhead stage timing and rolling pipeline metrics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
import time

class StageTimer:
    def __init__(self): self.started = None; self.elapsed_us = 0.0
    def start(self): self.started = time.perf_counter(); return self
    def stop(self):
        if self.started is None: raise RuntimeError("timer has not been started")
        self.elapsed_us = (time.perf_counter() - self.started) * 1_000_000; self.started = None; return self.elapsed_us
    def __enter__(self): return self.start()
    def __exit__(self, *_): self.stop()

@dataclass
class StageLatency:
    capture_ms: float = 0.0; preprocess_ms: float = 0.0; inference_ms: float = 0.0
    smoothing_ms: float = 0.0; gestures_ms: float = 0.0; render_ms: float = 0.0; total_ms: float = 0.0
    def as_dict(self): return asdict(self)

class PipelineTelemetry:
    def __init__(self, window_size: int = 120):
        self.window_size = window_size; self.samples = deque(maxlen=window_size); self.timestamps = deque(maxlen=window_size); self.frame_drops = 0
        self.latency = StageLatency(); self._last_timestamp = None
    def record(self, latency: StageLatency, timestamp: float | None = None, frame_drops: int = 0):
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self.samples.append(latency); self.timestamps.append(now); self.latency = latency; self.frame_drops += int(frame_drops); self._last_timestamp = now; return latency
    def add(self, latency: StageLatency, timestamp: float | None = None, frame_drops: int = 0): return self.record(latency, timestamp, frame_drops)
    @property
    def frame_drop_count(self): return self.frame_drops
    @property
    def instant_fps(self):
        if len(self.timestamps) < 2: return 0.0
        dt = self.timestamps[-1] - self.timestamps[-2]; return 1.0 / dt if dt > 0 else 0.0
    @property
    def fps(self): return self.smoothed_fps
    @property
    def smoothed_fps(self):
        if len(self.timestamps) < 2: return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]; return (len(self.timestamps)-1) / dt if dt > 0 else 0.0
    def averages(self) -> StageLatency:
        if not self.samples: return StageLatency()
        fields = asdict(self.samples[0]); return StageLatency(**{k: sum(asdict(x)[k] for x in self.samples)/len(self.samples) for k in fields})
    @property
    def stage_averages(self): return self.averages()
