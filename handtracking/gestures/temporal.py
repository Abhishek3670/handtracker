"""Sliding-window 3D trajectory and temporal gesture recognition."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable
from ..inference.models import Landmark3D

@dataclass(frozen=True)
class TrajectorySample:
    timestamp: float
    point: Landmark3D

class TrajectoryBuffer:
    def __init__(self, max_length: int = 30, max_size: int | None = None):
        self.samples = deque(maxlen=max_size or max_length)

    def add(self, point: Landmark3D | Iterable[float], timestamp: float):
        if not isinstance(point, Landmark3D):
            point = Landmark3D(*point)
        self.samples.append(TrajectorySample(float(timestamp), point))
        return point

    append = add

    def clear(self):
        self.samples.clear()

    def __len__(self):
        return len(self.samples)

    @property
    def points(self):
        return tuple(x.point for x in self.samples)

    @property
    def latest(self):
        return self.samples[-1].point if self.samples else None

    def displacement(self):
        if len(self.samples) < 2:
            return (0.0, 0.0, 0.0)
        a, b = self.samples[0].point, self.samples[-1].point
        return (b.x - a.x, b.y - a.y, b.z - a.z)

    def velocity(self):
        if len(self.samples) < 2:
            return (0.0, 0.0, 0.0)
        dt = self.samples[-1].timestamp - self.samples[0].timestamp
        return tuple(x / dt for x in self.displacement()) if dt > 0 else (0.0, 0.0, 0.0)

    def direction(self):
        dx, dy, dz = self.displacement()
        if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
            return "right" if dx >= 0 else "left"
        return "down" if dy >= 0 else "up"

    def path_length(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        total = 0.0
        samples = list(self.samples)
        for a, b in zip(samples, samples[1:]):
            total += math.hypot(b.point.x - a.point.x, b.point.y - a.point.y)
        return total

    def linearity(self) -> float:
        pl = self.path_length()
        if pl <= 1e-6:
            return 0.0
        dx, dy, _ = self.displacement()
        return math.hypot(dx, dy) / pl

    def max_direction_deviation(self) -> float:
        if len(self.samples) < 3:
            return 0.0
        samples = list(self.samples)
        step_angles = []
        for a, b in zip(samples, samples[1:]):
            dx = b.point.x - a.point.x
            dy = b.point.y - a.point.y
            if math.hypot(dx, dy) >= 0.003:
                step_angles.append(math.atan2(dy, dx))
        if len(step_angles) < 2:
            return 0.0
        ref_angle = step_angles[0]
        max_dev = 0.0
        for ang in step_angles[1:]:
            diff = abs((ang - ref_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff > max_dev:
                max_dev = diff
        return max_dev

class CircleDetector:
    """Detects continuous clockwise or counter-clockwise circular motion."""
    def __init__(self, window_size: int = 30, min_radius: float = 0.03, min_angle: float = 4.5):
        self.window_size = window_size
        self.min_radius = min_radius
        self.min_angle = min_angle
        self.buffer = TrajectoryBuffer(window_size)

    def update(self, point: Landmark3D | Iterable[float], timestamp: float) -> str | None:
        self.buffer.add(point, timestamp)
        return self.detect()

    def detect(self) -> str | None:
        if len(self.buffer) < 8:
            return None
        points = self.buffer.points
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        radii = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
        avg_radius = sum(radii) / len(radii)
        if avg_radius < self.min_radius:
            return None

        angles = [math.atan2(y - cy, x - cx) for x, y in zip(xs, ys)]
        total_angle = 0.0
        for i in range(1, len(angles)):
            d = angles[i] - angles[i - 1]
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            total_angle += d

        if total_angle >= self.min_angle:
            return "Circle CW"
        elif total_angle <= -self.min_angle:
            return "Circle CCW"
        return None

    def clear(self):
        self.buffer.clear()

class WaveDetector:
    """Detects horizontal oscillatory waving gestures."""
    def __init__(self, window_size: int = 20, min_reversals: int = 2, min_span: float = 0.04):
        self.window_size = window_size
        self.min_reversals = min_reversals
        self.min_span = min_span
        self.buffer = TrajectoryBuffer(window_size)

    def update(self, point: Landmark3D | Iterable[float], timestamp: float) -> bool:
        self.buffer.add(point, timestamp)
        return self.detect()

    def detect(self) -> bool:
        if len(self.buffer) < 5:
            return False
        points = self.buffer.points
        xs = [p.x for p in points]
        span = max(xs) - min(xs)
        if span < self.min_span:
            return False

        deltas = []
        for i in range(1, len(xs)):
            d = xs[i] - xs[i - 1]
            if abs(d) >= 0.003:
                deltas.append(1 if d > 0 else -1)

        if len(deltas) < 3:
            return False

        reversals = sum(1 for i in range(1, len(deltas)) if deltas[i] != deltas[i - 1])
        return reversals >= self.min_reversals

    def clear(self):
        self.buffer.clear()

def _evaluate_swipe_samples(
    samples: Sequence[TrajectorySample],
    min_duration: float,
    swipe_threshold: float,
    min_linearity: float = 0.85,
    max_dev: float = 0.55,
) -> str | None:
    if len(samples) < 3:
        return None
    dt = samples[-1].timestamp - samples[0].timestamp
    if dt < min_duration:
        return None

    a, b = samples[0].point, samples[-1].point
    dx, dy = b.x - a.x, b.y - a.y
    if max(abs(dx), abs(dy)) < swipe_threshold:
        return None

    total = 0.0
    for s1, s2 in zip(samples, samples[1:]):
        total += math.hypot(s2.point.x - s1.point.x, s2.point.y - s1.point.y)
    if total <= 1e-6:
        return None

    disp = math.hypot(dx, dy)
    linearity = disp / total
    if linearity < min_linearity:
        return None

    step_angles = []
    for s1, s2 in zip(samples, samples[1:]):
        step_dx = s2.point.x - s1.point.x
        step_dy = s2.point.y - s1.point.y
        if math.hypot(step_dx, step_dy) >= 0.003:
            step_angles.append(math.atan2(step_dy, step_dx))
    if len(step_angles) >= 2:
        ref_angle = step_angles[0]
        max_deviation = 0.0
        for ang in step_angles[1:]:
            diff = abs((ang - ref_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff > max_deviation:
                max_deviation = diff
        if max_deviation > max_dev:
            return None

    if abs(dx) >= abs(dy):
        direction = "Right" if dx >= 0 else "Left"
    else:
        direction = "Down" if dy >= 0 else "Up"
    return f"Swipe {direction}"


class TemporalGestureRecognizer:
    """Sliding-window dynamic gesture recognizer supporting swipes, circles, and waves."""
    def __init__(
        self,
        window_size: int = 30,
        swipe_threshold: float = 0.08,
        min_duration: float = 0.05,
        sub_window_range: tuple[int, int] = (10, 14),
    ):
        self.window_size = window_size
        self.swipe_threshold = swipe_threshold
        self.min_duration = min_duration
        self.sub_window_range = sub_window_range
        self.buffers: dict[Any, TrajectoryBuffer] = {}
        self.circle_detectors: dict[Any, CircleDetector] = {}
        self.wave_detectors: dict[Any, WaveDetector] = {}

    def update(self, hand_id: Any, point: Landmark3D | Iterable[float], timestamp: float) -> str | None:
        buffer = self.buffers.setdefault(hand_id, TrajectoryBuffer(self.window_size))
        circle_det = self.circle_detectors.setdefault(hand_id, CircleDetector(self.window_size))
        wave_det = self.wave_detectors.setdefault(hand_id, WaveDetector(self.window_size))

        buffer.add(point, timestamp)
        circle_res = circle_det.update(point, timestamp)
        wave_res = wave_det.update(point, timestamp)

        if circle_res:
            self._reset_hand(hand_id, point, timestamp)
            return circle_res

        if wave_res:
            self._reset_hand(hand_id, point, timestamp)
            return "Wave"

        if len(buffer) < 3:
            return None

        samples = list(buffer.samples)
        # Check sub-windows (10-14 frames) as well as full buffer for responsive flick detection
        sub_min, sub_max = self.sub_window_range
        candidate_lengths: list[int] = []
        if len(samples) >= sub_min:
            for w in range(min(len(samples), sub_max), sub_min - 1, -1):
                candidate_lengths.append(w)
            if len(samples) not in candidate_lengths:
                candidate_lengths.append(len(samples))
        else:
            candidate_lengths.append(len(samples))

        for w in candidate_lengths:
            sub = samples[-w:]
            thresh = self.swipe_threshold if w >= sub_min else max(self.swipe_threshold, 0.15)
            res = _evaluate_swipe_samples(sub, self.min_duration, thresh)
            if res:
                self._reset_hand(hand_id, point, timestamp)
                return res

        return None

    def _reset_hand(self, hand_id: Any, point: Landmark3D | Iterable[float] | None = None, timestamp: float | None = None):
        if hand_id in self.buffers:
            self.buffers[hand_id].clear()
            if point is not None and timestamp is not None:
                self.buffers[hand_id].add(point, timestamp)
        if hand_id in self.circle_detectors:
            self.circle_detectors[hand_id].clear()
            if point is not None and timestamp is not None:
                self.circle_detectors[hand_id].buffer.add(point, timestamp)
        if hand_id in self.wave_detectors:
            self.wave_detectors[hand_id].clear()
            if point is not None and timestamp is not None:
                self.wave_detectors[hand_id].buffer.add(point, timestamp)

    process = update
    recognize = update

    def reset(self, hand_id: Any = None):
        if hand_id is None:
            self.buffers.clear()
            self.circle_detectors.clear()
            self.wave_detectors.clear()
        else:
            self.buffers.pop(hand_id, None)
            self.circle_detectors.pop(hand_id, None)
            self.wave_detectors.pop(hand_id, None)


TemporalGestureTracker = TemporalGestureRecognizer

