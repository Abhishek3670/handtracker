"""Low-latency adaptive smoothing for hand landmark streams.

The implementation follows the 1 Euro filter described by Casiez et al. and
keeps the scalar filter deliberately allocation-free on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Hashable, Iterable

from ..inference.models import BoundingBox, HandLandmarks, Landmark3D


class LowPassFilter:
    """A first-order exponential low-pass filter."""

    def __init__(self, alpha: float | None = None) -> None:
        self._value: float | None = None
        self.alpha = 1.0 if alpha is None else self._check_alpha(alpha)

    @staticmethod
    def _check_alpha(alpha: float) -> float:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        return float(alpha)

    @property
    def initialized(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> float | None:
        return self._value

    def filter(self, value: float, alpha: float | None = None) -> float:
        """Filter *value*, optionally using a per-sample alpha."""
        coefficient = self.alpha if alpha is None else self._check_alpha(alpha)
        value = float(value)
        if self._value is None:
            self._value = value
        else:
            self._value = coefficient * value + (1.0 - coefficient) * self._value
        return self._value

    # ``apply`` is convenient for callers that use the terminology from the
    # original 1 Euro reference implementation.
    apply = filter

    def reset(self) -> None:
        self._value = None


class OneEuroFilter:
    """Adaptive scalar low-pass filter with timestamped samples."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        if beta < 0:
            raise ValueError("beta must not be negative")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._value_filter = LowPassFilter()
        self._derivative_filter = LowPassFilter()
        self._last_timestamp: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @property
    def last_timestamp(self) -> float | None:
        return self._last_timestamp

    def filter(self, value: float, timestamp: float) -> float:
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._value_filter.filter(value)
            self._derivative_filter.filter(0.0)
            return self._value_filter.value  # type: ignore[return-value]

        dt = timestamp - self._last_timestamp
        if dt <= 0.0:
            # Duplicate timestamps are common when frames are batched. They
            # carry no velocity information but should still be deterministic.
            return self._value_filter.value  # type: ignore[return-value]
        previous = self._value_filter.value  # initialized on the first sample
        raw_derivative = (float(value) - previous) / dt  # type: ignore[operator]
        derivative = self._derivative_filter.filter(
            raw_derivative, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(derivative)
        result = self._value_filter.filter(float(value), self._alpha(cutoff, dt))
        self._last_timestamp = timestamp
        return result

    def reset(self) -> None:
        self._value_filter.reset()
        self._derivative_filter.reset()
        self._last_timestamp = None


class LandmarkSmoother3D:
    """Three independent 1 Euro filters for one 3D landmark."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        args = (min_cutoff, beta, d_cutoff)
        self.x_filter, self.y_filter, self.z_filter = (OneEuroFilter(*args) for _ in range(3))

    def filter(self, landmark: Landmark3D, timestamp: float) -> Landmark3D:
        return Landmark3D(self.x_filter.filter(landmark.x, timestamp),
                          self.y_filter.filter(landmark.y, timestamp),
                          self.z_filter.filter(landmark.z, timestamp),
                          landmark.visibility)

    apply = filter

    def reset(self) -> None:
        self.x_filter.reset(); self.y_filter.reset(); self.z_filter.reset()


@dataclass
class _HandBank:
    filters: tuple[LandmarkSmoother3D, ...]
    world_filters: tuple[LandmarkSmoother3D, ...] | None = None
    last_timestamp: float | None = None


class HandSmoother:
    """Smooth complete hands while isolating temporal state per hand."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0, timeout: float = 0.5,
                 timeout_seconds: float | None = None) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.timeout = float(timeout if timeout_seconds is None else timeout_seconds)
        if self.timeout < 0:
            raise ValueError("timeout must not be negative")
        self._banks: dict[Hashable, _HandBank] = {}

    def _new_bank(self, with_world: bool = False) -> _HandBank:
        params = (self.min_cutoff, self.beta, self.d_cutoff)
        make = lambda: LandmarkSmoother3D(*params)
        return _HandBank(tuple(make() for _ in range(21)),
                         tuple(make() for _ in range(21)) if with_world else None)

    def smooth_hand(self, hand: HandLandmarks, timestamp: float,
                    track_id: Hashable | None = None) -> HandLandmarks:
        key = track_id if track_id is not None else hand.handedness.label
        bank = self._banks.get(key)
        if bank is None or (bank.last_timestamp is not None and
                            float(timestamp) - bank.last_timestamp > self.timeout):
            bank = self._new_bank(hand.world_landmarks is not None)
            self._banks[key] = bank
        points = tuple(f.filter(p, timestamp) for f, p in zip(bank.filters, hand.landmarks))
        world = None
        if hand.world_landmarks is not None:
            if bank.world_filters is None:
                bank.world_filters = tuple(LandmarkSmoother3D(self.min_cutoff, self.beta, self.d_cutoff)
                                           for _ in range(21))
            world = tuple(f.filter(p, timestamp) for f, p in zip(bank.world_filters, hand.world_landmarks))
        bank.last_timestamp = float(timestamp)
        return HandLandmarks(points, hand.handedness, BoundingBox.from_landmarks(points), world)

    def smooth(self, hands: HandLandmarks | Iterable[HandLandmarks], timestamp: float,
               track_ids: Iterable[Hashable] | None = None) -> HandLandmarks | tuple[HandLandmarks, ...]:
        if isinstance(hands, HandLandmarks):
            return self.smooth_hand(hands, timestamp)
        sequence = tuple(hands)
        ids = tuple(track_ids) if track_ids is not None else (None,) * len(sequence)
        if len(ids) != len(sequence):
            raise ValueError("track_ids must match hands")
        return tuple(self.smooth_hand(hand, timestamp, track_id) for hand, track_id in zip(sequence, ids))

    process = smooth

    def reset(self, track_id: Hashable | None = None) -> None:
        if track_id is None:
            self._banks.clear()
        else:
            self._banks.pop(track_id, None)

