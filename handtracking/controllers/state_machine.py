"""Wake / Sleep State Machine for touchless controller."""
from __future__ import annotations
from enum import Enum
import time
from typing import Iterable


class ControllerState(str, Enum):
    SLEEPING = "sleeping"
    WAKING = "waking"
    ACTIVE = "active"


class ControllerStateMachine:
    """Manages the SLEEPING -> WAKING -> ACTIVE -> SLEEPING lifecycle with continuous hold progress."""

    def __init__(self, wake_gesture: str = "open_palm", wake_duration_s: float = 1.0, idle_timeout_s: float = 4.0):
        self.wake_gesture = self._normalize(wake_gesture)
        self.wake_duration_s = max(0.05, float(wake_duration_s))
        self.idle_timeout_s = max(0.1, float(idle_timeout_s))
        self.state = ControllerState.SLEEPING
        self.wake_start_time: float | None = None
        self.last_activity_time: float | None = None
        self.hold_progress: float = 0.0

    @staticmethod
    def _normalize(name: str | None) -> str:
        if not name:
            return ""
        return str(name).lower().strip().replace(" ", "_").replace("-", "_")

    def _matches_wake_gesture(self, gestures: Iterable[str] | str | None) -> bool:
        if not gestures:
            return False
        if isinstance(gestures, str):
            gesture_list = [gestures]
        else:
            gesture_list = list(gestures)

        for g in gesture_list:
            norm = self._normalize(g)
            if norm == self.wake_gesture:
                return True
            if self.wake_gesture == "peace_sign" and norm == "peace":
                return True
            if self.wake_gesture == "peace" and norm == "peace_sign":
                return True
        return False

    def update(self, current_gestures: Iterable[str] | str | None, timestamp: float) -> ControllerState:
        """Update state machine based on current gestures and timestamp."""
        ts = float(timestamp)
        wake_detected = self._matches_wake_gesture(current_gestures)

        if self.state == ControllerState.SLEEPING:
            if wake_detected:
                self.state = ControllerState.WAKING
                self.wake_start_time = ts
                self.hold_progress = 0.0
            else:
                self.hold_progress = 0.0

        elif self.state == ControllerState.WAKING:
            if wake_detected:
                if self.wake_start_time is None:
                    self.wake_start_time = ts
                elapsed = ts - self.wake_start_time
                self.hold_progress = min(1.0, max(0.0, elapsed / self.wake_duration_s))
                if self.hold_progress >= 1.0:
                    self.state = ControllerState.ACTIVE
                    self.last_activity_time = ts
                    self.wake_start_time = None
            else:
                # Interrupted before 1 second threshold
                self.state = ControllerState.SLEEPING
                self.wake_start_time = None
                self.hold_progress = 0.0

        elif self.state == ControllerState.ACTIVE:
            self.hold_progress = 1.0
            if self.last_activity_time is not None and (ts - self.last_activity_time) >= self.idle_timeout_s:
                # Idle watchdog timed out
                self.state = ControllerState.SLEEPING
                self.wake_start_time = None
                self.last_activity_time = None
                self.hold_progress = 0.0

        return self.state

    def record_activity(self, timestamp: float) -> None:
        """Reset idle timer when a valid action / command occurs."""
        if self.state == ControllerState.ACTIVE:
            self.last_activity_time = float(timestamp)

    def wake(self, timestamp: float | None = None) -> None:
        """Forcibly transition to ACTIVE state."""
        self.state = ControllerState.ACTIVE
        self.hold_progress = 1.0
        self.last_activity_time = float(timestamp if timestamp is not None else time.time())
        self.wake_start_time = None

    def sleep(self) -> None:
        """Forcibly transition to SLEEPING state."""
        self.state = ControllerState.SLEEPING
        self.hold_progress = 0.0
        self.wake_start_time = None
        self.last_activity_time = None

    def reset(self) -> None:
        self.sleep()

    @property
    def is_active(self) -> bool:
        return self.state == ControllerState.ACTIVE

    @property
    def is_waking(self) -> bool:
        return self.state == ControllerState.WAKING

    @property
    def is_sleeping(self) -> bool:
        return self.state == ControllerState.SLEEPING

    def time_until_sleep(self, timestamp: float) -> float:
        if not self.is_active or self.last_activity_time is None:
            return 0.0
        return max(0.0, self.idle_timeout_s - (float(timestamp) - self.last_activity_time))
