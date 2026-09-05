"""Touchless media & entertainment controller."""
from __future__ import annotations
import time
from typing import Any, Iterable

from ..config.settings import MediaConfig
from .state_machine import ControllerState, ControllerStateMachine
from .synthesizer import KeySynthesizer


class MediaController:
    """Coordinates gesture inputs, wake state machine, volume state, and OS media dispatching."""

    def __init__(
        self,
        config: MediaConfig | None = None,
        state_machine: ControllerStateMachine | None = None,
        synthesizer: KeySynthesizer | None = None,
        initial_volume: int = 50,
    ):
        self.config = config or MediaConfig()
        self.config.normalize()
        self.state_machine = state_machine or ControllerStateMachine(
            wake_gesture=self.config.wake_gesture,
            wake_duration_s=self.config.wake_duration_s,
            idle_timeout_s=self.config.idle_timeout_s,
        )
        self.synthesizer = synthesizer or KeySynthesizer()
        self.volume = max(0, min(100, int(initial_volume)))
        self.is_muted = False
        self.last_action: str | None = None
        self.last_action_time: float = 0.0
        self.last_action_timestamp_map: dict[str, float] = {}
        self.last_toast: str | None = None
        self.toast_expires_at: float = 0.0

        # Action cooldowns in seconds to prevent spamming
        self.cooldowns: dict[str, float] = {
            "volume_up": 0.12,
            "volume_down": 0.12,
            "play_pause": 0.5,
            "next_track": 0.6,
            "prev_track": 0.6,
            "mute": 0.5,
        }

    def set_toast(self, text: str, duration: float = 1.5, timestamp: float | None = None) -> None:
        now = time.time() if timestamp is None else float(timestamp)
        self.last_toast = text
        self.toast_expires_at = now + duration

    def get_active_toast(self, timestamp: float | None = None) -> str | None:
        now = time.time() if timestamp is None else float(timestamp)
        if self.last_toast is not None and self.toast_expires_at >= now:
            return self.last_toast
        return None

    def get_cooldown(self, action: str) -> float:
        norm = str(action).lower().strip().replace(" ", "_").replace("-", "_")
        return self.cooldowns.get(norm, 0.25)

    def process_gestures(
        self,
        static_gestures: Iterable[str] = (),
        temporal_gestures: Iterable[str] = (),
        timestamp: float | None = None,
    ) -> list[str]:
        """Process observed gestures and trigger actions if controller is in ACTIVE state."""
        ts = time.time() if timestamp is None else float(timestamp)
        static_list = [g for g in static_gestures if g]
        temporal_list = [g for g in temporal_gestures if g]

        # Update state machine with static gestures (e.g. open palm)
        self.state_machine.update(static_list, ts)

        dispatched_actions: list[str] = []
        if not self.state_machine.is_active:
            return dispatched_actions

        # In ACTIVE state, evaluate actions: temporal first, then static
        candidates = temporal_list + static_list
        for candidate in candidates:
            action = self.config.get_action_for_gesture(candidate)
            if not action:
                continue

            cooldown = self.get_cooldown(action)
            last_time = self.last_action_timestamp_map.get(action, 0.0)
            if (ts - last_time) < cooldown:
                continue

            # Execute action logic
            toast = self._apply_action(action)
            self.synthesizer.send_action(action)
            self.state_machine.record_activity(ts)
            self.last_action = action
            self.last_action_time = ts
            self.last_action_timestamp_map[action] = ts
            self.set_toast(toast, duration=1.5, timestamp=ts)
            dispatched_actions.append(action)

        return dispatched_actions

    def _apply_action(self, action: str) -> str:
        norm = str(action).lower().strip().replace(" ", "_").replace("-", "_")
        if norm == "volume_up":
            self.volume = min(100, self.volume + self.config.volume_step)
            self.is_muted = False
            return f"Volume Up 🔊 {self.volume}%"
        elif norm == "volume_down":
            self.volume = max(0, self.volume - self.config.volume_step)
            self.is_muted = False
            return f"Volume Down 🔉 {self.volume}%"
        elif norm == "mute":
            self.is_muted = not self.is_muted
            return "Mute 🔇" if self.is_muted else f"Unmute 🔊 {self.volume}%"
        elif norm == "play_pause":
            return "Play / Pause ⏯"
        elif norm == "next_track":
            return "Next Track ⏭"
        elif norm == "prev_track":
            return "Previous Track ⏮"
        return f"Action: {action}"
