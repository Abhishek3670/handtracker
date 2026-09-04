"""Debounced gesture lifecycle events."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from typing import Callable, Any
from .recognizer import GestureType, GestureResult

class EventState(str, Enum): START="start"; HOLD="hold"; END="end"
@dataclass(frozen=True)
class GestureEvent:
    gesture: GestureType | str
    hand_id: Any
    state: EventState
    hold_duration: float
    timestamp: float
    result: GestureResult | None = None
    @property
    def gesture_type(self): return self.gesture

class GestureEventDispatcher:
    def __init__(self, debounce_time: float=.05, hold_threshold: float=.2):
        self.debounce_time, self.hold_threshold = debounce_time, hold_threshold
        self._callbacks = defaultdict(list); self._active = {}
    def on(self, gesture, callback): self._callbacks[gesture].append(callback); return callback
    def on_start(self, gesture, callback): return self.on((gesture, EventState.START), callback)
    def on_hold(self, gesture, callback): return self.on((gesture, EventState.HOLD), callback)
    def on_end(self, gesture, callback): return self.on((gesture, EventState.END), callback)
    def on_gesture_start(self, gesture_or_callback, callback=None):
        return self.on_start(gesture_or_callback, callback) if callback else self.on((None, EventState.START), gesture_or_callback)
    def on_gesture_hold(self, gesture_or_callback, callback=None):
        return self.on_hold(gesture_or_callback, callback) if callback else self.on((None, EventState.HOLD), gesture_or_callback)
    def on_gesture_end(self, gesture_or_callback, callback=None):
        return self.on_end(gesture_or_callback, callback) if callback else self.on((None, EventState.END), gesture_or_callback)
    def _emit(self, event):
        for key in (event.gesture, (event.gesture, event.state), (None, event.state)):
            for cb in self._callbacks[key]: cb(event)
    def update(self, hand_id, result: GestureResult | None, timestamp: float) -> list[GestureEvent]:
        now=float(timestamp); events=[]; old=self._active.get(hand_id)
        gesture = result.gesture if result else None
        if old and gesture is not None and gesture != old[0] and now - old[2] < self.debounce_time:
            return events
        if old and (gesture != old[0] or result is None):
            event=GestureEvent(old[0], hand_id, EventState.END, now-old[1], now, result); events.append(event); self._emit(event); del self._active[hand_id]
        if result is not None and gesture is not None:
            if not old or old[0] != gesture:
                self._active[hand_id]=(gesture, now, now); event=GestureEvent(gesture, hand_id, EventState.START, 0., now, result); events.append(event); self._emit(event)
            elif now-old[1] >= self.hold_threshold:
                event=GestureEvent(gesture, hand_id, EventState.HOLD, now-old[1], now, result); events.append(event); self._emit(event)
        return events
    process = update
    def end(self, hand_id, timestamp: float): return self.update(hand_id, None, timestamp)
    def reset(self, hand_id=None, timestamp: float=0.):
        if hand_id is None:
            for key in tuple(self._active): self.end(key, timestamp)
        elif hand_id in self._active: self.end(hand_id, timestamp)
