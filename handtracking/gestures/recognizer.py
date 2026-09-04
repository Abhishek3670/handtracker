"""Rule-based gesture recognition."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Any, Callable
from .finger_state import FingerPoseAnalyzer, FingerState, FingerStates
from ..inference.models import HandLandmarks

class GestureType(str, Enum):
    OPEN_PALM="open_palm"; FIST="fist"; PINCH="pinch"; PEACE="peace"; POINTING="pointing"
    THUMBS_UP="thumbs_up"; THUMBS_DOWN="thumbs_down"; OK_SIGN="ok_sign"; ROCK_ON="rock_on"; CALL_ME="call_me"; UNKNOWN="unknown"

@dataclass(frozen=True)
class GestureDefinition:
    states: dict[str, FingerState | str] | None = None
    predicate: Callable[[HandLandmarks, FingerStates, float], bool] | None = None

@dataclass(frozen=True)
class GestureResult:
    gesture: GestureType | str
    confidence: float
    finger_states: FingerStates
    pinch_distance: float
    hand: HandLandmarks
    @property
    def gesture_type(self): return self.gesture
    @property
    def is_pinch(self): return self.gesture == GestureType.PINCH or str(self.gesture).lower().endswith("pinch")
    @property
    def handedness(self): return self.hand.handedness

def _dist(a,b): return sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)

class GestureRecognizer:
    def __init__(self, analyzer: FingerPoseAnalyzer | None = None, pinch_threshold: float = .35):
        self.analyzer = analyzer or FingerPoseAnalyzer(); self.pinch_threshold = pinch_threshold; self.custom: dict[str, GestureDefinition] = {}
    def register_custom_gesture(self, name: str, definition: GestureDefinition | dict[str, Any] | Callable) -> None:
        if callable(definition): definition = GestureDefinition(predicate=definition)
        elif isinstance(definition, dict): definition = GestureDefinition(states=definition.get("states", definition))
        if not isinstance(definition, GestureDefinition): raise TypeError("definition must be GestureDefinition, dict, or callable")
        self.custom[name] = definition
    def recognize(self, hand: HandLandmarks) -> GestureResult:
        states = self.analyzer.analyze(hand); s = states.as_dict()
        scale = max(_dist(hand.landmarks[0], hand.landmarks[i]) for i in (5,9,13,17)) or 1.0
        pinch = _dist(hand.landmarks[4], hand.landmarks[8]) / scale
        name: GestureType | str = GestureType.UNKNOWN
        for custom_name, definition in self.custom.items():
            if definition.predicate:
                try: matched = definition.predicate(hand, states, pinch)
                except TypeError: matched = definition.predicate(hand)
            else:
                matched = all(s[k] == FingerState(v) for k,v in (definition.states or {}).items())
            if matched: name = custom_name; break
        if name == GestureType.UNKNOWN:
            e = FingerState.EXTENDED; c = FingerState.CURLED
            vals = [s[x] for x in ("thumb","index","middle","ring","pinky")]
            if pinch < self.pinch_threshold: name = GestureType.PINCH
            elif vals == [e,e,e,e,e]: name = GestureType.OPEN_PALM
            elif vals == [c,c,c,c,c]: name = GestureType.FIST
            elif vals[0] == c and vals[1] == c and vals[2:] == [e,e,e]: name = GestureType.OK_SIGN
            elif vals[1:3] == [e,e] and vals[3:] == [c,c]: name = GestureType.PEACE
            elif vals[1] == e and vals[2:] == [c,c,c]: name = GestureType.POINTING
            elif vals[0] == e and vals[1:] == [c,c,c,c]: name = GestureType.THUMBS_UP if hand.landmarks[4].y < hand.landmarks[0].y else GestureType.THUMBS_DOWN
            elif vals[0] == c and vals[1] == e and vals[2] == c and vals[3] == c and vals[4] == e: name = GestureType.ROCK_ON
            elif vals[0] == e and vals[1] == c and vals[2] == c and vals[3] == c and vals[4] == e: name = GestureType.CALL_ME
        confidence = 1.0 if name != GestureType.UNKNOWN else .0
        return GestureResult(name, confidence, states, pinch, hand)
    process = recognize
