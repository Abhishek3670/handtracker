"""Fast geometric finger-pose classification."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import acos, sqrt
from ..inference.models import HandLandmarks, Landmark3D

class FingerState(str, Enum):
    EXTENDED = "extended"
    FLEXED = "flexed"
    CURLED = "curled"

@dataclass(frozen=True)
class FingerStates:
    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    def __getitem__(self, name: str) -> FingerState:
        return getattr(self, name)
    def as_dict(self) -> dict[str, FingerState]:
        return {name: getattr(self, name) for name in ("thumb", "index", "middle", "ring", "pinky")}
    def values(self):
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

def _distance(a: Landmark3D, b: Landmark3D) -> float:
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def _angle(a: Landmark3D, b: Landmark3D, c: Landmark3D) -> float:
    u = (a.x-b.x, a.y-b.y, a.z-b.z); v = (c.x-b.x, c.y-b.y, c.z-b.z)
    nu = sqrt(sum(x*x for x in u)); nv = sqrt(sum(x*x for x in v))
    if not nu or not nv: return 0.0
    value = max(-1.0, min(1.0, sum(x*y for x,y in zip(u,v))/(nu*nv)))
    return acos(value)

class FingerPoseAnalyzer:
    """Classify poses using joint angle and wrist-relative extension."""
    _chains = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    def __init__(self, extended_angle: float = 2.35, curled_ratio: float = .9) -> None:
        self.extended_angle, self.curled_ratio = extended_angle, curled_ratio
    def _finger(self, points, chain) -> FingerState:
        mcp, pip, dip, tip = (points[i] for i in chain)
        wrist = points[0]
        reach = _distance(tip, wrist) / max(_distance(pip, wrist), 1e-9)
        angle = (_angle(mcp, pip, dip) + _angle(pip, dip, tip)) / 2
        if angle >= self.extended_angle and reach > 1.04: return FingerState.EXTENDED
        if reach < self.curled_ratio or angle < 1.55: return FingerState.CURLED
        return FingerState.FLEXED
    def analyze(self, hand: HandLandmarks | tuple[Landmark3D, ...]) -> FingerStates:
        points = hand.landmarks if isinstance(hand, HandLandmarks) else hand
        fingers = {name: self._finger(points, chain) for name, chain in self._chains.items()}
        # Thumb is best described by its tip-to-wrist reach and the CMC/MCP bend.
        thumb_reach = _distance(points[4], points[0]) / max(_distance(points[2], points[0]), 1e-9)
        thumb_angle = _angle(points[1], points[2], points[3])
        fingers["thumb"] = (FingerState.EXTENDED if thumb_reach > 1.25 and thumb_angle > 1.5
                             else FingerState.CURLED if thumb_reach < .9 or thumb_angle < 1.0
                             else FingerState.FLEXED)
        return FingerStates(fingers["thumb"], fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"])
    classify = analyze
