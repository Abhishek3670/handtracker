"""Persistent lightweight air-drawing canvas."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass
class Stroke:
    points: list[tuple[float,float]]
    color: tuple[int,int,int]
    thickness: int
class AirCanvas:
    def __init__(self, color=(0,255,0), thickness=2): self.color=tuple(color); self.thickness=thickness; self.strokes=[]; self._active=None
    def start_stroke(self, point, color=None, thickness=None): self._active=Stroke([tuple(point[:2])],tuple(color or self.color),thickness or self.thickness); self.strokes.append(self._active); return self._active
    def add_point(self, point): self.start_stroke(point) if self._active is None else self._active.points.append(tuple(point[:2]))
    def update(self, point, drawing=True): self.add_point(point) if drawing else self.end_stroke()
    def end_stroke(self): self._active=None
    def clear(self): self.strokes.clear(); self._active=None
    reset=clear
    def set_color(self,color): self.color=tuple(color)
    def render(self, frame):
        try:
            import cv2
            h,w=frame.shape[:2]
            for stroke in self.strokes:
                points=[(round(x*(w-1)),round(y*(h-1))) for x,y in stroke.points]
                for a,b in zip(points,points[1:]): cv2.line(frame,a,b,stroke.color,stroke.thickness,cv2.LINE_AA)
        except ImportError: pass
        return frame
