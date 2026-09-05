"""3D hand physics colliders and velocity tracking."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from ..inference.models import HandLandmarks, Landmark3D


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Sequence[float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Sequence[float]) -> tuple[float, float, float]:
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-6 else (0.0, 0.0, 1.0)


@dataclass
class PalmCollider:
    """3D Palm plane collider derived from Wrist (0), Index MCP (5), and Pinky MCP (17)."""
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float

    @classmethod
    def from_hand(cls, hand: HandLandmarks) -> PalmCollider:
        lm = hand.landmarks
        p0 = (lm[0].x, lm[0].y, lm[0].z)
        p5 = (lm[5].x, lm[5].y, lm[5].z)
        p17 = (lm[17].x, lm[17].y, lm[17].z)

        # Centroid of palm base triangle
        cx = (p0[0] + p5[0] + p17[0]) / 3.0
        cy = (p0[1] + p5[1] + p17[1]) / 3.0
        cz = (p0[2] + p5[2] + p17[2]) / 3.0
        origin = (cx, cy, cz)

        v1 = _sub(p5, p0)
        v2 = _sub(p17, p0)
        raw_normal = _cross(v1, v2)
        normal = _normalize(raw_normal)

        # Ensure palm normal points slightly toward camera / upward (negative z / negative y)
        if normal[2] > 0:
            normal = _scale(normal, -1.0)

        # Palm span radius
        r0 = _norm(_sub(p0, origin))
        r5 = _norm(_sub(p5, origin))
        r17 = _norm(_sub(p17, origin))
        radius = max(r0, r5, r17, 0.05) * 1.35
        return cls(origin=origin, normal=normal, radius=radius)

    def distance_to_plane(self, point: Sequence[float]) -> float:
        """Signed distance from a 3D point to the palm plane."""
        diff = _sub(point, self.origin)
        return _dot(diff, self.normal)

    def closest_point_on_plane(self, point: Sequence[float]) -> tuple[float, float, float]:
        dist = self.distance_to_plane(point)
        return _sub(point, _scale(self.normal, dist))

    def check_collision(
        self,
        ball_pos: Sequence[float],
        ball_radius: float,
    ) -> tuple[bool, float, tuple[float, float, float]]:
        """
        Returns (is_colliding, signed_distance, plane_normal).
        """
        dist = self.distance_to_plane(ball_pos)
        proj = self.closest_point_on_plane(ball_pos)
        dist_to_center = _norm(_sub(proj, self.origin))

        if abs(dist) <= ball_radius and dist_to_center <= self.radius:
            return True, dist, self.normal
        return False, dist, self.normal


@dataclass
class FingertipCollider:
    """Spherical colliders for 5 fingertips (Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20)."""
    tips: dict[int, tuple[float, float, float]]
    tip_radius: float = 0.035

    @classmethod
    def from_hand(cls, hand: HandLandmarks, tip_radius: float = 0.035) -> FingertipCollider:
        lm = hand.landmarks
        tips = {
            4: (lm[4].x, lm[4].y, lm[4].z),
            8: (lm[8].x, lm[8].y, lm[8].z),
            12: (lm[12].x, lm[12].y, lm[12].z),
            16: (lm[16].x, lm[16].y, lm[16].z),
            20: (lm[20].x, lm[20].y, lm[20].z),
        }
        return cls(tips=tips, tip_radius=tip_radius)

    def check_collision(
        self,
        ball_pos: Sequence[float],
        ball_radius: float,
    ) -> tuple[bool, int, tuple[float, float, float]]:
        """
        Returns (is_colliding, hit_tip_index, collision_normal) for the closest colliding fingertip.
        """
        closest_idx = -1
        min_dist = float("inf")
        closest_normal = (0.0, 0.0, 0.0)

        for tip_idx, tip_pos in self.tips.items():
            diff = _sub(ball_pos, tip_pos)
            dist = _norm(diff)
            if dist <= (ball_radius + self.tip_radius) and dist < min_dist:
                min_dist = dist
                closest_idx = tip_idx
                closest_normal = _normalize(diff) if dist > 1e-6 else (0.0, -1.0, 0.0)

        if closest_idx != -1:
            return True, closest_idx, closest_normal
        return False, -1, (0.0, 0.0, 0.0)


class HandVelocityTracker:
    """Estimates smoothed 3D linear velocity (vx, vy, vz) from frame history."""

    def __init__(self, smoothing_factor: float = 0.65):
        self.smoothing_factor = smoothing_factor
        self.history: dict[Any, tuple[tuple[float, float, float], float]] = {}
        self.velocities: dict[Any, tuple[float, float, float]] = {}

    def update(
        self,
        hand_id: Any,
        pos: Sequence[float],
        timestamp: float,
    ) -> tuple[float, float, float]:
        curr_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        ts = float(timestamp)

        if hand_id not in self.history:
            self.history[hand_id] = (curr_pos, ts)
            self.velocities[hand_id] = (0.0, 0.0, 0.0)
            return (0.0, 0.0, 0.0)

        prev_pos, prev_ts = self.history[hand_id]
        dt = ts - prev_ts

        if dt > 1e-4:
            raw_vx = (curr_pos[0] - prev_pos[0]) / dt
            raw_vy = (curr_pos[1] - prev_pos[1]) / dt
            raw_vz = (curr_pos[2] - prev_pos[2]) / dt
            prev_v = self.velocities.get(hand_id, (0.0, 0.0, 0.0))

            # Exponential smoothing
            vx = self.smoothing_factor * raw_vx + (1 - self.smoothing_factor) * prev_v[0]
            vy = self.smoothing_factor * raw_vy + (1 - self.smoothing_factor) * prev_v[1]
            vz = self.smoothing_factor * raw_vz + (1 - self.smoothing_factor) * prev_v[2]
            vel = (vx, vy, vz)
        else:
            vel = self.velocities.get(hand_id, (0.0, 0.0, 0.0))

        self.history[hand_id] = (curr_pos, ts)
        self.velocities[hand_id] = vel
        return vel

    def get_velocity(self, hand_id: Any) -> tuple[float, float, float]:
        return self.velocities.get(hand_id, (0.0, 0.0, 0.0))

    def reset(self, hand_id: Any = None) -> None:
        if hand_id is None:
            self.history.clear()
            self.velocities.clear()
        else:
            self.history.pop(hand_id, None)
            self.velocities.pop(hand_id, None)
