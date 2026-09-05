"""Real-time 3D rigid-body AR physics engine for interactive ball simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Iterable, Sequence

from ..inference.models import HandLandmarks, Landmark3D
from .colliders import FingertipCollider, HandVelocityTracker, PalmCollider, PointCollider, _add, _dot, _norm, _normalize, _scale, _sub


class BallInteractionState(str, Enum):
    FREE_FLIGHT = "free_flight"
    PALM_BOUNCE = "palm_bounce"
    FINGERTIP_VOLLEY = "fingertip_volley"
    GRABBED = "grabbed"


@dataclass
class ImpactRipple:
    """Expanding contact shockwave ripple ring."""
    center: tuple[float, float, float]
    birth_time: float
    radius: float = 0.015
    max_radius: float = 0.08
    lifetime: float = 0.35
    alpha: float = 1.0

    def update(self, current_time: float) -> bool:
        """Update ripple animation. Returns False when ripple has expired."""
        elapsed = current_time - self.birth_time
        if elapsed >= self.lifetime:
            self.alpha = 0.0
            return False
        progress = elapsed / self.lifetime
        self.radius = 0.015 + (self.max_radius - 0.015) * progress
        self.alpha = max(0.0, 1.0 - progress)
        return True


@dataclass
class BallState:
    """3D rigid body state of the AR ball."""
    position: tuple[float, float, float] = (0.5, 0.3, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.055
    mass: float = 1.0
    restitution: float = 0.82
    state: BallInteractionState = BallInteractionState.FREE_FLIGHT
    grabbed_hand_id: Any = None

    def reset(self, x: float = 0.5, y: float = 0.3, z: float = 0.0) -> None:
        self.position = (float(x), float(y), float(z))
        self.velocity = (0.0, 0.0, 0.0)
        self.state = BallInteractionState.FREE_FLIGHT
        self.grabbed_hand_id = None


class ARPhysicsEngine:
    """60 FPS 3D rigid-body numerical integrator with hand collision and grab dynamics."""

    def __init__(
        self,
        gravity: tuple[float, float, float] = (0.0, 1.25, 0.0),
        air_drag: float = 0.12,
        restitution: float = 0.82,
        enable_gravity: bool = True,
    ):
        self.ball = BallState(restitution=restitution)
        self.gravity = gravity
        self.air_drag = air_drag
        self.enable_gravity = enable_gravity
        self.velocity_tracker = HandVelocityTracker()
        self.ripples: list[ImpactRipple] = []
        self.last_step_time: float | None = None

        # Normalized screen boundaries
        self.bounds_min = (0.05, 0.05, -0.6)
        self.bounds_max = (0.95, 0.95, 0.6)
        self.grab_threshold = 0.09

    def spawn_ripple(self, center: tuple[float, float, float], timestamp: float) -> None:
        self.ripples.append(ImpactRipple(center=center, birth_time=timestamp))

    def step(
        self,
        hands: Iterable[HandLandmarks] = (),
        gestures: Iterable[Any] = (),
        dt: float | None = None,
        timestamp: float | None = None,
    ) -> BallState:
        """Advance physics state by dt seconds."""
        ts = time.time() if timestamp is None else float(timestamp)
        if dt is None:
            if self.last_step_time is not None:
                dt = max(0.001, min(0.05, ts - self.last_step_time))
            else:
                dt = 1.0 / 60.0
        self.last_step_time = ts
        dt = max(0.001, min(0.05, float(dt)))

        hands_list = list(hands)

        # 1. Update hand velocities
        for hand in hands_list:
            hand_id = hand.handedness.label
            p0 = (hand.landmarks[0].x, hand.landmarks[0].y, hand.landmarks[0].z)
            self.velocity_tracker.update(hand_id, p0, ts)

        # 2. Check Pinch-to-Grab and Throw
        is_pinching = False
        active_pinch_point: tuple[float, float, float] | None = None
        active_hand_id: Any = None

        for hand in hands_list:
            hand_id = hand.handedness.label
            lm = hand.landmarks
            thumb_tip = (lm[4].x, lm[4].y, lm[4].z)
            index_tip = (lm[8].x, lm[8].y, lm[8].z)
            p_dx = thumb_tip[0] - index_tip[0]
            p_dy = thumb_tip[1] - index_tip[1]
            p_dz = thumb_tip[2] - index_tip[2]
            pinch_dist_2d = math.hypot(p_dx, p_dy)
            pinch_dist_3d = math.sqrt(p_dx * p_dx + p_dy * p_dy + p_dz * p_dz)

            # 2.5D pinch check
            if pinch_dist_3d <= 0.075 or (pinch_dist_2d <= 0.065 and abs(p_dz) <= 0.18):
                pinch_mid = (
                    (thumb_tip[0] + index_tip[0]) * 0.5,
                    (thumb_tip[1] + index_tip[1]) * 0.5,
                    (thumb_tip[2] + index_tip[2]) * 0.5,
                )
                b_dx = pinch_mid[0] - self.ball.position[0]
                b_dy = pinch_mid[1] - self.ball.position[1]
                b_dz = pinch_mid[2] - self.ball.position[2]
                b_dist_2d = math.hypot(b_dx, b_dy)
                b_dist_3d = math.sqrt(b_dx * b_dx + b_dy * b_dy + b_dz * b_dz)

                is_near = (b_dist_3d <= self.grab_threshold) or (
                    b_dist_2d <= self.grab_threshold and abs(b_dz) <= 0.25
                )

                if is_near or (
                    self.ball.state == BallInteractionState.GRABBED and self.ball.grabbed_hand_id == hand_id
                ):
                    is_pinching = True
                    active_pinch_point = pinch_mid
                    active_hand_id = hand_id
                    break

        if is_pinching and active_pinch_point is not None:
            self.ball.state = BallInteractionState.GRABBED
            self.ball.grabbed_hand_id = active_hand_id
            self.ball.position = active_pinch_point
            hand_vel = self.velocity_tracker.get_velocity(active_hand_id)
            self.ball.velocity = hand_vel
            self._update_ripples(ts)
            return self.ball

        elif self.ball.state == BallInteractionState.GRABBED and not is_pinching:
            # Released / Thrown!
            self.ball.state = BallInteractionState.FREE_FLIGHT
            prev_hand_id = self.ball.grabbed_hand_id
            hand_vel = self.velocity_tracker.get_velocity(prev_hand_id)
            # Momentum transfer boost
            self.ball.velocity = _scale(hand_vel, 1.35)
            self.ball.grabbed_hand_id = None

        # 3. Free flight physics integration (Gravity + Drag)
        vx, vy, vz = self.ball.velocity
        if self.enable_gravity:
            vx += self.gravity[0] * dt
            vy += self.gravity[1] * dt
            vz += self.gravity[2] * dt

        # Velocity damping / air drag
        drag_mult = max(0.0, 1.0 - self.air_drag * dt)
        vx *= drag_mult
        vy *= drag_mult
        vz *= drag_mult

        px = self.ball.position[0] + vx * dt
        py = self.ball.position[1] + vy * dt
        pz = self.ball.position[2] + vz * dt
        self.ball.position = (px, py, pz)
        self.ball.velocity = (vx, vy, vz)

        # 4. Hand Collisions (Palms & Fingertips)
        r = self.ball.radius
        e = self.ball.restitution

        for hand in hands_list:
            hand_id = hand.handedness.label
            hand_vel = self.velocity_tracker.get_velocity(hand_id)

            # A. Palm Collision
            palm = PalmCollider.from_hand(hand)
            colliding, dist, normal = palm.check_collision(self.ball.position, r)
            if colliding:
                # Correct penetration
                penetration = r - abs(dist)
                if penetration > 0:
                    px += normal[0] * penetration
                    py += normal[1] * penetration
                    pz += normal[2] * penetration
                    self.ball.position = (px, py, pz)

                rel_v = _sub(self.ball.velocity, hand_vel)
                vn = _dot(rel_v, normal)
                if vn < 0:  # moving into palm
                    # Reflect velocity with restitution + hand impulse
                    impulse = -(1.0 + e) * vn
                    new_vx = rel_v[0] + normal[0] * impulse + hand_vel[0] * 0.8
                    new_vy = rel_v[1] + normal[1] * impulse + hand_vel[1] * 0.8
                    new_vz = rel_v[2] + normal[2] * impulse + hand_vel[2] * 0.8
                    self.ball.velocity = (new_vx, new_vy, new_vz)
                    self.ball.state = BallInteractionState.PALM_BOUNCE
                    self.spawn_ripple(self.ball.position, ts)
                continue

            # B. Fingertip Volley Collisions
            fingertips = FingertipCollider.from_hand(hand)
            tip_hit, tip_idx, tip_normal = fingertips.check_collision(self.ball.position, r)
            if tip_hit:
                rel_v = _sub(self.ball.velocity, hand_vel)
                vn = _dot(rel_v, tip_normal)
                if vn < 0:
                    impulse = -(1.0 + e) * vn
                    new_vx = rel_v[0] + tip_normal[0] * impulse + hand_vel[0] * 1.1
                    new_vy = rel_v[1] + tip_normal[1] * impulse + hand_vel[1] * 1.1
                    new_vz = rel_v[2] + tip_normal[2] * impulse + hand_vel[2] * 1.1
                    self.ball.velocity = (new_vx, new_vy, new_vz)
                    self.ball.state = BallInteractionState.FINGERTIP_VOLLEY
                    self.spawn_ripple(self.ball.position, ts)

        # 5. Screen Boundaries Elastic Collisions
        px, py, pz = self.ball.position
        vx, vy, vz = self.ball.velocity

        # Floor bounce
        if py >= (self.bounds_max[1] - r):
            py = self.bounds_max[1] - r
            vy = -abs(vy) * e
            vx *= 0.96
            vz *= 0.96
            if abs(vy) > 0.15:
                self.spawn_ripple((px, py, pz), ts)

        # Ceiling bounce
        elif py <= (self.bounds_min[1] + r):
            py = self.bounds_min[1] + r
            vy = abs(vy) * e

        # Left wall
        if px <= (self.bounds_min[0] + r):
            px = self.bounds_min[0] + r
            vx = abs(vx) * e

        # Right wall
        elif px >= (self.bounds_max[0] - r):
            px = self.bounds_max[0] - r
            vx = -abs(vx) * e

        # Depth limits (z)
        if pz <= self.bounds_min[2]:
            pz = self.bounds_min[2]
            vz = abs(vz) * e
        elif pz >= self.bounds_max[2]:
            pz = self.bounds_max[2]
            vz = -abs(vz) * e

        self.ball.position = (px, py, pz)
        self.ball.velocity = (vx, vy, vz)

        # 6. Update ripples
        self._update_ripples(ts)
        return self.ball

    def _update_ripples(self, current_time: float) -> None:
        self.ripples = [r for r in self.ripples if r.update(current_time)]
