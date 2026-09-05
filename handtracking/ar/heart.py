"""Digital AR Baby-Pink Heart on Palm with Dynamic Open/Close Scaling and Smooth Animation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Sequence

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

from ..inference.depth import estimate_hand_depth
from ..inference.models import HandLandmarks, Landmark3D
from .colliders import _add, _cross, _dot, _norm, _normalize, _scale, _sub


@dataclass
class HeartState:
    """State tracking for AR digital heart hovering over user's palm."""
    position: tuple[float, float, float] = (0.5, 0.5, 0.0)
    target_position: tuple[float, float, float] = (0.5, 0.5, 0.0)
    normal: tuple[float, float, float] = (0.0, -1.0, 0.0)
    scale: float = 1.0
    target_scale: float = 1.0
    openness: float = 1.0
    pulse_phase: float = 0.0
    hand_id: Any | None = None
    is_visible: bool = False
    is_activated: bool = False
    alpha: float = 1.0
    min_scale: float = 0.075
    max_scale: float = 1.0
    base_radius: float = 0.140
    color_bgr: tuple[int, int, int] = (193, 182, 255)       # Baby Pink #FFB6C1
    core_color_bgr: tuple[int, int, int] = (220, 210, 255)  # Pastel Light Pink
    glow_color_bgr: tuple[int, int, int] = (190, 120, 255)  # Radiant Aura Pink
    highlight_color_bgr: tuple[int, int, int] = (255, 242, 248)  # Specular Sheen


class PalmOpennessEstimator:
    """Computes continuous, smooth palm openness ratio in [0.0, 1.0] from 5-finger tip distances."""

    @staticmethod
    def is_palm_facing_camera(hand: HandLandmarks) -> bool:
        """
        Check whether the front palmar surface (inside palm) is facing the camera.
        Returns False when the dorsal surface (back of hand / knuckles) faces the camera.
        """
        lm = hand.landmarks
        if len(lm) < 21:
            return True

        p0 = (lm[0].x, lm[0].y)
        p5 = (lm[5].x, lm[5].y)
        p17 = (lm[17].x, lm[17].y)

        # 2D cross product of (P5 - P0) x (P17 - P0) in image coordinates (x right, y down)
        # Right hand: Palm facing camera -> cross_z > 0; Back of hand -> cross_z < 0
        # Left hand:  Palm facing camera -> cross_z < 0; Back of hand -> cross_z > 0
        cross_z = (p5[0] - p0[0]) * (p17[1] - p0[1]) - (p5[1] - p0[1]) * (p17[0] - p0[0])

        label = hand.handedness.label.strip().title() if hand.handedness else "Right"
        if label == "Right":
            return cross_z > -0.002
        else:
            return cross_z < 0.002

    @staticmethod
    def compute_openness(hand: HandLandmarks) -> float:
        """
        Calculate normalized continuous palm openness.
        
        Returns:
            1.0 for fully spread open palm,
            0.0 for tightly closed fist,
            continuous intermediate values for organic scaling.
        """
        lm = hand.landmarks
        if len(lm) < 21:
            return 1.0

        p0 = (lm[0].x, lm[0].y, lm[0].z)
        p5 = (lm[5].x, lm[5].y, lm[5].z)
        p9 = (lm[9].x, lm[9].y, lm[9].z)
        p17 = (lm[17].x, lm[17].y, lm[17].z)

        # Reference scale: wrist to middle MCP distance
        palm_ref = _norm(_sub(p9, p0))
        if palm_ref < 0.02:
            palm_ref = 0.15

        # 4 Long Fingers (Index 8, Middle 12, Ring 16, Pinky 20) vs wrist (0)
        d_idx = _norm(_sub((lm[8].x, lm[8].y, lm[8].z), p0)) / palm_ref
        d_mid = _norm(_sub((lm[12].x, lm[12].y, lm[12].z), p0)) / palm_ref
        d_rng = _norm(_sub((lm[16].x, lm[16].y, lm[16].z), p0)) / palm_ref
        d_pnk = _norm(_sub((lm[20].x, lm[20].y, lm[20].z), p0)) / palm_ref

        # Thumb (4) vs Pinky MCP (17) span
        d_thb = _norm(_sub((lm[4].x, lm[4].y, lm[4].z), p17)) / palm_ref

        # Normalization mappings (open ~ 1.8-2.3, closed ~ 0.7-0.9)
        c_idx = max(0.0, min(1.0, (d_idx - 0.90) / 1.05))
        c_mid = max(0.0, min(1.0, (d_mid - 0.95) / 1.10))
        c_rng = max(0.0, min(1.0, (d_rng - 0.85) / 1.05))
        c_pnk = max(0.0, min(1.0, (d_pnk - 0.75) / 0.95))
        c_thb = max(0.0, min(1.0, (d_thb - 0.55) / 0.70))

        # Weighted combination across all 5 fingers
        openness = 0.25 * c_idx + 0.25 * c_mid + 0.20 * c_rng + 0.15 * c_pnk + 0.15 * c_thb
        return max(0.0, min(1.0, float(openness)))



def generate_heart_mesh_2d(num_points: int = 48) -> list[tuple[float, float]]:
    """
    Generate parametric 2D normalized heart vertices centered at its visual area centroid.
    Returns array of (x, y) coordinates with x, y in [-1.0, 1.0].
    """
    points = []
    # Parametric equations:
    # x(t) = 16 * sin^3(t)  -> range [-16, 16]
    # y(t) = -(13 * cos(t) - 5 * cos(2t) - 2 * cos(3t) - cos(4t))
    for i in range(num_points):
        t = 2.0 * math.pi * i / num_points
        sin_t = math.sin(t)
        cos_t = math.cos(t)

        x = 16.0 * (sin_t ** 3)
        y = -(13.0 * cos_t - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t))

        # Normalize to [-1.0, 1.0] and align visual area centroid directly to (0, 0)
        norm_x = x / 16.0
        norm_y = (y + 0.823) / 17.823
        points.append((norm_x, norm_y))
    return points




_HEART_CONTOUR_2D = generate_heart_mesh_2d(48)


class ARHeartEngine:
    """Interactive AR engine rendering a responsive Baby-Pink Heart floating over palm."""

    def __init__(
        self,
        enabled: bool = True,
        min_scale: float = 0.075,
        max_scale: float = 1.0,
        base_radius: float = 0.140,
        pulse_bpm: float = 75.0,
    ):
        self.enabled = enabled
        self.state = HeartState(
            min_scale=min_scale,
            max_scale=max_scale,
            base_radius=base_radius,
        )
        self.pulse_bpm = pulse_bpm
        self._last_timestamp: float | None = None
        self._sparkles: list[dict[str, float]] = []
        self._init_sparkles()

    def _init_sparkles(self) -> None:
        """Initialize orbiting glitter sparkle particles."""
        self._sparkles = [
            {"angle": i * (2.0 * math.pi / 5), "dist": 1.25 + 0.15 * (i % 3), "speed": 1.2 + 0.3 * (i % 2), "phase": i * 1.3}
            for i in range(5)
        ]

    def toggle(self) -> bool:
        """Toggle AR Heart active state."""
        self.enabled = not self.enabled
        return self.enabled

    def reset(self) -> None:
        """Reset heart state."""
        self.state.is_visible = False
        self.state.is_activated = False
        self.state.scale = self.state.max_scale
        self.state.target_scale = self.state.max_scale
        self.state.openness = 1.0
        self.state.alpha = 0.0
        self._last_timestamp = None

    def step(self, hands: Iterable[HandLandmarks] = (), timestamp: float | None = None) -> HeartState:
        """Update palm tracking, normal orientation, continuous openness scaling, and heartbeat pulse."""
        ts = time.time() if timestamp is None else float(timestamp)
        dt = (ts - self._last_timestamp) if self._last_timestamp is not None else 0.033
        dt = max(0.001, min(0.5, dt))
        self._last_timestamp = ts

        hands_list = list(hands)
        if not self.enabled or not hands_list:
            # Smooth fade-out when no hands present
            self.state.alpha = max(0.0, self.state.alpha - dt * 4.0)
            if self.state.alpha <= 0.01:
                self.state.is_visible = False
                self.state.is_activated = False
            return self.state

        # Track first available hand
        hand = hands_list[0]
        lm = hand.landmarks
        self.state.hand_id = hand.handedness.label

        # 1. Check whether palm faces camera (suppress when showing back of hand)
        palm_facing = PalmOpennessEstimator.is_palm_facing_camera(hand)
        if not palm_facing:
            # Back of hand is facing camera -> immediately fade out and deactivate
            self.state.alpha = max(0.0, self.state.alpha - dt * 10.0)
            if self.state.alpha <= 0.01:
                self.state.is_visible = False
                self.state.is_activated = False
            return self.state

        # 2. Continuous Palm Openness Estimation
        raw_openness = PalmOpennessEstimator.compute_openness(hand)

        # Heart appears on open palm only
        if not self.state.is_activated:
            if raw_openness >= 0.55:
                self.state.is_activated = True
                self.state.is_visible = True
            else:
                # Hand is present but not yet an open palm -> remain hidden
                self.state.alpha = max(0.0, self.state.alpha - dt * 6.0)
                if self.state.alpha <= 0.01:
                    self.state.is_visible = False
                return self.state

        self.state.is_visible = True
        self.state.alpha = min(1.0, self.state.alpha + dt * 6.0)


        # 2. Compute Geometric Center of Palm Pad
        z_offset = estimate_hand_depth(hand)
        p0 = (lm[0].x, lm[0].y, z_offset + lm[0].z)
        p5 = (lm[5].x, lm[5].y, z_offset + lm[5].z)
        p9 = (lm[9].x, lm[9].y, z_offset + lm[9].z)
        p17 = (lm[17].x, lm[17].y, z_offset + lm[17].z)

        # True center of the palm pad: intersection of wrist-to-middle and index-to-pinky lines
        cx = 0.5 * (0.5 * (p0[0] + p9[0]) + 0.5 * (p5[0] + p17[0]))
        cy = 0.5 * (0.5 * (p0[1] + p9[1]) + 0.5 * (p5[1] + p17[1]))
        cz = 0.5 * (0.5 * (p0[2] + p9[2]) + 0.5 * (p5[2] + p17[2]))

        v1 = _sub(p5, p0)
        v2 = _sub(p17, p0)
        raw_norm = _cross(v1, v2)
        norm = _normalize(raw_norm)
        if norm[2] > 0:
            norm = _scale(norm, -1.0)
        self.state.normal = norm

        # Direct palm center position without lateral offset drift
        target_pos = (cx, cy, cz)
        self.state.target_position = target_pos

        # Exponential smoothing for position (responsive tracking)
        pos_alpha = 1.0 - math.exp(-22.0 * dt)
        cur_pos = self.state.position
        self.state.position = (
            cur_pos[0] + (target_pos[0] - cur_pos[0]) * pos_alpha,
            cur_pos[1] + (target_pos[1] - cur_pos[1]) * pos_alpha,
            cur_pos[2] + (target_pos[2] - cur_pos[2]) * pos_alpha,
        )

        # 3. Smooth Openness Metric
        open_alpha = 1.0 - math.exp(-16.0 * dt)
        self.state.openness += (raw_openness - self.state.openness) * open_alpha
        self.state.openness = max(0.0, min(1.0, self.state.openness))

        # 4. Dynamic Scale Interpolation
        target_s = self.state.min_scale + self.state.openness * (self.state.max_scale - self.state.min_scale)
        self.state.target_scale = target_s

        scale_alpha = 1.0 - math.exp(-14.0 * dt)
        self.state.scale += (target_s - self.state.scale) * scale_alpha

        # 5. Rhythmic Organic Heartbeat Pulse
        omega = 2.0 * math.pi * (self.pulse_bpm / 60.0)
        self.state.pulse_phase += omega * dt
        if self.state.pulse_phase > 2.0 * math.pi * 100.0:
            self.state.pulse_phase -= 2.0 * math.pi * 100.0

        return self.state

    def draw(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
        projection_fn: Any = None,
        focal_depth: float = 0.85,
    ) -> np.ndarray:
        """Render glowing baby-pink digital heart, aura, specular gloss, and sparkles on frame."""
        if not self.enabled or not self.state.is_visible or self.state.alpha <= 0.01:
            return frame

        h, w = frame.shape[:2]
        ts = time.time() if timestamp is None else float(timestamp)
        pos = self.state.position

        # 1. Screen Projection
        if projection_fn is not None:
            px, py = projection_fn(pos[0], pos[1], pos[2], w, h)
            depth_scale = 1.0
        else:
            # Direct 2D screen coordinate mapping pinned to landmark palm center
            px = round(pos[0] * (w - 1))
            py = round(pos[1] * (h - 1))
            # Depth scaling modulates size, not 2D screen position
            depth_scale = 1.0 / max(0.35, 1.0 + pos[2] * focal_depth)

        if not (-150 <= px < w + 150 and -150 <= py < h + 150):
            return frame

        # 2. Compute Animated Heart Radius with Heartbeat Pulse & Depth Scale
        phase = self.state.pulse_phase
        lub_dub = math.sin(phase) + 0.35 * math.sin(2.0 * phase - 0.5)
        pulse_amp = 0.05 * self.state.openness  # Pulsing dampens when closed into seed
        effective_scale = self.state.scale * (1.0 + pulse_amp * lub_dub)

        base_px_radius = self.state.base_radius * min(w, h)
        radius = max(6.0, base_px_radius * effective_scale * depth_scale)

        if cv2 is None:
            # Fallback direct pixel dot
            if 0 <= py < h and 0 <= px < w:
                frame[py, px] = self.state.color_bgr
            return frame

        # 3. Render Layered Heart Visuals (OpenCV AA)
        self._render_aura_glow(frame, px, py, radius, self.state.alpha)
        self._render_heart_body(frame, px, py, radius, self.state.alpha)
        self._render_specular_sheen(frame, px, py, radius, self.state.alpha)
        if self.state.openness > 0.4:
            self._render_sparkles(frame, px, py, radius, ts, self.state.alpha * self.state.openness)

        return frame

    render = draw

    def _render_aura_glow(self, frame: np.ndarray, cx: int, cy: int, radius: float, alpha: float) -> None:
        """Render concentric soft radial glowing auras behind the heart."""
        glow_layers = [
            (radius * 1.50, (200, 140, 255), 0.12 * alpha),  # Outer ethereal aura
            (radius * 1.25, (190, 120, 255), 0.22 * alpha),  # Mid pink glow
            (radius * 1.10, (210, 160, 255), 0.35 * alpha),  # Inner halo
        ]

        for r_layer, col, layer_alpha in glow_layers:
            if layer_alpha <= 0.01:
                continue
            poly = np.array(
                [(round(cx + pt[0] * r_layer), round(cy + pt[1] * r_layer)) for pt in _HEART_CONTOUR_2D],
                dtype=np.int32,
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], col, cv2.LINE_AA)
            cv2.addWeighted(overlay, layer_alpha, frame, 1.0 - layer_alpha, 0, frame)

    def _render_heart_body(self, frame: np.ndarray, cx: int, cy: int, radius: float, alpha: float) -> None:
        """Render solid anti-aliased baby-pink heart with pastel gradient core."""
        poly_main = np.array(
            [(round(cx + pt[0] * radius), round(cy + pt[1] * radius)) for pt in _HEART_CONTOUR_2D],
            dtype=np.int32,
        )

        # Core heart fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_main], self.state.color_bgr, cv2.LINE_AA)
        
        # Inner lighter pastel core (concentric)
        r_inner = radius * 0.68
        poly_inner = np.array(
            [(round(cx + pt[0] * r_inner), round(cy + pt[1] * r_inner)) for pt in _HEART_CONTOUR_2D],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [poly_inner], self.state.core_color_bgr, cv2.LINE_AA)


        # Smooth border outline
        cv2.polylines(overlay, [poly_main], isClosed=True, color=(240, 210, 255), thickness=max(1, round(radius * 0.04)), lineType=cv2.LINE_AA)

        body_alpha = min(1.0, 0.88 * alpha)
        cv2.addWeighted(overlay, body_alpha, frame, 1.0 - body_alpha, 0, frame)

    def _render_specular_sheen(self, frame: np.ndarray, cx: int, cy: int, radius: float, alpha: float) -> None:
        """Render glossy specular highlight reflection on the upper-left lobe."""
        if radius < 12.0:
            return

        hx = round(cx - radius * 0.32)
        hy = round(cy - radius * 0.28)
        ax_maj = max(2, round(radius * 0.24))
        ax_min = max(1, round(radius * 0.12))
        angle = -35

        overlay = frame.copy()
        cv2.ellipse(overlay, (hx, hy), (ax_maj, ax_min), angle, 0, 360, self.state.highlight_color_bgr, -1, cv2.LINE_AA)
        sheen_alpha = 0.55 * alpha
        cv2.addWeighted(overlay, sheen_alpha, frame, 1.0 - sheen_alpha, 0, frame)

    def _render_sparkles(self, frame: np.ndarray, cx: int, cy: int, radius: float, ts: float, alpha: float) -> None:
        """Render twinkling micro-sparkles orbiting the heart when open."""
        for sp in self._sparkles:
            angle = sp["angle"] + ts * sp["speed"]
            dist = radius * sp["dist"]
            sx = round(cx + math.cos(angle) * dist)
            sy = round(cy + math.sin(angle) * dist * 0.85)

            sparkle_twinkle = 0.5 + 0.5 * math.sin(ts * 6.0 + sp["phase"])
            sp_alpha = alpha * sparkle_twinkle * 0.75
            if sp_alpha <= 0.05:
                continue

            r_sp = max(1, round(2.5 * sparkle_twinkle))
            overlay = frame.copy()
            # 4-pointed sparkle star
            col = (255, 235, 245)
            cv2.line(overlay, (sx - r_sp * 2, sy), (sx + r_sp * 2, sy), col, 1, cv2.LINE_AA)
            cv2.line(overlay, (sx, sy - r_sp * 2), (sx, sy + r_sp * 2), col, 1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), r_sp, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, sp_alpha, frame, 1.0 - sp_alpha, 0, frame)
