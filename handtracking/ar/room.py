"""Digital 3D Cyber-Space Environment, Holographic Hands, and PIP Camera Overlay."""
from __future__ import annotations
import math
import time
from typing import Any, Iterable, Sequence

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

from ..inference.models import HAND_CONNECTIONS, HandLandmarks, HandednessLabel
from .colliders import PalmCollider
from .physics import ARPhysicsEngine, BallState


class Virtual3DRoomRenderer:
    """Renders a digital 3D perspective cyber-space environment, holographic hands, and PIP webcam feed."""

    def __init__(
        self,
        show_pip: bool = True,
        pip_scale: float = 0.20,
        focal_depth: float = 0.85,
        bounds_min: tuple[float, float, float] = (0.05, 0.05, -0.6),
        bounds_max: tuple[float, float, float] = (0.95, 0.95, 0.6),
    ):
        self.show_pip = show_pip
        self.pip_scale = pip_scale
        self.focal_depth = focal_depth
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.wall_glow_time: float = 0.0
        self.wall_glow_color: tuple[int, int, int] = (0, 220, 255)

    def project_3d(self, x: float, y: float, z: float, width: int, height: int) -> tuple[int, int]:
        """Project normalized 3D coordinates (x, y, z) to screen pixels using perspective frustum."""
        scale = 1.0 / max(0.25, 1.0 + z * self.focal_depth)
        u = round((0.5 + (x - 0.5) * scale) * (width - 1))
        v = round((0.5 + (y - 0.5) * scale) * (height - 1))
        return (max(-width, min(2 * width, u)), max(-height, min(2 * height, v)))

    def trigger_wall_pulse(self, timestamp: float | None = None, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        self.wall_glow_time = time.time() if timestamp is None else float(timestamp)
        self.wall_glow_color = color

    def render_room(
        self,
        frame: np.ndarray,
        engine: ARPhysicsEngine,
        hands: Iterable[HandLandmarks] = (),
        raw_webcam: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> np.ndarray:
        """Render the complete digital 3D room background, holographic hands, drop-indicators, and PIP."""
        ts = time.time() if timestamp is None else float(timestamp)
        h, w = frame.shape[:2]

        # 1. Fill cyber gradient background (Deep slate navy -> dark cosmic purple)
        self._render_cyber_background(frame, w, h)

        # 2. Render 3D Perspective Floor, Back Wall, and Boundary Grids
        self._render_3d_grids(frame, w, h, ts)

        # 3. Render 3D Holographic Hand Skeletons & Palm Avatars
        self._render_holographic_hands(frame, hands, w, h)

        # 4. Render Ball Spatial Altitude Drop-Line & Dynamic Floor Shadow
        self._render_ball_spatial_indicators(frame, engine.ball, w, h)

        # 5. Render Picture-in-Picture (PIP) Webcam Feed
        if self.show_pip and raw_webcam is not None:
            self._render_pip_webcam(frame, raw_webcam, w, h)

        return frame

    def _render_cyber_background(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw deep vertical cosmic slate gradient."""
        # Top color: (20, 14, 28) BGR, Bottom color: (40, 24, 52) BGR
        top_color = np.array([20, 14, 28], dtype=np.float32)
        bottom_color = np.array([42, 26, 56], dtype=np.float32)
        ratios = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        gradient = (top_color * (1.0 - ratios) + bottom_color * ratios).astype(np.uint8)
        frame[:] = np.repeat(gradient, width, axis=1)

    def _render_3d_grids(self, frame: np.ndarray, width: int, height: int, current_time: float) -> None:
        """Render perspective floor, back wall, and boundary wireframes."""
        if cv2 is None:
            return

        b_min_x, b_min_y, b_min_z = self.bounds_min
        b_max_x, b_max_y, b_max_z = self.bounds_max

        # Grid colors
        grid_col = (75, 50, 95)  # Purple-slate grid lines
        grid_accent = (140, 90, 180)  # Brighter accents

        # A. Perspective Floor Grid (Y = b_max_y)
        floor_y = b_max_y
        num_x_lines = 9
        for i in range(num_x_lines):
            x = b_min_x + (b_max_x - b_min_x) * (i / (num_x_lines - 1))
            p_front = self.project_3d(x, floor_y, b_min_z, width, height)
            p_back = self.project_3d(x, floor_y, b_max_z, width, height)
            col = grid_accent if i in (0, num_x_lines - 1, num_x_lines // 2) else grid_col
            cv2.line(frame, p_front, p_back, col, 1, cv2.LINE_AA)

        num_z_lines = 7
        for j in range(num_z_lines):
            z = b_min_z + (b_max_z - b_min_z) * (j / (num_z_lines - 1))
            p_left = self.project_3d(b_min_x, floor_y, z, width, height)
            p_right = self.project_3d(b_max_x, floor_y, z, width, height)
            cv2.line(frame, p_left, p_right, grid_col, 1, cv2.LINE_AA)

        # B. Back Wall Grid (Z = b_max_z)
        for i in range(num_x_lines):
            x = b_min_x + (b_max_x - b_min_x) * (i / (num_x_lines - 1))
            p_bottom = self.project_3d(x, b_max_y, b_max_z, width, height)
            p_top = self.project_3d(x, b_min_y, b_max_z, width, height)
            cv2.line(frame, p_bottom, p_top, grid_col, 1, cv2.LINE_AA)

        num_y_lines = 6
        for k in range(num_y_lines):
            y = b_min_y + (b_max_y - b_min_y) * (k / (num_y_lines - 1))
            p_left = self.project_3d(b_min_x, y, b_max_z, width, height)
            p_right = self.project_3d(b_max_x, y, b_max_z, width, height)
            cv2.line(frame, p_left, p_right, grid_col, 1, cv2.LINE_AA)

        # C. Side Wall & Ceiling Bounding Lines
        corners_front = [
            self.project_3d(b_min_x, b_min_y, b_min_z, width, height),
            self.project_3d(b_max_x, b_min_y, b_min_z, width, height),
            self.project_3d(b_max_x, b_max_y, b_min_z, width, height),
            self.project_3d(b_min_x, b_max_y, b_min_z, width, height),
        ]
        corners_back = [
            self.project_3d(b_min_x, b_min_y, b_max_z, width, height),
            self.project_3d(b_max_x, b_min_y, b_max_z, width, height),
            self.project_3d(b_max_x, b_max_y, b_max_z, width, height),
            self.project_3d(b_min_x, b_max_y, b_max_z, width, height),
        ]

        # Check impact pulse
        elapsed_glow = current_time - self.wall_glow_time
        glow_active = 0.0 <= elapsed_glow <= 0.35
        glow_col = self.wall_glow_color if glow_active else (90, 60, 120)
        thickness = 2 if glow_active else 1

        # Connect front to back corners (room depth edges)
        for cf, cb in zip(corners_front, corners_back):
            cv2.line(frame, cf, cb, glow_col, thickness, cv2.LINE_AA)

        # Ceiling front edge
        cv2.line(frame, corners_front[0], corners_front[1], (100, 70, 140), 1, cv2.LINE_AA)

    def _render_holographic_hands(
        self,
        frame: np.ndarray,
        hands: Iterable[HandLandmarks],
        width: int,
        height: int,
    ) -> None:
        """Render glowing 3D holographic skeletons and palm collision planes."""
        if cv2 is None:
            return

        for hand in hands:
            is_left = hand.handedness.label == HandednessLabel.LEFT
            bone_color = (255, 220, 0) if is_left else (0, 240, 255)  # Cyan or Gold
            joint_color = (255, 255, 255)  # Bright core

            # Project all 21 joints
            projected_pts = [
                self.project_3d(lm.x, lm.y, lm.z, width, height)
                for lm in hand.landmarks
            ]

            # 1. Semi-transparent Palm Avatar
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_poly = np.array([projected_pts[i] for i in palm_indices], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [palm_poly], (50, 120, 100) if not is_left else (120, 80, 40))
            cv2.polylines(overlay, [palm_poly], True, bone_color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # 2. Cyber-Bones
            for a, b in HAND_CONNECTIONS:
                pt_a = projected_pts[a]
                pt_b = projected_pts[b]
                avg_z = (hand.landmarks[a].z + hand.landmarks[b].z) * 0.5
                thickness = max(1, round(2.5 * (1.0 / max(0.3, 1.0 + avg_z))))
                cv2.line(frame, pt_a, pt_b, bone_color, thickness, cv2.LINE_AA)

            # 3. Glowing Joint Spheres
            for i, (lm, pt) in enumerate(zip(hand.landmarks, projected_pts)):
                z_scale = 1.0 / max(0.3, 1.0 + lm.z)
                base_r = 5 if i in (4, 8, 12, 16, 20) else 3
                radius = max(2, round(base_r * z_scale))
                cv2.circle(frame, pt, radius + 2, bone_color, 1, cv2.LINE_AA)
                cv2.circle(frame, pt, radius, joint_color, -1, cv2.LINE_AA)

    def _render_ball_spatial_indicators(
        self,
        frame: np.ndarray,
        ball: BallState,
        width: int,
        height: int,
    ) -> None:
        """Render vertical altitude drop-indicator line and animated 3D floor shadow ellipse."""
        if cv2 is None:
            return

        bx, by, bz = ball.position
        floor_y = self.bounds_max[1]

        p_ball = self.project_3d(bx, by, bz, width, height)
        p_floor = self.project_3d(bx, floor_y, bz, width, height)

        altitude = max(0.0, floor_y - by)
        max_alt = self.bounds_max[1] - self.bounds_min[1]
        alt_ratio = min(1.0, altitude / max_alt)

        # 1. Vertical Drop-Line (Altitude Indicator)
        if p_ball[1] < p_floor[1] - 4:
            # Draw subtle dashed/dotted altitude guide line
            cv2.line(frame, p_ball, p_floor, (120, 100, 160), 1, cv2.LINE_AA)

        # 2. Dynamic 3D Floor Shadow
        # Scale shadow radius and opacity with altitude
        z_scale = 1.0 / max(0.25, 1.0 + bz * self.focal_depth)
        base_r = ball.radius * width * z_scale
        shadow_rx = max(4, round(base_r * (0.8 + 0.8 * alt_ratio)))
        shadow_ry = max(2, round(shadow_rx * 0.42))  # Perspective flattened ellipse
        shadow_alpha = max(0.12, 0.70 * (1.0 - alt_ratio * 0.75))

        overlay = frame.copy()
        cv2.ellipse(overlay, p_floor, (shadow_rx, shadow_ry), 0, 0, 360, (10, 6, 15), -1)
        cv2.addWeighted(overlay, shadow_alpha, frame, 1.0 - shadow_alpha, 0, frame)

        # Subtle neon concentric target ring at floor point
        ring_r = max(6, round(base_r * 0.9))
        cv2.ellipse(frame, p_floor, (ring_r, round(ring_r * 0.42)), 0, 0, 360, (160, 110, 220), 1, cv2.LINE_AA)

    def _render_pip_webcam(
        self,
        frame: np.ndarray,
        raw_webcam: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """Render mini Picture-in-Picture webcam feed in the top-right corner."""
        if cv2 is None or raw_webcam is None or raw_webcam.size == 0:
            return

        pip_w = max(80, min(width // 2, round(width * self.pip_scale)))
        pip_h = max(60, min(height // 2, round(pip_w * (raw_webcam.shape[0] / max(1, raw_webcam.shape[1])))))

        margin = 12
        x2 = width - margin
        x1 = x2 - pip_w
        y1 = margin + 20  # below telemetry line
        y2 = y1 + pip_h

        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return

        resized_cam = cv2.resize(raw_webcam, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = resized_cam

        # Cyber Border & Title Banner
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1 - 14), (x1 + 64, y1), (18, 18, 18), -1)
        cv2.rectangle(frame, (x1, y1 - 14), (x1 + 64, y1), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "CAM FEED", (x1 + 4, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 255), 1, cv2.LINE_AA)
