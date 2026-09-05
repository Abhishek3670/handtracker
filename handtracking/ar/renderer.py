"""Photorealistic 3D AR Ball Renderer with Blinn-Phong lighting, skins, palm shadows, and contact ripples."""
from __future__ import annotations
from enum import Enum
import math
import time
from typing import Any, Iterable, Sequence

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

from ..inference.models import HandLandmarks
from .colliders import PalmCollider, _dot, _norm, _normalize, _sub
from .physics import ARPhysicsEngine, BallState, ImpactRipple


class BallSkin(str, Enum):
    BASKETBALL = "basketball"
    CHROME = "chrome"
    TENNIS = "tennis"
    NEON = "neon"


SKIN_CYCLE = [
    BallSkin.BASKETBALL,
    BallSkin.CHROME,
    BallSkin.TENNIS,
    BallSkin.NEON,
]


class BallRenderer:
    """Renders 3D shaded ball with Blinn-Phong lighting, materials, dynamic palm shadows, and ripples."""

    def __init__(self, skin: BallSkin = BallSkin.BASKETBALL, light_dir: tuple[float, float, float] = (-0.5, -0.7, -0.5)):
        self.skin = skin
        self.light_dir = _normalize(light_dir)
        self.view_dir = (0.0, 0.0, -1.0)
        # Halfway vector for Blinn-Phong
        self.half_vector = _normalize((-self.light_dir[0] + self.view_dir[0], -self.light_dir[1] + self.view_dir[1], -self.light_dir[2] + self.view_dir[2]))
        self._sphere_cache: dict[tuple[int, str], np.ndarray] = {}

    def cycle_skin(self) -> BallSkin:
        """Cycle to next material skin."""
        idx = SKIN_CYCLE.index(self.skin)
        self.skin = SKIN_CYCLE[(idx + 1) % len(SKIN_CYCLE)]
        return self.skin

    def set_skin(self, skin: BallSkin | str) -> None:
        if isinstance(skin, BallSkin):
            self.skin = skin
        else:
            self.skin = BallSkin(str(getattr(skin, "value", skin)).lower())

    def draw(
        self,
        frame: np.ndarray,
        engine: ARPhysicsEngine,
        hands: Iterable[HandLandmarks] = (),
        timestamp: float | None = None,
        virtual_room: bool = False,
        projection_fn: Any = None,
        focal_depth: float = 0.85,
    ) -> np.ndarray:
        """Draw complete AR physics scene: shadows, ball, ripples, and HUD in-place on frame."""
        ts = time.time() if timestamp is None else float(timestamp)
        h, w = frame.shape[:2]
        ball = engine.ball

        # 1. Draw dynamic palm drop shadows (suppressed in virtual 3D room to avoid dual shadows)
        if not virtual_room:
            self._draw_palm_shadows(frame, ball, hands, w, h)

        # 2. Draw impact ripples
        self._draw_ripples(frame, engine.ripples, w, h)

        # 3. Draw 3D shaded ball
        self._draw_ball(
            frame,
            ball,
            w,
            h,
            ts,
            virtual_room=virtual_room,
            projection_fn=projection_fn,
            focal_depth=focal_depth,
        )
        return frame

    render = draw

    def _draw_palm_shadows(
        self,
        frame: np.ndarray,
        ball: BallState,
        hands: Iterable[HandLandmarks],
        width: int,
        height: int,
    ) -> None:
        for hand in hands:
            palm = PalmCollider.from_hand(hand)
            dist = palm.distance_to_plane(ball.position)
            # Only cast shadow if ball is above palm surface within reasonable range
            if 0 < dist < 0.35:
                proj = palm.closest_point_on_plane(ball.position)
                px = round(proj[0] * (width - 1))
                py = round(proj[1] * (height - 1))

                # Shadow shrinks and darkens as ball gets closer
                dist_factor = dist / 0.35
                shadow_r = max(6, round(ball.radius * width * (0.6 + 0.6 * dist_factor)))
                shadow_alpha = max(0.15, 0.65 * (1.0 - dist_factor))

                if cv2 is not None:
                    overlay = frame.copy()
                    cv2.ellipse(overlay, (px, py), (shadow_r, round(shadow_r * 0.6)), 0, 0, 360, (15, 15, 15), -1)
                    cv2.addWeighted(overlay, shadow_alpha, frame, 1.0 - shadow_alpha, 0, frame)
                else:
                    y_min = max(0, py - round(shadow_r * 0.6))
                    y_max = min(height, py + round(shadow_r * 0.6))
                    x_min = max(0, px - shadow_r)
                    x_max = min(width, px + shadow_r)
                    if y_max > y_min and x_max > x_min:
                        frame[y_min:y_max, x_min:x_max] = (
                            frame[y_min:y_max, x_min:x_max] * (1.0 - shadow_alpha) + np.array([15, 15, 15]) * shadow_alpha
                        ).astype(np.uint8)

    def _draw_ripples(
        self,
        frame: np.ndarray,
        ripples: Sequence[ImpactRipple],
        width: int,
        height: int,
    ) -> None:
        if cv2 is None or not ripples:
            return

        for r in ripples:
            if r.alpha <= 0.01:
                continue
            cx = round(r.center[0] * (width - 1))
            cy = round(r.center[1] * (height - 1))
            pix_r = max(3, round(r.radius * width))

            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), pix_r, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, r.alpha * 0.75, frame, 1.0 - (r.alpha * 0.75), 0, frame)

    def _generate_shaded_sphere(self, radius_px: int, skin: BallSkin) -> np.ndarray:
        """Render Blinn-Phong shaded 3D sphere sprite with alpha channel."""
        cache_key = (radius_px, skin.value)
        if cache_key in self._sphere_cache:
            return self._sphere_cache[cache_key]

        size = radius_px * 2 + 1
        sprite = np.zeros((size, size, 4), dtype=np.uint8)
        c = radius_px

        y_indices, x_indices = np.ogrid[:size, :size]
        dx = (x_indices - c) / float(radius_px)
        dy = (y_indices - c) / float(radius_px)
        r2 = dx * dx + dy * dy
        mask = r2 <= 1.0

        if not np.any(mask):
            return sprite

        nz = -np.sqrt(np.maximum(0.0, 1.0 - r2))
        nx = dx
        ny = dy

        # Blinn-Phong Dot products
        lx, ly, lz = self.light_dir
        hx, hy, hz = self.half_vector

        # Diffuse: N dot L
        ndotl = np.maximum(0.0, -(nx * lx + ny * ly + nz * lz))
        # Specular: N dot H
        ndoth = np.maximum(0.0, -(nx * hx + ny * hy + nz * hz))

        if skin == BallSkin.BASKETBALL:
            base_b, base_g, base_r = 30, 100, 230
            ambient = 0.22
            diffuse_weight = 0.70
            spec_weight = 0.25
            shininess = 16

            # Basketball black seam curves
            seams = (np.abs(dx) < 0.06) | (np.abs(dy) < 0.06) | (np.abs(np.abs(dx) - 0.55) < 0.04)
            diffuse = ambient + diffuse_weight * ndotl
            specular = spec_weight * (ndoth ** shininess)

            b = np.where(seams, 20, np.clip(base_b * diffuse + 255 * specular, 0, 255))
            g = np.where(seams, 20, np.clip(base_g * diffuse + 255 * specular, 0, 255))
            r = np.where(seams, 20, np.clip(base_r * diffuse + 255 * specular, 0, 255))

        elif skin == BallSkin.CHROME:
            # Metallic mirror reflection
            ambient = 0.15
            spec_weight = 0.85
            shininess = 48
            horizon = np.sin(ny * 4.0) * 0.5 + 0.5
            diffuse = ambient + 0.4 * ndotl + 0.3 * horizon
            specular = spec_weight * (ndoth ** shininess)

            b = np.clip(210 * diffuse + 255 * specular, 0, 255)
            g = np.clip(220 * diffuse + 255 * specular, 0, 255)
            r = np.clip(235 * diffuse + 255 * specular, 0, 255)

        elif skin == BallSkin.TENNIS:
            base_b, base_g, base_r = 45, 230, 205
            ambient = 0.30
            diffuse_weight = 0.65
            spec_weight = 0.15
            shininess = 8

            # Tennis curved seam
            seam = (np.abs(dx * dx - dy * 0.7) < 0.05) | (np.abs(dy * dy - dx * 0.7) < 0.05)
            diffuse = ambient + diffuse_weight * ndotl
            specular = spec_weight * (ndoth ** shininess)

            b = np.where(seam, 240, np.clip(base_b * diffuse + 255 * specular, 0, 255))
            g = np.where(seam, 240, np.clip(base_g * diffuse + 255 * specular, 0, 255))
            r = np.where(seam, 240, np.clip(base_r * diffuse + 255 * specular, 0, 255))

        else:  # NEON
            base_b, base_g, base_r = 255, 220, 0  # Cyan electric core
            ambient = 0.65
            diffuse_weight = 0.35
            spec_weight = 0.90
            shininess = 32

            rim = (1.0 - np.abs(nz)) ** 2
            diffuse = ambient + diffuse_weight * ndotl + 0.4 * rim
            specular = spec_weight * (ndoth ** shininess)

            b = np.clip(base_b * diffuse + 255 * specular, 0, 255)
            g = np.clip(base_g * diffuse + 255 * specular, 0, 255)
            r = np.clip(base_r * diffuse + 255 * specular, 0, 255)

        sprite[:, :, 0] = np.where(mask, b.astype(np.uint8), 0)
        sprite[:, :, 1] = np.where(mask, g.astype(np.uint8), 0)
        sprite[:, :, 2] = np.where(mask, r.astype(np.uint8), 0)
        sprite[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

        self._sphere_cache[cache_key] = sprite
        return sprite

    def _draw_ball(
        self,
        frame: np.ndarray,
        ball: BallState,
        width: int,
        height: int,
        ts: float,
        virtual_room: bool = False,
        projection_fn: Any = None,
        focal_depth: float = 0.85,
    ) -> None:
        if virtual_room and projection_fn is not None:
            cx, cy = projection_fn(ball.position[0], ball.position[1], ball.position[2], width, height)
            z_scale = 1.0 / max(0.25, 1.0 + ball.position[2] * focal_depth)
        else:
            cx = round(ball.position[0] * (width - 1))
            cy = round(ball.position[1] * (height - 1))
            # Perspective scaling by z-depth
            z_scale = 1.0 / max(0.4, 1.0 + ball.position[2] * 1.5)

        radius_px = max(8, round(ball.radius * width * z_scale))

        if cv2 is not None:
            sprite = self._generate_shaded_sphere(radius_px, self.skin)
            size = sprite.shape[0]
            x1 = cx - radius_px
            y1 = cy - radius_px
            x2 = x1 + size
            y2 = y1 + size

            # Source and destination bounding box clipping
            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            src_x2 = size - max(0, x2 - width)
            src_y2 = size - max(0, y2 - height)

            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(width, x2)
            dst_y2 = min(height, y2)

            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                sub_sprite = sprite[src_y1:src_y2, src_x1:src_x2]
                alpha = sub_sprite[:, :, 3:4] / 255.0
                frame_roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
                frame[dst_y1:dst_y2, dst_x1:dst_x2] = (
                    sub_sprite[:, :, :3] * alpha + frame_roi * (1.0 - alpha)
                ).astype(np.uint8)

            # Draw skin label indicator
            skin_label = f"Skin: {self.skin.value.title()}"
            cv2.putText(frame, skin_label, (15, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            # Fallback numpy circle
            y_min = max(0, cy - radius_px)
            y_max = min(height, cy + radius_px)
            x_min = max(0, cx - radius_px)
            x_max = min(width, cx + radius_px)
            frame[y_min:y_max, x_min:x_max] = (0, 165, 255)
