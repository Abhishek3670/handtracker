"""Monocular hand depth estimation from palm scale geometry."""
from __future__ import annotations
import math
from typing import Any, Sequence

from .models import HandLandmarks, Landmark3D


def estimate_hand_depth(
    hand: HandLandmarks | Sequence[Any],
    ref_span: float = 0.18,
    gain: float = 0.85,
    min_depth: float = -0.55,
    max_depth: float = 0.55,
) -> float:
    """
    Estimate normalized 3D camera distance / room depth offset Z_hand from palm scale.

    Measures normalized 2D span between Wrist (P0) and Middle MCP (P9):
    L_palm = ||P9 - P0|| = sqrt((x9 - x0)^2 + (y9 - y0)^2).

    Mapped against calibrated baseline ref_span (default 0.18) with scaling gain (default 0.85):
    Z_hand = clamp(((ref_span / L_palm) - 1.0) * gain, min_depth, max_depth).

    - Closer hand (large on screen, L_palm > ref_span) -> Z_hand < 0 (front of room / near).
    - Farther hand (small on screen, L_palm < ref_span) -> Z_hand > 0 (deep in room / far).
    """
    if isinstance(hand, HandLandmarks):
        landmarks = hand.landmarks
    elif hasattr(hand, "landmarks"):
        landmarks = hand.landmarks
    else:
        landmarks = hand

    if landmarks is None or len(landmarks) < 10:
        return 0.0

    p0 = landmarks[0]
    p9 = landmarks[9]

    p0_x = getattr(p0, "x", p0[0] if isinstance(p0, (list, tuple)) else 0.0)
    p0_y = getattr(p0, "y", p0[1] if isinstance(p0, (list, tuple)) else 0.0)
    p9_x = getattr(p9, "x", p9[0] if isinstance(p9, (list, tuple)) else 0.0)
    p9_y = getattr(p9, "y", p9[1] if isinstance(p9, (list, tuple)) else 0.0)

    dx = float(p9_x) - float(p0_x)
    dy = float(p9_y) - float(p0_y)
    palm_span = math.hypot(dx, dy)

    if palm_span < 1e-4:
        return 0.0

    raw_z = ((ref_span / palm_span) - 1.0) * gain
    return max(min_depth, min(max_depth, float(raw_z)))
