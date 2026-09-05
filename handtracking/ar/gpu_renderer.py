"""Hardware-accelerated ModernGL GPU Shader Engine for 3D Cyber Room, Mesh Shading & Holographic Hands."""
from __future__ import annotations
import math
import time
from typing import Any, Iterable, Sequence

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

try:
    import moderngl
except ImportError:
    moderngl = None

from ..inference.depth import estimate_hand_depth
from ..inference.models import HAND_CONNECTIONS, HandLandmarks, HandednessLabel
from .physics import ARPhysicsEngine, BallState
from .renderer import BallRenderer, BallSkin
from .room import Virtual3DRoomRenderer


def build_uv_sphere_mesh(rings: int = 24, sectors: int = 32, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate tessellated 3D UV sphere geometry (positions, normals, UVs, and triangle indices).

    Returns:
        vertices: np.ndarray of shape (num_verts, 8) with [x, y, z, nx, ny, nz, u, v] as float32.
        indices: np.ndarray of shape (num_indices,) as uint32.
    """
    verts = []
    for r in range(rings + 1):
        phi = math.pi * r / rings  # 0 to pi
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors  # 0 to 2pi
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            nx = sin_phi * cos_theta
            ny = cos_phi
            nz = sin_phi * sin_theta

            x = radius * nx
            y = radius * ny
            z = radius * nz

            u = float(s) / float(sectors)
            v = float(r) / float(rings)

            verts.append([x, y, z, nx, ny, nz, u, v])

    verts_arr = np.array(verts, dtype=np.float32)

    indices = []
    for r in range(rings):
        for s in range(sectors):
            cur = r * (sectors + 1) + s
            nxt = cur + (sectors + 1)

            indices.extend([cur, nxt, cur + 1])
            indices.extend([cur + 1, nxt, nxt + 1])

    indices_arr = np.array(indices, dtype=np.uint32)
    return verts_arr, indices_arr


def make_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Construct 3x3 rotation matrix from pitch (rx), yaw (ry), and roll (rz)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx,  cx],
    ], dtype=np.float32)

    rot_y = np.array([
        [ cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float32)

    rot_z = np.array([
        [cz, -sz, 0.0],
        [sz,  cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return np.dot(rot_z, np.dot(rot_y, rot_x))


def make_ortho_or_perspective_matrix(focal_depth: float = 0.85) -> np.ndarray:
    """
    Construct 4x4 perspective MVP matrix aligning 3D room coordinates with camera vanishing point.

    World mapping:
        xw = (x - 0.5) * 2.0
        yw = -(y - 0.5) * 2.0
        zw = z
        scale = 1.0 / (1.0 + z * focal_depth) = d / (z + d), where d = 1.0 / focal_depth
    """
    f = float(focal_depth)
    d = 1.0 / max(0.1, f)

    # In row-major form before column-major transpose:
    # X_clip = d * xw
    # Y_clip = d * yw
    # Z_clip = zw
    # W_clip = zw + d
    mat = np.array([
        [d,   0.0, 0.0, 0.0],
        [0.0, d,   0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, d],
    ], dtype=np.float32)
    return mat


SPHERE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_mat;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec3 v_local_pos;
out vec2 v_uv;

void main() {
    vec4 wp = u_model * vec4(in_pos, 1.0);
    v_world_pos = wp.xyz;
    v_world_normal = normalize(u_normal_mat * in_normal);
    v_local_pos = in_pos;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

SPHERE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_world_normal;
in vec3 v_local_pos;
in vec2 v_uv;

uniform vec3 u_view_pos;
uniform int u_skin;
uniform float u_time;
uniform vec4 u_base_color;
uniform int u_is_hand_joint;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(u_view_pos - v_world_pos);

    // ─── 3-Point Studio Lighting Setup ───────────────────────
    // 1. Key Light: Warm bright studio key light (top-right-front)
    vec3 key_pos = vec3(0.40, 0.60, -2.2);
    vec3 L_key = normalize(key_pos - v_world_pos);
    vec3 H_key = normalize(L_key + V);
    float NdotL_key = max(dot(N, L_key), 0.0);
    vec3 key_color = vec3(1.0, 0.97, 0.92) * 1.45;

    // 2. Fill Light: Soft cool cyan-blue fill light (front-left)
    vec3 fill_pos = vec3(-0.70, 0.30, -1.8);
    vec3 L_fill = normalize(fill_pos - v_world_pos);
    vec3 H_fill = normalize(L_fill + V);
    float NdotL_fill = max(dot(N, L_fill), 0.0);
    vec3 fill_color = vec3(0.75, 0.88, 1.0) * 0.85;

    // 3. Rim / Ground Bounce Light: Soft purple/violet floor bounce (bottom-back)
    vec3 rim_pos = vec3(0.0, -0.85, 0.6);
    vec3 L_rim = normalize(rim_pos - v_world_pos);
    float NdotL_rim = max(dot(N, L_rim), 0.0);
    vec3 rim_color = vec3(0.85, 0.60, 1.0) * 0.55;

    // 4. Hemispheric Sky/Ground Ambient
    float hemi = N.y * 0.5 + 0.5;
    vec3 ambient_light = mix(vec3(0.32, 0.26, 0.40), vec3(0.60, 0.55, 0.68), hemi);

    float NdotV = max(dot(N, V), 0.0);
    float fresnel = pow(1.0 - NdotV, 3.0);

    // ─── Holographic Hand Joint Avatar ───────────────────────
    if (u_is_hand_joint == 1) {
        float diff = NdotL_key * 0.7 + NdotL_fill * 0.3;
        float spec = pow(max(dot(N, H_key), 0.0), 24.0) * 0.8;
        vec3 col = u_base_color.rgb * (0.65 + 0.35 * diff) + vec3(1.0) * spec + u_base_color.rgb * fresnel * 1.5;
        fragColor = vec4(col, u_base_color.a);
        return;
    }

    vec3 albedo = vec3(0.98, 0.42, 0.08);
    float shininess = 32.0;
    float spec_strength = 0.50;

    vec3 p = normalize(v_local_pos);

    // ─── Material Skins ──────────────────────────────────────
    if (u_skin == 0) { // BASKETBALL (Wilson Evolution / NBA Pro Leather)
        // 1. Precise 8-panel basketball seams in local 3D coordinates
        float d_equator = abs(p.y);
        float d_meridian1 = abs(p.x);
        float d_rib1 = abs(abs(p.x) - 0.707 * sqrt(max(0.001, 1.0 - p.y * p.y)));
        float d_rib2 = abs(abs(p.z) - 0.707 * sqrt(max(0.001, 1.0 - p.y * p.y)));

        float d_seam = min(d_equator, min(d_meridian1, min(d_rib1, d_rib2)));

        // Seam zones:
        // Core groove: black rubber channel (width ~0.024)
        float seam_core = 1.0 - smoothstep(0.016, 0.030, d_seam);
        // Groove shadow falloff: ambient darkening inside channel
        float seam_shadow = 1.0 - smoothstep(0.025, 0.055, d_seam);
        // Leather bevel highlight alongside groove
        float bevel_factor = sin(clamp((d_seam - 0.022) / 0.038, 0.0, 1.0) * 3.14159);

        // 2. High-Frequency Micro-Pebble Leather Bump Mapping
        float p1 = sin(p.x * 220.0) * sin(p.y * 220.0) * sin(p.z * 220.0);
        float p2 = sin(p.x * 440.0 + 1.2) * sin(p.y * 440.0 + 1.2) * sin(p.z * 440.0 + 1.2);
        float pebble = p1 * 0.7 + p2 * 0.3;
        float grain = 0.94 + 0.14 * pebble;

        // Normal perturbation for true tactile leather bump glints
        vec3 bump = vec3(
            cos(p.x * 220.0) * sin(p.y * 220.0),
            cos(p.y * 220.0) * sin(p.z * 220.0),
            cos(p.z * 220.0) * sin(p.x * 220.0)
        );
        N = normalize(N + bump * 0.10 * (1.0 - seam_core));
        // Recompute lighting with perturbed normal
        NdotL_key = max(dot(N, L_key), 0.0);
        NdotL_fill = max(dot(N, L_fill), 0.0);
        H_key = normalize(L_key + V);
        H_fill = normalize(L_fill + V);

        // 3. Premium Wilson Terracotta Leather Color Palette
        vec3 leather_base = vec3(0.98, 0.42, 0.08) * grain;
        // Panel center gradient (warmer, brighter towards center of panels)
        vec3 panel_warmth = vec3(1.0, 0.50, 0.14) * (1.0 - seam_shadow * 0.25);
        leather_base = mix(leather_base, panel_warmth, 0.35 + 0.20 * bevel_factor);

        vec3 rubber_black = vec3(0.05, 0.04, 0.04);
        albedo = mix(leather_base, rubber_black, seam_core);

        // Groove normal dip for embossed 3D channel depth
        if (seam_shadow > 0.01) {
            N = normalize(N - seam_shadow * 0.25 * N);
        }

        spec_strength = mix(0.45, 0.06, seam_core);
        shininess = mix(28.0, 6.0, seam_core);

        // Golden leather Fresnel grazing sheen
        albedo += vec3(1.0, 0.65, 0.20) * pow(1.0 - NdotV, 3.0) * 0.45 * (1.0 - seam_core);

    } else if (u_skin == 1) { // CHROME (Liquid Mirror Metallics)
        shininess = 256.0;
        spec_strength = 2.2;
        vec3 refl = reflect(-V, N);

        // Simulated cyber room grid reflection
        float grid_rx = abs(fract(refl.x * 4.0) - 0.5);
        float grid_ry = abs(fract(refl.y * 4.0) - 0.5);
        float grid_lines = smoothstep(0.05, 0.02, min(grid_rx, grid_ry));

        vec3 sky_refl = mix(vec3(0.25, 0.35, 0.55), vec3(0.95, 0.98, 1.0), refl.y * 0.5 + 0.5);
        sky_refl += vec3(0.0, 0.9, 1.0) * grid_lines * 0.5;

        // Chromatic dispersion (RGB split on grazing angles)
        float f_r = pow(1.0 - NdotV, 2.5);
        float f_g = pow(1.0 - NdotV, 3.0);
        float f_b = pow(1.0 - NdotV, 3.5);
        vec3 dispersion = vec3(f_r, f_g, f_b) * vec3(0.3, 0.6, 1.0);

        albedo = sky_refl + dispersion;

    } else if (u_skin == 2) { // TENNIS (US Open Felt)
        // Fuzzy micro-fiber velvet procedural texture
        float felt_noise = sin(p.x * 180.0) * sin(p.y * 180.0) * sin(p.z * 180.0);
        float felt_grain = 0.94 + 0.12 * felt_noise;

        // Curved 3D tennis ball seam
        float tennis_dist = abs(p.x * p.y - 0.38 * p.z);
        float t_seam = 1.0 - smoothstep(0.022, 0.050, tennis_dist);

        vec3 felt_color = vec3(0.82, 0.94, 0.14) * felt_grain;
        vec3 seam_white = vec3(0.94, 0.95, 0.92);

        albedo = mix(felt_color, seam_white, t_seam);
        spec_strength = mix(0.18, 0.35, t_seam);
        shininess = mix(10.0, 20.0, t_seam);

        // Inverted Fresnel velvet sheen (fuzzy rim glow)
        albedo += felt_color * (1.0 - NdotV) * 0.45;

    } else if (u_skin == 3) { // NEON CYBER CORE (Holographic Plasma)
        vec3 cyan = vec3(0.0, 0.96, 1.0);
        vec3 magenta = vec3(1.0, 0.15, 0.85);
        vec3 gold = vec3(1.0, 0.85, 0.10);

        float pulse = 0.5 + 0.5 * sin(u_time * 4.0);
        float plasma_wave = sin(p.x * 16.0 + u_time * 3.0) * sin(p.y * 16.0 - u_time * 2.0) * sin(p.z * 16.0);
        float shield_grid = smoothstep(0.15, 0.35, abs(plasma_wave));

        vec3 core_color = mix(cyan, magenta, pulse * 0.6 + (p.y * 0.5 + 0.5) * 0.4);
        albedo = mix(core_color * 0.45, core_color, shield_grid);

        spec_strength = 1.6;
        shininess = 96.0;

        // Multi-color Fresnel energy halo (Cyan -> Magenta -> Gold)
        vec3 rim_halo = mix(magenta, gold, pulse) * fresnel * 2.8;
        albedo += rim_halo;
    }

    // ─── Dual Specular Highlights (Key + Fill) ────────────────
    float spec_key = pow(max(dot(N, H_key), 0.0), shininess) * spec_strength;
    float spec_fill = pow(max(dot(N, H_fill), 0.0), shininess * 0.75) * (spec_strength * 0.35);

    // ─── Final Combined Radiance ─────────────────────────────
    vec3 diffuse = (NdotL_key * key_color + NdotL_fill * fill_color + NdotL_rim * rim_color);
    vec3 specular = (spec_key * key_color + spec_fill * fill_color);
    vec3 color = (ambient_light + diffuse) * albedo + specular;

    fragColor = vec4(color, 1.0);
}
"""

LINE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec4 in_color;

uniform mat4 u_mvp;
out vec4 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330 core
in vec4 v_color;
uniform vec4 u_tint;
out vec4 fragColor;

void main() {
    fragColor = v_color * u_tint;
}
"""

BG_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec4 in_color;
out vec4 v_color;

void main() {
    v_color = in_color;
    gl_Position = vec4(in_pos, 0.999, 1.0);
}
"""

BG_FRAGMENT_SHADER = """
#version 330 core
in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""


class GPURoomRenderer:
    """Hardware-accelerated ModernGL GPU Shader Engine for 3D Cyber Room, Mesh Shading & Holographic Hands."""

    def __init__(
        self,
        show_pip: bool = True,
        pip_scale: float = 0.20,
        focal_depth: float = 0.85,
        bounds_min: tuple[float, float, float] = (0.05, 0.05, -0.6),
        bounds_max: tuple[float, float, float] = (0.95, 0.95, 0.6),
        prefer_gpu: bool = True,
    ):
        self.show_pip = show_pip
        self.pip_scale = pip_scale
        self.focal_depth = focal_depth
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.wall_glow_time: float = 0.0
        self.wall_glow_color: tuple[int, int, int] = (0, 220, 255)

        self.prefer_gpu = prefer_gpu
        self.is_gpu_available: bool = False
        self.ctx: Any = None
        self.fbo: Any = None
        self.fbo_color: Any = None
        self.fbo_depth: Any = None
        self.fbo_size: tuple[int, int] = (0, 0)

        # CPU Fallback renderers
        self.cpu_room_renderer = Virtual3DRoomRenderer(
            show_pip=show_pip,
            pip_scale=pip_scale,
            focal_depth=focal_depth,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        self.cpu_ball_renderer = BallRenderer()

        if self.prefer_gpu and moderngl is not None:
            self._init_gpu_context()

    def _init_gpu_context(self) -> None:
        """Initialize ModernGL standalone context, shaders, and geometry buffers."""
        try:
            self.ctx = moderngl.create_context(standalone=True)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)

            # 1. Compile Shaders
            self.sphere_prog = self.ctx.program(
                vertex_shader=SPHERE_VERTEX_SHADER,
                fragment_shader=SPHERE_FRAGMENT_SHADER,
            )
            self.line_prog = self.ctx.program(
                vertex_shader=LINE_VERTEX_SHADER,
                fragment_shader=LINE_FRAGMENT_SHADER,
            )
            self.bg_prog = self.ctx.program(
                vertex_shader=BG_VERTEX_SHADER,
                fragment_shader=BG_FRAGMENT_SHADER,
            )

            # 2. Build UV Sphere Geometry
            sphere_v, sphere_i = build_uv_sphere_mesh(rings=24, sectors=32, radius=1.0)
            self.sphere_vbo = self.ctx.buffer(sphere_v.tobytes())
            self.sphere_ibo = self.ctx.buffer(sphere_i.tobytes())
            self.sphere_vao = self.ctx.vertex_array(
                self.sphere_prog,
                [(self.sphere_vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")],
                index_buffer=self.sphere_ibo,
            )
            self.sphere_index_count = len(sphere_i)

            # 3. Background Quad Geometry (Top: slate navy (20, 14, 28) -> Bottom: dark cosmic purple (42, 26, 56))
            # Normalized RGB colors
            c_top = [28.0 / 255.0, 14.0 / 255.0, 20.0 / 255.0, 1.0]      # R, G, B, A
            c_bot = [56.0 / 255.0, 26.0 / 255.0, 42.0 / 255.0, 1.0]
            bg_verts = np.array([
                # x, y, r, g, b, a
                [-1.0, -1.0, *c_bot],
                [ 1.0, -1.0, *c_bot],
                [-1.0,  1.0, *c_top],
                [-1.0,  1.0, *c_top],
                [ 1.0, -1.0, *c_bot],
                [ 1.0,  1.0, *c_top],
            ], dtype=np.float32)
            self.bg_vbo = self.ctx.buffer(bg_verts.tobytes())
            self.bg_vao = self.ctx.vertex_array(
                self.bg_prog,
                [(self.bg_vbo, "2f 4f", "in_pos", "in_color")],
            )

            self.is_gpu_available = True
        except Exception:
            self.is_gpu_available = False
            self.ctx = None

    def _ensure_fbo(self, width: int, height: int) -> None:
        """Allocate or resize offscreen Framebuffer Object."""
        if self.fbo_size == (width, height) and self.fbo is not None:
            return

        if self.fbo is not None:
            self.fbo.release()
            self.fbo_color.release()
            self.fbo_depth.release()

        self.fbo_color = self.ctx.texture((width, height), 4)
        self.fbo_depth = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.fbo_color],
            depth_attachment=self.fbo_depth,
        )
        self.fbo_size = (width, height)

    def project_3d(self, x: float, y: float, z: float, width: int, height: int) -> tuple[int, int]:
        """Project normalized 3D coordinates (x, y, z) to screen pixels using perspective frustum."""
        scale = 1.0 / max(0.25, 1.0 + z * self.focal_depth)
        u = round((0.5 + (x - 0.5) * scale) * (width - 1))
        v = round((0.5 + (y - 0.5) * scale) * (height - 1))
        return (max(-width, min(2 * width, u)), max(-height, min(2 * height, v)))

    def trigger_wall_pulse(self, timestamp: float | None = None, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        self.wall_glow_time = time.time() if timestamp is None else float(timestamp)
        self.wall_glow_color = color
        self.cpu_room_renderer.trigger_wall_pulse(timestamp, color)

    def _to_world_coords(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Convert normalized (0..1, 0..1, -0.6..0.6) to centered world coords (-1..1, -1..1, -0.6..0.6)."""
        return ((x - 0.5) * 2.0, -(y - 0.5) * 2.0, z)

    def _draw_lines(self, lines: list[tuple[Sequence[float], Sequence[float], Sequence[float]]], mvp_mat: np.ndarray, tint=(1.0, 1.0, 1.0, 1.0)) -> None:
        """Render batch of 3D colored line segments."""
        if not lines:
            return

        verts = []
        for p1, p2, col in lines:
            w1 = self._to_world_coords(p1[0], p1[1], p1[2])
            w2 = self._to_world_coords(p2[0], p2[1], p2[2])
            rgba = [col[0], col[1], col[2], col[3] if len(col) > 3 else 1.0]
            verts.extend([w1[0], w1[1], w1[2], *rgba])
            verts.extend([w2[0], w2[1], w2[2], *rgba])

        line_arr = np.array(verts, dtype=np.float32)
        vbo = self.ctx.buffer(line_arr.tobytes())
        vao = self.ctx.vertex_array(
            self.line_prog,
            [(vbo, "3f 4f", "in_pos", "in_color")],
        )

        self.line_prog["u_mvp"].write(mvp_mat.T.copy().tobytes())
        self.line_prog["u_tint"].value = tint
        vao.render(moderngl.LINES)
        vbo.release()
        vao.release()

    def _draw_sphere(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        mvp_mat: np.ndarray,
        skin: int = 0,
        base_color=(1.0, 1.0, 1.0, 1.0),
        is_hand_joint: bool = False,
        timestamp: float = 0.0,
        aspect: float = 1.0,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Render 3D sphere at world position with model transform, 3D rotation, aspect compensation, and GLSL lighting."""
        wx, wy, wz = self._to_world_coords(x, y, z)
        # Isotropic aspect-corrected scaling:
        # In NDC space, X spans [-1, 1] across width W, Y spans [-1, 1] across height H.
        # To maintain 1:1 pixel aspect ratio on non-square viewports (aspect = width / height),
        # the vertical radius in NDC must be scaled by aspect so that DeltaX_px == DeltaY_px.
        rx = radius * 2.0
        ry = radius * 2.0 * aspect
        rz = radius * 2.0

        # Model matrix: Translate * AspectScale * 3D Rotation
        rot_mat = make_rotation_matrix(rotation[0], rotation[1], rotation[2])
        col0 = rot_mat[:, 0] * rx
        col1 = rot_mat[:, 1] * ry
        col2 = rot_mat[:, 2] * rz

        model_mat = np.array([
            [col0[0], col1[0], col2[0], wx],
            [col0[1], col1[1], col2[1], wy],
            [col0[2], col1[2], col2[2], wz],
            [0.0,     0.0,     0.0,     1.0],
        ], dtype=np.float32)

        # Normal matrix (inv transpose 3x3)
        scale_diag = np.diag([1.0 / max(1e-5, rx), 1.0 / max(1e-5, ry), 1.0 / max(1e-5, rz)])
        normal_mat = np.dot(rot_mat, scale_diag).astype(np.float32)

        final_mvp = np.dot(mvp_mat, model_mat)

        # Transpose row-major NumPy matrices to OpenGL column-major byte buffer
        self.sphere_prog["u_mvp"].write(final_mvp.T.copy().tobytes())
        self.sphere_prog["u_model"].write(model_mat.T.copy().tobytes())
        self.sphere_prog["u_normal_mat"].write(normal_mat.T.copy().tobytes())
        self.sphere_prog["u_view_pos"].value = (0.0, 0.0, -1.0 / self.focal_depth)
        self.sphere_prog["u_skin"].value = int(skin)
        self.sphere_prog["u_time"].value = float(timestamp)
        self.sphere_prog["u_base_color"].value = tuple(base_color)
        self.sphere_prog["u_is_hand_joint"].value = 1 if is_hand_joint else 0

        self.sphere_vao.render(moderngl.TRIANGLES)

    def render_room(
        self,
        frame: np.ndarray,
        engine: ARPhysicsEngine,
        hands: Iterable[HandLandmarks] = (),
        raw_webcam: np.ndarray | None = None,
        timestamp: float | None = None,
        skin: BallSkin = BallSkin.BASKETBALL,
    ) -> np.ndarray:
        """Render full 3D room, holographic hands, and shaded ball via GPU shader (or CPU fallback)."""
        ts = time.time() if timestamp is None else float(timestamp)
        h, w = frame.shape[:2]

        if not self.is_gpu_available or self.ctx is None:
            # Safe CPU Fallback
            self.cpu_room_renderer.render_room(frame, engine, hands, raw_webcam=raw_webcam, timestamp=ts)
            self.cpu_ball_renderer.set_skin(skin)
            self.cpu_ball_renderer.draw(
                frame,
                engine,
                hands=hands,
                timestamp=ts,
                virtual_room=True,
                projection_fn=self.cpu_room_renderer.project_3d,
                focal_depth=self.focal_depth,
            )
            return frame

        # Sync wall impact glow pulse
        if getattr(engine, "last_wall_impact_time", None) is not None:
            time_since_impact = ts - engine.last_wall_impact_time
            if 0.0 <= time_since_impact <= 0.35 and (ts - self.wall_glow_time) > 0.35:
                self.trigger_wall_pulse(timestamp=engine.last_wall_impact_time, color=(0, 255, 255))

        # Viewport aspect ratio for isotropic 1:1 circular geometry on non-square screens
        aspect = float(w) / float(max(1, h))

        # 1. Setup Offscreen FBO & Viewport
        self._ensure_fbo(w, h)
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        # 2. Render Cosmic Slate Gradient Background Quad
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.bg_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)

        # 3. Compute Perspective MVP Matrix
        mvp_mat = make_ortho_or_perspective_matrix(self.focal_depth)

        # 4. Generate 3D Room Grid & Wireframe Lines
        b_min_x, b_min_y, b_min_z = self.bounds_min
        b_max_x, b_max_y, b_max_z = self.bounds_max
        grid_lines = []

        grid_col = (75.0 / 255.0, 50.0 / 255.0, 95.0 / 255.0, 1.0)
        grid_accent = (140.0 / 255.0, 90.0 / 255.0, 180.0 / 255.0, 1.0)

        # A. Perspective Floor Grid (Y = b_max_y)
        floor_y = b_max_y
        num_x = 9
        for i in range(num_x):
            x = b_min_x + (b_max_x - b_min_x) * (i / (num_x - 1))
            col = grid_accent if i in (0, num_x - 1, num_x // 2) else grid_col
            grid_lines.append(((x, floor_y, b_min_z), (x, floor_y, b_max_z), col))

        num_z = 7
        for j in range(num_z):
            z = b_min_z + (b_max_z - b_min_z) * (j / (num_z - 1))
            grid_lines.append(((b_min_x, floor_y, z), (b_max_x, floor_y, z), grid_col))

        # B. Back Wall Grid (Z = b_max_z)
        elapsed_glow = ts - self.wall_glow_time
        glow_active = 0.0 <= elapsed_glow <= 0.35
        wall_col = (
            (self.wall_glow_color[0] / 255.0, self.wall_glow_color[1] / 255.0, self.wall_glow_color[2] / 255.0, 1.0)
            if glow_active else grid_col
        )
        for i in range(num_x):
            x = b_min_x + (b_max_x - b_min_x) * (i / (num_x - 1))
            grid_lines.append(((x, b_max_y, b_max_z), (x, b_min_y, b_max_z), wall_col))

        num_y = 6
        for k in range(num_y):
            y = b_min_y + (b_max_y - b_min_y) * (k / (num_y - 1))
            grid_lines.append(((b_min_x, y, b_max_z), (b_max_x, y, b_max_z), wall_col))

        # C. Bounding Depth Lines
        corners_front = [
            (b_min_x, b_min_y, b_min_z), (b_max_x, b_min_y, b_min_z),
            (b_max_x, b_max_y, b_min_z), (b_min_x, b_max_y, b_min_z),
        ]
        corners_back = [
            (b_min_x, b_min_y, b_max_z), (b_max_x, b_min_y, b_max_z),
            (b_max_x, b_max_y, b_max_z), (b_min_x, b_max_y, b_max_z),
        ]
        edge_col = wall_col if glow_active else (90.0 / 255.0, 60.0 / 255.0, 120.0 / 255.0, 1.0)
        for cf, cb in zip(corners_front, corners_back):
            grid_lines.append((cf, cb, edge_col))

        # Ceiling front edge
        grid_lines.append((corners_front[0], corners_front[1], (100.0 / 255.0, 70.0 / 255.0, 140.0 / 255.0, 1.0)))

        # D. Ball Spatial Grounding: Altitude Drop-Line & 3D Floor Target Ring
        bx, by, bz = engine.ball.position
        if by < floor_y - 0.01:
            grid_lines.append(((bx, by, bz), (bx, floor_y, bz), (120.0 / 255.0, 100.0 / 255.0, 160.0 / 255.0, 0.8)))

        # Concentric 3D Floor Target Ring underneath the ball
        ring_radius = max(0.02, engine.ball.radius * 0.85)
        num_ring_pts = 16
        ring_pts = []
        for k in range(num_ring_pts):
            theta = 2.0 * math.pi * k / num_ring_pts
            rx_k = bx + ring_radius * math.cos(theta)
            rz_k = bz + ring_radius * 0.65 * math.sin(theta)
            ring_pts.append((rx_k, floor_y, rz_k))

        for k in range(num_ring_pts):
            p_a = ring_pts[k]
            p_b = ring_pts[(k + 1) % num_ring_pts]
            grid_lines.append((p_a, p_b, (160.0 / 255.0, 110.0 / 255.0, 220.0 / 255.0, 0.75)))

        self._draw_lines(grid_lines, mvp_mat)

        # 5. Render 3D Holographic Hand Skeletons
        hand_lines = []
        for hand in hands:
            is_left = hand.handedness.label == HandednessLabel.LEFT
            bone_col = (0.0, 0.95, 1.0, 1.0) if not is_left else (1.0, 0.85, 0.0, 1.0)
            joint_col = (1.0, 1.0, 1.0, 1.0)

            z_hand = estimate_hand_depth(hand)
            joint_pts_3d = [(lm.x, lm.y, z_hand + lm.z) for lm in hand.landmarks]

            # Cyber-Bones
            for a, b in HAND_CONNECTIONS:
                hand_lines.append((joint_pts_3d[a], joint_pts_3d[b], bone_col))

            # Joint Spheres (with aspect compensation)
            for i, p in enumerate(joint_pts_3d):
                r = 0.012 if i in (4, 8, 12, 16, 20) else 0.008
                self._draw_sphere(
                    p[0], p[1], p[2],
                    radius=r,
                    mvp_mat=mvp_mat,
                    base_color=joint_col,
                    is_hand_joint=True,
                    timestamp=ts,
                    aspect=aspect,
                )

        if hand_lines:
            self._draw_lines(hand_lines, mvp_mat)

        # 6. Render 3D Shaded AR Physics Ball (with aspect compensation)
        skin_id = 0
        if skin == BallSkin.CHROME or (isinstance(skin, str) and skin.lower() == "chrome"):
            skin_id = 1
        elif skin == BallSkin.TENNIS or (isinstance(skin, str) and skin.lower() == "tennis"):
            skin_id = 2
        elif skin == BallSkin.NEON or (isinstance(skin, str) and skin.lower() == "neon"):
            skin_id = 3

        ball_rot = getattr(engine.ball, "rotation", (0.38, 0.55, 0.22))
        self._draw_sphere(
            bx, by, bz,
            radius=engine.ball.radius,
            mvp_mat=mvp_mat,
            skin=skin_id,
            is_hand_joint=False,
            timestamp=ts,
            aspect=aspect,
            rotation=ball_rot,
        )

        # 7. Readout RGBA Frame from FBO
        raw_bytes = self.fbo.read(components=4)
        rendered_rgba = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 4))

        # ModernGL reads bottom-to-top -> flip vertically
        rendered_rgba = np.flipud(rendered_rgba)

        # Convert RGBA to BGR
        if cv2 is not None:
            rendered_bgr = cv2.cvtColor(rendered_rgba, cv2.COLOR_RGBA2BGR)
        else:
            # Fallback numpy channel slice (RGBA -> BGR)
            rendered_bgr = rendered_rgba[:, :, [2, 1, 0]]

        frame[:] = rendered_bgr

        # 8. Render Picture-in-Picture (PIP) live camera preview
        if self.show_pip and raw_webcam is not None:
            self.cpu_room_renderer._render_pip_webcam(frame, raw_webcam, w, h)

        return frame

    def release(self) -> None:
        """Release ModernGL resources."""
        if self.fbo is not None:
            self.fbo.release()
            self.fbo_color.release()
            self.fbo_depth.release()
            self.fbo = None
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
        self.is_gpu_available = False
