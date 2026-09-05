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
    z_near = -0.65
    z_far = 0.70

    # Perspective matrix with view translation d
    # X_clip = xw * d
    # Y_clip = yw * d
    # Z_clip = ((z_far + d)/(z_far - z_near)) * (zw + d) - (2*(z_far+d)*(z_near+d)/(z_far - z_near))
    # W_clip = zw + d
    a = (z_far + d) / (z_far - z_near)
    b = -2.0 * (z_far + d) * (z_near + d) / (z_far - z_near)

    mat = np.array([
        [d,   0.0, 0.0, 0.0],
        [0.0, d,   0.0, 0.0],
        [0.0, 0.0, a,   b],
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
out vec2 v_uv;

void main() {
    vec4 wp = u_model * vec4(in_pos, 1.0);
    v_world_pos = wp.xyz;
    v_world_normal = normalize(u_normal_mat * in_normal);
    v_uv = in_uv;
    gl_Position = u_mvp * wp;
}
"""

SPHERE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform int u_skin;
uniform float u_time;
uniform vec4 u_base_color;
uniform int u_is_hand_joint;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 L = normalize(u_light_pos - v_world_pos);
    vec3 V = normalize(u_view_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float fresnel = pow(1.0 - NdotV, 3.0);

    if (u_is_hand_joint == 1) {
        vec3 col = u_base_color.rgb * (0.65 + 0.35 * NdotL) + vec3(1.0) * pow(NdotH, 16.0) * 0.7 + u_base_color.rgb * fresnel * 1.5;
        fragColor = vec4(col, u_base_color.a);
        return;
    }

    vec3 albedo = vec3(0.9, 0.4, 0.1);
    float shininess = 32.0;
    float spec_strength = 0.5;
    float ambient = 0.25;

    if (u_skin == 0) { // BASKETBALL
        float u_seam = abs(sin(v_uv.x * 3.14159 * 2.0));
        float v_seam = abs(cos(v_uv.y * 3.14159 * 4.0));
        bool is_rib = (u_seam < 0.05 || abs(v_uv.y - 0.5) < 0.03 || (v_seam < 0.07 && abs(v_uv.y - 0.5) < 0.35));
        if (is_rib) {
            albedo = vec3(0.08, 0.05, 0.05);
            spec_strength = 0.1;
        } else {
            albedo = vec3(0.88, 0.38, 0.12);
            spec_strength = 0.35;
        }
        shininess = 20.0;
    } else if (u_skin == 1) { // CHROME
        albedo = vec3(0.78, 0.82, 0.88);
        shininess = 128.0;
        spec_strength = 1.4;
        ambient = 0.35;
        albedo += vec3(0.2, 0.3, 0.4) * fresnel;
    } else if (u_skin == 2) { // TENNIS
        float seam = abs(sin(v_uv.x * 6.283) * 0.3 - (v_uv.y - 0.5));
        if (seam < 0.05) {
            albedo = vec3(0.92, 0.92, 0.88);
            spec_strength = 0.1;
        } else {
            albedo = vec3(0.75, 0.88, 0.18);
            spec_strength = 0.2;
        }
        shininess = 8.0;
    } else if (u_skin == 3) { // NEON
        vec3 cyan = vec3(0.0, 0.9, 1.0);
        vec3 magenta = vec3(1.0, 0.1, 0.8);
        float pulse = 0.5 + 0.5 * sin(u_time * 4.0);
        albedo = mix(cyan, magenta, pulse * 0.7 + v_uv.y * 0.3);
        spec_strength = 1.2;
        shininess = 64.0;
        ambient = 0.6;
        albedo += mix(magenta, cyan, pulse) * fresnel * 2.0;
    }

    float diff = NdotL;
    float spec = pow(NdotH, shininess) * spec_strength;
    vec3 color = (ambient + diff) * albedo + vec3(spec);
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

    def _draw_lines(self, lines: list[tuple[Sequence[float], Sequence[float], Sequence[float]]], mvp_bytes: bytes, tint=(1.0, 1.0, 1.0, 1.0)) -> None:
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

        self.line_prog["u_mvp"].write(mvp_bytes)
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
    ) -> None:
        """Render 3D sphere at world position with model transform and GLSL lighting."""
        wx, wy, wz = self._to_world_coords(x, y, z)
        rx = radius * 2.0
        ry = radius * 2.0
        rz = radius

        # Model matrix: Translate * Scale
        model_mat = np.array([
            [rx,  0.0, 0.0, wx],
            [0.0, ry,  0.0, wy],
            [0.0, 0.0, rz,  wz],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Normal matrix (inv transpose 3x3)
        normal_mat = np.array([
            [1.0 / max(1e-5, rx), 0.0, 0.0],
            [0.0, 1.0 / max(1e-5, ry), 0.0],
            [0.0, 0.0, 1.0 / max(1e-5, rz)],
        ], dtype=np.float32)

        final_mvp = np.dot(mvp_mat, model_mat)

        self.sphere_prog["u_mvp"].write(final_mvp.tobytes())
        self.sphere_prog["u_model"].write(model_mat.tobytes())
        self.sphere_prog["u_normal_mat"].write(normal_mat.tobytes())
        self.sphere_prog["u_light_pos"].value = (0.3, 0.8, -0.7)
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
        mvp_bytes = mvp_mat.tobytes()

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

        # D. Ball Altitude Vertical Drop-Line
        bx, by, bz = engine.ball.position
        if by < floor_y - 0.02:
            grid_lines.append(((bx, by, bz), (bx, floor_y, bz), (120.0 / 255.0, 100.0 / 255.0, 160.0 / 255.0, 0.8)))

        self._draw_lines(grid_lines, mvp_bytes)

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

            # Joint Spheres
            for i, p in enumerate(joint_pts_3d):
                r = 0.012 if i in (4, 8, 12, 16, 20) else 0.008
                self._draw_sphere(
                    p[0], p[1], p[2],
                    radius=r,
                    mvp_mat=mvp_mat,
                    base_color=joint_col,
                    is_hand_joint=True,
                    timestamp=ts,
                )

        if hand_lines:
            self._draw_lines(hand_lines, mvp_bytes)

        # 6. Render 3D Shaded AR Physics Ball
        skin_id = 0
        if skin == BallSkin.CHROME or (isinstance(skin, str) and skin.lower() == "chrome"):
            skin_id = 1
        elif skin == BallSkin.TENNIS or (isinstance(skin, str) and skin.lower() == "tennis"):
            skin_id = 2
        elif skin == BallSkin.NEON or (isinstance(skin, str) and skin.lower() == "neon"):
            skin_id = 3

        self._draw_sphere(
            bx, by, bz,
            radius=engine.ball.radius,
            mvp_mat=mvp_mat,
            skin=skin_id,
            is_hand_joint=False,
            timestamp=ts,
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
