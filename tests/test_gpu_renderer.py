import numpy as np
import pytest

from handtracking.ar.gpu_renderer import GPURoomRenderer, build_uv_sphere_mesh, make_ortho_or_perspective_matrix
from handtracking.ar.physics import ARPhysicsEngine
from handtracking.ar.renderer import BallSkin
from handtracking.demo import build_parser
from handtracking.inference.models import BoundingBox, HandLandmarks, Handedness, Landmark3D
from handtracking.pipeline import HandTrackingPipeline


def make_test_hand(x=0.5, y=0.5, z=0.0):
    points = [Landmark3D(x, y, z)] * 21
    points[0] = Landmark3D(x, y + 0.1, z)
    points[5] = Landmark3D(x - 0.05, y, z)
    points[9] = Landmark3D(x, y - 0.08, z)
    points[17] = Landmark3D(x + 0.05, y, z)
    # Tips
    points[4] = Landmark3D(x - 0.08, y - 0.05, z)
    points[8] = Landmark3D(x - 0.04, y - 0.1, z)
    points[12] = Landmark3D(x, y - 0.12, z)
    points[16] = Landmark3D(x + 0.04, y - 0.1, z)
    points[20] = Landmark3D(x + 0.08, y - 0.05, z)
    return HandLandmarks(tuple(points), Handedness("Right", 0.95), BoundingBox.from_landmarks(points))


def test_uv_sphere_mesh_generation():
    rings = 16
    sectors = 24
    radius = 1.0
    verts, indices = build_uv_sphere_mesh(rings=rings, sectors=sectors, radius=radius)

    # Vertex buffer shape: (num_verts, 8) -> [x, y, z, nx, ny, nz, u, v]
    expected_verts = (rings + 1) * (sectors + 1)
    assert verts.shape == (expected_verts, 8)
    assert verts.dtype == np.float32

    # Verify positions on sphere surface
    positions = verts[:, :3]
    radii = np.linalg.norm(positions, axis=1)
    assert np.allclose(radii, radius, atol=1e-5)

    # Verify unit normals
    normals = verts[:, 3:6]
    norm_lens = np.linalg.norm(normals, axis=1)
    assert np.allclose(norm_lens, 1.0, atol=1e-5)

    # Verify UV range [0, 1]
    uvs = verts[:, 6:8]
    assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0)

    # Verify indices
    expected_indices = rings * sectors * 6
    assert indices.shape == (expected_indices,)
    assert indices.dtype == np.uint32
    assert np.all(indices < expected_verts)


def test_perspective_matrix_construction():
    focal_depth = 0.85
    mat = make_ortho_or_perspective_matrix(focal_depth=focal_depth)
    assert mat.shape == (4, 4)
    assert mat.dtype == np.float32
    d = 1.0 / focal_depth
    assert np.isclose(mat[0, 0], d)
    assert np.isclose(mat[1, 1], d)
    assert np.isclose(mat[2, 2], 1.0)
    assert np.isclose(mat[3, 2], 1.0)
    assert np.isclose(mat[3, 3], d)

    # Verify perspective division matches CPU formula: x / (1 + z * focal_depth)
    for xw, yw, zw in [(0.2, -0.3, 0.4), (-0.5, 0.1, -0.2), (0.0, 0.0, 0.5)]:
        p_world = np.array([xw, yw, zw, 1.0], dtype=np.float32)
        p_clip = np.dot(mat, p_world)
        ndc_x = p_clip[0] / p_clip[3]
        ndc_y = p_clip[1] / p_clip[3]

        expected_scale = 1.0 / (1.0 + zw * focal_depth)
        assert np.isclose(ndc_x, xw * expected_scale, atol=1e-5)
        assert np.isclose(ndc_y, yw * expected_scale, atol=1e-5)


def test_isotropic_sphere_model_transform():
    focal_depth = 0.85
    mvp_mat = make_ortho_or_perspective_matrix(focal_depth=focal_depth)

    # Ball at (x=0.5, y=0.5, z=0.0) -> world coords (0.0, 0.0, 0.0)
    radius = 0.05
    rx = ry = rz = radius * 2.0
    model_mat = np.array([
        [rx,  0.0, 0.0, 0.0],
        [0.0, ry,  0.0, 0.0],
        [0.0, 0.0, rz,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    final_mvp = np.dot(mvp_mat, model_mat)

    # Vertex on X pole: (1, 0, 0)
    vx = np.dot(final_mvp, np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    ndc_vx = vx[0] / vx[3]

    # Vertex on Y pole: (0, 1, 0)
    vy = np.dot(final_mvp, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    ndc_vy = vy[1] / vy[3]

    # Vertex on Z pole: (0, 0, 1)
    vz = np.dot(final_mvp, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    # Both X and Y extents in NDC space are identical (isotropic aspect ratio 1.0)
    assert np.isclose(abs(ndc_vx), abs(ndc_vy), atol=1e-5)
    assert np.isclose(abs(ndc_vx), rx / 1.0, atol=1e-5)



def test_gpu_room_renderer_init_and_properties():
    renderer = GPURoomRenderer(focal_depth=0.85)
    assert renderer.focal_depth == 0.85
    assert renderer.bounds_min == (0.05, 0.05, -0.6)
    assert renderer.bounds_max == (0.95, 0.95, 0.6)

    # project_3d center point
    u0, v0 = renderer.project_3d(0.5, 0.5, 0.0, 640, 480)
    assert abs(u0 - 320) <= 2
    assert abs(v0 - 240) <= 2


def test_gpu_room_renderer_render_room_execution():
    renderer = GPURoomRenderer(focal_depth=0.85)
    engine = ARPhysicsEngine()
    engine.ball.position = (0.5, 0.4, 0.1)
    hand = make_test_hand()

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    raw_cam = np.ones((240, 320, 3), dtype=np.uint8) * 128

    out = renderer.render_room(
        frame,
        engine,
        hands=[hand],
        raw_webcam=raw_cam,
        timestamp=1.0,
        skin=BallSkin.BASKETBALL,
    )

    assert out is frame
    assert frame.shape == (240, 320, 3)
    assert np.any(frame > 0)


def test_gpu_room_renderer_all_skins():
    renderer = GPURoomRenderer(focal_depth=0.85)
    engine = ARPhysicsEngine()
    hand = make_test_hand()

    for skin in (BallSkin.BASKETBALL, BallSkin.CHROME, BallSkin.TENNIS, BallSkin.NEON):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = renderer.render_room(frame, engine, hands=[hand], timestamp=1.5, skin=skin)
        assert np.any(out > 0)


def test_gpu_room_renderer_cpu_fallback():
    # Force GPU off to test CPU fallback path
    renderer = GPURoomRenderer(prefer_gpu=False)
    assert renderer.is_gpu_available is False

    engine = ARPhysicsEngine()
    hand = make_test_hand()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    out = renderer.render_room(frame, engine, hands=[hand], timestamp=1.0)
    assert out is frame
    assert np.any(frame > 0)


def test_gpu_room_renderer_wall_pulse():
    renderer = GPURoomRenderer()
    renderer.trigger_wall_pulse(timestamp=2.5, color=(0, 255, 200))
    assert renderer.wall_glow_time == 2.5
    assert renderer.wall_glow_color == (0, 255, 200)


def test_pipeline_gpu_rendering_integration():
    class MockDetector:
        def detect(self, frame):
            from handtracking.inference.models import DetectionResult
            return DetectionResult(hands=(make_test_hand(),), timestamp=1.0)

    pipe = HandTrackingPipeline(
        detector=MockDetector(),
        ar_physics=ARPhysicsEngine(),
        virtual_room=True,
        use_gpu_render=True,
    )

    assert pipe.use_gpu_render is True
    assert pipe.virtual_room is True

    toggled = pipe.toggle_gpu_render()
    assert toggled is False
    assert pipe.use_gpu_render is False

    pipe.toggle_gpu_render()
    assert pipe.use_gpu_render is True

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out, gestures, telemetry = pipe.process_frame(frame)

    assert out.shape == (240, 320, 3)
    assert np.any(out > 0)
    pipe.close()


def test_demo_parser_gpu_flags():
    parser = build_parser()

    args1 = parser.parse_args(["--gpu-render"])
    assert args1.gpu_render is True

    args2 = parser.parse_args(["--gpu"])
    assert args2.gpu_render is True


def test_aspect_ratio_isotropic_pixel_scaling():
    """Verify that scaling ry by viewport aspect ratio produces equal pixel dimensions on any display."""
    focal_depth = 0.85
    mvp_mat = make_ortho_or_perspective_matrix(focal_depth=focal_depth)

    # Test wide viewports: 1280x720 (16:9), 640x320 (2:1), 640x480 (4:3)
    for width, height in [(1280, 720), (640, 320), (640, 480), (1920, 1080)]:
        aspect = float(width) / float(height)
        radius = 0.05

        rx = radius * 2.0
        ry = radius * 2.0 * aspect
        rz = radius * 2.0

        model_mat = np.array([
            [rx,  0.0, 0.0, 0.0],
            [0.0, ry,  0.0, 0.0],
            [0.0, 0.0, rz,  0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        final_mvp = np.dot(mvp_mat, model_mat)

        # X pole vertex (1, 0, 0)
        vx = np.dot(final_mvp, np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
        ndc_x = vx[0] / vx[3]
        pix_x = ndc_x * (width / 2.0)

        # Y pole vertex (0, 1, 0)
        vy = np.dot(final_mvp, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
        ndc_y = vy[1] / vy[3]
        pix_y = ndc_y * (height / 2.0)

        # Pixel span horizontally must exactly match pixel span vertically (1:1 circular sphere)
        assert np.isclose(abs(pix_x), abs(pix_y), atol=1e-4)
        assert np.isclose(abs(pix_x), radius * width, atol=1e-4)


def test_gpu_floor_indicators_and_shadows():
    """Verify that floor shadow and altitude drop-line indicators are rendered in GPURoomRenderer."""
    renderer = GPURoomRenderer(focal_depth=0.85)
    engine = ARPhysicsEngine()
    # Place ball at altitude above floor (floor is at y = 0.95)
    engine.ball.position = (0.5, 0.5, 0.0)

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = renderer.render_room(frame, engine, hands=[], timestamp=2.0, skin=BallSkin.BASKETBALL)

    assert out is frame
    assert frame.shape == (360, 640, 3)
    # Ensure floor area has rendered shadow/grid pixels
    floor_pixel_y = int(0.95 * (360 - 1))
    assert np.any(frame[floor_pixel_y - 20:floor_pixel_y + 10, :] > 0)

