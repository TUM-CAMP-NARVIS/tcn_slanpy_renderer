"""Diagnostic tests: ArcBall controller vs look_at rendering comparison.

Renders a unit cube from multiple known viewpoints using both the ArcBall
controller and the look_at() function, then compares view matrices and
rendered center-pixel colors to identify rendering discrepancies.

The phong shader with renderStaticColor=True encodes world-space normals
as colors: color = 0.5 * (N + 1).  Each cube face has a distinct normal
and therefore a distinct color, making it easy to verify which face is
visible from a given camera position.

Expected face colors (uint8, approx):
  +X  (1,0,0)  → (255, 127, 127)    -X  (-1,0,0)  → (0, 127, 127)
  +Y  (0,1,0)  → (127, 255, 127)    -Y  (0,-1,0)  → (127, 0, 127)
  +Z  (0,0,1)  → (127, 127, 255)    -Z  (0,0,-1)  → (127, 127, 0)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from slangpy_renderer import Mesh, OffscreenContext
from slangpy_renderer.controllers import ArcBall
from slangpy_renderer.offscreen import look_at, vulkan_rh_zo_perspective

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIDTH = 256
HEIGHT = 256
FOV_Y = 60.0
NEAR = 0.1
FAR = 100.0
DISTANCE = 3.0
CENTER = np.array([0.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])

# Expected center-pixel RGB for each face (tolerance ±10)
FACE_COLORS = {
    "+X": np.array([255, 127, 127]),
    "-X": np.array([0, 127, 127]),
    "+Y": np.array([127, 255, 127]),
    "-Y": np.array([127, 0, 127]),
    "+Z": np.array([127, 127, 255]),
    "-Z": np.array([127, 127, 0]),
}

# Camera positions and the face they should see
AXIS_VIEWS = [
    # (eye,                up_hint,             expected_face)
    (np.array([0, 0, DISTANCE]),  UP,                  "+Z"),
    (np.array([0, 0, -DISTANCE]), UP,                  "-Z"),
    (np.array([DISTANCE, 0, 0]),  UP,                  "+X"),
    (np.array([-DISTANCE, 0, 0]), UP,                  "-X"),
    (np.array([0, DISTANCE, 0]),  np.array([0, 0, -1.0]),  "+Y"),
    (np.array([0, -DISTANCE, 0]), np.array([0, 0, 1.0]),   "-Y"),
]


def _find_cube_obj() -> str:
    import slangpy_renderer
    pkg_dir = Path(slangpy_renderer.__file__).parent
    candidate = pkg_dir / "assets" / "models" / "cube.obj"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"cube.obj not found at {candidate}")


def _proj_matrix() -> np.ndarray:
    return vulkan_rh_zo_perspective(FOV_Y, float(WIDTH) / HEIGHT, NEAR, FAR)


def _render_center_pixel(
    ctx: OffscreenContext,
    cube: Mesh,
    view: np.ndarray,
    proj: np.ndarray,
) -> np.ndarray:
    """Render and return the center pixel RGBA as uint8 array."""
    ctx.clear()
    cube_copy = Mesh.from_obj(ctx.device, _find_cube_obj())
    cube_copy.pose = np.eye(4, dtype=np.float32)
    cube_copy.renderer = ctx.mesh_renderer
    ctx.add_renderable("cube", cube_copy)

    color = ctx.render_frame(
        view.astype(np.float32),
        proj.astype(np.float32),
        clear_color=(0.0, 0.0, 0.0, 0.0),
        extra_args={"renderStaticColor": True, "pointSize": 3.0},
    )
    ctx.remove_renderable("cube")
    return color[HEIGHT // 2, WIDTH // 2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx() -> OffscreenContext:
    return OffscreenContext(width=WIDTH, height=HEIGHT)


@pytest.fixture(scope="module")
def proj() -> np.ndarray:
    return _proj_matrix()


# ---------------------------------------------------------------------------
# Test: ArcBall vs look_at matrix comparison
# ---------------------------------------------------------------------------

class TestArcBallVsLookAt:
    """Compare ArcBall view matrix with look_at for identical parameters."""

    @pytest.mark.parametrize(
        "eye, up_hint, face_name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_view_matrix_match(self, eye, up_hint, face_name):
        """ArcBall.view_matrix() should match look_at() for the same parameters."""
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()

        V_arcball = arcball.view_matrix()
        V_lookat = look_at(eye, CENTER, up_hint).astype(np.float32)

        log.info(
            "\n=== Camera from %s (face %s) ===\n"
            "ArcBall view:\n%s\n"
            "look_at view:\n%s\n"
            "Difference:\n%s",
            eye, face_name,
            np.array2string(V_arcball, precision=4, suppress_small=True),
            np.array2string(V_lookat, precision=4, suppress_small=True),
            np.array2string(V_arcball - V_lookat, precision=6, suppress_small=True),
        )

        np.testing.assert_allclose(
            V_arcball, V_lookat, atol=1e-4,
            err_msg=f"View matrix mismatch for camera from {face_name}",
        )

    @pytest.mark.parametrize(
        "eye, up_hint, face_name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_look_at_renders_expected_face(self, ctx, proj, eye, up_hint, face_name):
        """look_at renders the correct face at the center pixel."""
        V = look_at(eye, CENTER, up_hint)
        pixel = _render_center_pixel(ctx, None, V, proj)
        expected = FACE_COLORS[face_name]

        log.info(
            "look_at from %s: center pixel RGB=%s, expected %s (%s)",
            eye, pixel[:3], expected, face_name,
        )

        np.testing.assert_allclose(
            pixel[:3].astype(float), expected.astype(float), atol=15,
            err_msg=f"look_at: wrong face color from {face_name}",
        )

    @pytest.mark.parametrize(
        "eye, up_hint, face_name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_arcball_renders_expected_face(self, ctx, proj, eye, up_hint, face_name):
        """ArcBall renders the correct face at the center pixel."""
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()
        V = arcball.view_matrix()
        pixel = _render_center_pixel(ctx, None, V, proj)
        expected = FACE_COLORS[face_name]

        log.info(
            "ArcBall from %s: center pixel RGB=%s, expected %s (%s)",
            eye, pixel[:3], expected, face_name,
        )

        np.testing.assert_allclose(
            pixel[:3].astype(float), expected.astype(float), atol=15,
            err_msg=f"ArcBall: wrong face color from {face_name}",
        )

    @pytest.mark.parametrize(
        "eye, up_hint, face_name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_rendered_images_match(self, ctx, proj, eye, up_hint, face_name):
        """Rendering with ArcBall vs look_at should produce the same center pixel."""
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()

        V_arcball = arcball.view_matrix()
        V_lookat = look_at(eye, CENTER, up_hint)

        pixel_arcball = _render_center_pixel(ctx, None, V_arcball, proj)
        pixel_lookat = _render_center_pixel(ctx, None, V_lookat, proj)

        log.info(
            "From %s (%s): ArcBall pixel=%s, look_at pixel=%s",
            eye, face_name, pixel_arcball[:3], pixel_lookat[:3],
        )

        np.testing.assert_allclose(
            pixel_arcball[:3].astype(float),
            pixel_lookat[:3].astype(float),
            atol=5,
            err_msg=f"Render mismatch for camera from {face_name}",
        )


# ---------------------------------------------------------------------------
# Test: Diagonal views
# ---------------------------------------------------------------------------

DIAGONAL_VIEWS = [
    (np.array([2.0, 2.0, 2.0]),   UP, "diag_ppp"),
    (np.array([-2.0, 2.0, 2.0]),  UP, "diag_npp"),
    (np.array([2.0, -2.0, 2.0]),  UP, "diag_pnp"),
    (np.array([2.0, 2.0, -2.0]),  UP, "diag_ppn"),
]


class TestDiagonalViews:
    """Verify ArcBall matches look_at for diagonal camera positions."""

    @pytest.mark.parametrize(
        "eye, up_hint, name",
        DIAGONAL_VIEWS,
        ids=[d[2] for d in DIAGONAL_VIEWS],
    )
    def test_diagonal_matrix_match(self, eye, up_hint, name):
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()

        V_arcball = arcball.view_matrix()
        V_lookat = look_at(eye, CENTER, up_hint).astype(np.float32)

        log.info(
            "\n=== Diagonal %s ===\nArcBall:\n%s\nlook_at:\n%s",
            name,
            np.array2string(V_arcball, precision=4),
            np.array2string(V_lookat, precision=4),
        )

        np.testing.assert_allclose(
            V_arcball, V_lookat, atol=1e-4,
            err_msg=f"Diagonal view {name}: matrix mismatch",
        )

    @pytest.mark.parametrize(
        "eye, up_hint, name",
        DIAGONAL_VIEWS,
        ids=[d[2] for d in DIAGONAL_VIEWS],
    )
    def test_diagonal_render_match(self, ctx, proj, eye, up_hint, name):
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()

        V_arcball = arcball.view_matrix()
        V_lookat = look_at(eye, CENTER, up_hint)

        pixel_arcball = _render_center_pixel(ctx, None, V_arcball, proj)
        pixel_lookat = _render_center_pixel(ctx, None, V_lookat, proj)

        log.info(
            "Diagonal %s: ArcBall pixel=%s, look_at pixel=%s",
            name, pixel_arcball[:3], pixel_lookat[:3],
        )

        # Both should show the same thing — a combination of face colors
        np.testing.assert_allclose(
            pixel_arcball[:3].astype(float),
            pixel_lookat[:3].astype(float),
            atol=5,
            err_msg=f"Diagonal view {name}: render mismatch",
        )


# ---------------------------------------------------------------------------
# Test: ArcBall view matrix properties
# ---------------------------------------------------------------------------

class TestViewMatrixProperties:
    """Verify mathematical properties of ArcBall view matrices."""

    @pytest.mark.parametrize(
        "eye, up_hint, name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_rotation_orthonormal(self, eye, up_hint, name):
        """Rotation part of view matrix should satisfy R@R^T = I."""
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()
        V = arcball.view_matrix()
        R = V[:3, :3]

        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-5,
            err_msg=f"{name}: rotation not orthonormal",
        )

    @pytest.mark.parametrize(
        "eye, up_hint, name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_rotation_determinant_positive(self, eye, up_hint, name):
        """det(R) should be +1 (right-handed)."""
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()
        V = arcball.view_matrix()
        R = V[:3, :3]
        det = np.linalg.det(R)

        log.info("%s: det(R) = %.6f", name, det)
        np.testing.assert_allclose(det, 1.0, atol=1e-5, err_msg=f"{name}: det(R) != +1")

    @pytest.mark.parametrize(
        "eye, up_hint, name",
        [(e, u, f) for e, u, f in AXIS_VIEWS],
        ids=[f[2] for f in AXIS_VIEWS],
    )
    def test_center_maps_to_negative_z(self, eye, up_hint, name):
        """View target should map to negative Z in view space."""
        arcball = ArcBall(eye, CENTER, up_hint, FOV_Y, (WIDTH, HEIGHT))
        arcball.update_transformation()
        V = arcball.view_matrix()
        center_view = V @ np.append(CENTER, 1.0)

        log.info("%s: center in view space = %s", name, center_view[:3])
        assert center_view[2] < 0, (
            f"{name}: center should be at -Z in view space, got z={center_view[2]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: Simulated ArcBall rotation
# ---------------------------------------------------------------------------

class TestArcBallRotation:
    """Simulate mouse drags on the ArcBall and verify rendered output."""

    def test_rotation_small_steps_stays_orthonormal(self):
        """Many small rotations should not accumulate numerical drift."""
        arcball = ArcBall(
            np.array([0, 0, DISTANCE]), CENTER, UP, FOV_Y, (WIDTH, HEIGHT),
        )
        arcball.update_transformation()

        # Simulate 100 small rotation steps (circular drag)
        n_steps = 100
        cx, cy = WIDTH // 2, HEIGHT // 2
        radius = 80
        for i in range(n_steps):
            angle = 2 * math.pi * i / n_steps
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            if i == 0:
                arcball.init_transformation((x, y))
            else:
                arcball.rotate((x, y))
            arcball.update_transformation()

        V = arcball.view_matrix()
        R = V[:3, :3]
        det = np.linalg.det(R)
        log.info("After %d rotation steps: det(R) = %.6f", n_steps, det)

        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-4,
            err_msg="Rotation is no longer orthonormal after many small rotations",
        )
        np.testing.assert_allclose(det, 1.0, atol=1e-4)

    def test_rotation_renders_different_face(self, ctx, proj):
        """After rotating away from +Z view, center pixel should change."""
        arcball = ArcBall(
            np.array([0, 0, DISTANCE]), CENTER, UP, FOV_Y, (WIDTH, HEIGHT),
        )
        arcball.update_transformation()

        # Render initial view (should be +Z face)
        V_initial = arcball.view_matrix()
        pixel_initial = _render_center_pixel(ctx, None, V_initial, proj)
        log.info("Initial center pixel (from +Z): %s", pixel_initial[:3])

        # Simulate a large horizontal drag (should rotate around Y)
        arcball.init_transformation((WIDTH // 2, HEIGHT // 2))
        # Drag from center to far left — big rotation
        for x in range(WIDTH // 2, 20, -5):
            arcball.rotate((x, HEIGHT // 2))
            arcball.update_transformation()

        V_rotated = arcball.view_matrix()
        pixel_rotated = _render_center_pixel(ctx, None, V_rotated, proj)
        log.info("After rotation center pixel: %s", pixel_rotated[:3])

        # The pixel should have changed (different face visible)
        diff = np.abs(pixel_initial[:3].astype(float) - pixel_rotated[:3].astype(float))
        log.info("Pixel difference: %s (max=%.1f)", diff, diff.max())

        # After a big horizontal drag, we should see a different face
        assert diff.max() > 20, (
            f"Expected different face after rotation, but pixels are too similar: "
            f"initial={pixel_initial[:3]}, rotated={pixel_rotated[:3]}"
        )

    def test_full_orbit_returns_to_start(self, ctx, proj):
        """A full 360° orbit should return approximately to the starting view."""
        arcball = ArcBall(
            np.array([0, 0, DISTANCE]), CENTER, UP, FOV_Y, (WIDTH, HEIGHT),
        )
        arcball.update_transformation()
        V_start = arcball.view_matrix().copy()

        # Simulate a full circular orbit (horizontal drag across the full width)
        n_steps = 200
        cx, cy = WIDTH // 2, HEIGHT // 2
        radius = 60

        arcball.init_transformation((cx + radius, cy))
        for i in range(1, n_steps + 1):
            angle = 2 * math.pi * i / n_steps
            x = int(cx + radius * math.cos(angle))
            y = cy  # pure horizontal orbit
            arcball.rotate((x, y))
            arcball.update_transformation()

        V_end = arcball.view_matrix()

        log.info(
            "\n=== Full orbit test ===\n"
            "Start view:\n%s\n"
            "End view:\n%s\n"
            "Difference:\n%s",
            np.array2string(V_start, precision=4),
            np.array2string(V_end, precision=4),
            np.array2string(V_start - V_end, precision=6, suppress_small=True),
        )

        # After a full circle the view should return close to the starting view
        # (some numerical drift is expected)
        np.testing.assert_allclose(
            V_start, V_end, atol=0.1,
            err_msg="Full orbit did not return to starting view",
        )
