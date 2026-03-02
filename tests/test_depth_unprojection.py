"""
Tests for GPU depth-to-pointcloud unprojection (DepthUnprojector).

Verifies XY lookup table computation, depth-to-pointcloud conversion,
the CameraIntrinsics / DepthParameters data types, validation against
real Azure Kinect sensor data with an e57 reference pointcloud, and
color camera UV projection.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from slangpy_renderer import CameraIntrinsics, DepthParameters, DepthUnprojector
from slangpy_renderer import ColorProjectionParameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def device(offscreen_ctx):
    """Vulkan device with shader include paths (from OffscreenContext)."""
    return offscreen_ctx.device


@pytest.fixture
def pinhole_params() -> DepthParameters:
    """Pinhole camera (zero distortion), 64x48 image."""
    return DepthParameters(
        width=64,
        height=48,
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0),
    )


@pytest.fixture
def azure_kinect_params() -> DepthParameters:
    """Azure Kinect-like camera with Brown-Conrady distortion, 640x576 image."""
    return DepthParameters(
        width=640,
        height=576,
        intrinsics=CameraIntrinsics(
            fx=504.0,
            fy=504.0,
            cx=320.0,
            cy=288.0,
            radial_distortion=[0.5, -2.7, 1.6, 0.4, -2.5, 1.5],
            tangential_distortion=[0.0, 0.0],
        ),
    )


def _load_calibration() -> dict:
    with open(DATA_DIR / "calibration_camera01.json") as f:
        return json.load(f)


def _depth_params_from_calibration(calib: dict) -> DepthParameters:
    """Build DepthParameters from the calibration JSON's depth_parameters section."""
    dp = calib["depth_parameters"]
    return DepthParameters(
        width=dp["width"],
        height=dp["height"],
        intrinsics=CameraIntrinsics(
            fx=dp["fx"],
            fy=dp["fy"],
            cx=dp["cx"],
            cy=dp["cy"],
            radial_distortion=dp["radial_distortion"],
            tangential_distortion=dp["tangential_distortion"],
            max_radius=dp["metric_radius"],
        ),
    )


@pytest.fixture(scope="module")
def sensor_calibration():
    """Calibration dict loaded from tests/data/calibration_camera01.json."""
    return _load_calibration()


@pytest.fixture(scope="module")
def sensor_depth_params(sensor_calibration) -> DepthParameters:
    """DepthParameters from the real Azure Kinect calibration."""
    return _depth_params_from_calibration(sensor_calibration)


@pytest.fixture(scope="module")
def sensor_depth_image() -> np.ndarray:
    """Real depth image (576x640, uint16) from Azure Kinect."""
    return np.load(DATA_DIR / "depth_camera01.npy")


@pytest.fixture(scope="module")
def sensor_color_image() -> np.ndarray:
    """Real color image (1080x1920, 3) uint8 from Azure Kinect."""
    return np.load(DATA_DIR / "color_camera01.npy")


@pytest.fixture(scope="module")
def e57_reference_points() -> np.ndarray:
    """Reference pointcloud (N, 3) float64 from e57 scan data."""
    return np.load(DATA_DIR / "e57_points_camera01.npy")


@pytest.fixture(scope="module")
def sensor_color_params(sensor_calibration) -> ColorProjectionParameters:
    """ColorProjectionParameters built from the calibration JSON."""
    return ColorProjectionParameters.from_calibration(
        sensor_calibration["color_parameters"],
        sensor_calibration["color2depth_transform"],
    )


# ---------------------------------------------------------------------------
# CPU reference: forward Brown-Conrady projection
# ---------------------------------------------------------------------------


def _cpu_forward_project(
    point_3d: np.ndarray,
    color_intrinsics: CameraIntrinsics,
    depth_to_color: np.ndarray,
    color_width: int,
    color_height: int,
) -> np.ndarray:
    """
    CPU reference implementation of the depth-to-color UV projection.

    Takes a 3D point in depth camera space and returns the normalized UV
    coordinates in the color image, using the same Brown-Conrady distortion
    model as the GPU shader.
    """
    # Transform to color camera space
    p_homo = np.append(point_3d, 1.0)
    p_color = (depth_to_color @ p_homo)[:3]

    # Perspective divide
    xp = p_color[0] / p_color[2]
    yp = p_color[1] / p_color[2]

    # Brown-Conrady forward projection
    k = color_intrinsics.k
    p = color_intrinsics.p

    xp2 = xp * xp
    yp2 = yp * yp
    xyp = xp * yp
    rs = xp2 + yp2

    if rs > color_intrinsics.max_radius ** 2:
        return np.array([0.0, 0.0])

    rss = rs * rs
    rsc = rss * rs
    a = 1.0 + k[0] * rs + k[1] * rss + k[2] * rsc
    b = 1.0 + k[3] * rs + k[4] * rss + k[5] * rsc
    bi = 1.0 / b if b != 0 else 1.0
    d = a * bi

    xp_d = xp * d
    yp_d = yp * d

    rs_2xp2 = rs + 2.0 * xp2
    rs_2yp2 = rs + 2.0 * yp2

    xp_d += rs_2xp2 * p[1] + 2.0 * xyp * p[0]
    yp_d += rs_2yp2 * p[0] + 2.0 * xyp * p[1]

    u = xp_d * color_intrinsics.fx + color_intrinsics.cx
    v = yp_d * color_intrinsics.fy + color_intrinsics.cy

    return np.array([u / color_width, v / color_height])


# ---------------------------------------------------------------------------
# Import / construction tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_camera_intrinsics(self):
        from slangpy_renderer import CameraIntrinsics

        assert callable(CameraIntrinsics)

    def test_import_depth_parameters(self):
        from slangpy_renderer import DepthParameters

        assert callable(DepthParameters)

    def test_import_depth_unprojector(self):
        from slangpy_renderer import DepthUnprojector

        assert callable(DepthUnprojector)

    def test_import_color_projection_parameters(self):
        from slangpy_renderer import ColorProjectionParameters

        assert callable(ColorProjectionParameters)


class TestCameraIntrinsics:
    def test_default_distortion(self):
        cam = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        assert cam.k == [0.0] * 6
        assert cam.p == [0.0, 0.0]
        assert cam.max_radius == 1.7

    def test_partial_radial_distortion_padded(self):
        cam = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            radial_distortion=[0.1, -0.2],
        )
        assert cam.k == [0.1, -0.2, 0.0, 0.0, 0.0, 0.0]

    def test_full_radial_distortion(self):
        k = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
        cam = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            radial_distortion=k,
        )
        assert cam.k == k


class TestDepthParameters:
    def test_default_depth_scale(self):
        params = DepthParameters(
            width=640, height=480,
            intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
        )
        assert params.depth_scale == 0.001

    def test_custom_depth_scale(self):
        params = DepthParameters(
            width=640, height=480,
            intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
            depth_scale=0.0001,
        )
        assert params.depth_scale == 0.0001


# ---------------------------------------------------------------------------
# XY Table tests
# ---------------------------------------------------------------------------


class TestXYTable:
    def test_pinhole_center_pixel_is_zero(self, device, pinhole_params):
        """Principal point should map to ray direction (0, 0)."""
        unprojector = DepthUnprojector(device, pinhole_params)
        xy = unprojector.xy_table_to_numpy()
        cx = int(pinhole_params.intrinsics.cx)
        cy = int(pinhole_params.intrinsics.cy)
        np.testing.assert_allclose(xy[cy, cx], [0.0, 0.0], atol=1e-6)

    def test_pinhole_matches_analytic_formula(self, device, pinhole_params):
        """With zero distortion, ray = ((px - cx) / fx, (py - cy) / fy)."""
        unprojector = DepthUnprojector(device, pinhole_params)
        xy = unprojector.xy_table_to_numpy()
        cam = pinhole_params.intrinsics

        # Check a grid of sample pixels
        for py in [0, 12, 24, 36, 47]:
            for px in [0, 16, 32, 48, 63]:
                expected_x = (px - cam.cx) / cam.fx
                expected_y = (py - cam.cy) / cam.fy
                np.testing.assert_allclose(
                    xy[py, px],
                    [expected_x, expected_y],
                    atol=1e-5,
                    err_msg=f"Mismatch at pixel ({px}, {py})",
                )

    def test_pinhole_xy_table_shape(self, device, pinhole_params):
        unprojector = DepthUnprojector(device, pinhole_params)
        xy = unprojector.xy_table_to_numpy()
        assert xy.shape == (48, 64, 2)
        assert xy.dtype == np.float32

    def test_distorted_center_pixel_is_zero(self, device, azure_kinect_params):
        """Principal point should still be (0, 0) with distortion."""
        unprojector = DepthUnprojector(device, azure_kinect_params)
        xy = unprojector.xy_table_to_numpy()
        cx = int(azure_kinect_params.intrinsics.cx)
        cy = int(azure_kinect_params.intrinsics.cy)
        np.testing.assert_allclose(xy[cy, cx], [0.0, 0.0], atol=1e-6)

    def test_distorted_corner_rays_nonzero(self, device, azure_kinect_params):
        """Corner pixels should have significant ray directions."""
        unprojector = DepthUnprojector(device, azure_kinect_params)
        xy = unprojector.xy_table_to_numpy()
        # Top-left corner should have negative x,y ray directions
        assert xy[0, 0, 0] < -0.3
        assert xy[0, 0, 1] < -0.3

    def test_distorted_most_pixels_valid(self, device, azure_kinect_params):
        """Nearly all pixels in a typical camera should have valid (non-zero) rays."""
        unprojector = DepthUnprojector(device, azure_kinect_params)
        xy = unprojector.xy_table_to_numpy()
        nonzero = np.sum(np.any(xy != 0, axis=2))
        total = azure_kinect_params.width * azure_kinect_params.height
        # At least 99% of pixels should be valid
        assert nonzero / total > 0.99

    def test_distorted_xy_table_differs_from_pinhole(self, device):
        """With distortion, the XY table should differ from the pinhole model."""
        cam_pinhole = CameraIntrinsics(fx=504.0, fy=504.0, cx=320.0, cy=288.0)
        cam_distorted = CameraIntrinsics(
            fx=504.0, fy=504.0, cx=320.0, cy=288.0,
            radial_distortion=[0.5, -2.7, 1.6, 0.4, -2.5, 1.5],
        )
        params_p = DepthParameters(width=640, height=576, intrinsics=cam_pinhole)
        params_d = DepthParameters(width=640, height=576, intrinsics=cam_distorted)

        xy_p = DepthUnprojector(device, params_p).xy_table_to_numpy()
        xy_d = DepthUnprojector(device, params_d).xy_table_to_numpy()

        # Center should match (distortion has no effect at optical axis)
        np.testing.assert_allclose(xy_p[288, 320], xy_d[288, 320], atol=1e-6)
        # Corner should differ
        assert not np.allclose(xy_p[0, 0], xy_d[0, 0], atol=1e-4)


# ---------------------------------------------------------------------------
# Pointcloud tests
# ---------------------------------------------------------------------------


class TestPointcloud:
    def test_constant_depth_z_values(self, device, pinhole_params):
        """Constant depth image should produce uniform z values."""
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.full(
            (pinhole_params.height, pinhole_params.width), 2000, dtype=np.uint16
        )
        unprojector.unproject(depth)
        points = unprojector.to_numpy()
        # All z values should be 2.0m (2000 * 0.001)
        np.testing.assert_allclose(points[:, :, 2], 2.0, atol=1e-5)

    def test_zero_depth_produces_origin(self, device, pinhole_params):
        """Zero-depth pixels should map to (0, 0, 0)."""
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.zeros(
            (pinhole_params.height, pinhole_params.width), dtype=np.uint16
        )
        unprojector.unproject(depth)
        points = unprojector.to_numpy()
        np.testing.assert_array_equal(points, 0.0)

    def test_mixed_depth_zero_and_valid(self, device, pinhole_params):
        """Zero-depth region should be (0,0,0) while valid region has correct z."""
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.full(
            (pinhole_params.height, pinhole_params.width), 1000, dtype=np.uint16
        )
        depth[10:20, 10:20] = 0

        unprojector.unproject(depth)
        points = unprojector.to_numpy()

        # Zero region
        np.testing.assert_array_equal(points[10:20, 10:20], 0.0)
        # Valid region center
        np.testing.assert_allclose(points[24, 32, 2], 1.0, atol=1e-5)

    def test_depth_scale_applied(self, device):
        """Custom depth_scale should scale z values accordingly."""
        params = DepthParameters(
            width=16,
            height=16,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=8.0),
            depth_scale=0.0001,  # 0.1mm resolution
        )
        unprojector = DepthUnprojector(device, params)
        depth = np.full((16, 16), 10000, dtype=np.uint16)  # 1.0m at 0.1mm scale
        unprojector.unproject(depth)
        points = unprojector.to_numpy()
        np.testing.assert_allclose(points[8, 8, 2], 1.0, atol=1e-5)

    def test_varying_depth(self, device, pinhole_params):
        """Different depth values should produce different z values."""
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.zeros(
            (pinhole_params.height, pinhole_params.width), dtype=np.uint16
        )
        depth[:24, :] = 1000  # top half: 1m
        depth[24:, :] = 5000  # bottom half: 5m

        unprojector.unproject(depth)
        points = unprojector.to_numpy()

        np.testing.assert_allclose(points[12, 32, 2], 1.0, atol=1e-5)
        np.testing.assert_allclose(points[36, 32, 2], 5.0, atol=1e-5)

    def test_pinhole_pointcloud_geometry(self, device, pinhole_params):
        """For pinhole camera, point = (ray_x * d, ray_y * d, d)."""
        unprojector = DepthUnprojector(device, pinhole_params)
        cam = pinhole_params.intrinsics
        depth = np.full(
            (pinhole_params.height, pinhole_params.width), 3000, dtype=np.uint16
        )  # 3m

        unprojector.unproject(depth)
        points = unprojector.to_numpy()

        # Check a non-center pixel
        px, py = 48, 36
        d = 3.0  # meters
        expected_x = ((px - cam.cx) / cam.fx) * d
        expected_y = ((py - cam.cy) / cam.fy) * d
        np.testing.assert_allclose(
            points[py, px], [expected_x, expected_y, d], atol=1e-4
        )

    def test_buffer_reuse(self, device, pinhole_params):
        """position_buffer should be the same object across unproject() calls."""
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.full(
            (pinhole_params.height, pinhole_params.width), 1000, dtype=np.uint16
        )
        buf1 = unprojector.unproject(depth)
        buf2 = unprojector.unproject(depth)
        assert buf1 is buf2
        assert buf1 is unprojector.position_buffer


# ---------------------------------------------------------------------------
# Properties / API tests
# ---------------------------------------------------------------------------


class TestAPI:
    def test_num_points(self, device, pinhole_params):
        unprojector = DepthUnprojector(device, pinhole_params)
        assert unprojector.num_points == 64 * 48

    def test_to_numpy_shape(self, device, pinhole_params):
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.full((48, 64), 1000, dtype=np.uint16)
        unprojector.unproject(depth)
        points = unprojector.to_numpy()
        assert points.shape == (48, 64, 3)
        assert points.dtype == np.float32

    def test_xy_table_to_numpy_shape(self, device, pinhole_params):
        unprojector = DepthUnprojector(device, pinhole_params)
        xy = unprojector.xy_table_to_numpy()
        assert xy.shape == (48, 64, 2)
        assert xy.dtype == np.float32

    def test_wrong_dtype_raises(self, device, pinhole_params):
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.full((48, 64), 1000, dtype=np.float32)
        with pytest.raises(AssertionError, match="uint16"):
            unprojector.unproject(depth)

    def test_wrong_shape_raises(self, device, pinhole_params):
        unprojector = DepthUnprojector(device, pinhole_params)
        depth = np.full((64, 48), 1000, dtype=np.uint16)  # swapped
        with pytest.raises(AssertionError, match="shape"):
            unprojector.unproject(depth)


# ---------------------------------------------------------------------------
# Reference data tests (real Azure Kinect capture vs e57 reference)
#
# The e57 reference pointcloud was generated from the same depth image using
# simple pinhole unprojection: point = ((px-cx)/fx * d, (py-cy)/fy * d, d)
# without distortion correction, in a (X-right, Y-up, Z-backward) convention:
#   e57_x =  cam_x
#   e57_y = -cam_y
#   e57_z = -cam_z
#
# Our DepthUnprojector applies full Brown-Conrady distortion correction, so
# the x/y values differ from the pinhole reference. The z values (pure depth
# scaling) are identical.
# ---------------------------------------------------------------------------


class TestE57Reference:
    """Validate GPU unprojection against real sensor data + e57 reference."""

    def test_valid_point_count_matches(
        self, device, sensor_depth_params, sensor_depth_image, e57_reference_points
    ):
        """Number of non-zero depth pixels must equal e57 point count."""
        n_valid = np.count_nonzero(sensor_depth_image)
        assert n_valid == e57_reference_points.shape[0]

    def test_depth_values_match_reference(
        self, device, sensor_depth_params, sensor_depth_image, e57_reference_points
    ):
        """Z values should match the e57 reference within float32 precision.

        Z is computed as raw_depth * depth_scale — no distortion involved —
        so our output and the reference must agree to float32 epsilon.
        """
        unprojector = DepthUnprojector(device, sensor_depth_params)
        unprojector.unproject(sensor_depth_image)
        gpu_points = unprojector.to_numpy()

        # Extract valid points in row-major order, apply coordinate convention
        valid_mask = sensor_depth_image > 0
        gpu_valid = gpu_points[valid_mask]
        gpu_z = gpu_valid[:, 2].astype(np.float64)
        ref_z = -e57_reference_points[:, 2]  # e57 z is negated

        np.testing.assert_allclose(gpu_z, ref_z, atol=5e-7, rtol=0)

    def test_pinhole_xy_matches_reference(
        self, device, sensor_depth_params, sensor_depth_image, e57_reference_points
    ):
        """With zero distortion, x/y should match the e57 pinhole reference.

        This confirms the e57 reference used pinhole unprojection and
        validates our basic projection math and coordinate convention.
        """
        # Create a pinhole version (same intrinsics, zero distortion)
        cam = sensor_depth_params.intrinsics
        pinhole_params = DepthParameters(
            width=sensor_depth_params.width,
            height=sensor_depth_params.height,
            intrinsics=CameraIntrinsics(
                fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
            ),
        )
        unprojector = DepthUnprojector(device, pinhole_params)
        unprojector.unproject(sensor_depth_image)
        gpu_points = unprojector.to_numpy()

        valid_mask = sensor_depth_image > 0
        gpu_valid = gpu_points[valid_mask].astype(np.float64)
        # Apply coordinate convention: e57 = (cam_x, -cam_y, -cam_z)
        gpu_e57 = gpu_valid.copy()
        gpu_e57[:, 1] *= -1
        gpu_e57[:, 2] *= -1

        np.testing.assert_allclose(gpu_e57, e57_reference_points, atol=5e-7, rtol=0)

    def test_distortion_correction_changes_xy(
        self, device, sensor_depth_params, sensor_depth_image, e57_reference_points
    ):
        """Full distortion correction should produce different x/y than pinhole.

        The Azure Kinect has significant rational distortion (k1~20, k4~21).
        Distortion correction spreads rays further from center, so the
        corrected x/y magnitudes should generally be larger.
        """
        unprojector = DepthUnprojector(device, sensor_depth_params)
        unprojector.unproject(sensor_depth_image)
        gpu_points = unprojector.to_numpy()

        valid_mask = sensor_depth_image > 0
        gpu_valid = gpu_points[valid_mask].astype(np.float64)
        gpu_e57 = gpu_valid.copy()
        gpu_e57[:, 1] *= -1
        gpu_e57[:, 2] *= -1

        # z must still match (distortion doesn't affect depth)
        np.testing.assert_allclose(
            gpu_e57[:, 2], e57_reference_points[:, 2], atol=5e-7, rtol=0
        )

        # x/y should differ from the pinhole reference
        xy_diff = np.abs(gpu_e57[:, :2] - e57_reference_points[:, :2])
        mean_diff = xy_diff.mean()
        # With k1~20 distortion, the mean xy difference should be substantial
        assert mean_diff > 0.01, (
            f"Distortion correction had negligible effect (mean diff={mean_diff:.6f})"
        )

    def test_distortion_increases_ray_magnitude(
        self, device, sensor_depth_params, sensor_depth_image, e57_reference_points
    ):
        """Distortion correction should increase ray spread for this lens.

        The Azure Kinect's rational model has numerator/denominator ratio < 1
        at the edges, meaning the image is barrel-distorted. Un-distorting
        should push rays outward (larger magnitude).
        """
        # Pinhole unprojection
        cam = sensor_depth_params.intrinsics
        pinhole_params = DepthParameters(
            width=sensor_depth_params.width,
            height=sensor_depth_params.height,
            intrinsics=CameraIntrinsics(
                fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
            ),
        )
        pinhole_unproj = DepthUnprojector(device, pinhole_params)

        # Full distortion unprojection
        distorted_unproj = DepthUnprojector(device, sensor_depth_params)

        xy_pinhole = pinhole_unproj.xy_table_to_numpy()
        xy_distorted = distorted_unproj.xy_table_to_numpy()

        # Compare ray magnitudes for non-center pixels
        # Exclude a small region around the principal point where distortion is minimal
        h, w = sensor_depth_params.height, sensor_depth_params.width
        cy_i, cx_i = int(cam.cy), int(cam.cx)
        margin = 50
        mask = np.ones((h, w), dtype=bool)
        mask[max(0, cy_i - margin):cy_i + margin, max(0, cx_i - margin):cx_i + margin] = False

        mag_pinhole = np.linalg.norm(xy_pinhole[mask], axis=1)
        mag_distorted = np.linalg.norm(xy_distorted[mask], axis=1)

        # Most edge pixels should have larger magnitude after distortion correction
        larger = np.sum(mag_distorted > mag_pinhole)
        total = mask.sum()
        ratio = larger / total
        assert ratio > 0.90, (
            f"Expected >90% of edge rays to increase in magnitude, got {ratio:.1%}"
        )


# ---------------------------------------------------------------------------
# Color projection tests
#
# Validates the depth-to-color camera UV projection pipeline:
# 1. Transform 3D depth-camera point to color camera space (rigid transform)
# 2. Perspective divide to get normalized image coords
# 3. Forward Brown-Conrady distortion through color camera intrinsics
# 4. Normalize pixel coords to [0, 1] UV range
# ---------------------------------------------------------------------------


class TestColorProjection:
    """Validate color camera UV projection with real Azure Kinect data."""

    def test_from_calibration(self, sensor_calibration):
        """ColorProjectionParameters.from_calibration parses all fields."""
        cp = ColorProjectionParameters.from_calibration(
            sensor_calibration["color_parameters"],
            sensor_calibration["color2depth_transform"],
        )
        assert cp.width == 1920
        assert cp.height == 1080
        assert cp.intrinsics.fx == pytest.approx(1121.701904296875)
        assert cp.depth_to_color.shape == (4, 4)

    def test_depth_to_color_is_valid_rigid_transform(self, sensor_color_params):
        """The depth-to-color matrix should be a valid rigid body transform."""
        M = sensor_color_params.depth_to_color
        R = M[:3, :3]
        # Rotation should be orthogonal: R @ R.T ≈ I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Determinant should be +1 (proper rotation, not reflection)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_allclose(M[3, :], [0, 0, 0, 1], atol=1e-15)

    def test_color2depth_inversion(self, sensor_calibration, sensor_color_params):
        """Depth-to-color matrix should be the inverse of color-to-depth."""
        from slangpy_renderer.utils.depth_unprojector import (
            _rigid_transform_to_matrix,
        )

        c2d_t = sensor_calibration["color2depth_transform"]
        rot = c2d_t["rotation"]
        trans = c2d_t["translation"]
        c2d = _rigid_transform_to_matrix(
            (rot["x"], rot["y"], rot["z"], rot["w"]),
            (trans["x"], trans["y"], trans["z"]),
        )
        d2c = sensor_color_params.depth_to_color

        # c2d @ d2c should be identity
        np.testing.assert_allclose(c2d @ d2c, np.eye(4), atol=1e-10)

    def test_has_color_projection_flag(
        self, device, sensor_depth_params, sensor_color_params
    ):
        """has_color_projection should reflect whether color params were provided."""
        without = DepthUnprojector(device, sensor_depth_params)
        assert not without.has_color_projection

        with_color = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        assert with_color.has_color_projection

    def test_texcoords_shape(
        self, device, sensor_depth_params, sensor_depth_image, sensor_color_params
    ):
        """Texcoord buffer should have (H, W, 2) float32 shape."""
        unprojector = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        unprojector.unproject(sensor_depth_image)
        uvs = unprojector.texcoords_to_numpy()
        assert uvs.shape == (576, 640, 2)
        assert uvs.dtype == np.float32

    def test_zero_depth_gives_zero_uv(
        self, device, sensor_depth_params, sensor_color_params
    ):
        """Zero-depth pixels should produce (0, 0) UVs."""
        unprojector = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        depth = np.zeros(
            (sensor_depth_params.height, sensor_depth_params.width), dtype=np.uint16
        )
        unprojector.unproject(depth)
        uvs = unprojector.texcoords_to_numpy()
        np.testing.assert_array_equal(uvs, 0.0)

    def test_no_color_params_gives_zero_uvs(
        self, device, sensor_depth_params, sensor_depth_image
    ):
        """Without color params, texcoords should be all zero."""
        unprojector = DepthUnprojector(device, sensor_depth_params)
        unprojector.unproject(sensor_depth_image)
        uvs = unprojector.texcoords_to_numpy()
        np.testing.assert_array_equal(uvs, 0.0)

    def test_texcoords_in_valid_range(
        self, device, sensor_depth_params, sensor_depth_image, sensor_color_params
    ):
        """For valid depth pixels, most UVs should be in [0, 1].

        The depth camera's FoV is wider than the color camera's FoV, so some
        edge pixels may project outside [0, 1]. But the majority should be inside.
        """
        unprojector = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        unprojector.unproject(sensor_depth_image)
        uvs = unprojector.texcoords_to_numpy()

        valid_mask = sensor_depth_image > 0
        valid_uvs = uvs[valid_mask]

        in_range = (
            (valid_uvs[:, 0] >= 0)
            & (valid_uvs[:, 0] <= 1)
            & (valid_uvs[:, 1] >= 0)
            & (valid_uvs[:, 1] <= 1)
        )
        ratio = in_range.sum() / len(valid_uvs)
        assert ratio > 0.80, (
            f"Expected >80% of valid UVs in [0,1], got {ratio:.1%}"
        )

    def test_center_pixel_projects_to_color_image(
        self, device, sensor_depth_params, sensor_depth_image, sensor_color_params
    ):
        """A point near the depth camera's optical axis should project into the
        central region of the color image.

        The Azure Kinect has a ~32mm baseline and ~3 degree rotation between
        depth and color cameras, so the exact projection depends on depth. We
        verify it falls within the central 80% of the color image.
        """
        unprojector = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        unprojector.unproject(sensor_depth_image)
        uvs = unprojector.texcoords_to_numpy()

        # Use the depth camera's principal point pixel
        cx_d = int(sensor_depth_params.intrinsics.cx)
        cy_d = int(sensor_depth_params.intrinsics.cy)
        uv_center = uvs[cy_d, cx_d]

        # The projected UV should be roughly in the central region
        assert 0.1 < uv_center[0] < 0.9, (
            f"Center pixel u={uv_center[0]:.3f} outside central 80%"
        )
        assert 0.1 < uv_center[1] < 0.9, (
            f"Center pixel v={uv_center[1]:.3f} outside central 80%"
        )

    def test_texcoords_match_cpu_reference(
        self, device, sensor_depth_params, sensor_depth_image, sensor_color_params
    ):
        """GPU UVs should match CPU forward projection at sampled points.

        Picks a set of representative depth pixels, computes their 3D positions,
        projects to color UVs using the CPU reference, and verifies the GPU
        produces matching results.
        """
        unprojector = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        unprojector.unproject(sensor_depth_image)
        gpu_points = unprojector.to_numpy()
        gpu_uvs = unprojector.texcoords_to_numpy()

        # Sample a grid of pixels with valid depth
        sample_pixels = []
        for py in range(50, 550, 80):
            for px in range(50, 600, 80):
                if sensor_depth_image[py, px] > 0:
                    sample_pixels.append((py, px))

        assert len(sample_pixels) >= 10, "Need at least 10 valid sample pixels"

        for py, px in sample_pixels:
            point_3d = gpu_points[py, px].astype(np.float64)
            gpu_uv = gpu_uvs[py, px].astype(np.float64)

            cpu_uv = _cpu_forward_project(
                point_3d,
                sensor_color_params.intrinsics,
                sensor_color_params.depth_to_color,
                sensor_color_params.width,
                sensor_color_params.height,
            )

            np.testing.assert_allclose(
                gpu_uv, cpu_uv, atol=1e-5,
                err_msg=f"UV mismatch at pixel ({px}, {py}): "
                f"GPU={gpu_uv}, CPU={cpu_uv}",
            )

    def test_color_distortion_has_effect(
        self, device, sensor_depth_params, sensor_depth_image, sensor_color_params
    ):
        """UVs with distorted color intrinsics should differ from pinhole color."""
        # With real distortion
        unprojector_d = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        unprojector_d.unproject(sensor_depth_image)
        uvs_distorted = unprojector_d.texcoords_to_numpy()

        # With pinhole color camera (same fx/fy/cx/cy, zero distortion)
        pinhole_color = ColorProjectionParameters(
            width=sensor_color_params.width,
            height=sensor_color_params.height,
            intrinsics=CameraIntrinsics(
                fx=sensor_color_params.intrinsics.fx,
                fy=sensor_color_params.intrinsics.fy,
                cx=sensor_color_params.intrinsics.cx,
                cy=sensor_color_params.intrinsics.cy,
            ),
            depth_to_color=sensor_color_params.depth_to_color,
        )
        unprojector_p = DepthUnprojector(
            device, sensor_depth_params, pinhole_color
        )
        unprojector_p.unproject(sensor_depth_image)
        uvs_pinhole = unprojector_p.texcoords_to_numpy()

        valid_mask = sensor_depth_image > 0
        diff = np.abs(uvs_distorted[valid_mask] - uvs_pinhole[valid_mask])
        mean_diff = diff.mean()
        # Color camera has mild distortion (k1~0.075), expect small but nonzero effect
        assert mean_diff > 1e-4, (
            f"Color distortion had negligible effect (mean diff={mean_diff:.8f})"
        )

    def test_color_image_sampling_produces_plausible_colors(
        self,
        device,
        sensor_depth_params,
        sensor_depth_image,
        sensor_color_params,
        sensor_color_image,
    ):
        """Sampling the color image at GPU-computed UVs should produce varied colors.

        This is a plausibility check — not pixel-exact, but verifies the UV
        mapping produces sensible results when used to texture-map the pointcloud.
        """
        unprojector = DepthUnprojector(
            device, sensor_depth_params, sensor_color_params
        )
        unprojector.unproject(sensor_depth_image)
        uvs = unprojector.texcoords_to_numpy()

        valid_mask = sensor_depth_image > 0
        valid_uvs = uvs[valid_mask]

        # Filter to UVs inside the color image
        in_range = (
            (valid_uvs[:, 0] >= 0)
            & (valid_uvs[:, 0] < 1)
            & (valid_uvs[:, 1] >= 0)
            & (valid_uvs[:, 1] < 1)
        )
        inside_uvs = valid_uvs[in_range]

        # Convert UVs to pixel coordinates and sample
        px = (inside_uvs[:, 0] * sensor_color_image.shape[1]).astype(int)
        py = (inside_uvs[:, 1] * sensor_color_image.shape[0]).astype(int)
        px = np.clip(px, 0, sensor_color_image.shape[1] - 1)
        py = np.clip(py, 0, sensor_color_image.shape[0] - 1)

        sampled_colors = sensor_color_image[py, px]  # (N, 3)

        # Colors should be varied (not all the same)
        color_std = sampled_colors.std(axis=0).mean()
        assert color_std > 10.0, (
            f"Sampled colors lack variation (std={color_std:.1f}), "
            f"suggesting UV mapping is wrong"
        )

        # Colors should not be predominantly black
        mean_brightness = sampled_colors.mean()
        assert mean_brightness > 20.0, (
            f"Sampled colors are too dark (mean={mean_brightness:.1f}), "
            f"suggesting UV mapping is wrong"
        )


# ---------------------------------------------------------------------------
# Normal computation tests
# ---------------------------------------------------------------------------


class TestNormalComputation:
    """Tests for compute_normals kernel via DepthUnprojector."""

    def test_normal_buffer_exists(self, device):
        """DepthUnprojector should expose a normal_buffer property."""
        params = DepthParameters(
            width=8,
            height=8,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=4.0, cy=4.0),
        )
        unprojector = DepthUnprojector(device, params)
        assert unprojector.normal_buffer is not None

    def test_invalid_depth_gives_zero_normal(self, device):
        """Points with zero depth should produce zero normals."""
        params = DepthParameters(
            width=8,
            height=8,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=4.0, cy=4.0),
        )
        unprojector = DepthUnprojector(device, params)
        depth_mm = np.zeros((8, 8), dtype=np.uint16)
        unprojector.unproject(depth_mm)
        normals = unprojector.normals_to_numpy()
        np.testing.assert_allclose(normals, 0.0, atol=1e-6)

    def test_flat_plane_normals(self, device):
        """A flat plane perpendicular to the camera should have normals in +z direction."""
        params = DepthParameters(
            width=16,
            height=16,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=8.0),
        )
        unprojector = DepthUnprojector(device, params)
        depth_mm = np.full((16, 16), 2000, dtype=np.uint16)
        unprojector.unproject(depth_mm)
        normals = unprojector.normals_to_numpy()
        assert normals.shape == (16, 16, 3)
        # Interior points should have normals approximately (0, 0, 1) — surface outward normal
        # cross(right, down) for fronto-parallel: right=(+,0,0), down=(0,+,0) -> cross = (0,0,+)
        interior = normals[2:-2, 2:-2]
        valid = np.linalg.norm(interior, axis=-1) > 0.5
        assert valid.all(), "All interior points should have valid normals"
        norms = interior / np.linalg.norm(interior, axis=-1, keepdims=True)
        np.testing.assert_allclose(norms[..., 0], 0.0, atol=0.02)
        np.testing.assert_allclose(norms[..., 1], 0.0, atol=0.02)
        np.testing.assert_allclose(norms[..., 2], 1.0, atol=0.02)

    def test_normals_shape_matches_pointcloud(self, device):
        """Normal buffer should have the same dimensions as position buffer."""
        params = DepthParameters(
            width=32,
            height=24,
            intrinsics=CameraIntrinsics(fx=200.0, fy=200.0, cx=16.0, cy=12.0),
        )
        unprojector = DepthUnprojector(device, params)
        depth_mm = np.full((24, 32), 1500, dtype=np.uint16)
        unprojector.unproject(depth_mm)
        normals = unprojector.normals_to_numpy()
        positions = unprojector.to_numpy()
        assert normals.shape == positions.shape
