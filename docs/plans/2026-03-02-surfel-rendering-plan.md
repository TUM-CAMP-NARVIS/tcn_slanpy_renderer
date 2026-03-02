# Surfel Rendering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add normal-oriented, textured hexagonal surfel rendering for structured depth pointclouds, with normals computed in the depth-unproject compute shader.

**Architecture:** Three new files (camera_math.slang, pointcloud_surfels.slang, pointcloud_surfel_renderer.py), modifications to depth_unproject.slang, depth_unprojector.py, and view_depth_pointcloud.py. Shared camera math is extracted to a common include. Normals are computed as a third compute kernel dispatched after pointcloud generation. The surfel geometry shader reads positions + normals + color camera params to emit oriented hexagons with per-vertex UV projection.

**Tech Stack:** Slang (GPU shaders), SlangPy (Python GPU framework), Vulkan rendering pipeline, NumPy, pytest.

---

### Task 1: Extract shared camera math to `camera_math.slang`

**Files:**
- Create: `slangpy_renderer/assets/shaders/camera_math.slang`
- Modify: `slangpy_renderer/assets/shaders/depth_unproject.slang`

**Step 1: Create `camera_math.slang` with shared types and functions**

Extract from `depth_unproject.slang` (lines 13-176) into a new file:

```slang
// camera_math.slang — Shared camera types and projection functions.
//
// Provides CameraIntrinsics, ColorProjectionParams structs and the
// project_forward() Brown-Conrady forward projection function.
// Used by both depth_unproject.slang and pointcloud_surfels.slang.

#pragma once

struct CameraIntrinsics
{
    float fx, fy;               // focal length (pixels)
    float cx, cy;               // principal point (pixels)
    float k1, k2, k3;          // radial distortion (numerator)
    float k4, k5, k6;          // radial distortion (denominator)
    float p1, p2;              // tangential distortion
    float max_radius;          // max normalized radius for valid projection (~1.7)
};

struct ColorProjectionParams
{
    CameraIntrinsics intrinsics;
    uint width;
    uint height;
    float4x4 depth_to_color;   // rigid transform: depth camera -> color camera space
};

struct ForwardProjectResult
{
    float2 uv;
    bool valid;
};

ForwardProjectResult project_forward(CameraIntrinsics cam, float2 xy)
{
    ForwardProjectResult result;
    result.valid = true;

    float xp = xy.x;
    float yp = xy.y;

    float xp2 = xp * xp;
    float yp2 = yp * yp;
    float xyp = xp * yp;
    float rs = xp2 + yp2;

    if (rs > cam.max_radius * cam.max_radius)
    {
        result.valid = false;
        result.uv = float2(0.f, 0.f);
        return result;
    }

    float rss = rs * rs;
    float rsc = rss * rs;
    float a = 1.f + cam.k1 * rs + cam.k2 * rss + cam.k3 * rsc;
    float b = 1.f + cam.k4 * rs + cam.k5 * rss + cam.k6 * rsc;
    float bi = (b != 0.f) ? (1.f / b) : 1.f;
    float d = a * bi;

    float xp_d = xp * d;
    float yp_d = yp * d;

    float rs_2xp2 = rs + 2.f * xp2;
    float rs_2yp2 = rs + 2.f * yp2;

    xp_d += rs_2xp2 * cam.p2 + 2.f * xyp * cam.p1;
    yp_d += rs_2yp2 * cam.p1 + 2.f * xyp * cam.p2;

    result.uv = float2(xp_d * cam.fx + cam.cx, yp_d * cam.fy + cam.cy);
    return result;
}
```

**Step 2: Update `depth_unproject.slang` to import shared code**

Replace the duplicated structs and `project_forward()` at lines 13-176 with an import. Keep the `DepthParameters` struct, `project_internal()` (with Jacobian), and `iterative_unproject()`/`unproject_pixel()` locally since they're only used by the depth unproject kernels.

At the top of `depth_unproject.slang` (after the header comment), add:

```slang
#include "camera_math.slang"
```

Then remove:
- The `CameraIntrinsics` struct definition (lines 17-25) — now from include
- The `ColorProjectionParams` struct definition (lines 35-41) — now from include
- The `ForwardProjectResult` struct and `project_forward()` function (lines 127-176) — now from include

Keep locally:
- `DepthParameters` struct (lines 27-33) — references `CameraIntrinsics` from the include
- `project_internal()` and everything below it — only used by depth unproject kernels

**Step 3: Verify the existing tests still pass**

Run: `cd /home/narvis/develop/rendering && source .venv-renderer/bin/activate && pytest tcn_slangpy_renderer/tests/test_depth_unprojection.py -v`

Expected: All 45 tests PASS (no behavior change, just code reorganization).

**Step 4: Commit**

```bash
cd /home/narvis/develop/rendering/tcn_slangpy_renderer
git add slangpy_renderer/assets/shaders/camera_math.slang slangpy_renderer/assets/shaders/depth_unproject.slang
git commit -m "refactor: extract shared camera math to camera_math.slang

Extract CameraIntrinsics, ColorProjectionParams, and project_forward()
into a shared include file so both depth_unproject.slang and the
upcoming pointcloud_surfels.slang can use them."
```

---

### Task 2: Add `compute_normals` kernel to `depth_unproject.slang`

**Files:**
- Modify: `slangpy_renderer/assets/shaders/depth_unproject.slang` (append kernel at end)

**Step 1: Write the test for normal computation**

Create test code in the existing test file. We'll test with a synthetic flat plane where normals should all point in the same direction.

File: `tests/test_depth_unprojection.py` — add at end:

```python
# ---------------------------------------------------------------------------
# Normal computation tests
# ---------------------------------------------------------------------------


class TestNormalComputation:
    """Tests for compute_normals kernel via DepthUnprojector."""

    def test_flat_plane_normals(self, device):
        """A flat plane perpendicular to the camera should have normals pointing back at the camera."""
        # Create a pinhole camera (no distortion), 16x16
        params = DepthParameters(
            width=16,
            height=16,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=8.0),
        )
        unprojector = DepthUnprojector(device, params)

        # Create a depth image where every pixel is at 2.0 meters
        depth_mm = np.full((16, 16), 2000, dtype=np.uint16)
        unprojector.unproject(depth_mm)

        normals = unprojector.normals_to_numpy()
        assert normals.shape == (16, 16, 3)

        # Interior points (away from boundaries) should have normals approximately (0, 0, -1)
        # pointing back at the camera (z-forward convention, normal faces viewer)
        # Actually: cross(right, down) with x-right, y-down, z-forward:
        # right vector ~ (+dx, 0, 0), down vector ~ (0, +dy, 0)
        # cross(right, down) = (0, 0, dx*dy) — wait, need to think about sign...
        # For a fronto-parallel plane at depth z:
        #   P(i,j) = ((i - cx)/fx * z, (j - cy)/fy * z, z)
        #   right = P(i+1,j) - P(i-1,j) = (2*z/fx, 0, 0)
        #   down  = P(i,j+1) - P(i,j-1) = (0, 2*z/fy, 0)
        #   cross(right, down) = (0, 0, 4*z^2/(fx*fy)) — points in +z (away from camera)
        # We want normals facing the camera, so we should negate: cross(down, right) or check sign.
        # The shader uses cross(right, down) which gives +z for a fronto-parallel plane.
        # In our convention (z-forward = into screen), +z normal means "pointing away from camera".
        # This is the outward surface normal for a surface facing the camera — which is correct
        # for the surfel use case (the normal defines the surface orientation, not the viewing direction).
        interior = normals[2:-2, 2:-2]
        valid = np.linalg.norm(interior, axis=-1) > 0.5
        assert valid.all(), "All interior points should have valid normals"

        # All normals should point in +z (into screen, away from camera = surface outward normal)
        # Normalize them and check
        norms = interior / np.linalg.norm(interior, axis=-1, keepdims=True)
        np.testing.assert_allclose(norms[..., 0], 0.0, atol=0.01)
        np.testing.assert_allclose(norms[..., 1], 0.0, atol=0.01)
        np.testing.assert_allclose(norms[..., 2], 1.0, atol=0.01)

    def test_invalid_depth_gives_zero_normal(self, device):
        """Points with zero depth should produce zero normals."""
        params = DepthParameters(
            width=8,
            height=8,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=4.0, cy=4.0),
        )
        unprojector = DepthUnprojector(device, params)

        # All zeros → all normals should be zero
        depth_mm = np.zeros((8, 8), dtype=np.uint16)
        unprojector.unproject(depth_mm)

        normals = unprojector.normals_to_numpy()
        np.testing.assert_allclose(normals, 0.0, atol=1e-6)

    def test_normal_buffer_exists(self, device):
        """DepthUnprojector should expose a normal_buffer property."""
        params = DepthParameters(
            width=8,
            height=8,
            intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=4.0, cy=4.0),
        )
        unprojector = DepthUnprojector(device, params)
        assert unprojector.normal_buffer is not None

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tcn_slangpy_renderer/tests/test_depth_unprojection.py::TestNormalComputation -v`

Expected: FAIL — `DepthUnprojector` has no `normal_buffer` property or `normals_to_numpy()` method yet.

**Step 3: Add the `compute_normals` kernel to `depth_unproject.slang`**

Append at the end of `depth_unproject.slang` (after `compute_pointcloud`):

```slang
// ============================================================================
// Kernel 3: Compute Normals from Structured Pointcloud
// Uses central-difference cross-product on the depth image grid.
// Dispatched after compute_pointcloud. Reads from pointcloud buffer.
// ============================================================================

uniform RWStructuredBuffer<float3> normals;    // normal output

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_normals(uint3 threadId : SV_DispatchThreadID)
{
    uint px = threadId.x;
    uint py = threadId.y;

    if (px >= depth_params.width || py >= depth_params.height)
        return;

    uint idx = py * depth_params.width + px;
    float3 center = pointcloud[idx];

    // Invalid point: zero normal
    if (center.z == 0.f)
    {
        normals[idx] = float3(0.f, 0.f, 0.f);
        return;
    }

    // Compute finite differences using central differences (one-sided at boundaries)
    float3 dx, dy;
    bool has_dx = false;
    bool has_dy = false;

    // Horizontal difference
    if (px > 0 && px < depth_params.width - 1)
    {
        float3 left  = pointcloud[idx - 1];
        float3 right = pointcloud[idx + 1];
        if (left.z > 0.f && right.z > 0.f)
        {
            dx = right - left;
            has_dx = true;
        }
    }
    if (!has_dx && px < depth_params.width - 1)
    {
        float3 right = pointcloud[idx + 1];
        if (right.z > 0.f && center.z > 0.f)
        {
            dx = right - center;
            has_dx = true;
        }
    }
    if (!has_dx && px > 0)
    {
        float3 left = pointcloud[idx - 1];
        if (left.z > 0.f && center.z > 0.f)
        {
            dx = center - left;
            has_dx = true;
        }
    }

    // Vertical difference
    if (py > 0 && py < depth_params.height - 1)
    {
        float3 up   = pointcloud[idx - depth_params.width];
        float3 down = pointcloud[idx + depth_params.width];
        if (up.z > 0.f && down.z > 0.f)
        {
            dy = down - up;
            has_dy = true;
        }
    }
    if (!has_dy && py < depth_params.height - 1)
    {
        float3 down = pointcloud[idx + depth_params.width];
        if (down.z > 0.f && center.z > 0.f)
        {
            dy = down - center;
            has_dy = true;
        }
    }
    if (!has_dy && py > 0)
    {
        float3 up = pointcloud[idx - depth_params.width];
        if (up.z > 0.f && center.z > 0.f)
        {
            dy = center - up;
            has_dy = true;
        }
    }

    if (!has_dx || !has_dy)
    {
        normals[idx] = float3(0.f, 0.f, 0.f);
        return;
    }

    float3 n = cross(dx, dy);
    float len = length(n);

    if (len < 1e-10f)
    {
        normals[idx] = float3(0.f, 0.f, 0.f);
        return;
    }

    normals[idx] = n / len;
}
```

Note: `cross(dx, dy)` where dx points right (+x) and dy points down (+y) gives a normal in the +z direction for a fronto-parallel surface. This is the outward surface normal (pointing away from the camera toward the scene), which is the standard convention for surfel orientation.

**Step 4: Update `DepthUnprojector` in `depth_unprojector.py`**

In `__init__()`, after the `m_pc_kernel` compilation (line 278), add:

```python
        normals_program = device.load_program(
            "depth_unproject.slang", ["compute_normals"]
        )
        self.m_normals_kernel = device.create_compute_kernel(normals_program)
```

After `m_texcoord_buffer` creation (line 303), add:

```python
        # Normal output: float3 per pixel (12 bytes per element)
        self.m_normal_buffer = device.create_buffer(
            element_count=w * h,
            struct_size=12,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )
```

Add a new method `_dispatch_normals()` after `_dispatch_pointcloud()`:

```python
    def _dispatch_normals(self) -> None:
        """Compute per-point normals from the structured pointcloud."""
        w, h = self.m_params.width, self.m_params.height
        groups_x = _group_count(w, _GROUP_SIZE)
        groups_y = _group_count(h, _GROUP_SIZE)

        command_encoder = self.m_device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_enc:
            shader_obj = pass_enc.bind_pipeline(self.m_normals_kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj)
            _bind_depth_params(cursor.depth_params, self.m_params)
            cursor.pointcloud = self.m_position_buffer
            cursor.normals = self.m_normal_buffer
            pass_enc.dispatch_compute(spy.uint3(groups_x, groups_y, 1))
        self.m_device.submit_command_buffer(command_encoder.finish())
        self.m_device.wait()
```

In `unproject()` (line 379), after `self._dispatch_pointcloud()`, add:

```python
        self._dispatch_normals()
```

Add new property and readback method after `texcoord_buffer` property:

```python
    @property
    def normal_buffer(self) -> spy.Buffer:
        """The output normal buffer (stable reference, reused every frame)."""
        return self.m_normal_buffer

    def normals_to_numpy(self) -> np.ndarray:
        """
        Read back the normal buffer as a numpy array.

        Returns:
            (H, W, 3) float32 numpy array of per-point normal vectors.
        """
        raw = self.m_normal_buffer.to_numpy()
        return raw.view(np.float32).reshape(
            self.m_params.height, self.m_params.width, 3
        )
```

**Step 5: Run tests to verify they pass**

Run: `pytest tcn_slangpy_renderer/tests/test_depth_unprojection.py -v`

Expected: All tests PASS (old + new).

**Step 6: Commit**

```bash
cd /home/narvis/develop/rendering/tcn_slangpy_renderer
git add slangpy_renderer/assets/shaders/depth_unproject.slang slangpy_renderer/utils/depth_unprojector.py tests/test_depth_unprojection.py
git commit -m "feat: add compute_normals kernel for structured pointcloud normals

Central-difference cross-product on the depth image grid. Handles
boundaries with one-sided differences and produces zero normals for
invalid (zero-depth) points. Dispatched automatically after
compute_pointcloud in DepthUnprojector.unproject()."
```

---

### Task 3: Create `pointcloud_surfels.slang` shader

**Files:**
- Create: `slangpy_renderer/assets/shaders/pointcloud_surfels.slang`

**Step 1: Write the surfel shader**

```slang
// pointcloud_surfels.slang — Normal-oriented textured hexagonal surfel rendering.
//
// Geometry shader expands each point into a hexagonal surfel oriented by the
// point's normal vector. Each surfel vertex gets its UV by projecting the 3D
// position through the color camera model (Brown-Conrady distortion).

import "slangpy";
#include "camera_math.slang"

// --- Uniforms ---
uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

// Point cloud data buffers
uniform StructuredBuffer<float3> vertices;
uniform StructuredBuffer<float3> normals;
uniform StructuredBuffer<float2> uvCoords;  // center UVs (fallback, from compute shader)

uniform Texture2D<float4> colorTex;
uniform SamplerState sampler_colorTex;

// Color camera parameters for per-vertex UV projection
uniform ColorProjectionParams color_camera;
uniform bool has_color_projection;

// Sprite sizing
uniform float depth_fy;       // depth camera fy (pixels) for auto-sizing
uniform float sprite_scale;   // multiplier on auto-computed radius (default 1.5)

// Depth image dimensions for structured grid indexing
uniform int depthWidth;
uniform int depthHeight;

// Debug: render with static color instead of texture
uniform bool useStaticColor;
uniform float4 staticColor;

// --- Inter-stage structures ---

struct VertexToGeometry {
    uint vid : VERTEXID;
};

struct GeometryToFragment {
    float2 uv     : TEXCOORD0;
    float4 pos    : SV_Position;
};

// --- Vertex shader ---

[shader("vertex")]
VertexToGeometry vertex_main(uint vid : SV_VertexID) {
    VertexToGeometry output;
    output.vid = vid;
    return output;
}

// --- Helper: transform to clip space ---

float4 world_to_clip(float3 world_pos) {
    float4 wp = float4(world_pos, 1.0);
    return mul(proj, mul(view, mul(model, wp)));
}

// --- Helper: project 3D point to color camera UV ---

float2 project_to_color_uv(float3 depth_pos) {
    // Transform from depth camera space to color camera space
    float4 p_color4 = mul(color_camera.depth_to_color, float4(depth_pos, 1.0));
    float3 p_color = p_color4.xyz;

    if (p_color.z <= 0.f)
        return float2(-1.f, -1.f);

    // Perspective divide
    float2 xy_norm = float2(p_color.x / p_color.z, p_color.y / p_color.z);

    // Forward project through distortion model
    ForwardProjectResult proj_result = project_forward(color_camera.intrinsics, xy_norm);

    if (!proj_result.valid)
        return float2(-1.f, -1.f);

    // Normalize to [0, 1]
    return proj_result.uv / float2((float)color_camera.width, (float)color_camera.height);
}

// --- Constants for hexagon ---

static const int HEXAGON_VERTS = 6;
// Precomputed cos/sin for 60-degree increments (0, 60, 120, 180, 240, 300)
static const float2 hex_offsets[6] = {
    float2( 1.0,       0.0),         // 0°
    float2( 0.5,       0.8660254),   // 60°
    float2(-0.5,       0.8660254),   // 120°
    float2(-1.0,       0.0),         // 180°
    float2(-0.5,      -0.8660254),   // 240°
    float2( 0.5,      -0.8660254),   // 300°
};

// --- Geometry shader ---

[shader("geometry")]
[maxvertexcount(18)]
void geometry_main(
    point VertexToGeometry input[1],
    inout TriangleStream<GeometryToFragment> outStream)
{
    uint idx = input[0].vid;
    float3 pos = vertices[idx];
    float3 normal = normals[idx];

    // Skip invalid points (zero position or zero normal)
    if (pos.z <= 0.f || length(normal) < 0.5f)
        return;

    // Auto-compute sprite radius from depth and camera intrinsics
    // At depth z, each pixel covers z/fy meters vertically.
    // sprite_scale controls overlap (1.5 = slight overlap to fill gaps).
    float radius = pos.z * (1.0f / depth_fy) * sprite_scale;

    // Build tangent frame from normal
    float3 N = normalize(normal);

    // Choose reference vector not parallel to N
    float3 ref = (abs(N.y) < 0.9f) ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
    float3 T = normalize(cross(ref, N));
    float3 B = cross(N, T);

    // Compute center UV
    float2 center_uv;
    if (has_color_projection)
        center_uv = project_to_color_uv(pos);
    else
        center_uv = uvCoords[idx];

    // Compute hexagon vertex positions and UVs
    float3 hex_world[6];
    float2 hex_uv[6];

    for (int i = 0; i < HEXAGON_VERTS; i++)
    {
        hex_world[i] = pos + radius * (hex_offsets[i].x * T + hex_offsets[i].y * B);

        if (has_color_projection)
            hex_uv[i] = project_to_color_uv(hex_world[i]);
        else
            hex_uv[i] = center_uv;  // fallback: same UV for all vertices
    }

    float4 center_clip = world_to_clip(pos);

    // Emit hexagon as triangle fan via triangle strips:
    // 6 triangles: center-v0-v1, center-v1-v2, ..., center-v5-v0
    for (int i = 0; i < HEXAGON_VERTS; i++)
    {
        int next = (i + 1) % HEXAGON_VERTS;

        GeometryToFragment out_v;

        // Center vertex
        out_v.pos = center_clip;
        out_v.uv = center_uv;
        outStream.Append(out_v);

        // Current hex vertex
        out_v.pos = world_to_clip(hex_world[i]);
        out_v.uv = hex_uv[i];
        outStream.Append(out_v);

        // Next hex vertex
        out_v.pos = world_to_clip(hex_world[next]);
        out_v.uv = hex_uv[next];
        outStream.Append(out_v);

        outStream.RestartStrip();
    }
}

// --- Fragment shader ---

[shader("fragment")]
float4 fragment_main(GeometryToFragment input) : SV_Target {
    // Discard invalid UVs
    if (input.uv.x < 0.f || input.uv.x > 1.f || input.uv.y < 0.f || input.uv.y > 1.f)
        discard;

    if (useStaticColor)
        return staticColor;

    float4 c = colorTex.Sample(sampler_colorTex, input.uv);
    c.a = 1.0f;
    return c;
}
```

**Step 2: Commit**

```bash
cd /home/narvis/develop/rendering/tcn_slangpy_renderer
git add slangpy_renderer/assets/shaders/pointcloud_surfels.slang
git commit -m "feat: add pointcloud_surfels.slang shader

Normal-oriented hexagonal surfel geometry shader with per-vertex
UV projection through the color camera's Brown-Conrady distortion
model. Auto-sizes surfels from depth and camera intrinsics."
```

---

### Task 4: Create `PointcloudSurfelRenderer`

**Files:**
- Create: `slangpy_renderer/renderers/pointcloud_surfel_renderer.py`
- Modify: `slangpy_renderer/renderers/__init__.py` (line 9)
- Modify: `slangpy_renderer/__init__.py` (lines 53-57, 89-93)

**Step 1: Write the renderer**

```python
import slangpy as spy
import numpy as np
import logging

from ..renderables.pointcloud import Pointcloud
from ..utils.depth_unprojector import (
    ColorProjectionParameters,
    _bind_intrinsics,
    _bind_color_projection_params,
)

log = logging.getLogger(__name__)


class PointcloudSurfelRenderer:
    """
    Renderer for normal-oriented textured hexagonal surfels.

    Uses a geometry shader to expand each point into a hexagonal surfel
    oriented by the point's normal vector. Each surfel vertex gets a UV
    by projecting its 3D position through the color camera model.
    """

    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.device = device
        self.program = device.load_program(
            "pointcloud_surfels.slang",
            ["vertex_main", "geometry_main", "fragment_main"],
        )

        self.sampler = device.create_sampler()

        self.pipeline = device.create_render_pipeline(
            program=self.program,
            input_layout=None,
            targets=[{"format": output_format}],
            primitive_topology=spy.PrimitiveTopology.triangle_strip,
            depth_stencil={
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
        )

    def render(
        self,
        pass_encoder: spy.RenderPassEncoder,
        pointcloud: Pointcloud,
        window_size: tuple[int, int],
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        model_matrix: np.ndarray,
        extra_args: dict = None,
    ):
        """
        Render a pointcloud as normal-oriented textured hexagonal surfels.

        Args:
            pass_encoder: Active render pass encoder
            pointcloud: Pointcloud with position_buffer, normal_buffer, and texture
            window_size: (width, height) tuple
            view_matrix: Camera view matrix (4x4)
            proj_matrix: Camera projection matrix (4x4)
            model_matrix: Object pose/model matrix (4x4)
            extra_args: Optional dict with keys:
                - color_params: ColorProjectionParameters (for per-vertex UV projection)
                - depth_fy: float (depth camera fy for auto-sizing)
                - sprite_scale: float (default 1.5, controls surfel overlap)
                - useStaticColor: bool (debug: render with flat color)
                - staticColor: np.ndarray (float4, default white)
                - depthWidth: int
                - depthHeight: int
        """
        if not pointcloud.has_vertices:
            log.debug("Pointcloud has no vertices, skipping surfel render")
            return

        if not pointcloud.has_normals:
            log.debug("Pointcloud has no normals, skipping surfel render")
            return

        extra_args = extra_args or {}

        shader_object = pass_encoder.bind_pipeline(self.pipeline)
        cursor = spy.ShaderCursor(shader_object)

        # Transformation matrices
        cursor.proj = proj_matrix
        cursor.view = view_matrix
        cursor.model = model_matrix

        # Structured buffers
        cursor.vertices = pointcloud.position_buffer
        cursor.normals = pointcloud.normal_buffer
        if pointcloud.has_texcoords:
            cursor.uvCoords = pointcloud.uv_buffer

        # Texture + sampler
        if pointcloud.has_texture:
            cursor.colorTex = pointcloud.texture
            cursor.sampler_colorTex = self.sampler

        # Color camera parameters for per-vertex UV projection
        color_params = extra_args.get("color_params")
        if color_params is not None:
            cursor.has_color_projection = True
            _bind_color_projection_params(cursor.color_camera, color_params)
        else:
            cursor.has_color_projection = False
            _bind_color_projection_params(cursor.color_camera, None)

        # Sprite sizing
        cursor.depth_fy = extra_args.get("depth_fy", 500.0)
        cursor.sprite_scale = extra_args.get("sprite_scale", 1.5)

        # Depth dimensions for structured grid
        cursor.depthWidth = extra_args.get("depthWidth", 0)
        cursor.depthHeight = extra_args.get("depthHeight", 0)

        # Debug options
        cursor.useStaticColor = extra_args.get("useStaticColor", False)
        cursor.staticColor = extra_args.get(
            "staticColor", np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )

        # Render state
        pass_encoder.set_render_state(
            {
                "viewports": [spy.Viewport.from_size(*window_size)],
                "scissor_rects": [spy.ScissorRect.from_size(*window_size)],
            }
        )

        # Draw: one vertex per point, geometry shader expands to hexagon
        vertex_count = (
            pointcloud.vertices.size
            if hasattr(pointcloud.vertices, "size")
            else len(pointcloud.vertices)
        )
        pass_encoder.draw({"vertex_count": vertex_count})
```

**Step 2: Update `renderers/__init__.py`**

Add at line 5 (after existing imports):

```python
from .pointcloud_surfel_renderer import PointcloudSurfelRenderer
```

Add `'PointcloudSurfelRenderer'` to the `__all__` list.

**Step 3: Update `slangpy_renderer/__init__.py`**

Add `PointcloudSurfelRenderer` to the import from `.renderers` and to `__all__`.

**Step 4: Commit**

```bash
cd /home/narvis/develop/rendering/tcn_slangpy_renderer
git add slangpy_renderer/renderers/pointcloud_surfel_renderer.py slangpy_renderer/renderers/__init__.py slangpy_renderer/__init__.py
git commit -m "feat: add PointcloudSurfelRenderer

Python renderer that binds the pointcloud_surfels.slang pipeline.
Passes positions, normals, color camera params, and auto-sizing
parameters to the geometry shader."
```

---

### Task 5: Update `view_depth_pointcloud.py` to use surfel renderer

**Files:**
- Modify: `examples/view_depth_pointcloud.py`

**Step 1: Update the example**

Changes needed:
1. Import `PointcloudSurfelRenderer` instead of `PointcloudSpritesRenderer` (line 42)
2. Pass `normal_buffer` from unprojector to pointcloud (after line 156)
3. Create `PointcloudSurfelRenderer` instead of `PointcloudSpritesRenderer` (line 177)
4. Pass `color_params` and `depth_fy` in extra_args (lines 305-307)
5. Add `--sprite-scale` CLI argument (after line 98)

Update import (line 42):
```python
from slangpy_renderer.renderers import PointcloudSurfelRenderer
```

After `pointcloud.uv_buffer = unprojector.texcoord_buffer` (line 156), add:
```python
    pointcloud.normal_buffer = unprojector.normal_buffer
```

Replace renderer creation (line 177):
```python
    renderer = PointcloudSurfelRenderer(device, output_format)
```

Update extra_args in render call (lines 305-307):
```python
            extra_args={
                "color_params": color_params,
                "depth_fy": depth_params.intrinsics.fy,
                "sprite_scale": args.sprite_scale,
                "depthWidth": dw,
                "depthHeight": dh,
            },
```

Add CLI argument (after `--point-size` arg):
```python
    parser.add_argument(
        "--sprite-scale", type=float, default=1.5,
        help="Surfel size multiplier (1.5 = slight overlap)"
    )
```

Remove the `--point-size` argument since it's replaced by `--sprite-scale`.

**Step 2: Test manually**

Run: `cd /home/narvis/develop/rendering && source .venv-renderer/bin/activate && python tcn_slangpy_renderer/examples/view_depth_pointcloud.py --sprite-scale 1.5`

Expected: Interactive pointcloud viewer opens showing textured hexagonal surfels oriented by surface normals.

**Step 3: Commit**

```bash
cd /home/narvis/develop/rendering/tcn_slangpy_renderer
git add examples/view_depth_pointcloud.py
git commit -m "feat: update view_depth_pointcloud to use surfel renderer

Switches from billboard sprites to normal-oriented textured hexagonal
surfels. Passes normals from depth unprojector and color camera params
for per-vertex UV projection in the geometry shader."
```

---

### Task 6: Run full test suite and fix any issues

**Step 1: Run all tests**

Run: `cd /home/narvis/develop/rendering && source .venv-renderer/bin/activate && pytest tcn_slangpy_renderer/tests/ -v`

Expected: All existing tests PASS, new normal computation tests PASS.

**Step 2: Fix any failures**

If the `camera_math.slang` refactor broke shader compilation for existing tests, check the `#include` path resolution and ensure the include is found via the `include_paths` compiler option.

**Step 3: Final commit (if fixes needed)**

Only if Step 2 required changes.
