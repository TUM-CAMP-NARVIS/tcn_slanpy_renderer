# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`slangpy-renderer` (import name: `slangpy_renderer`) is a pure-Python 3D rendering library built on SlangPy (Vulkan backend). It provides interactive windowed rendering, headless offscreen rendering, and optional CUDA/Vulkan interop for streaming data pipelines. 

## Commands

```bash
# Activate the development venv
source /home/narvis/develop/rendering/.venv-renderer/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_arcball_rendering.py -v

# Run a single test
pytest tests/test_pipeline_validation.py::TestPixelValidation::test_cube_front_face_normal_color -v

# Run with log output (useful for rendering diagnostics)
pytest tests/test_arcball_rendering.py -v -s --log-cli-level=INFO

# Run pipeline validation with report
./scripts/validate_pipeline.sh
```

## Architecture

**Renderable/Renderer separation** is the core pattern: renderables own geometry + GPU buffers, renderers are stateless and shared across renderables of the same type.

```
User code  ──►  Renderable.update()  ──►  Renderable.sync_gpu()  ──►  Renderer.render()
              (any thread, stages data)  (render thread, uploads)    (render thread, draws)
```

**Renderables** (`renderables/`): `Mesh`, `Pointcloud`, `ColoredMesh` — thread-safe `update()`, lazy `sync_gpu()`, 4x4 `pose` matrix.

**Renderers** (`renderers/`): `MeshRenderer` (phong.slang), `PointcloudRenderer`, `PointcloudSpritesRenderer`, `ColoredMeshRenderer` (color_drawable.slang, line_list topology). Each is initialized with `(device, output_format)` and creates a Vulkan render pipeline.

**Contexts**: `SlangWindow` (interactive, requires CuPy for CUDA interop) and `OffscreenContext` (headless, no CuPy needed). Both create all 4 renderer types, manage a scene graph, and render in a single pass.

**Controllers** (`controllers/`): `ArcBall` (quaternion-based orbital camera) and `FirstPersonView`.

**Shaders** (`assets/shaders/`): Slang shaders compiled at device creation via `compiler_options["include_paths"]`. `phong.slang` supports `renderStaticColor=True` debug mode that encodes world-space normals as colors: `0.5*(N+1)`.

## Vulkan/Slang Conventions

These are verified by the test suite — do not change them without updating tests:

- **Row-major matrices**: Slang `mul(M, v)` = numpy `M @ v`
- **Shader transform chain**: `clip = mul(proj, mul(view, mul(model, vertex)))`
- **Projection**: `vulkan_rh_zo_perspective()` in `offscreen.py` — right-handed, depth [0,1], `P[1,1] > 0` (no Y-flip)
- **Y axis**: world Y+ maps to screen bottom (Vulkan native Y-down NDC)
- **Depth**: Vulkan ZO [0,1], near=0, far=1, closer objects have smaller depth
- **Viewport**: `Viewport.from_size(w, h)` = positive height, no flip

## Critical Implementation Details

**Depth format must be specified in render pipelines.** Every `create_render_pipeline()` call with depth testing must include `"format": spy.Format.d32_float` in the `depth_stencil` dict. Without it, SlangPy defaults to `Format::undefined`, which silently disables depth testing in Vulkan. All 4 renderers have this set correctly — do not remove it.

```python
depth_stencil={
    "format": spy.Format.d32_float,  # REQUIRED for depth testing to work
    "depth_test_enable": True,
    "depth_write_enable": True,
    "depth_func": spy.ComparisonFunc.less,
},
```

**CuPy is optional.** `SlangWindow` requires CuPy (CUDA/Vulkan interop), but `OffscreenContext` and the examples in `examples/` work without it. CuPy imports are guarded in `window.py`, `pointcloud.py`, and `cuda_helpers.py`.

**Mesh loading** uses trimesh. `Mesh.from_obj(device, path)` loads vertices, normals, UVs, and texture. The unit cube is at `assets/models/cube.obj` (vertices in [-0.5, 0.5]).

## Testing Patterns

- `conftest.py` provides session-scoped `offscreen_ctx` (256x256), `assets_path`, `view_matrix`, `proj_matrix` fixtures
- Use `OffscreenContext` for headless rendering tests, NOT `SlangWindow`
- Parametrize by renderable type for rendering tests (see `test_offscreen_rendering.py`)
- Use `renderStaticColor=True` to get normal-encoded face colors for pixel validation
- `look_at()` and `vulkan_rh_zo_perspective()` from `slangpy_renderer.offscreen` are the canonical matrix functions
- Call `ctx.clear()` between unrelated renders; `ctx.remove_renderable(name)` to clean up

## File Layout

```
slangpy_renderer/
  __init__.py          # All public exports
  window.py            # SlangWindow (interactive, needs CuPy)
  offscreen.py         # OffscreenContext (headless) + look_at() + vulkan_rh_zo_perspective()
  renderables/         # Mesh, Pointcloud, ColoredMesh
  renderers/           # MeshRenderer, PointcloudRenderer, etc.
  controllers/         # ArcBall, FirstPersonView
  utils/               # cuda_helpers.py
  debug/               # RenderDoc integration (optional, renderdoc_api.py + capture_analysis.py)
  assets/shaders/      # 6 Slang shaders (phong, pointcloud, color_drawable, etc.)
  assets/models/       # cube.obj, monkey.obj, textures
tests/                 # pytest suite (89 tests across 4 modules)
examples/              # view_cube.py, show_axis3d.py, capture_cube.py
```
