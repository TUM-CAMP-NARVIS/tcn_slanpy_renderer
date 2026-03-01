"""
Tests for package imports and basic object construction.
"""
import pytest


def test_import_all_public_classes():
    """All public symbols should be importable from the top-level package."""
    from slangpy_renderer import (
        SlangWindow,
        Pointcloud,
        Mesh,
        ColoredMesh,
        PointcloudRenderer,
        PointcloudSpritesRenderer,
        MeshRenderer,
        ColoredMeshRenderer,
        ArcBall,
        FirstPersonView,
        OffscreenContext,
        copy_cupy_array_into_slangpy_buffer,
        CameraIntrinsics,
        ColorProjectionParameters,
        DepthParameters,
        DepthUnprojector,
    )
    assert callable(SlangWindow)
    assert callable(Pointcloud)
    assert callable(Mesh)
    assert callable(ColoredMesh)
    assert callable(PointcloudRenderer)
    assert callable(PointcloudSpritesRenderer)
    assert callable(MeshRenderer)
    assert callable(ColoredMeshRenderer)
    assert callable(ArcBall)
    assert callable(FirstPersonView)
    assert callable(OffscreenContext)
    assert callable(copy_cupy_array_into_slangpy_buffer)
    assert callable(CameraIntrinsics)
    assert callable(ColorProjectionParameters)
    assert callable(DepthParameters)
    assert callable(DepthUnprojector)


def test_version():
    """Package version should be defined."""
    from slangpy_renderer import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_create_offscreen_context():
    """OffscreenContext should create a Vulkan device and render targets."""
    from slangpy_renderer import OffscreenContext

    ctx = OffscreenContext(width=64, height=64)
    assert ctx.device is not None
    assert ctx.color_texture is not None
    assert ctx.depth_texture is not None
    assert ctx.width == 64
    assert ctx.height == 64


def test_create_colored_mesh(offscreen_ctx):
    """ColoredMesh.create_axis3d should produce a renderable with geometry."""
    from slangpy_renderer import ColoredMesh

    axes = ColoredMesh.create_axis3d(offscreen_ctx.device, scale=2.0)
    assert axes.has_geometry
    assert axes.position_buffer is not None
    assert axes.color_buffer is not None
    assert axes.index_buffer is not None


def test_create_mesh_from_data(offscreen_ctx):
    """Mesh constructed from synthetic numpy data should have GPU buffers."""
    import numpy as np
    from slangpy_renderer import Mesh

    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
    )
    indices = np.array([0, 1, 2], dtype=np.uint16)
    normals = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32
    )

    mesh = Mesh(
        device=offscreen_ctx.device,
        positions=positions,
        indices=indices,
        normals=normals,
        sync_gpu=True,
    )
    assert mesh.position_buffer is not None
    assert mesh.index_buffer is not None
    assert mesh.normal_buffer is not None


def test_create_renderers(offscreen_ctx):
    """All four renderer types should be constructable."""
    assert offscreen_ctx.mesh_renderer is not None
    assert offscreen_ctx.pointcloud_renderer is not None
    assert offscreen_ctx.pointcloud_sprites_renderer is not None
    assert offscreen_ctx.colored_mesh_renderer is not None
