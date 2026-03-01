#!/usr/bin/env python3
"""View a depth camera pointcloud with color texture projection.

Loads depth and color images from test data, unprojects the depth image to a
3D pointcloud using the GPU XY lookup table pipeline, projects each point into
the color camera's image plane to produce UV coordinates, and renders the
result with textured billboard sprites.

Controls:
    Left-click + drag          Rotate
    Shift + left-click + drag  Pan
    Scroll wheel               Zoom (shift for fine zoom)
    ESC                        Quit

Usage::

    python examples/view_depth_pointcloud.py
    python examples/view_depth_pointcloud.py --width 1280 --height 960 --point-size 0.005

No CuPy/CUDA required — uses a plain Vulkan device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import slangpy as spy
import numpy as np

from slangpy_renderer import (
    CameraIntrinsics,
    ColorProjectionParameters,
    DepthParameters,
    DepthUnprojector,
    Pointcloud,
)
from slangpy_renderer.controllers import ArcBall
from slangpy_renderer.renderers import PointcloudSpritesRenderer


DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


def vulkan_rh_zo_perspective(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Right-handed Vulkan perspective projection (depth 0..1, row-major)."""
    fovy = math.radians(fov_y_deg)
    f = 1.0 / math.tan(0.5 * fovy)
    A = far / (near - far)
    B = (far * near) / (near - far)

    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = A
    P[2, 3] = B
    P[3, 2] = -1.0
    return P


def load_calibration() -> dict:
    """Load the Azure Kinect calibration from test data."""
    with open(DATA_DIR / "calibration_camera01.json") as f:
        return json.load(f)


def build_depth_params(calib: dict) -> DepthParameters:
    """Build DepthParameters from the calibration JSON."""
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="View depth pointcloud with color texture projection."
    )
    parser.add_argument("--width", type=int, default=1024, help="Window width")
    parser.add_argument("--height", type=int, default=768, help="Window height")
    parser.add_argument(
        "--point-size", type=float, default=0.004, help="Sprite size in clip space"
    )
    args = parser.parse_args(argv)

    width, height = args.width, args.height

    # --- Create window and device ---
    window = spy.Window(width, height, "Depth Pointcloud Viewer", resizable=True)

    asset_root = Path(__file__).resolve().parent.parent / "slangpy_renderer" / "assets"
    if not asset_root.exists():
        import slangpy_renderer

        asset_root = Path(slangpy_renderer.__file__).parent / "assets"

    device = spy.Device(
        type=spy.DeviceType.vulkan,
        enable_debug_layers=True,
        compiler_options={
            "include_paths": [
                str(asset_root / "shaders"),
                os.path.join(os.path.dirname(spy.__file__), "slang"),
            ],
        },
    )

    surface = device.create_surface(window)
    surface.configure(width, height)
    output_format = surface.config.format

    # --- Load calibration and sensor data ---
    print("Loading calibration and sensor data...")
    calib = load_calibration()
    depth_image = np.load(DATA_DIR / "depth_camera01.npy")
    color_image = np.load(DATA_DIR / "color_camera01.npy")

    depth_params = build_depth_params(calib)
    color_params = ColorProjectionParameters.from_calibration(
        calib["color_parameters"],
        calib["color2depth_transform"],
    )

    print(
        f"Depth image: {depth_image.shape} {depth_image.dtype}, "
        f"Color image: {color_image.shape} {color_image.dtype}"
    )

    # --- Create unprojector and compute pointcloud + UVs ---
    print("Computing XY lookup table and unprojecting depth...")
    unprojector = DepthUnprojector(device, depth_params, color_params)
    unprojector.unproject(depth_image)

    dw, dh = depth_params.width, depth_params.height
    print(f"Pointcloud: {dw}x{dh} = {dw * dh} points")

    # --- Create the Pointcloud renderable ---
    # We bypass sync_gpu() and set GPU buffers directly from the unprojector
    pointcloud = Pointcloud(device, sync_gpu=False)
    pointcloud.position_buffer = unprojector.position_buffer
    pointcloud.uv_buffer = unprojector.texcoord_buffer
    # vertices attribute provides vertex count (.size) and depth dimensions (.shape)
    pointcloud.vertices = np.empty((dh, dw), dtype=np.uint8)

    # --- Upload color texture ---
    # Convert (H, W, 3) uint8 → (H, W, 4) uint8 (add alpha = 255)
    ch, cw = color_image.shape[:2]
    rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
    rgba[:, :, :3] = color_image
    rgba[:, :, 3] = 255

    color_texture = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=cw,
        height=ch,
        usage=spy.TextureUsage.shader_resource,
    )
    color_texture.copy_from_numpy(rgba)
    pointcloud.texture = color_texture

    # --- Create renderer ---
    renderer = PointcloudSpritesRenderer(device, output_format)
    pointcloud.renderer = renderer

    # --- Depth buffer ---
    def create_depth_texture() -> spy.Texture:
        return device.create_texture(
            format=spy.Format.d32_float,
            width=window.width,
            height=window.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    depth_texture = create_depth_texture()

    # --- Set up arcball camera ---
    # Depth camera coordinates: x-right, y-down, z-forward
    # Scene is at z > 0, roughly z ∈ [0.5, 4.0]
    camera_pos = np.array([0.0, -1.0, -0.5], dtype=np.float32)
    camera_target = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    fov = 60.0

    arcball = ArcBall(camera_pos, camera_target, camera_up, fov, (width, height))

    # --- Mouse/keyboard state ---
    current_button = None
    needs_init = False
    dirty = False
    should_render = True

    def on_mouse_event(event: spy.MouseEvent):
        nonlocal current_button, needs_init, should_render

        if event.type == spy.MouseEventType.button_down:
            if current_button != event.button:
                needs_init = True
            current_button = event.button

        elif event.type == spy.MouseEventType.button_up:
            current_button = None

        elif event.type == spy.MouseEventType.move:
            pos = (int(event.pos.x), int(event.pos.y))
            if current_button == spy.MouseButton.left:
                if needs_init:
                    needs_init = False
                    arcball.init_transformation(pos)

                if event.mods == spy.KeyModifierFlags.shift:
                    arcball.translate(pos)
                else:
                    arcball.rotate(pos)
                should_render = True

        elif event.type == spy.MouseEventType.scroll:
            delta = event.scroll.y / 5.0
            if event.mods == spy.KeyModifierFlags.shift:
                delta /= 3.0
            arcball.zoom(delta)
            should_render = True

    def on_keyboard_event(event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                window.close()

    def on_resize(w: int, h: int):
        nonlocal dirty, should_render
        dirty = True
        should_render = True

    window.on_mouse_event = on_mouse_event
    window.on_keyboard_event = on_keyboard_event
    window.on_resize = on_resize

    print("Ready. Use mouse to orbit, shift+drag to pan, scroll to zoom.")

    # --- Render loop ---
    while not window.should_close():
        window.process_events()

        if not should_render:
            continue
        should_render = False

        if dirty:
            del depth_texture
            device.wait()
            surface.configure(window.width, window.height)
            depth_texture = create_depth_texture()
            arcball.reshape((window.width, window.height))
            dirty = False

        arcball.update_transformation()

        surface_texture = surface.acquire_next_image()
        if not surface_texture:
            continue

        view_matrix = arcball.view_matrix()
        aspect = float(window.width) / float(window.height)
        proj_matrix = vulkan_rh_zo_perspective(fov, aspect, 0.01, 20.0)

        command_encoder = device.create_command_encoder()

        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": surface_texture.create_view(),
                        "clear_value": [0.05, 0.05, 0.05, 1.0],
                        "load_op": spy.LoadOp.clear,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": depth_texture.create_view(),
                    "depth_clear_value": 1.0,
                    "depth_load_op": spy.LoadOp.clear,
                    "depth_store_op": spy.StoreOp.store,
                    "depth_read_only": False,
                },
            }
        ) as pass_encoder:
            pointcloud.render(
                pass_encoder,
                (window.width, window.height),
                view_matrix,
                proj_matrix,
                extra_args={
                    "pointSize": args.point_size,
                },
            )

        device.submit_command_buffer(command_encoder.finish())
        surface.present()


if __name__ == "__main__":
    main()
