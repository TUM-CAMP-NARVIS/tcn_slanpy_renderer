#!/usr/bin/env python3
"""Interactive cube viewer with arcball camera control.

A minimal example that renders a unit cube in a window using the ArcBall
controller for mouse-driven camera manipulation.

Controls:
    Left-click + drag        Rotate
    Shift + left-click + drag  Pan
    Scroll wheel             Zoom (shift for fine zoom)
    ESC                      Quit

Usage::

    python examples/view_cube.py
    python examples/view_cube.py --width 1024 --height 768

No CuPy/CUDA required — uses a plain Vulkan device.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import slangpy as spy
import numpy as np

from slangpy_renderer import Mesh
from slangpy_renderer.controllers import ArcBall
from slangpy_renderer.renderers import MeshRenderer


def vulkan_rh_zo_perspective(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Right-handed Vulkan perspective projection (depth 0..1, row-major)."""
    fovy = math.radians(fov_y_deg)
    f = 1.0 / math.tan(0.5 * fovy)
    A = far / (near - far)
    B = (far * near) / (near - far)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = A
    P[2, 3] = B
    P[3, 2] = -1.0
    return P


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive cube viewer with arcball control.")
    parser.add_argument("--width", type=int, default=800, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    args = parser.parse_args(argv)

    width, height = args.width, args.height

    # --- Create window and device ---
    window = spy.Window(width, height, "Cube Viewer", resizable=True)

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

    # --- Load cube mesh ---
    cube_path = asset_root / "models" / "cube.obj"
    cube = Mesh.from_obj(device, str(cube_path))
    cube.pose = np.eye(4, dtype=np.float32)

    # --- Create renderer and depth buffer ---
    renderer = MeshRenderer(device, output_format)
    cube.renderer = renderer

    def create_depth_texture() -> spy.Texture:
        return device.create_texture(
            format=spy.Format.d32_float,
            width=window.width,
            height=window.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    depth_texture = create_depth_texture()

    # --- Set up arcball camera ---
    camera_pos = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov = 60.0

    arcball = ArcBall(camera_pos, camera_target, camera_up, fov, (width, height))

    # --- Mouse state ---
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
        proj_matrix = vulkan_rh_zo_perspective(fov, aspect, 0.1, 100.0)

        command_encoder = device.create_command_encoder()

        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": surface_texture.create_view(),
                        "clear_value": [0.1, 0.1, 0.1, 1.0],
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
            cube.render(
                pass_encoder,
                (window.width, window.height),
                view_matrix,
                proj_matrix,
                extra_args={"renderStaticColor": True},
            )

        device.submit_command_buffer(command_encoder.finish())
        surface.present()


if __name__ == "__main__":
    main()
