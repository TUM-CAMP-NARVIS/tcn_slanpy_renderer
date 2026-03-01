# Slangpy Renderer - Standalone 3D Rendering Library

A standalone Python library for real-time 3D rendering using Slangpy (Vulkan backend) with CUDA interoperability. This library has been extracted from the TCN rendering pipeline to provide a clean, reusable rendering solution.

## Features

- **Real-time 3D Rendering**: Hardware-accelerated rendering using Vulkan via Slangpy
- **CUDA Interoperability**: Direct GPU-to-GPU data transfer for high-performance streaming
- **Multiple Renderable Types**:
  - Point clouds with texture mapping
  - Textured meshes with Phong shading
  - Colored wireframes and coordinate axes
- **Interactive Camera Controls**:
  - ArcBall controller for intuitive 3D navigation
  - First-person view controller
- **Thread-Safe Updates**: Safely update renderables from external data pipelines
- **Clean API**: Explicit interface/implementation separation for easy integration

## Installation

### Prerequisites

```bash
pip install slangpy numpy cupy pyglm trimesh pillow
```

### Add to Python Path

```python
import sys
sys.path.insert(0, '/path/to/applications/tcn_artekmed/tcn_render_tests/standalone')
```

## Quick Start

### Basic Window with Mesh

```python
from slangpy_renderer import SlangWindow, Mesh

# Create rendering window
window = SlangWindow(1024, 768, "My 3D Viewer")

# Load a mesh from OBJ file
mesh = Mesh.from_obj(window.get_device(), "path/to/model.obj")

# Add to scene
window.add_renderable("my_mesh", mesh)

# Run rendering loop
window.run()
```

### Rendering Point Clouds with Streaming Data

```python
from slangpy_renderer import SlangWindow, Pointcloud
import numpy as np
import cupy as cp

# Create window
window = SlangWindow(1024, 768, "Point Cloud Viewer")

# Create initial pointcloud
positions = np.random.randn(480, 640, 3).astype(np.float32)
texcoords = np.random.rand(480, 640, 2).astype(np.float32)
image = np.random.rand(480, 640, 4).astype(np.float32)

pointcloud = Pointcloud(
    device=window.get_device(),
    positions=positions,
    texcoords=texcoords,
    image=image,
    sync_gpu=True
)

window.add_renderable("stream_pointcloud", pointcloud)

# Update from another thread (e.g., camera stream)
def update_pointcloud(new_positions, new_texcoords, new_image):
    pointcloud.update(
        positions=new_positions,
        texcoords=new_texcoords,
        image=new_image
    )
    window.request_redraw()

# Run rendering
window.run()
```

### Creating Colored Coordinate Axes

```python
from slangpy_renderer import SlangWindow, ColoredMesh
import numpy as np

window = SlangWindow(800, 600, "Axes Viewer")

# Create 3D coordinate axes
axes = ColoredMesh.create_axis3d(window.get_device(), scale=2.0)
window.add_renderable("world_axes", axes)

# Set pose
pose = np.eye(4, dtype=np.float32)
pose[:3, 3] = [1, 2, 3]  # Translation
window.set_pose("world_axes", pose)

window.run()
```

## Architecture

### Core Components

```
slangpy_renderer/
├── __init__.py              # Public API exports
├── window.py                # SlangWindow - main rendering window
├── renderables/             # Renderable data objects
│   ├── base.py             # Renderable base class
│   ├── pointcloud.py       # Pointcloud implementation
│   ├── mesh.py             # Mesh implementation
│   └── colored_mesh.py     # ColoredMesh implementation
├── renderers/               # Renderer implementations
│   ├── pointcloud_renderer.py
│   ├── pointcloud_sprites_renderer.py
│   ├── mesh_renderer.py
│   └── colored_mesh_renderer.py
├── controllers/             # Camera controllers
│   ├── arcball.py          # ArcBall camera
│   └── fpv.py              # First-person view camera
├── utils/                   # Utility functions
│   └── cuda_helpers.py     # CUDA/CuPy utilities
└── assets/                  # Shaders and models
    ├── shaders/
    └── models/
```

### API Design

The library follows a clear separation between interface and implementation:

1. **Renderable Objects** (`Pointcloud`, `Mesh`, `ColoredMesh`):
   - Store geometry, texture, and GPU buffer data
   - Provide thread-safe `update()` method for external data sources
   - Implement `sync_gpu()` for GPU buffer synchronization
   - Delegate rendering to associated renderer

2. **Renderers** (`PointcloudRenderer`, `MeshRenderer`, etc.):
   - Stateless rendering implementations
   - Shared across multiple renderables of the same type
   - Handle shader compilation, pipeline setup, and draw calls

3. **SlangWindow**:
   - Manages window, surface, and device
   - Owns all renderers (one instance per type)
   - Provides scene management (add/remove/update renderables)
   - Handles camera control and input events

## Advanced Usage

### Custom Rendering Parameters

```python
# Add renderable with custom rendering settings
window.add_renderable("my_pointcloud", pointcloud)

# Configure rendering parameters via window UI controls
# or programmatically through extra_args in render loop
```

### Multiple Renderables

```python
# Add multiple objects to the scene
window.add_renderable("pointcloud_1", pointcloud1)
window.add_renderable("pointcloud_2", pointcloud2)
window.add_renderable("reference_axes", axes)

# Toggle visibility
window.set_visible("pointcloud_2", False)

# Update poses independently
window.set_pose("reference_axes", transform_matrix)
```

### Camera Control

```python
# Access camera controller
camera = window.arc_ball

# Programmatic camera control
camera.zoom(5.0)
camera.translate_delta(np.array([0.1, 0.0]))

# Or use keyboard/mouse (automatic)
# - Left mouse: Rotate
# - Shift + Left mouse: Translate
# - Scroll: Zoom
# - Keys 1-9: Toggle renderable visibility
# - ESC: Close window
```

### Integration with Data Pipelines

```python
import threading

window = SlangWindow(1024, 768, "Pipeline Viewer")
pointcloud = Pointcloud(device=window.get_device())
window.add_renderable("stream", pointcloud)

def data_processing_thread():
    while True:
        # Get data from your pipeline (e.g., camera, network)
        positions, texcoords, image = get_next_frame()

        # Update renderable (thread-safe)
        pointcloud.update(
            positions=positions,
            texcoords=texcoords,
            image=image
        )

        # Request redraw
        window.request_redraw()

# Start data thread
threading.Thread(target=data_processing_thread, daemon=True).start()

# Run rendering loop (main thread)
window.run()
```

## Holoscan Integration

To integrate back into Holoscan applications:

```python
from slangpy_renderer import SlangWindow, Pointcloud
import holoscan as hs

class RenderOperator(hs.core.Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.window = None

    def start(self):
        self.window = SlangWindow(1024, 768, "Holoscan Viewer")
        self.render_thread = threading.Thread(target=self.window.run)
        self.render_thread.start()

    def compute(self, op_input, op_output, context):
        # Receive data from Holoscan pipeline
        data = op_input.receive("input")

        # Update renderable
        renderable = self.window.get_renderable("my_object")
        if renderable:
            renderable.update(**data)
            self.window.request_redraw()

    def stop(self):
        self.window.close()
        self.render_thread.join()
```

## Performance Considerations

- **CUDA Interop**: Data is transferred directly from CUDA memory to Vulkan without CPU round-trip
- **Thread Safety**: Updates from external threads are queued and synchronized before rendering
- **Lazy GPU Sync**: GPU buffers are only updated when data is marked dirty
- **Single Render Pass**: All renderables are rendered in a single pass to minimize state changes

## Troubleshooting

### Shader Compilation Errors
Ensure the `assets/shaders` directory is accessible. Set custom path:
```python
window = SlangWindow(1024, 768, "Title", assets_path="/custom/path/to/assets")
```

### CUDA/Vulkan Interop Issues
Check that both CUDA and Vulkan drivers are properly installed:
```bash
nvidia-smi  # Verify CUDA
vulkaninfo  # Verify Vulkan
```

### Performance Issues
- Reduce point cloud resolution
- Check GPU memory usage
- Ensure data updates are not too frequent

## License

This code is derived from the TCN project and maintains the same license.

## Authors

- Original implementation: Ulrich Eck, TUM
- Standalone extraction: 2026
