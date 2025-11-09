# GeoProps - gpu module
"""GPU-accelerated implementations of proportion calculators.

GPU implementation uses ModernGL for cross-platform OpenGL rendering.
This provides compatibility with NVIDIA, AMD, and Intel GPUs.

Primary method: Depth Peeling (uses GPU rasterization pipeline)
Fallback method: GPU Raycasting (uses compute shaders)
"""

from geoprops.gpu.context import GPUContext, is_gpu_available, get_gpu_info
from geoprops.gpu.depth_peeling import DepthPeelingCalculator
from geoprops.gpu.gpu_raycasting import GPURayCastingCalculator

__all__ = [
    "GPUContext",
    "is_gpu_available",
    "get_gpu_info",
    "DepthPeelingCalculator",
    "GPURayCastingCalculator",
]
