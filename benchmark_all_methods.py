"""Benchmark all calculation methods: CPU, GPU Raycasting, GPU Depth Peeling.

This script compares the performance of all three methods across
different block model sizes.
"""

import time
import geoprops as gp
import numpy as np

# Load test mesh
cube_mesh = gp.mesh.load("tests/fixtures/cube.obj")
print("=" * 70)
print("GeoProps: CPU vs GPU Raycasting vs GPU Depth Peeling Benchmark")
print("=" * 70)
print(f"\nTest mesh: {cube_mesh.n_vertices} vertices, {cube_mesh.n_faces} faces")

# Test different grid sizes
test_cases = [
    (5, 5, 5, "Small (125 blocks)"),
    (10, 10, 10, "Medium (1,000 blocks)"),
    (15, 15, 10, "Large (2,250 blocks)"),
    (20, 20, 10, "Very Large (4,000 blocks)"),
]

print("\n" + "=" * 70)
print("Benchmark Results (resolution=10 for all methods):")
print("=" * 70)
print(f"{'Grid Size':<22} {'CPU':<12} {'GPU-Ray':<12} {'GPU-DP':<12} {'Ray/CPU':<10} {'DP/CPU':<10}")
print("-" * 70)

for nx, ny, nz, name in test_cases:
    blocks = gp.BlockModel(
        origin=(-1.0, -1.0, -1.0),
        size=(0.1, 0.1, 0.1),
        extent=(nx, ny, nz),
    )

    n_blocks = nx * ny * nz

    # Warm-up (compile shaders, etc.)
    if name == "Small (125 blocks)":
        _ = gp.calculate(blocks, cube_mesh, method='cpu', resolution=10)
        _ = gp.calculate(blocks, cube_mesh, method='gpu-raycast', resolution=10)
        _ = gp.calculate(blocks, cube_mesh, method='gpu', resolution=10)

    # CPU benchmark
    start = time.time()
    result_cpu = gp.calculate(blocks, cube_mesh, method='cpu', resolution=10)
    time_cpu = time.time() - start

    # GPU Raycasting benchmark
    start = time.time()
    result_gpu_ray = gp.calculate(blocks, cube_mesh, method='gpu-raycast', resolution=10)
    time_gpu_ray = time.time() - start

    # GPU Depth Peeling benchmark
    start = time.time()
    result_gpu_dp = gp.calculate(blocks, cube_mesh, method='gpu', resolution=10)
    time_gpu_dp = time.time() - start

    speedup_ray = time_cpu / time_gpu_ray if time_gpu_ray > 0 else 0
    speedup_dp = time_cpu / time_gpu_dp if time_gpu_dp > 0 else 0

    print(f"{name:<22} {time_cpu:>8.3f}s   {time_gpu_ray:>8.3f}s   {time_gpu_dp:>8.3f}s   {speedup_ray:>6.2f}x    {speedup_dp:>6.2f}x")

print("=" * 70)
print("\nMethod Details:")
print("-" * 70)
print("CPU:             NumPy raycasting (float64)")
print("GPU-Raycast:     OpenGL compute shader raycasting (float32)")
print("GPU-DepthPeel:   ** Currently uses CPU fallback **")
print("                 (Proper rasterization pipeline not yet implemented)")
print("\nNote:")
print("- GPU-DepthPeel currently falls back to CPU, so times match CPU")
print("- Once implemented, GPU-DepthPeel should be 10-100x faster via rasterization")
print("- GPU-Raycast gives ~4x speedup but is limited by triangle count")
print("=" * 70)

# Show method metadata
print("\nMethod Metadata Check:")
print("-" * 70)
small_blocks = gp.BlockModel(origin=(-1, -1, -1), size=(1, 1, 1), extent=(2, 2, 2))
r1 = gp.calculate(small_blocks, cube_mesh, method='cpu', resolution=10)
r2 = gp.calculate(small_blocks, cube_mesh, method='gpu-raycast', resolution=10)
r3 = gp.calculate(small_blocks, cube_mesh, method='gpu', resolution=10)

print(f"CPU method:          {r1.metadata['method']}")
print(f"GPU-Raycast method:  {r2.metadata['method']}")
print(f"GPU-DepthPeel method: {r3.metadata['method']}")
if 'note' in r3.metadata:
    print(f"  Note: {r3.metadata['note']}")
print("=" * 70)
