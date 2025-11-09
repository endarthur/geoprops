"""Benchmark GPU vs CPU performance.

This script compares the performance of GPU and CPU calculators across
different block model sizes to demonstrate the speedup.
"""

import time
import geoprops as gp
import numpy as np

# Load test mesh
cube_mesh = gp.mesh.load("tests/fixtures/cube.obj")
print("=" * 60)
print("GeoProps GPU vs CPU Performance Benchmark")
print("=" * 60)
print(f"\nTest mesh: {cube_mesh.n_vertices} vertices, {cube_mesh.n_faces} faces")

# Test different grid sizes
test_cases = [
    (5, 5, 5, "Small (125 blocks)"),
    (10, 10, 10, "Medium (1,000 blocks)"),
    (20, 20, 10, "Large (4,000 blocks)"),
    (30, 30, 10, "Very Large (9,000 blocks)"),
]

print("\n" + "=" * 60)
print("Benchmark Results:")
print("=" * 60)
print(f"{'Grid Size':<20} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<10}")
print("-" * 60)

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
        _ = gp.calculate(blocks, cube_mesh, method='gpu', resolution=10)

    # CPU benchmark
    start_cpu = time.time()
    result_cpu = gp.calculate(blocks, cube_mesh, method='cpu', resolution=10)
    time_cpu = time.time() - start_cpu

    # GPU benchmark
    start_gpu = time.time()
    result_gpu = gp.calculate(blocks, cube_mesh, method='gpu', resolution=10)
    time_gpu = time.time() - start_gpu

    speedup = time_cpu / time_gpu

    print(f"{name:<20} {time_cpu:>10.3f}s    {time_gpu:>10.3f}s    {speedup:>6.2f}x")

print("=" * 60)
print("\nNote: Speedup will vary based on GPU model and CPU.")
print("Larger grids typically show better GPU performance.")
print("=" * 60)
