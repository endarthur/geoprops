"""GPU raycasting calculator using compute shaders.

This module implements GPU-accelerated proportion calculations using ModernGL
compute shaders. It parallelizes raycasting across the GPU, testing each sample
point against all triangles.

NOTE: This is a fallback method. The primary GPU method should be depth peeling,
which uses the GPU's rasterization pipeline for much better performance.
"""

import struct
import time
from typing import Union

import numpy as np

from geoprops.blocks import BlockModel
from geoprops.core.base import Calculator
from geoprops.gpu.context import GPUContext, is_gpu_available
from geoprops.mesh import Mesh
from geoprops.polygons import Polygon
from geoprops.result import Result


class GPURayCastingCalculator(Calculator):
    """GPU-accelerated raycasting calculator using compute shaders.

    This calculator uses ModernGL compute shaders to parallelize raycasting
    on the GPU. Each sample point is tested against all triangles using
    ray-triangle intersection.

    Performance: ~4x faster than CPU for medium grids, but slower than
    depth peeling for large meshes.

    Args:
        resolution: Grid resolution per block (e.g., 10 = 10×10×10 grid)
                   Default: 15
    """

    def __init__(self, resolution: int = 15):
        """Initialize GPU calculator.

        Args:
            resolution: Number of sample points along each axis per block

        Raises:
            RuntimeError: If GPU not available
        """
        if not is_gpu_available():
            try:
                import moderngl
                raise RuntimeError(
                    "ModernGL is installed but no OpenGL context available. "
                    "Please check your GPU drivers."
                )
            except ImportError:
                raise ImportError(
                    "ModernGL is required for GPU acceleration. "
                    "Install with: conda install -c conda-forge moderngl"
                )

        if resolution < 1:
            raise ValueError(f"resolution must be >= 1, got {resolution}")

        self.resolution = resolution
        self.n_samples = resolution ** 3

    @property
    def method_name(self) -> str:
        """Method name."""
        return f"GPU-RayCasting(resolution={self.resolution})"

    def can_handle(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> bool:
        """Check if this calculator can handle the inputs.

        Args:
            blocks: BlockModel
            geometry: Mesh or Polygon

        Returns:
            True if geometry is a Mesh and GPU is available
        """
        return isinstance(geometry, Mesh) and is_gpu_available()

    def calculate(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> Result:
        """Calculate block proportions using GPU acceleration.

        Args:
            blocks: BlockModel
            geometry: Mesh

        Returns:
            Result with proportions

        Raises:
            TypeError: If geometry is not a Mesh
        """
        # Validate
        self.validate_inputs(blocks, geometry)

        if not isinstance(geometry, Mesh):
            raise TypeError(
                f"GPURayCastingCalculator only handles Mesh, got {type(geometry)}"
            )

        # Start timer
        start_time = time.time()

        # Use GPU context
        with GPUContext() as gpu_ctx:
            result = self._calculate_gpu(blocks, geometry, gpu_ctx)

        # Calculate time
        elapsed_time = time.time() - start_time
        result.metadata["elapsed_time"] = elapsed_time

        return result

    def _calculate_gpu(
        self,
        blocks: BlockModel,
        geometry: Mesh,
        gpu_ctx: GPUContext,
    ) -> Result:
        """Perform GPU calculation using compute shader.

        Args:
            blocks: BlockModel
            geometry: Mesh
            gpu_ctx: GPU context

        Returns:
            Result with proportions
        """
        import moderngl

        ctx = gpu_ctx.gl_context

        # Prepare mesh data (convert to float32 for GPU)
        vertices = geometry.vertices.astype(np.float32)
        faces = geometry.faces.astype(np.int32)

        # Create compute shader
        compute_shader = ctx.compute_shader(COMPUTE_SHADER_SOURCE)

        # Upload mesh data to GPU
        vertex_buffer = ctx.buffer(vertices.tobytes())
        face_buffer = ctx.buffer(faces.tobytes())

        # Prepare output arrays
        nx, ny, nz = blocks.extent
        n_blocks = nx * ny * nz
        block_ids = []
        proportions = []

        # Process blocks
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Generate sample points for this block
                    sample_points = self._generate_grid_points(blocks, i, j, k)
                    n_points = len(sample_points)

                    # Upload points to GPU
                    point_buffer = ctx.buffer(sample_points.astype(np.float32).tobytes())

                    # Create output buffer for results
                    result_buffer = ctx.buffer(reserve=n_points * 4)  # int32 = 4 bytes

                    # Bind buffers
                    point_buffer.bind_to_storage_buffer(0)
                    vertex_buffer.bind_to_storage_buffer(1)
                    face_buffer.bind_to_storage_buffer(2)
                    result_buffer.bind_to_storage_buffer(3)

                    # Set uniforms
                    compute_shader['num_points'].value = n_points
                    compute_shader['num_faces'].value = len(faces)

                    # Run compute shader
                    # Work group size is 64, so dispatch ceil(n_points / 64) groups
                    num_groups = (n_points + 63) // 64
                    compute_shader.run(num_groups, 1, 1)

                    # Read back results
                    results_data = struct.unpack(f'{n_points}i', result_buffer.read())
                    n_inside = sum(results_data)

                    # Calculate proportion
                    proportion = n_inside / n_points

                    # Store result
                    block_ids.append([i, j, k])
                    proportions.append(proportion)

                    # Clean up buffers for this block
                    point_buffer.release()
                    result_buffer.release()

        # Clean up mesh buffers
        vertex_buffer.release()
        face_buffer.release()

        # Convert to arrays
        block_ids = np.array(block_ids, dtype=np.int32)
        proportions = np.array(proportions, dtype=np.float64)

        # Create result
        return Result(
            block_ids=block_ids,
            proportions=proportions,
            metadata={
                "method": self.method_name,
                "n_blocks": n_blocks,
                "n_samples_per_block": self.n_samples,
            },
        )

    def _generate_grid_points(
        self,
        blocks: BlockModel,
        i: int,
        j: int,
        k: int,
    ) -> np.ndarray:
        """Generate regular grid of sample points within a block.

        Args:
            blocks: BlockModel
            i: Block index in X direction
            j: Block index in Y direction
            k: Block index in Z direction

        Returns:
            (resolution^3, 3) array of sample points
        """
        # Get block size
        dx, dy, dz = blocks.size

        # Create 1D grids along each axis (in local block coordinates)
        x_local = np.linspace(
            dx / (2 * self.resolution),
            dx * (1 - 1 / (2 * self.resolution)),
            self.resolution,
        )
        y_local = np.linspace(
            dy / (2 * self.resolution),
            dy * (1 - 1 / (2 * self.resolution)),
            self.resolution,
        )
        z_local = np.linspace(
            dz / (2 * self.resolution),
            dz * (1 - 1 / (2 * self.resolution)),
            self.resolution,
        )

        # Create 3D meshgrid
        xx, yy, zz = np.meshgrid(x_local, y_local, z_local, indexing='ij')

        # Flatten to list of points
        points_local = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Transform to world coordinates
        if blocks.rotation == 0:
            # Simple case: no rotation
            origin_x = blocks.origin[0] + i * dx
            origin_y = blocks.origin[1] + j * dy
            origin_z = blocks.origin[2] + k * dz

            points_world = points_local + np.array([origin_x, origin_y, origin_z])

        else:
            # Rotated blocks
            cos_rot = blocks._cos_rot
            sin_rot = blocks._sin_rot

            # Local block offset
            local_block_x = i * dx
            local_block_y = j * dy

            # Add local block offset
            lx = local_block_x + points_local[:, 0]
            ly = local_block_y + points_local[:, 1]
            lz = k * dz + points_local[:, 2]

            # Rotate
            points_world = np.column_stack([
                lx * cos_rot - ly * sin_rot + blocks.origin[0],
                lx * sin_rot + ly * cos_rot + blocks.origin[1],
                lz + blocks.origin[2]
            ])

        return points_world


# Compute shader for point-in-mesh testing
# Note: Using flat float arrays to avoid vec3 padding issues in std430 layout
COMPUTE_SHADER_SOURCE = """
#version 430

layout(local_size_x = 64) in;

// Input buffers - using flat float arrays to avoid vec3 alignment issues
layout(std430, binding = 0) buffer Points {
    float points[];  // Flat array: x0,y0,z0, x1,y1,z1, ...
};

layout(std430, binding = 1) buffer Vertices {
    float vertices[];  // Flat array: x0,y0,z0, x1,y1,z1, ...
};

layout(std430, binding = 2) buffer Faces {
    int faces[];  // Flat array: i0,j0,k0, i1,j1,k1, ...
};

// Output buffer
layout(std430, binding = 3) buffer Results {
    int inside[];
};

uniform int num_points;
uniform int num_faces;

// Helper to get vec3 from flat array
vec3 getPoint(int idx) {
    int base = idx * 3;
    return vec3(points[base], points[base+1], points[base+2]);
}

vec3 getVertex(int idx) {
    int base = idx * 3;
    return vec3(vertices[base], vertices[base+1], vertices[base+2]);
}

ivec3 getFace(int idx) {
    int base = idx * 3;
    return ivec3(faces[base], faces[base+1], faces[base+2]);
}

// Ray-triangle intersection using Möller-Trumbore algorithm
bool rayIntersectsTriangle(vec3 origin, vec3 dir, vec3 v0, vec3 v1, vec3 v2) {
    const float EPSILON = 0.0000001;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(dir, edge2);
    float a = dot(edge1, h);

    if (abs(a) < EPSILON)
        return false;

    float f = 1.0 / a;
    vec3 s = origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    vec3 q = cross(s, edge1);
    float v = f * dot(dir, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * dot(edge2, q);

    return t > EPSILON;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= num_points)
        return;

    vec3 point = getPoint(int(idx));
    vec3 ray_dir = vec3(0.0, 0.0, 1.0);  // +Z direction

    int intersection_count = 0;

    // Test against all triangles
    for (int i = 0; i < num_faces; i++) {
        ivec3 face = getFace(i);
        vec3 v0 = getVertex(face.x);
        vec3 v1 = getVertex(face.y);
        vec3 v2 = getVertex(face.z);

        if (rayIntersectsTriangle(point, ray_dir, v0, v1, v2)) {
            intersection_count++;
        }
    }

    // Odd intersections = inside
    inside[idx] = (intersection_count % 2 == 1) ? 1 : 0;
}
"""
