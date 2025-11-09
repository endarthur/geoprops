"""CPU raycasting calculator for general meshes.

This module implements a regular grid discretization approach for calculating
block proportions. It samples points on a regular grid within each block
and uses raycasting to determine if they're inside the mesh.
"""

import time
from typing import Union

import numpy as np
import trimesh

from geoprops.blocks import BlockModel
from geoprops.core.base import Calculator
from geoprops.mesh import Mesh
from geoprops.polygons import Polygon
from geoprops.result import Result


class RayCastingCalculator(Calculator):
    """CPU-based raycasting calculator with regular grid discretization.

    This calculator samples points on a regular 3D grid within each block
    and uses raycasting to determine if they're inside the mesh. The proportion
    is calculated as (points inside) / (total points sampled).

    This approach is:
    - Deterministic: Same inputs always give same outputs
    - Reproducible: No random sampling variance
    - Predictable: Accuracy directly relates to grid resolution

    This works for any mesh geometry but is relatively slow. For better
    performance, use GPU calculators.

    Args:
        resolution: Grid resolution per block (e.g., 10 = 10×10×10 grid = 1000 points)
                   Higher values give better accuracy but slower calculation.
                   Recommended: 15-20 for good accuracy
                   Default: 15
    """

    def __init__(self, resolution: int = 15):
        """Initialize calculator.

        Args:
            resolution: Number of sample points along each axis per block
                       (total points = resolution^3)
        """
        if resolution < 1:
            raise ValueError(f"resolution must be >= 1, got {resolution}")

        self.resolution = resolution
        self.n_samples = resolution ** 3  # Total points per block

    @property
    def method_name(self) -> str:
        """Method name."""
        return f"CPU-RayCasting(resolution={self.resolution})"

    def can_handle(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> bool:
        """Check if this calculator can handle the inputs.

        The raycasting calculator can handle any Mesh.

        Args:
            blocks: BlockModel
            geometry: Mesh or Polygon

        Returns:
            True if geometry is a Mesh
        """
        return isinstance(geometry, Mesh)

    def calculate(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> Result:
        """Calculate block proportions using Monte Carlo sampling.

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
                f"RayCastingCalculator only handles Mesh, got {type(geometry)}"
            )

        # Start timer
        start_time = time.time()

        # Convert mesh to trimesh for contains() queries
        mesh_tri = trimesh.Trimesh(
            vertices=geometry.vertices,
            faces=geometry.faces
        )

        # Prepare output arrays
        n_blocks = blocks.n_blocks
        nx, ny, nz = blocks.extent

        # Generate all block indices
        block_ids = []
        proportions = []

        # Iterate through all blocks
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Generate regular grid of sample points within this block
                    sample_points = self._generate_grid_points(blocks, i, j, k)

                    # Check how many are inside the mesh
                    inside = mesh_tri.contains(sample_points)
                    n_inside = inside.sum()

                    # Calculate proportion
                    proportion = n_inside / self.n_samples

                    # Store result
                    block_ids.append([i, j, k])
                    proportions.append(proportion)

        # Convert to arrays
        block_ids = np.array(block_ids, dtype=np.int32)
        proportions = np.array(proportions, dtype=np.float64)

        # Calculate time
        elapsed_time = time.time() - start_time

        # Create Result
        return Result(
            block_ids=block_ids,
            proportions=proportions,
            metadata={
                "method": self.method_name,
                "n_blocks": n_blocks,
                "n_samples_per_block": self.n_samples,
                "elapsed_time": elapsed_time,
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

        Creates a regular 3D grid of points within the block. For rotated blocks,
        generates grid in local block coordinates then transforms to world coordinates.

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
        # Sample at regular intervals, offset by half-spacing from edges
        x_local = np.linspace(dx / (2 * self.resolution), dx * (1 - 1 / (2 * self.resolution)), self.resolution)
        y_local = np.linspace(dy / (2 * self.resolution), dy * (1 - 1 / (2 * self.resolution)), self.resolution)
        z_local = np.linspace(dz / (2 * self.resolution), dz * (1 - 1 / (2 * self.resolution)), self.resolution)

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
            # Rotated blocks: transform each point
            cos_rot = blocks._cos_rot
            sin_rot = blocks._sin_rot

            # Rotate in XY plane around block origin
            points_world = np.zeros_like(points_local)

            # Apply rotation
            local_block_x = i * dx
            local_block_y = j * dy

            for idx, (px, py, pz) in enumerate(points_local):
                # Add local block offset
                lx = local_block_x + px
                ly = local_block_y + py
                lz = k * dz + pz

                # Rotate
                points_world[idx, 0] = lx * cos_rot - ly * sin_rot + blocks.origin[0]
                points_world[idx, 1] = lx * sin_rot + ly * cos_rot + blocks.origin[1]
                points_world[idx, 2] = lz + blocks.origin[2]

        return points_world
