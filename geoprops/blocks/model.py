"""BlockModel class for regular 3D grids.

This module provides the BlockModel class representing regular 3D block
grids used in resource estimation and mine planning.
"""

from typing import Any

import numpy as np


class BlockModel:
    """Regular 3D grid of blocks for resource estimation.

    A block model is a regular grid of rectangular blocks, optionally rotated,
    used to discretize a deposit for resource estimation and mine planning.

    Attributes:
        origin: Lower-left-front corner (x, y, z)
        size: Block dimensions (dx, dy, dz)
        extent: Grid size (nx, ny, nz) - number of blocks in each direction
        rotation: Grid rotation in degrees (counterclockwise around Z-axis)
        crs: Coordinate reference system (EPSG code or WKT string)

    Examples:
        >>> # Simple aligned grid
        >>> blocks = gp.BlockModel(
        ...     origin=(500000, 9200000, 100),
        ...     size=(10, 10, 16),
        ...     extent=(200, 300, 50),
        ...     crs='EPSG:31983'
        ... )

        >>> # Rotated grid
        >>> blocks = gp.BlockModel(
        ...     origin=(500000, 9200000, 100),
        ...     size=(10, 10, 16),
        ...     extent=(200, 300, 50),
        ...     rotation=45,  # 45° rotation
        ...     crs='EPSG:31983'
        ... )

        >>> print(blocks.n_blocks)  # Total number of blocks
        >>> print(blocks.bounds)    # Bounding box
    """

    def __init__(
        self,
        origin: tuple[float, float, float],
        size: tuple[float, float, float],
        extent: tuple[int, int, int],
        rotation: float = 0.0,
        crs: str | None = None,
    ):
        """Initialize BlockModel.

        Args:
            origin: Lower-left-front corner (x, y, z)
            size: Block dimensions (dx, dy, dz), must be positive
            extent: Grid size (nx, ny, nz), must be positive integers
            rotation: Grid rotation in degrees (counterclockwise around Z)
            crs: Optional coordinate reference system

        Raises:
            ValueError: If inputs have invalid values
        """
        # Validate origin
        origin_arr = np.array(origin, dtype=np.float64)
        if origin_arr.shape != (3,):
            raise ValueError(f"origin must be (x, y, z), got {origin}")

        # Validate size
        size_arr = np.array(size, dtype=np.float64)
        if size_arr.shape != (3,):
            raise ValueError(f"size must be (dx, dy, dz), got {size}")
        if np.any(size_arr <= 0):
            raise ValueError(f"size must be positive, got {size}")

        # Validate extent
        extent_arr = np.array(extent, dtype=np.int32)
        if extent_arr.shape != (3,):
            raise ValueError(f"extent must be (nx, ny, nz), got {extent}")
        if np.any(extent_arr <= 0):
            raise ValueError(f"extent must be positive, got {extent}")

        # Validate rotation
        if not isinstance(rotation, (int, float)):
            raise ValueError(f"rotation must be a number, got {type(rotation)}")

        self.origin = tuple(origin_arr)
        self.size = tuple(size_arr)
        self.extent = tuple(extent_arr)
        self.rotation = float(rotation)
        self.crs = crs

        # Precompute rotation matrix if needed
        self._rotation_rad = np.radians(self.rotation)
        self._cos_rot = np.cos(self._rotation_rad)
        self._sin_rot = np.sin(self._rotation_rad)

    @property
    def n_blocks(self) -> int:
        """Total number of blocks in the model.

        Returns:
            nx * ny * nz
        """
        return self.extent[0] * self.extent[1] * self.extent[2]

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box of the block model.

        Returns:
            Array of shape (2, 3) with [[xmin, ymin, zmin], [xmax, ymax, zmax]]

        Note:
            For rotated grids, this returns the axis-aligned bounding box
            of the rotated grid.
        """
        # Get all 8 corners of the grid
        corners = self._get_grid_corners()

        # Return min/max
        return np.array([corners.min(axis=0), corners.max(axis=0)])

    def _get_grid_corners(self) -> np.ndarray:
        """Get 8 corner points of the grid.

        Returns:
            Array of shape (8, 3) with corner coordinates
        """
        nx, ny, nz = self.extent
        dx, dy, dz = self.size
        x0, y0, z0 = self.origin

        # Define 8 corners in local coordinates
        local_corners = np.array(
            [
                [0, 0, 0],
                [nx * dx, 0, 0],
                [0, ny * dy, 0],
                [nx * dx, ny * dy, 0],
                [0, 0, nz * dz],
                [nx * dx, 0, nz * dz],
                [0, ny * dy, nz * dz],
                [nx * dx, ny * dy, nz * dz],
            ]
        )

        # Apply rotation if needed
        if self.rotation != 0:
            # Rotate around origin in XY plane
            corners = np.zeros_like(local_corners)
            corners[:, 0] = (
                local_corners[:, 0] * self._cos_rot
                - local_corners[:, 1] * self._sin_rot
                + x0
            )
            corners[:, 1] = (
                local_corners[:, 0] * self._sin_rot
                + local_corners[:, 1] * self._cos_rot
                + y0
            )
            corners[:, 2] = local_corners[:, 2] + z0
        else:
            corners = local_corners + np.array([x0, y0, z0])

        return corners

    def block_center(self, i: int, j: int, k: int) -> np.ndarray:
        """Get center coordinates of block (i, j, k).

        Args:
            i: Block index in X direction (0 to nx-1)
            j: Block index in Y direction (0 to ny-1)
            k: Block index in Z direction (0 to nz-1)

        Returns:
            Array [x, y, z] of block center

        Raises:
            ValueError: If indices are out of range

        Examples:
            >>> center = blocks.block_center(10, 20, 5)
            >>> print(center)  # [x, y, z]
        """
        # Validate indices
        if not (0 <= i < self.extent[0]):
            raise ValueError(f"i must be in [0, {self.extent[0]-1}], got {i}")
        if not (0 <= j < self.extent[1]):
            raise ValueError(f"j must be in [0, {self.extent[1]-1}], got {j}")
        if not (0 <= k < self.extent[2]):
            raise ValueError(f"k must be in [0, {self.extent[2]-1}], got {k}")

        # Local coordinates (offset from origin, centered in block)
        local_x = (i + 0.5) * self.size[0]
        local_y = (j + 0.5) * self.size[1]
        local_z = (k + 0.5) * self.size[2]

        # Apply rotation if needed
        if self.rotation != 0:
            x = local_x * self._cos_rot - local_y * self._sin_rot + self.origin[0]
            y = local_x * self._sin_rot + local_y * self._cos_rot + self.origin[1]
            z = local_z + self.origin[2]
        else:
            x = local_x + self.origin[0]
            y = local_y + self.origin[1]
            z = local_z + self.origin[2]

        return np.array([x, y, z])

    def block_corners(self, i: int, j: int, k: int) -> np.ndarray:
        """Get 8 corner coordinates of block (i, j, k).

        Args:
            i: Block index in X direction
            j: Block index in Y direction
            k: Block index in Z direction

        Returns:
            Array of shape (8, 3) with corner coordinates

        Raises:
            ValueError: If indices are out of range

        Examples:
            >>> corners = blocks.block_corners(10, 20, 5)
            >>> print(corners.shape)  # (8, 3)
        """
        # Validate indices
        if not (0 <= i < self.extent[0]):
            raise ValueError(f"i must be in [0, {self.extent[0]-1}], got {i}")
        if not (0 <= j < self.extent[1]):
            raise ValueError(f"j must be in [0, {self.extent[1]-1}], got {j}")
        if not (0 <= k < self.extent[2]):
            raise ValueError(f"k must be in [0, {self.extent[2]-1}], got {k}")

        # Define 8 corners in local coordinates relative to block origin
        dx, dy, dz = self.size
        local_x = i * dx
        local_y = j * dy
        local_z = k * dz

        # 8 corners (lower 4, then upper 4)
        offsets = np.array(
            [
                [0, 0, 0],
                [dx, 0, 0],
                [0, dy, 0],
                [dx, dy, 0],
                [0, 0, dz],
                [dx, 0, dz],
                [0, dy, dz],
                [dx, dy, dz],
            ]
        )

        # Apply rotation if needed
        if self.rotation != 0:
            corners = np.zeros((8, 3))
            for idx, (ox, oy, oz) in enumerate(offsets):
                lx = local_x + ox
                ly = local_y + oy
                lz = local_z + oz

                corners[idx, 0] = lx * self._cos_rot - ly * self._sin_rot + self.origin[0]
                corners[idx, 1] = lx * self._sin_rot + ly * self._cos_rot + self.origin[1]
                corners[idx, 2] = lz + self.origin[2]
        else:
            corners = offsets + np.array([local_x + self.origin[0],
                                          local_y + self.origin[1],
                                          local_z + self.origin[2]])

        return corners

    def all_block_centers(self) -> np.ndarray:
        """Get centers of all blocks.

        Returns:
            Array of shape (n_blocks, 3) with all block centers

        Examples:
            >>> centers = blocks.all_block_centers()
            >>> print(centers.shape)  # (3000000, 3) for 200x300x50 grid
        """
        # Generate all (i, j, k) indices
        i_indices = np.arange(self.extent[0])
        j_indices = np.arange(self.extent[1])
        k_indices = np.arange(self.extent[2])

        # Create meshgrid
        i_grid, j_grid, k_grid = np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')

        # Flatten to get all indices
        i_flat = i_grid.ravel()
        j_flat = j_grid.ravel()
        k_flat = k_grid.ravel()

        # Local coordinates (centered in blocks)
        local_x = (i_flat + 0.5) * self.size[0]
        local_y = (j_flat + 0.5) * self.size[1]
        local_z = (k_flat + 0.5) * self.size[2]

        # Apply rotation if needed
        if self.rotation != 0:
            x = local_x * self._cos_rot - local_y * self._sin_rot + self.origin[0]
            y = local_x * self._sin_rot + local_y * self._cos_rot + self.origin[1]
            z = local_z + self.origin[2]
        else:
            x = local_x + self.origin[0]
            y = local_y + self.origin[1]
            z = local_z + self.origin[2]

        return np.column_stack([x, y, z])

    @property
    def volume(self) -> float:
        """Total volume of the block model.

        Returns:
            Total volume (n_blocks × block_volume)
        """
        block_volume = self.size[0] * self.size[1] * self.size[2]
        return self.n_blocks * block_volume

    def __repr__(self) -> str:
        bounds = self.bounds
        return (
            f"BlockModel(\n"
            f"  origin={self.origin},\n"
            f"  size={self.size},\n"
            f"  extent={self.extent},\n"
            f"  rotation={self.rotation}°,\n"
            f"  n_blocks={self.n_blocks:,},\n"
            f"  bounds=[({bounds[0,0]:.1f}, {bounds[0,1]:.1f}, {bounds[0,2]:.1f}), "
            f"({bounds[1,0]:.1f}, {bounds[1,1]:.1f}, {bounds[1,2]:.1f})],\n"
            f"  crs={self.crs!r}\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()
