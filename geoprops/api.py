"""Main API functions for GeoProps.

This module provides the primary user-facing API for calculating geometric
proportions and point containment queries.
"""

from typing import Union

import numpy as np

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from geoprops.blocks import BlockModel
from geoprops.mesh import Mesh
from geoprops.polygons import Polygon
from geoprops.result import Result


def calculate(
    target: Union[np.ndarray, "BlockModel", "pl.DataFrame", "pd.DataFrame"],
    geometry: Union[Mesh, Polygon],
    **kwargs,
) -> Result:
    """Calculate geometric proportions or point containment.

    This is the main entry point for GeoProps calculations. It automatically
    detects the input types and routes to the appropriate calculation method.

    Usage:
        - calculate(points, mesh) → Point containment (are points inside mesh?)
        - calculate(blocks, mesh) → Block proportions (% of each block in mesh)
        - calculate(blocks, polygon) → Polygon proportions with extrusion

    Args:
        target: Points (N, 3), BlockModel, or DataFrame with X/Y/Z columns
        geometry: Mesh or Polygon defining the domain
        **kwargs: Additional arguments for calculation method

    Returns:
        Result object with proportions/containment

    Raises:
        TypeError: If inputs are of unsupported types
        ValueError: If inputs have invalid shapes or incompatible types

    Examples:
        >>> # Point containment
        >>> points = np.array([[0, 0, 0], [1, 1, 1]])
        >>> result = gp.calculate(points, ore_mesh)
        >>> print(result.selected)  # Boolean mask of points inside

        >>> # With DataFrame
        >>> composites = pl.read_parquet('composites.parquet')
        >>> result = gp.calculate(composites[['X', 'Y', 'Z']], ore_mesh)
        >>> composites = composites.with_columns(
        ...     pl.Series('in_ore', result.selected)
        ... )

        >>> # Block model proportions (not yet implemented)
        >>> blocks = gp.BlockModel(...)
        >>> result = gp.calculate(blocks, ore_mesh)
        >>> df = result.to_dataframe()
    """
    # Handle DataFrames
    if POLARS_AVAILABLE and isinstance(target, pl.DataFrame):
        # Convert Polars DataFrame to numpy array
        target = target.to_numpy()
    elif PANDAS_AVAILABLE and isinstance(target, pd.DataFrame):
        # Convert Pandas DataFrame to numpy array
        target = target.values

    # Route to appropriate function
    if isinstance(target, np.ndarray):
        return _calculate_point_containment(target, geometry, **kwargs)
    elif isinstance(target, BlockModel):
        return _calculate_block_proportions(target, geometry, **kwargs)
    else:
        raise TypeError(
            f"Unsupported target type: {type(target)}. "
            f"Expected np.ndarray, BlockModel, or DataFrame"
        )


def _calculate_point_containment(
    points: np.ndarray,
    geometry: Union[Mesh, Polygon],
    **kwargs,
) -> Result:
    """Calculate point containment (are points inside geometry?).

    Args:
        points: (N, 3) array of point coordinates
        geometry: Mesh or Polygon
        **kwargs: Additional arguments

    Returns:
        Result with proportions = 1.0 (inside) or 0.0 (outside)

    Raises:
        ValueError: If points have invalid shape
    """
    # Validate points
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Points must be numpy array, got {type(points)}")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Points must be (N, 3) array, got shape {points.shape}"
        )

    n_points = len(points)

    # Handle empty points
    if n_points == 0:
        return Result(
            block_ids=np.empty((0, 1), dtype=np.int32),
            proportions=np.empty(0, dtype=np.float64),
            metadata={"method": "point_containment", "n_points": 0, "n_inside": 0},
        )

    # Calculate containment based on geometry type
    if isinstance(geometry, Mesh):
        inside = _point_in_mesh(points, geometry)
    elif isinstance(geometry, Polygon):
        inside = _point_in_polygon(points, geometry)
    else:
        raise TypeError(
            f"Unsupported geometry type: {type(geometry)}. "
            f"Expected Mesh or Polygon"
        )

    # Create Result
    # For points, block_ids is just the point index
    block_ids = np.arange(n_points, dtype=np.int32).reshape(-1, 1)
    proportions = inside.astype(np.float64)

    return Result(
        block_ids=block_ids,
        proportions=proportions,
        metadata={
            "method": "point_containment",
            "n_points": n_points,
            "n_inside": int(inside.sum()),
        },
    )


def _point_in_mesh(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Check if points are inside a mesh.

    Uses trimesh's contains() method which uses ray casting.

    Args:
        points: (N, 3) array of points
        mesh: Mesh to test against

    Returns:
        (N,) boolean array, True if point is inside
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for point containment queries. "
            "Install with: pip install trimesh"
        )

    # Convert to trimesh object
    mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # Use trimesh's optimized contains() method
    inside = mesh_tri.contains(points)

    return inside


def _point_in_polygon(points: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Check if points are inside a polygon.

    Args:
        points: (N, 3) array of points
        polygon: Polygon to test against

    Returns:
        (N,) boolean array, True if point is inside

    Raises:
        NotImplementedError: Polygon containment not yet implemented
    """
    # TODO: Implement in Phase 4 of roadmap
    raise NotImplementedError(
        "Point-in-polygon queries not yet implemented. "
        "This will be added in Phase 4. "
        "For now, please use Mesh objects for containment queries."
    )


def _calculate_block_proportions(
    blocks: BlockModel,
    geometry: Union[Mesh, Polygon],
    **kwargs,
) -> Result:
    """Calculate proportions of blocks occupied by geometry.

    Args:
        blocks: BlockModel
        geometry: Mesh or Polygon
        **kwargs: Additional arguments (method, n_samples, etc.)

    Returns:
        Result with block proportions

    Raises:
        ValueError: If no suitable calculator is available
    """
    from geoprops.core import RayCastingCalculator
    from geoprops.gpu import DepthPeelingCalculator, GPURayCastingCalculator, is_gpu_available

    # Get method preference
    method = kwargs.get('method', 'auto')

    # Get resolution parameter
    # Note: old n_samples parameter is converted to resolution (backward compat)
    if 'n_samples' in kwargs:
        # Convert n_samples to approximate resolution (cube root)
        n_samples = kwargs['n_samples']
        resolution = max(1, int(np.cbrt(n_samples)))
    else:
        resolution = kwargs.get('resolution', 15)  # Default: 15 (3375 samples)

    # Automatic method selection
    if method == 'auto':
        # Use GPU if available, otherwise fall back to CPU
        if is_gpu_available():
            method = 'gpu'
        else:
            method = 'cpu'

    # Create appropriate calculator
    if method == 'cpu':
        calculator = RayCastingCalculator(resolution=resolution)

    elif method == 'gpu':
        # Use depth peeling as primary GPU method (faster via rasterization)
        try:
            calculator = DepthPeelingCalculator(resolution=resolution)
        except (ImportError, RuntimeError) as e:
            # GPU not available, fall back to CPU
            import warnings
            warnings.warn(
                f"GPU requested but not available ({e}). Falling back to CPU."
            )
            calculator = RayCastingCalculator(resolution=resolution)

    elif method == 'gpu-raycast':
        # Explicit GPU raycasting (compute shader approach)
        try:
            calculator = GPURayCastingCalculator(resolution=resolution)
        except (ImportError, RuntimeError) as e:
            import warnings
            warnings.warn(
                f"GPU raycasting requested but not available ({e}). Falling back to CPU."
            )
            calculator = RayCastingCalculator(resolution=resolution)

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Use 'auto', 'cpu', 'gpu', or 'gpu-raycast'."
        )

    # Check it can handle this geometry
    if not calculator.can_handle(blocks, geometry):
        raise ValueError(
            f"{type(calculator).__name__} cannot handle {type(geometry).__name__}. "
            f"Only Mesh is supported for now."
        )

    # Calculate
    return calculator.calculate(blocks, geometry)
