"""GeoProps: GPU-Accelerated Geometric Proportions for Resource Estimation.

GeoProps provides fast, GPU-accelerated calculations of geometric proportions
between meshes/polygons and block models, with fuzzy logic operations for
combining domains without mesh boolean operations.

Quick Start:
    >>> import geoprops as gp
    >>>
    >>> # Load mesh
    >>> mesh = gp.mesh.load('ore_domain.obj')
    >>>
    >>> # Define block model
    >>> blocks = gp.BlockModel(
    ...     origin=(500000, 9200000, 100),
    ...     size=(10, 10, 16),
    ...     extent=(200, 300, 50),
    ...     crs='EPSG:31983'
    ... )
    >>>
    >>> # Calculate proportions
    >>> result = gp.calculate(blocks, mesh)
    >>>
    >>> # Export results
    >>> df = result.to_dataframe()
    >>> df.write_parquet('proportions.parquet')

Main Features:
    - GPU-accelerated calculation (100-1000x speedup)
    - Fuzzy logic operations on results
    - Point containment queries
    - CRS-aware transformations
    - Polygon extrusion
"""

from geoprops.version import __version__, __version_info__
from geoprops.config import config

# Core classes
from geoprops.result import Result
from geoprops.mesh import Mesh
from geoprops.blocks import BlockModel
from geoprops.polygons import Polygon

# Submodules
from geoprops import mesh
from geoprops import blocks
from geoprops import polygons

# Main API functions
from geoprops.api import calculate
# calculate_batch will be implemented later

__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    # Configuration
    "config",
    # Core classes
    "Result",
    "Mesh",
    "BlockModel",
    "Polygon",
    # Submodules
    "mesh",
    "blocks",
    "polygons",
    # API functions
    "calculate",
    # "calculate_batch",  # To be added later
]

# Package metadata
__author__ = "Arthur Endlein"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/geoprops"
__description__ = "GPU-accelerated geometric proportions for resource estimation"
