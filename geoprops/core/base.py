"""Base classes for proportion calculators.

This module defines abstract base classes that all calculators must implement.
"""

from abc import ABC, abstractmethod
from typing import Union

from geoprops.blocks import BlockModel
from geoprops.mesh import Mesh
from geoprops.polygons import Polygon
from geoprops.result import Result


class Calculator(ABC):
    """Base class for all proportion calculators.

    All calculators (CPU and GPU) must inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def calculate(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> Result:
        """Calculate proportions for each block.

        Args:
            blocks: BlockModel defining the grid
            geometry: Mesh or Polygon defining the domain

        Returns:
            Result object with block proportions
        """
        pass

    @abstractmethod
    def can_handle(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> bool:
        """Check if this calculator can handle these inputs.

        Args:
            blocks: BlockModel
            geometry: Mesh or Polygon

        Returns:
            True if this calculator can process these inputs
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Human-readable name for this calculation method.

        Returns:
            Method name (e.g., 'CPU-RayCasting', 'GPU-DepthPeeling')
        """
        pass

    def validate_inputs(
        self,
        blocks: BlockModel,
        geometry: Union[Mesh, Polygon],
    ) -> None:
        """Validate inputs before calculation.

        Args:
            blocks: BlockModel
            geometry: Mesh or Polygon

        Raises:
            ValueError: If inputs are invalid
        """
        # Basic validation
        if not isinstance(blocks, BlockModel):
            raise TypeError(f"Expected BlockModel, got {type(blocks)}")

        if not isinstance(geometry, (Mesh, Polygon)):
            raise TypeError(
                f"Expected Mesh or Polygon, got {type(geometry)}"
            )

        # Check mesh is valid
        if isinstance(geometry, Mesh) and not geometry.is_valid():
            raise ValueError("Mesh is not valid (no vertices or faces)")

        # Check block model has blocks
        if blocks.n_blocks == 0:
            raise ValueError("BlockModel has zero blocks")
