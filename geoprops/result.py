"""Result object for proportion calculations and point queries.

This module provides the Result class that stores calculation results and
supports fuzzy logic operations for combining domains.
"""

from typing import Any, Literal

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


class Result:
    """Unified result from proportion calculation or point queries.

    The Result class stores block/point indices and their corresponding
    proportion values. It supports fuzzy logic operations for combining
    multiple domain proportions without mesh boolean operations.

    Attributes:
        block_ids: Block or point indices (N, 3) for blocks or (N, 1) for points
        proportions: Proportion values in range [0, 1]
        metadata: Dictionary with method info, timing, statistics

    Examples:
        >>> # Calculate proportions
        >>> result = gp.calculate(blocks, mesh)
        >>> print(result.proportions.mean())

        >>> # Fuzzy logic operations
        >>> ore = high_grade | low_grade  # Union
        >>> oxide_ore = ore & oxide       # Intersection
        >>> waste = ~ore                  # Complement

        >>> # Export to DataFrame
        >>> df = result.to_dataframe()
        >>> df.write_parquet('proportions.parquet')
    """

    def __init__(
        self,
        block_ids: np.ndarray,
        proportions: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize Result.

        Args:
            block_ids: Block/point indices, shape (N, 3) for blocks or (N, 1) for points
            proportions: Proportion values, shape (N,)
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If inputs have incompatible shapes
        """
        # Validate inputs
        if block_ids.ndim != 2:
            raise ValueError(
                f"block_ids must be 2D array (N, 3) or (N, 1), got shape {block_ids.shape}"
            )

        if proportions.ndim != 1:
            raise ValueError(
                f"proportions must be 1D array (N,), got shape {proportions.shape}"
            )

        if block_ids.shape[0] != proportions.shape[0]:
            raise ValueError(
                f"block_ids and proportions must have same length, "
                f"got {block_ids.shape[0]} and {proportions.shape[0]}"
            )

        if not np.all((proportions >= 0) & (proportions <= 1)):
            raise ValueError("proportions must be in range [0, 1]")

        self.block_ids = block_ids
        self.proportions = proportions
        self.metadata = metadata or {}

    @property
    def selected(self) -> np.ndarray:
        """Boolean mask where proportion > 0.

        Returns:
            Boolean array of shape (N,)

        Examples:
            >>> result = gp.calculate(blocks, mesh)
            >>> n_blocks_with_ore = result.selected.sum()
        """
        return self.proportions > 0

    def threshold(self, value: float) -> np.ndarray:
        """Boolean mask at custom threshold.

        Args:
            value: Threshold value in range [0, 1]

        Returns:
            Boolean array of shape (N,)

        Examples:
            >>> # Get blocks that are at least 50% ore
            >>> mostly_ore = result.threshold(0.5)
        """
        if not 0 <= value <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        return self.proportions > value

    # Fuzzy logic operations
    def union(self, other: "Result") -> "Result":
        """Fuzzy OR operation: max(A, B).

        Combines two results taking the maximum proportion at each location.

        Args:
            other: Another Result to combine with

        Returns:
            New Result with union proportions

        Examples:
            >>> ore = high_grade.union(low_grade)
            >>> # Or use operator: ore = high_grade | low_grade
        """
        self._validate_compatible(other)
        return Result(
            block_ids=self.block_ids.copy(),
            proportions=np.maximum(self.proportions, other.proportions),
            metadata={
                "operation": "union",
                "operands": [self.metadata.get("name"), other.metadata.get("name")],
            },
        )

    def intersection(self, other: "Result") -> "Result":
        """Fuzzy AND operation: min(A, B).

        Combines two results taking the minimum proportion at each location.

        Args:
            other: Another Result to combine with

        Returns:
            New Result with intersection proportions

        Examples:
            >>> oxide_ore = ore.intersection(oxide)
            >>> # Or use operator: oxide_ore = ore & oxide
        """
        self._validate_compatible(other)
        return Result(
            block_ids=self.block_ids.copy(),
            proportions=np.minimum(self.proportions, other.proportions),
            metadata={
                "operation": "intersection",
                "operands": [self.metadata.get("name"), other.metadata.get("name")],
            },
        )

    def difference(self, other: "Result") -> "Result":
        """Fuzzy DIFFERENCE operation: min(A, 1-B).

        Subtracts the second result from the first.

        Args:
            other: Result to subtract

        Returns:
            New Result with difference proportions

        Examples:
            >>> fresh_ore = ore.difference(weathered)
            >>> # Or use operator: fresh_ore = ore - weathered
        """
        self._validate_compatible(other)
        return Result(
            block_ids=self.block_ids.copy(),
            proportions=np.minimum(self.proportions, 1.0 - other.proportions),
            metadata={
                "operation": "difference",
                "operands": [self.metadata.get("name"), other.metadata.get("name")],
            },
        )

    def complement(self) -> "Result":
        """Fuzzy NOT operation: 1 - A.

        Inverts the proportions (ore becomes waste, waste becomes ore).

        Returns:
            New Result with complement proportions

        Examples:
            >>> waste = ore.complement()
            >>> # Or use operator: waste = ~ore
        """
        return Result(
            block_ids=self.block_ids.copy(),
            proportions=1.0 - self.proportions,
            metadata={
                "operation": "complement",
                "operand": self.metadata.get("name"),
            },
        )

    # Operator overloading for intuitive syntax
    def __or__(self, other: "Result") -> "Result":
        """Union operator: result_a | result_b."""
        return self.union(other)

    def __and__(self, other: "Result") -> "Result":
        """Intersection operator: result_a & result_b."""
        return self.intersection(other)

    def __sub__(self, other: "Result") -> "Result":
        """Difference operator: result_a - result_b."""
        return self.difference(other)

    def __invert__(self) -> "Result":
        """Complement operator: ~result."""
        return self.complement()

    # Export methods
    def to_dataframe(
        self, format: Literal["polars", "pandas"] = "polars"
    ) -> "pl.DataFrame | pd.DataFrame":
        """Export to DataFrame (Polars default, Pandas optional).

        Args:
            format: 'polars' or 'pandas'

        Returns:
            DataFrame with block IDs and proportions

        Raises:
            ImportError: If requested format library is not installed

        Examples:
            >>> df = result.to_dataframe()  # Polars by default
            >>> df.write_parquet('proportions.parquet')

            >>> df_pd = result.to_dataframe(format='pandas')
            >>> df_pd.to_csv('proportions.csv')
        """
        # Prepare data
        if self.block_ids.shape[1] == 3:
            # Block model indices (i, j, k)
            data = {
                "i": self.block_ids[:, 0],
                "j": self.block_ids[:, 1],
                "k": self.block_ids[:, 2],
                "proportion": self.proportions,
            }
        else:
            # Point indices
            data = {
                "point_id": self.block_ids[:, 0],
                "proportion": self.proportions,
            }

        if format == "polars":
            if not POLARS_AVAILABLE:
                raise ImportError(
                    "Polars is not installed. Install with: pip install polars"
                )
            return pl.DataFrame(data)

        elif format == "pandas":
            if not PANDAS_AVAILABLE:
                raise ImportError(
                    "Pandas is not installed. Install with: pip install pandas"
                )
            return pd.DataFrame(data)

        else:
            raise ValueError(f"Unknown format: {format}. Use 'polars' or 'pandas'")

    def to_array_3d(self, blocks: "BlockModel") -> np.ndarray:
        """Export as 3D array matching block model shape.

        Args:
            blocks: BlockModel that defines the grid shape

        Returns:
            3D array of proportions with shape matching block model

        Raises:
            ValueError: If this result is not from a block model calculation

        Examples:
            >>> result = gp.calculate(blocks, mesh)
            >>> arr_3d = result.to_array_3d(blocks)
            >>> print(arr_3d.shape)  # (nx, ny, nz)
        """
        if self.block_ids.shape[1] != 3:
            raise ValueError(
                "to_array_3d() only works with block model results, not point queries"
            )

        # Import here to avoid circular import
        from geoprops.blocks import BlockModel

        if not isinstance(blocks, BlockModel):
            raise TypeError(f"Expected BlockModel, got {type(blocks)}")

        # Create 3D array
        arr_3d = np.zeros(blocks.extent, dtype=np.float32)

        # Fill with proportions
        i = self.block_ids[:, 0]
        j = self.block_ids[:, 1]
        k = self.block_ids[:, 2]
        arr_3d[i, j, k] = self.proportions

        return arr_3d

    # Validation
    def _validate_compatible(self, other: "Result") -> None:
        """Check if two results are compatible for operations.

        Args:
            other: Another Result

        Raises:
            TypeError: If other is not a Result
            ValueError: If results have incompatible shapes
        """
        if not isinstance(other, Result):
            raise TypeError(f"Can only combine with another Result, got {type(other)}")

        if self.block_ids.shape != other.block_ids.shape:
            raise ValueError(
                f"Incompatible block_ids shapes: {self.block_ids.shape} vs {other.block_ids.shape}"
            )

        if not np.array_equal(self.block_ids, other.block_ids):
            raise ValueError(
                "Results have different block_ids. They must be from the same block model."
            )

    # Statistics
    @property
    def n_blocks(self) -> int:
        """Total number of blocks/points."""
        return len(self.proportions)

    @property
    def n_selected(self) -> int:
        """Number of blocks/points with proportion > 0."""
        return self.selected.sum()

    @property
    def mean_proportion(self) -> float:
        """Mean proportion across all blocks."""
        return float(self.proportions.mean())

    @property
    def total_volume_fraction(self) -> float:
        """Total volume as fraction of block model.

        This is the sum of all proportions divided by total number of blocks.
        For a block model with uniform block size, this represents the volume
        fraction occupied by the geometry.
        """
        return float(self.proportions.sum() / len(self.proportions))

    def __repr__(self) -> str:
        n_sel = self.n_selected
        pct = 100 * n_sel / self.n_blocks if self.n_blocks > 0 else 0
        return (
            f"Result(\n"
            f"  n_blocks={self.n_blocks},\n"
            f"  n_selected={n_sel} ({pct:.1f}%),\n"
            f"  mean_proportion={self.mean_proportion:.4f},\n"
            f"  method={self.metadata.get('method', 'unknown')}\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()
