"""Test point containment queries."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixtures_dir():
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def cube_mesh(fixtures_dir):
    """Load cube mesh for testing."""
    import geoprops as gp

    return gp.mesh.load(fixtures_dir / "cube.obj")


class TestPointContainment:
    """Test point-in-mesh queries."""

    def test_points_inside_cube(self, cube_mesh):
        """Test that points inside cube are detected."""
        import geoprops as gp

        # Points clearly inside the cube (-0.5 to 0.5 in all dimensions)
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Center
                [0.2, 0.2, 0.2],  # Inside
                [-0.2, -0.2, -0.2],  # Inside
            ]
        )

        result = gp.calculate(points, cube_mesh)

        # All points should be inside (proportion = 1.0)
        assert result.n_blocks == 3
        assert result.n_selected == 3
        assert np.all(result.proportions == 1.0)

    def test_points_outside_cube(self, cube_mesh):
        """Test that points outside cube are detected."""
        import geoprops as gp

        # Points clearly outside the cube
        points = np.array(
            [
                [1.0, 0.0, 0.0],  # Outside on +X
                [0.0, 1.0, 0.0],  # Outside on +Y
                [0.0, 0.0, 1.0],  # Outside on +Z
                [-1.0, -1.0, -1.0],  # Outside far corner
            ]
        )

        result = gp.calculate(points, cube_mesh)

        # All points should be outside (proportion = 0.0)
        assert result.n_blocks == 4
        assert result.n_selected == 0  # None selected
        assert np.all(result.proportions == 0.0)

    def test_mixed_inside_outside(self, cube_mesh):
        """Test mixture of inside and outside points."""
        import geoprops as gp

        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Inside
                [1.0, 1.0, 1.0],  # Outside
                [-0.2, 0.2, 0.0],  # Inside
                [0.6, 0.0, 0.0],  # Outside
            ]
        )

        result = gp.calculate(points, cube_mesh)

        assert result.n_blocks == 4
        assert result.n_selected == 2  # Two inside
        assert result.proportions[0] == 1.0  # First is inside
        assert result.proportions[1] == 0.0  # Second is outside
        assert result.proportions[2] == 1.0  # Third is inside
        assert result.proportions[3] == 0.0  # Fourth is outside

    def test_points_on_boundary(self, cube_mesh):
        """Test points exactly on boundary (edge case)."""
        import geoprops as gp

        # Points on the cube surface
        points = np.array(
            [
                [0.5, 0.0, 0.0],  # On +X face
                [0.0, 0.5, 0.0],  # On +Y face
                [0.0, 0.0, 0.5],  # On +Z face
            ]
        )

        result = gp.calculate(points, cube_mesh)

        # Boundary points behavior depends on trimesh implementation
        # Usually considered outside or on boundary
        assert result.n_blocks == 3
        # Just verify we get valid proportions (0 or 1)
        assert np.all(np.isin(result.proportions, [0.0, 1.0]))

    def test_single_point(self, cube_mesh):
        """Test querying a single point."""
        import geoprops as gp

        point = np.array([[0.0, 0.0, 0.0]])  # Single point inside

        result = gp.calculate(point, cube_mesh)

        assert result.n_blocks == 1
        assert result.proportions[0] == 1.0

    def test_empty_points_array(self, cube_mesh):
        """Test querying empty points array."""
        import geoprops as gp

        points = np.empty((0, 3))  # Empty array

        result = gp.calculate(points, cube_mesh)

        assert result.n_blocks == 0
        assert len(result.proportions) == 0

    def test_invalid_points_shape(self, cube_mesh):
        """Test that invalid point shapes raise errors."""
        import geoprops as gp

        # 1D array (should be 2D)
        with pytest.raises(ValueError, match="must be.*array"):
            gp.calculate(np.array([0, 0, 0]), cube_mesh)

        # Wrong number of columns
        with pytest.raises(ValueError, match="must be.*3"):
            gp.calculate(np.array([[0, 0]]), cube_mesh)


class TestPointContainmentDataFrames:
    """Test point containment with DataFrames."""

    def test_polars_dataframe(self, cube_mesh):
        """Test point containment with Polars DataFrame."""
        import geoprops as gp

        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Create DataFrame with point coordinates
        df = pl.DataFrame(
            {
                "X": [0.0, 1.0, -0.2],
                "Y": [0.0, 1.0, 0.2],
                "Z": [0.0, 1.0, 0.0],
            }
        )

        result = gp.calculate(df[["X", "Y", "Z"]], cube_mesh)

        assert result.n_blocks == 3
        assert result.proportions[0] == 1.0  # Inside
        assert result.proportions[1] == 0.0  # Outside
        assert result.proportions[2] == 1.0  # Inside

    def test_pandas_dataframe(self, cube_mesh):
        """Test point containment with Pandas DataFrame."""
        import geoprops as gp

        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not installed")

        # Create DataFrame with point coordinates
        df = pd.DataFrame(
            {
                "X": [0.0, 1.0, -0.2],
                "Y": [0.0, 1.0, 0.2],
                "Z": [0.0, 1.0, 0.0],
            }
        )

        result = gp.calculate(df[["X", "Y", "Z"]], cube_mesh)

        assert result.n_blocks == 3
        assert result.proportions[0] == 1.0  # Inside
        assert result.proportions[1] == 0.0  # Outside
        assert result.proportions[2] == 1.0  # Inside

    def test_add_result_to_dataframe(self, cube_mesh):
        """Test adding containment result to DataFrame."""
        import geoprops as gp

        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Create DataFrame
        df = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "X": [0.0, 1.0, -0.2],
                "Y": [0.0, 1.0, 0.2],
                "Z": [0.0, 1.0, 0.0],
            }
        )

        # Calculate containment
        result = gp.calculate(df[["X", "Y", "Z"]], cube_mesh)

        # Add to DataFrame
        df = df.with_columns(
            pl.Series("in_domain", result.selected), pl.Series("proportion", result.proportions)
        )

        assert "in_domain" in df.columns
        assert "proportion" in df.columns
        assert df["in_domain"][0] == True  # First point inside
        assert df["in_domain"][1] == False  # Second point outside


class TestResultProperties:
    """Test Result object properties for point queries."""

    def test_result_metadata(self, cube_mesh):
        """Test that result contains appropriate metadata."""
        import geoprops as gp

        points = np.array([[0.0, 0.0, 0.0]])
        result = gp.calculate(points, cube_mesh)

        assert "method" in result.metadata
        assert "n_points" in result.metadata or "n_blocks" in result.metadata

    def test_result_selected_property(self, cube_mesh):
        """Test the .selected property."""
        import geoprops as gp

        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Inside
                [1.0, 1.0, 1.0],  # Outside
            ]
        )

        result = gp.calculate(points, cube_mesh)

        selected = result.selected
        assert len(selected) == 2
        assert selected[0] == True
        assert selected[1] == False

    def test_result_threshold(self, cube_mesh):
        """Test the .threshold() method."""
        import geoprops as gp

        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Inside
                [1.0, 1.0, 1.0],  # Outside
            ]
        )

        result = gp.calculate(points, cube_mesh)

        # Threshold at 0.5 should give same as .selected for binary values
        thresh = result.threshold(0.5)
        assert np.array_equal(thresh, result.selected)
