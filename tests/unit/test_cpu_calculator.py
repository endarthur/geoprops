"""Test CPU calculator for block model proportions."""

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


class TestCPUBlockCalculations:
    """Test CPU-based block model proportion calculations."""

    def test_cube_in_single_block(self, cube_mesh):
        """Test cube mesh that fits entirely in one block."""
        import geoprops as gp

        # Create block model with one large block that contains the entire cube
        # Cube is -0.5 to 0.5, so block from -1 to 1 contains it fully
        blocks = gp.BlockModel(
            origin=(-1.0, -1.0, -1.0),
            size=(2.0, 2.0, 2.0),  # Single 2×2×2 block
            extent=(1, 1, 1),
            crs=None
        )

        # Use higher resolution for accurate results, explicitly request CPU
        result = gp.calculate(blocks, cube_mesh, method='cpu', resolution=20)

        # Should have 1 block
        assert result.n_blocks == 1

        # Cube volume is 1.0, block volume is 8.0
        # So proportion should be 1.0 / 8.0 = 0.125
        # Regular grid discretization is deterministic and accurate
        expected_proportion = 1.0 / 8.0
        assert np.isclose(result.proportions[0], expected_proportion, rtol=0.05)

    def test_cube_spans_multiple_blocks(self, cube_mesh):
        """Test cube that spans multiple blocks."""
        import geoprops as gp

        # Create 2×2×2 grid of small blocks
        # Each block is 0.5×0.5×0.5, grid covers -0.5 to 0.5 in all dims
        blocks = gp.BlockModel(
            origin=(-0.5, -0.5, -0.5),
            size=(0.5, 0.5, 0.5),
            extent=(2, 2, 2),
            crs=None
        )

        result = gp.calculate(blocks, cube_mesh, method='cpu')

        # Should have 8 blocks total
        assert result.n_blocks == 8

        # All 8 blocks should be fully occupied (cube fills all of them)
        assert result.n_selected == 8
        assert np.allclose(result.proportions, 1.0)

    def test_blocks_outside_mesh(self, cube_mesh):
        """Test blocks that don't intersect mesh at all."""
        import geoprops as gp

        # Create blocks far from the cube
        blocks = gp.BlockModel(
            origin=(10.0, 10.0, 10.0),
            size=(1.0, 1.0, 1.0),
            extent=(3, 3, 3),
            crs=None
        )

        result = gp.calculate(blocks, cube_mesh)

        # Should have 27 blocks
        assert result.n_blocks == 27

        # None should be selected (all proportions = 0)
        assert result.n_selected == 0
        assert np.all(result.proportions == 0.0)

    def test_partial_block_occupation(self, cube_mesh):
        """Test blocks that are partially occupied."""
        import geoprops as gp

        # Create larger blocks where cube only partially fills them
        blocks = gp.BlockModel(
            origin=(-1.0, -1.0, -1.0),
            size=(1.0, 1.0, 1.0),
            extent=(2, 2, 2),
            crs=None
        )

        result = gp.calculate(blocks, cube_mesh, resolution=15)

        # Should have 8 blocks
        assert result.n_blocks == 8

        # All 8 blocks should have some occupation (cube touches all)
        assert result.n_selected == 8

        # Proportions should be between 0 and 1
        assert np.all(result.proportions > 0)
        assert np.all(result.proportions <= 1.0)

        # Sum of (proportion × block_volume) should equal cube volume
        block_volume = 1.0 * 1.0 * 1.0
        total_volume = (result.proportions * block_volume).sum()
        expected_volume = 1.0  # Cube volume
        # Regular discretization with resolution=15 gives good accuracy
        # Some error expected due to discretization at block boundaries
        assert np.isclose(total_volume, expected_volume, rtol=0.20)

    def test_fine_grid(self, cube_mesh):
        """Test with fine grid to verify volume conservation."""
        import geoprops as gp

        # Create fine grid covering the cube
        blocks = gp.BlockModel(
            origin=(-0.5, -0.5, -0.5),
            size=(0.1, 0.1, 0.1),
            extent=(10, 10, 10),
            crs=None
        )

        # Use default resolution (15) which is sufficient for these small blocks
        result = gp.calculate(blocks, cube_mesh)

        # Should have 1000 blocks
        assert result.n_blocks == 1000

        # Many blocks should be occupied
        assert result.n_selected > 100

        # Volume conservation: sum of (proportion × block_volume) ≈ cube volume
        block_volume = 0.1 * 0.1 * 0.1
        total_volume = (result.proportions * block_volume).sum()
        expected_volume = 1.0
        assert np.isclose(total_volume, expected_volume, rtol=0.10)

    def test_empty_block_model(self):
        """Test edge case with zero blocks."""
        import geoprops as gp

        # This should raise an error (can't have 0 blocks)
        with pytest.raises(ValueError):
            blocks = gp.BlockModel(
                origin=(0, 0, 0),
                size=(1, 1, 1),
                extent=(0, 0, 0),  # Invalid!
            )

    def test_result_to_dataframe(self, cube_mesh):
        """Test converting result to DataFrame."""
        import geoprops as gp

        blocks = gp.BlockModel(
            origin=(-0.5, -0.5, -0.5),
            size=(0.5, 0.5, 0.5),
            extent=(2, 2, 2),
        )

        result = gp.calculate(blocks, cube_mesh)
        df = result.to_dataframe()

        # Should have columns for i, j, k, proportion
        assert 'i' in df.columns
        assert 'j' in df.columns
        assert 'k' in df.columns
        assert 'proportion' in df.columns

        # Should have 8 rows (8 blocks)
        assert len(df) == 8

    def test_result_to_3d_array(self, cube_mesh):
        """Test converting result to 3D array."""
        import geoprops as gp

        blocks = gp.BlockModel(
            origin=(-0.5, -0.5, -0.5),
            size=(0.5, 0.5, 0.5),
            extent=(2, 2, 2),
        )

        result = gp.calculate(blocks, cube_mesh)
        arr_3d = result.to_array_3d(blocks)

        # Should have shape matching block model
        assert arr_3d.shape == (2, 2, 2)

        # All values should be proportions (0 to 1)
        assert np.all(arr_3d >= 0)
        assert np.all(arr_3d <= 1.0)


class TestMethodSelection:
    """Test that CPU method is used when appropriate."""

    def test_method_metadata(self, cube_mesh):
        """Test that result contains method metadata."""
        import geoprops as gp

        blocks = gp.BlockModel(
            origin=(-1, -1, -1),
            size=(1, 1, 1),
            extent=(2, 2, 2),
        )

        result = gp.calculate(blocks, cube_mesh)

        # Should have method in metadata
        assert 'method' in result.metadata

        # Auto should select either CPU or GPU (depending on availability)
        method = result.metadata['method'].lower()
        assert any(keyword in method for keyword in ['cpu', 'ray', 'gpu', 'depth'])

    def test_explicit_cpu_method(self, cube_mesh):
        """Test explicitly requesting CPU method."""
        import geoprops as gp

        blocks = gp.BlockModel(
            origin=(-1, -1, -1),
            size=(1, 1, 1),
            extent=(2, 2, 2),
        )

        # Explicitly request CPU
        result = gp.calculate(blocks, cube_mesh, method='cpu')

        assert 'cpu' in result.metadata['method'].lower() or 'ray' in result.metadata['method'].lower()


class TestVolumeConservation:
    """Test that volume is conserved across different grid resolutions."""

    def test_volume_conservation_coarse(self, cube_mesh):
        """Test volume conservation with coarse grid."""
        import geoprops as gp

        blocks = gp.BlockModel(
            origin=(-1, -1, -1),
            size=(0.5, 0.5, 0.5),
            extent=(4, 4, 4),
        )

        # Use default resolution (15)
        result = gp.calculate(blocks, cube_mesh)

        block_volume = 0.5 * 0.5 * 0.5
        total_volume = (result.proportions * block_volume).sum()

        # Should be close to 1.0 (cube volume)
        # Regular discretization gives good accuracy
        assert np.isclose(total_volume, 1.0, rtol=0.10)

    def test_volume_conservation_medium(self, cube_mesh):
        """Test volume conservation with medium grid."""
        import geoprops as gp

        blocks = gp.BlockModel(
            origin=(-0.6, -0.6, -0.6),
            size=(0.2, 0.2, 0.2),
            extent=(6, 6, 6),
        )

        # Use default resolution (15)
        result = gp.calculate(blocks, cube_mesh)

        block_volume = 0.2 * 0.2 * 0.2
        total_volume = (result.proportions * block_volume).sum()

        # Should be close to 1.0
        # Regular discretization gives good accuracy
        assert np.isclose(total_volume, 1.0, rtol=0.10)


class TestRotatedBlocks:
    """Test with rotated block models."""

    def test_rotated_blocks_basic(self, cube_mesh):
        """Test basic calculation with rotated blocks."""
        import geoprops as gp

        # Create rotated block model
        blocks = gp.BlockModel(
            origin=(0, 0, -0.5),
            size=(0.5, 0.5, 0.5),
            extent=(3, 3, 2),
            rotation=45,  # 45° rotation
        )

        result = gp.calculate(blocks, cube_mesh)

        # Should complete without error
        assert result.n_blocks == 18

        # Some blocks should be occupied
        assert result.n_selected > 0

        # Proportions should be valid
        assert np.all(result.proportions >= 0)
        assert np.all(result.proportions <= 1.0)
