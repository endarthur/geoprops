"""Test basic package imports and core class initialization."""

import numpy as np
import pytest


def test_import_geoprops():
    """Test that geoprops can be imported."""
    import geoprops as gp

    assert gp.__version__ is not None


def test_import_core_classes():
    """Test that core classes can be imported."""
    from geoprops import Mesh, BlockModel, Polygon, Result

    assert Mesh is not None
    assert BlockModel is not None
    assert Polygon is not None
    assert Result is not None


def test_mesh_creation():
    """Test creating a simple mesh."""
    from geoprops import Mesh

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = Mesh(vertices=vertices, faces=faces)

    assert mesh.n_vertices == 3
    assert mesh.n_faces == 1
    assert mesh.is_valid()


def test_blockmodel_creation():
    """Test creating a simple block model."""
    from geoprops import BlockModel

    blocks = BlockModel(
        origin=(0, 0, 0), size=(10, 10, 10), extent=(10, 10, 10), crs="EPSG:31983"
    )

    assert blocks.n_blocks == 1000
    assert blocks.extent == (10, 10, 10)


def test_polygon_creation_2d():
    """Test creating a 2D polygon."""
    from geoprops import Polygon

    points = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)

    poly = Polygon(points=points, crs="EPSG:31983")

    assert poly.is_2d
    assert not poly.is_3d
    assert poly.n_points == 4


def test_polygon_creation_3d():
    """Test creating a 3D polygon."""
    from geoprops import Polygon

    points = np.array(
        [[0, 0, 0], [100, 0, 10], [100, 50, 20], [0, 50, 15]], dtype=np.float64
    )

    poly = Polygon(points=points, crs="EPSG:31983")

    assert poly.is_3d
    assert not poly.is_2d
    assert poly.n_points == 4


def test_result_creation():
    """Test creating a Result object."""
    from geoprops import Result

    block_ids = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    proportions = np.array([0.5, 0.8, 0.3], dtype=np.float64)

    result = Result(block_ids=block_ids, proportions=proportions)

    assert result.n_blocks == 3
    assert result.n_selected == 3  # All have proportion > 0
    assert result.mean_proportion == pytest.approx(0.5333, rel=1e-3)


def test_result_fuzzy_union():
    """Test fuzzy union operation."""
    from geoprops import Result

    block_ids = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    a = Result(block_ids=block_ids, proportions=np.array([0.3, 0.7]))
    b = Result(block_ids=block_ids, proportions=np.array([0.5, 0.4]))

    c = a | b

    assert np.array_equal(c.proportions, [0.5, 0.7])  # max(0.3, 0.5), max(0.7, 0.4)


def test_result_fuzzy_intersection():
    """Test fuzzy intersection operation."""
    from geoprops import Result

    block_ids = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    a = Result(block_ids=block_ids, proportions=np.array([0.3, 0.7]))
    b = Result(block_ids=block_ids, proportions=np.array([0.5, 0.4]))

    c = a & b

    assert np.array_equal(c.proportions, [0.3, 0.4])  # min(0.3, 0.5), min(0.7, 0.4)


def test_result_complement():
    """Test fuzzy complement operation."""
    from geoprops import Result

    block_ids = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    a = Result(block_ids=block_ids, proportions=np.array([0.3, 0.7]))

    b = ~a

    assert np.allclose(b.proportions, [0.7, 0.3])  # 1 - 0.3, 1 - 0.7


def test_config():
    """Test configuration object."""
    from geoprops import config

    # Check defaults
    assert config.default_method in ("auto", "gpu", "cpu")
    assert config.verbose >= 0
    assert config.cpu_workers > 0

    # Test setter
    original_verbose = config.verbose
    config.verbose = 2
    assert config.verbose == 2

    # Reset
    config.verbose = original_verbose
