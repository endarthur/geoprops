"""Test mesh I/O functionality."""

import os
from pathlib import Path

import numpy as np
import pytest


# Fixture for test data directory
@pytest.fixture
def fixtures_dir():
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def cube_obj(fixtures_dir):
    """Get path to cube.obj test file."""
    return fixtures_dir / "cube.obj"


@pytest.fixture
def triangle_obj(fixtures_dir):
    """Get path to triangle.obj test file."""
    return fixtures_dir / "triangle.obj"


@pytest.fixture
def tetrahedron_obj(fixtures_dir):
    """Get path to tetrahedron.obj test file."""
    return fixtures_dir / "tetrahedron.obj"


class TestMeshLoad:
    """Test mesh loading functionality."""

    def test_load_obj_basic(self, triangle_obj):
        """Test loading a simple OBJ file."""
        import geoprops as gp

        mesh = gp.mesh.load(str(triangle_obj))

        assert mesh is not None
        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1
        assert mesh.is_valid()

    def test_load_obj_cube(self, cube_obj):
        """Test loading a cube mesh."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj))

        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12
        assert mesh.is_valid()

    def test_load_obj_tetrahedron(self, tetrahedron_obj):
        """Test loading a tetrahedron mesh."""
        import geoprops as gp

        mesh = gp.mesh.load(str(tetrahedron_obj))

        assert mesh.n_vertices == 4
        assert mesh.n_faces == 4
        assert mesh.is_valid()

    def test_load_with_crs(self, cube_obj):
        """Test loading mesh with CRS specification."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj), crs="EPSG:31983")

        assert mesh.crs == "EPSG:31983"

    def test_load_with_metadata(self, cube_obj):
        """Test loading mesh with metadata."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj), name="test_cube", category="test")

        assert mesh.metadata.get("name") == "test_cube"
        assert mesh.metadata.get("category") == "test"

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        import geoprops as gp

        with pytest.raises(FileNotFoundError):
            gp.mesh.load("nonexistent_file.obj")

    def test_load_from_path_object(self, cube_obj):
        """Test loading from pathlib.Path object."""
        import geoprops as gp

        mesh = gp.mesh.load(cube_obj)  # Pass Path object directly

        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12

    def test_mesh_bounds(self, cube_obj):
        """Test that loaded mesh has correct bounds."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj))
        bounds = mesh.bounds

        # Cube is -0.5 to 0.5 in all dimensions
        assert np.allclose(bounds[0], [-0.5, -0.5, -0.5])
        assert np.allclose(bounds[1], [0.5, 0.5, 0.5])

    def test_cube_is_closed(self, cube_obj):
        """Test that cube mesh is detected as closed."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj))

        # Cube should be a closed (watertight) mesh
        assert mesh.is_closed()

    def test_triangle_is_not_closed(self, triangle_obj):
        """Test that single triangle is not closed."""
        import geoprops as gp

        mesh = gp.mesh.load(str(triangle_obj))

        # Single triangle is not closed
        assert not mesh.is_closed()


class TestMeshSave:
    """Test mesh saving functionality."""

    def test_save_obj(self, tmp_path, cube_obj):
        """Test saving mesh to OBJ format."""
        import geoprops as gp

        # Load mesh
        mesh = gp.mesh.load(str(cube_obj))

        # Save to temp file
        output_file = tmp_path / "output.obj"
        gp.mesh.save(mesh, str(output_file))

        assert output_file.exists()

        # Load it back and verify
        mesh2 = gp.mesh.load(str(output_file))
        assert mesh2.n_vertices == mesh.n_vertices
        assert mesh2.n_faces == mesh.n_faces

    def test_save_with_format(self, tmp_path, cube_obj):
        """Test saving mesh with explicit format specification."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj))

        # Save as OBJ
        output_file = tmp_path / "output_explicit.obj"
        gp.mesh.save(mesh, str(output_file), format="obj")

        assert output_file.exists()

    def test_roundtrip_preserves_data(self, tmp_path, cube_obj):
        """Test that save/load roundtrip preserves mesh data."""
        import geoprops as gp

        # Load original
        mesh1 = gp.mesh.load(str(cube_obj))

        # Save and reload
        output_file = tmp_path / "roundtrip.obj"
        gp.mesh.save(mesh1, str(output_file))
        mesh2 = gp.mesh.load(str(output_file))

        # Compare
        assert np.allclose(mesh1.vertices, mesh2.vertices)
        assert np.array_equal(mesh1.faces, mesh2.faces)


class TestMeshVolume:
    """Test mesh volume calculation."""

    def test_cube_volume(self, cube_obj):
        """Test that cube volume is calculated correctly."""
        import geoprops as gp

        mesh = gp.mesh.load(str(cube_obj))

        # Cube with side length 1.0 should have volume 1.0
        # Use abs() since negative volume just means inverted normals
        volume = abs(mesh.volume())
        assert np.isclose(volume, 1.0, rtol=1e-3)

    def test_tetrahedron_volume(self, tetrahedron_obj):
        """Test tetrahedron volume calculation."""
        import geoprops as gp

        mesh = gp.mesh.load(str(tetrahedron_obj))

        # Should have some positive volume
        volume = mesh.volume()
        assert volume > 0

    def test_open_mesh_volume_raises(self, triangle_obj):
        """Test that calculating volume of open mesh raises error."""
        import geoprops as gp

        mesh = gp.mesh.load(str(triangle_obj))

        with pytest.raises(ValueError, match="non-watertight"):
            mesh.volume()
