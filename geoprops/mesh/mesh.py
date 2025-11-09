"""Mesh class for triangular surface meshes.

This module provides the Mesh class representing 3D triangular meshes
used for geological domains, surfaces, and volumes.
"""

from typing import Any

import numpy as np


class Mesh:
    """Triangular mesh representing a 3D domain or surface.

    A mesh consists of vertices (3D points) and faces (triangles defined by
    vertex indices). Meshes can represent closed volumes (geological domains)
    or open surfaces (topography, fault planes).

    Attributes:
        vertices: (N, 3) array of vertex coordinates [x, y, z]
        faces: (M, 3) array of triangle vertex indices
        metadata: Dictionary for name, category, custom properties
        crs: Coordinate reference system (EPSG code or WKT string)

    Examples:
        >>> # Load from file
        >>> mesh = gp.mesh.load('ore_domain.obj')

        >>> # Create from arrays
        >>> mesh = gp.Mesh(
        ...     vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        ...     faces=np.array([[0, 1, 2]]),
        ...     crs='EPSG:31983'
        ... )

        >>> # Check properties
        >>> print(mesh.is_closed())
        >>> print(mesh.volume())
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        metadata: dict[str, Any] | None = None,
        crs: str | None = None,
    ):
        """Initialize Mesh.

        Args:
            vertices: (N, 3) float array of vertex coordinates
            faces: (M, 3) int array of triangle vertex indices
            metadata: Optional metadata dictionary
            crs: Optional coordinate reference system

        Raises:
            ValueError: If inputs have invalid shapes
        """
        # Validate vertices
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices, dtype=np.float64)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must be (N, 3) array, got shape {vertices.shape}"
            )

        # Validate faces
        if not isinstance(faces, np.ndarray):
            faces = np.array(faces, dtype=np.int32)

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3) array, got shape {faces.shape}")

        # Check indices are valid
        if faces.size > 0:
            if faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError(
                    f"Face indices must be in range [0, {len(vertices)-1}], "
                    f"got range [{faces.min()}, {faces.max()}]"
                )

        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int32)
        self.metadata = metadata or {}
        self.crs = crs

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces (triangles)."""
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box of the mesh.

        Returns:
            Array of shape (2, 3) with [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        """
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def is_valid(self) -> bool:
        """Check if mesh is valid (has vertices and faces).

        Returns:
            True if mesh has at least one vertex and one face
        """
        return self.n_vertices > 0 and self.n_faces > 0

    def is_closed(self) -> bool:
        """Check if mesh is watertight (closed manifold).

        A mesh is closed if every edge is shared by exactly two faces.

        Returns:
            True if mesh is watertight

        Notes:
            This uses trimesh for the actual check. Falls back to False
            if trimesh is not available.
        """
        try:
            import trimesh

            mesh_tri = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            return mesh_tri.is_watertight
        except ImportError:
            # If trimesh not available, can't check
            return False

    def volume(self) -> float:
        """Calculate enclosed volume.

        Returns:
            Volume of the mesh (0 if not closed)

        Raises:
            ValueError: If mesh is not closed

        Notes:
            Uses trimesh for volume calculation.
        """
        try:
            import trimesh

            mesh_tri = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

            if not mesh_tri.is_watertight:
                raise ValueError(
                    "Cannot calculate volume of non-watertight mesh. "
                    "Use mesh.is_closed() to check first."
                )

            return float(mesh_tri.volume)
        except ImportError:
            raise ImportError("trimesh is required for volume calculation")

    def transform(self, to_crs: str, z_offset: float = 0.0) -> "Mesh":
        """Transform to different CRS with optional vertical offset.

        Args:
            to_crs: Target coordinate reference system (EPSG or WKT)
            z_offset: Vertical offset to add after transformation (meters)

        Returns:
            New Mesh in target CRS

        Raises:
            ImportError: If pyproj is not available
            ValueError: If source CRS is not set

        Examples:
            >>> mesh_utm23 = mesh.transform('EPSG:31983')
            >>> mesh_utm24 = mesh.transform('EPSG:31984', z_offset=10.0)
        """
        if self.crs is None:
            raise ValueError(
                "Source CRS is not set. Set mesh.crs before transforming."
            )

        try:
            from pyproj import Transformer
        except ImportError:
            raise ImportError("pyproj is required for CRS transformations")

        # Create transformer
        transformer = Transformer.from_crs(self.crs, to_crs, always_xy=True)

        # Transform X, Y
        x_new, y_new = transformer.transform(
            self.vertices[:, 0], self.vertices[:, 1]
        )

        # Manual Z offset
        z_new = self.vertices[:, 2] + z_offset

        # Create new mesh
        vertices_new = np.column_stack([x_new, y_new, z_new])

        return Mesh(
            vertices=vertices_new,
            faces=self.faces.copy(),
            metadata=self.metadata.copy(),
            crs=to_crs,
        )

    def translate(self, offset: tuple[float, float, float]) -> "Mesh":
        """Translate mesh by offset vector.

        Args:
            offset: Translation vector (dx, dy, dz)

        Returns:
            New translated Mesh

        Examples:
            >>> shifted = mesh.translate((100, 200, 0))
        """
        offset_arr = np.array(offset, dtype=np.float64)
        if offset_arr.shape != (3,):
            raise ValueError(f"offset must be (dx, dy, dz), got {offset}")

        vertices_new = self.vertices + offset_arr

        return Mesh(
            vertices=vertices_new,
            faces=self.faces.copy(),
            metadata=self.metadata.copy(),
            crs=self.crs,
        )

    @property
    def __geo_interface__(self) -> dict:
        """GeoJSON-like representation for ecosystem compatibility.

        Returns a simplified representation that can be used with
        Shapely, GeoPandas, and other tools that support __geo_interface__.

        Note: This represents the mesh as a MultiPolygon of triangles,
        which is not ideal but provides basic compatibility.
        """
        # Convert triangles to polygons
        coordinates = []
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            # Close the polygon
            coords = [
                tuple(v0),
                tuple(v1),
                tuple(v2),
                tuple(v0),
            ]
            coordinates.append([coords])

        return {
            "type": "MultiPolygon",
            "coordinates": coordinates,
        }

    @classmethod
    def from_trimesh(cls, mesh_tri: "trimesh.Trimesh", **kwargs) -> "Mesh":
        """Create Mesh from trimesh.Trimesh object.

        Args:
            mesh_tri: trimesh.Trimesh object
            **kwargs: Additional arguments (metadata, crs)

        Returns:
            New Mesh instance

        Examples:
            >>> import trimesh
            >>> tm = trimesh.load('file.obj')
            >>> mesh = gp.Mesh.from_trimesh(tm, crs='EPSG:31983')
        """
        return cls(
            vertices=mesh_tri.vertices,
            faces=mesh_tri.faces,
            **kwargs,
        )

    def to_trimesh(self) -> "trimesh.Trimesh":
        """Convert to trimesh.Trimesh object.

        Returns:
            trimesh.Trimesh object

        Raises:
            ImportError: If trimesh is not available
        """
        try:
            import trimesh

            return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        except ImportError:
            raise ImportError("trimesh is required for to_trimesh()")

    def copy(self) -> "Mesh":
        """Create a deep copy of this mesh.

        Returns:
            New Mesh with copied data
        """
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            metadata=self.metadata.copy(),
            crs=self.crs,
        )

    def __repr__(self) -> str:
        name = self.metadata.get("name", "unnamed")
        bounds = self.bounds
        return (
            f"Mesh('{name}', "
            f"n_vertices={self.n_vertices}, "
            f"n_faces={self.n_faces}, "
            f"bounds=[({bounds[0,0]:.1f}, {bounds[0,1]:.1f}, {bounds[0,2]:.1f}), "
            f"({bounds[1,0]:.1f}, {bounds[1,1]:.1f}, {bounds[1,2]:.1f})], "
            f"crs={self.crs!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
