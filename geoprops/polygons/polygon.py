"""Polygon class for 2D and 3D polygons.

This module provides the Polygon class for representing 2D polygons
(footprints) and 3D polygons (fault planes, vein traces) with optional
extrusion to create 3D volumes.
"""

from typing import Any

import numpy as np


class Polygon:
    """2D or 3D polygon for selections and extrusions.

    A polygon can be 2D (footprint with X,Y coordinates) or 3D (fault plane,
    vein trace with X,Y,Z coordinates). Polygons can be extruded to create
    3D mesh volumes.

    Attributes:
        points: (N, 2) or (N, 3) array of polygon vertices
        holes: Optional list of hole polygons (same dimension as outer polygon)
        crs: Coordinate reference system (EPSG code or WKT string)

    Examples:
        >>> # 2D polygon (footprint)
        >>> poly = gp.Polygon(
        ...     points=np.array([[0, 0], [100, 0], [100, 50], [0, 50]]),
        ...     crs='EPSG:31983'
        ... )

        >>> # 3D polygon (fault plane)
        >>> fault = gp.Polygon(
        ...     points=np.array([[x1, y1, z1], [x2, y2, z2], ...]),
        ...     crs='EPSG:31983'
        ... )

        >>> # Extrude to 3D volume
        >>> volume = poly.extrude(distance=(0, 100))
    """

    def __init__(
        self,
        points: np.ndarray,
        holes: list[np.ndarray] | None = None,
        crs: str | None = None,
    ):
        """Initialize Polygon.

        Args:
            points: (N, 2) or (N, 3) array of polygon vertices
            holes: Optional list of hole polygons
            crs: Optional coordinate reference system

        Raises:
            ValueError: If inputs have invalid shapes
        """
        # Validate points
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                f"points must be (N, 2) or (N, 3) array, got shape {points.shape}"
            )

        if points.shape[0] < 3:
            raise ValueError(
                f"polygon must have at least 3 points, got {points.shape[0]}"
            )

        # Validate holes
        if holes is not None:
            if not isinstance(holes, list):
                holes = [holes]

            for i, hole in enumerate(holes):
                if not isinstance(hole, np.ndarray):
                    holes[i] = np.array(hole, dtype=np.float64)

                if holes[i].shape[1] != points.shape[1]:
                    raise ValueError(
                        f"hole {i} has dimension {holes[i].shape[1]}, "
                        f"but outer polygon has dimension {points.shape[1]}"
                    )

                if holes[i].shape[0] < 3:
                    raise ValueError(
                        f"hole {i} must have at least 3 points, got {holes[i].shape[0]}"
                    )

        self.points = points.astype(np.float64)
        self.holes = holes
        self.crs = crs

    @property
    def is_2d(self) -> bool:
        """Check if polygon is 2D.

        Returns:
            True if polygon has (N, 2) shape
        """
        return self.points.shape[1] == 2

    @property
    def is_3d(self) -> bool:
        """Check if polygon is 3D.

        Returns:
            True if polygon has (N, 3) shape
        """
        return self.points.shape[1] == 3

    @property
    def n_points(self) -> int:
        """Number of points in the polygon."""
        return len(self.points)

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box of the polygon.

        Returns:
            For 2D: array of shape (2, 2) with [[xmin, ymin], [xmax, ymax]]
            For 3D: array of shape (2, 3) with [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        """
        return np.array([self.points.min(axis=0), self.points.max(axis=0)])

    def extrude(
        self,
        distance: float | tuple[float, float] | None = None,
        direction: np.ndarray | None = None,
    ) -> "Mesh":
        """Extrude polygon to create 3D volume.

        For 2D polygons:
            - distance=(base, top): Vertical extrusion to elevation range
            - direction=[dx, dy, dz]: Extrude along vector

        For 3D polygons:
            - distance=value: Extrude along polygon normal
            - direction=[dx, dy, dz]: Extrude along vector

        Args:
            distance: Extrusion distance or (min, max) elevation range
            direction: Optional extrusion direction vector

        Returns:
            Mesh representing the extruded volume

        Raises:
            ValueError: If neither distance nor direction is specified
            NotImplementedError: Placeholder for now

        Examples:
            >>> # 2D polygon, vertical extrusion
            >>> poly_2d.extrude(distance=(0, 100))  # 0 to 100m elevation

            >>> # 3D polygon (fault plane), extrude along normal
            >>> fault_poly.extrude(distance=50)  # 50m thick fault zone

            >>> # 3D polygon, extrude in specific direction
            >>> poly_3d.extrude(direction=[10, 0, 5])  # Offset ribbon
        """
        # Import here to avoid circular import
        from geoprops.mesh import Mesh

        # This is a placeholder - full implementation will be done later
        # For now, we'll raise NotImplementedError
        raise NotImplementedError(
            "Polygon extrusion is not yet implemented. "
            "This will be added in Phase 4 of the roadmap."
        )

    @property
    def __geo_interface__(self) -> dict:
        """GeoJSON representation for Shapely/GeoPandas compatibility.

        Returns:
            GeoJSON-like dictionary representing the polygon

        Note:
            For 3D polygons, Z coordinates are included as the third element
            in each coordinate tuple.
        """
        # Convert points to list of tuples
        if self.is_2d:
            # Ensure polygon is closed
            if not np.array_equal(self.points[0], self.points[-1]):
                points_closed = np.vstack([self.points, self.points[0:1]])
            else:
                points_closed = self.points

            coordinates = [tuple(p) for p in points_closed]

            # Add holes if any
            if self.holes:
                coords_with_holes = [coordinates]
                for hole in self.holes:
                    if not np.array_equal(hole[0], hole[-1]):
                        hole_closed = np.vstack([hole, hole[0:1]])
                    else:
                        hole_closed = hole
                    coords_with_holes.append([tuple(p) for p in hole_closed])
                coordinates = coords_with_holes
            else:
                coordinates = [coordinates]

            return {
                "type": "Polygon",
                "coordinates": coordinates,
            }

        else:  # 3D polygon
            # Ensure polygon is closed
            if not np.array_equal(self.points[0], self.points[-1]):
                points_closed = np.vstack([self.points, self.points[0:1]])
            else:
                points_closed = self.points

            coordinates = [tuple(p) for p in points_closed]

            return {
                "type": "LineString",  # 3D polygon as LineString
                "coordinates": coordinates,
            }

    @classmethod
    def from_geo_interface(cls, geo: dict, **kwargs) -> "Polygon":
        """Create Polygon from __geo_interface__ object.

        Args:
            geo: Dictionary with __geo_interface__ protocol
            **kwargs: Additional arguments (crs, etc.)

        Returns:
            New Polygon instance

        Raises:
            ValueError: If geo has unsupported type

        Examples:
            >>> # From Shapely polygon (if available)
            >>> import shapely.geometry
            >>> shapely_poly = shapely.geometry.Polygon([...])
            >>> poly = gp.Polygon.from_geo_interface(shapely_poly.__geo_interface__)
        """
        # Handle object with __geo_interface__
        if hasattr(geo, "__geo_interface__"):
            geo = geo.__geo_interface__

        geom_type = geo.get("type")

        if geom_type == "Polygon":
            coords = geo["coordinates"]
            # First ring is outer, rest are holes
            outer = np.array(coords[0], dtype=np.float64)
            holes = [np.array(h, dtype=np.float64) for h in coords[1:]] if len(coords) > 1 else None

            return cls(points=outer, holes=holes, **kwargs)

        elif geom_type == "LineString":
            # Treat as 3D polygon
            coords = np.array(geo["coordinates"], dtype=np.float64)
            return cls(points=coords, **kwargs)

        else:
            raise ValueError(
                f"Unsupported geometry type: {geom_type}. "
                f"Expected 'Polygon' or 'LineString'"
            )

    def copy(self) -> "Polygon":
        """Create a deep copy of this polygon.

        Returns:
            New Polygon with copied data
        """
        holes_copy = [h.copy() for h in self.holes] if self.holes else None
        return Polygon(
            points=self.points.copy(),
            holes=holes_copy,
            crs=self.crs,
        )

    def __repr__(self) -> str:
        dim = "2D" if self.is_2d else "3D"
        bounds = self.bounds
        n_holes = len(self.holes) if self.holes else 0

        if self.is_2d:
            bounds_str = (
                f"[({bounds[0,0]:.1f}, {bounds[0,1]:.1f}), "
                f"({bounds[1,0]:.1f}, {bounds[1,1]:.1f})]"
            )
        else:
            bounds_str = (
                f"[({bounds[0,0]:.1f}, {bounds[0,1]:.1f}, {bounds[0,2]:.1f}), "
                f"({bounds[1,0]:.1f}, {bounds[1,1]:.1f}, {bounds[1,2]:.1f})]"
            )

        return (
            f"Polygon({dim}, "
            f"n_points={self.n_points}, "
            f"n_holes={n_holes}, "
            f"bounds={bounds_str}, "
            f"crs={self.crs!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
