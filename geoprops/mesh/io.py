"""Mesh I/O functionality.

This module handles loading and saving mesh files in various formats,
delegating to trimesh for standard formats and providing custom loaders
for mining-specific formats.
"""

import os
from pathlib import Path
from typing import Any

import trimesh

from geoprops.mesh.mesh import Mesh


def load(
    file_path: str | Path,
    crs: str | None = None,
    **kwargs: Any,
) -> Mesh:
    """Load a mesh from file.

    Supports standard formats via trimesh (OBJ, STL, PLY, OFF, GLTF)
    and custom formats for mining software (MSH, LFM).

    Args:
        file_path: Path to mesh file
        crs: Optional coordinate reference system (EPSG or WKT)
        **kwargs: Additional metadata to attach to mesh (name, category, etc.)

    Returns:
        Loaded Mesh object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported

    Examples:
        >>> mesh = gp.mesh.load('domain.obj')
        >>> mesh = gp.mesh.load('domain.stl', crs='EPSG:31983')
        >>> mesh = gp.mesh.load('domain.obj', name='ore', category='geology')
    """
    # Convert to Path object
    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    # Get file extension
    ext = file_path.suffix.lower()

    # Handle custom formats
    if ext == ".msh":
        return _load_seequent_msh(file_path, crs=crs, **kwargs)
    elif ext == ".lfm":
        return _load_leapfrog_lfm(file_path, crs=crs, **kwargs)

    # Use trimesh for standard formats
    try:
        mesh_tri = trimesh.load(str(file_path))

        # Handle case where trimesh returns a Scene instead of a Mesh
        if isinstance(mesh_tri, trimesh.Scene):
            # For scenes with multiple geometries, combine them
            if len(mesh_tri.geometry) == 0:
                raise ValueError(f"No geometry found in file: {file_path}")
            elif len(mesh_tri.geometry) == 1:
                mesh_tri = list(mesh_tri.geometry.values())[0]
            else:
                # Combine multiple meshes
                mesh_tri = trimesh.util.concatenate(list(mesh_tri.geometry.values()))

        # Extract vertices and faces
        vertices = mesh_tri.vertices
        faces = mesh_tri.faces

        # Build metadata
        metadata = kwargs.copy()
        if "name" not in metadata:
            metadata["name"] = file_path.stem

        # Create Mesh object
        return Mesh(
            vertices=vertices,
            faces=faces,
            metadata=metadata,
            crs=crs,
        )

    except Exception as e:
        raise ValueError(f"Failed to load mesh from {file_path}: {e}") from e


def save(
    mesh: Mesh,
    file_path: str | Path,
    format: str | None = None,
) -> None:
    """Save a mesh to file.

    Args:
        mesh: Mesh to save
        file_path: Output file path
        format: Optional format override (e.g., 'obj', 'stl', 'ply')

    Raises:
        ValueError: If format is not supported

    Examples:
        >>> gp.mesh.save(mesh, 'output.obj')
        >>> gp.mesh.save(mesh, 'output.stl', format='stl')
    """
    # Convert to Path object
    file_path = Path(file_path)

    # Determine format
    if format is None:
        format = file_path.suffix.lower().lstrip(".")

    # Create trimesh object
    mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # Export using trimesh
    try:
        mesh_tri.export(str(file_path), file_type=format)
    except Exception as e:
        raise ValueError(f"Failed to save mesh to {file_path}: {e}") from e


def _load_seequent_msh(
    file_path: Path,
    crs: str | None = None,
    **kwargs: Any,
) -> Mesh:
    """Load Seequent .msh format (custom loader).

    The .msh format is a simple text format:
    - Header line
    - Vertex count, face count
    - Vertices (x y z per line)
    - Faces (v1 v2 v3 per line, 0-indexed or 1-indexed)

    Args:
        file_path: Path to .msh file
        crs: Optional CRS
        **kwargs: Additional metadata

    Returns:
        Loaded Mesh

    Raises:
        NotImplementedError: Placeholder for now
    """
    # TODO: Implement in Phase 4 of roadmap
    raise NotImplementedError(
        "Seequent .msh format not yet implemented. "
        "This will be added in Phase 4. "
        "For now, please convert to OBJ, STL, or PLY format."
    )


def _load_leapfrog_lfm(
    file_path: Path,
    crs: str | None = None,
    **kwargs: Any,
) -> Mesh:
    """Load Leapfrog .lfm format (custom loader).

    Args:
        file_path: Path to .lfm file
        crs: Optional CRS
        **kwargs: Additional metadata

    Returns:
        Loaded Mesh

    Raises:
        NotImplementedError: Placeholder for now
    """
    # TODO: Implement in Phase 4 of roadmap
    raise NotImplementedError(
        "Leapfrog .lfm format not yet implemented. "
        "This will be added in Phase 4. "
        "For now, please convert to OBJ, STL, or PLY format."
    )
