# GeoProps - mesh module
"""Mesh handling: I/O, validation, and mesh operations."""

from geoprops.mesh.mesh import Mesh
from geoprops.mesh.io import load, save

__all__ = ["Mesh", "load", "save"]
