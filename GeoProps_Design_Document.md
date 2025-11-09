# GeoProps - Design Document
## GPU-Accelerated Geometric Proportions Library

**Version:** 1.0  
**Date:** November 5, 2025  
**Status:** Design Complete - Ready for Implementation

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Fundamental Objects](#2-fundamental-objects)
3. [API Design](#3-api-design)
4. [Internal Architecture](#4-internal-architecture)
5. [Calculation Methods](#5-calculation-methods)
6. [Fuzzy Logic Operations](#6-fuzzy-logic-operations)
7. [File Format Support](#7-file-format-support)
8. [CRS Handling](#8-crs-handling)
9. [Performance Strategy](#9-performance-strategy)
10. [Testing Strategy](#10-testing-strategy)
11. [Technology Stack](#11-technology-stack)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. PROJECT OVERVIEW

### 1.1 Purpose

**GeoProps** is a GPU-accelerated Python library for calculating geometric proportions between meshes/polygons and regular block models. Primary use cases include:

- Resource estimation (domain proportions in blocks)
- Mine reconciliation (production vs. model)
- Civil engineering (cut/fill calculations)  
- Hydrology (watershed-aquifer intersections)
- Point cloud containment queries (samples in domains)

### 1.2 Key Features

- ⚡ **GPU-accelerated**: 100-1000x faster than CPU methods
- 📐 **Universal input**: Meshes, polygons, or point clouds
- 🎯 **Fuzzy logic operations**: Boolean operations without mesh operations
- 🔄 **Automatic optimization**: Detects 2.5D vs general meshes
- 🌍 **CRS-aware**: Coordinate system transformations via pyproj
- 📦 **Ecosystem compatible**: `__geo_interface__` protocol support
- ✅ **Extensively validated**: Known test cases and benchmarks

### 1.3 Performance Targets

| Dataset | Method | Target Time |
|---------|--------|-------------|
| 3M blocks × 1 mesh (2.5D) | GPU | <1 second |
| 3M blocks × 1 mesh (general) | GPU | <5 seconds |
| 3M blocks × 20 meshes | GPU batch | <30 seconds |
| 3M blocks × 1 mesh | CPU | <2 minutes |
| 50k points × 1 mesh | CPU/GPU | <100ms |

### 1.4 Non-Goals

- Mesh boolean operations (use specialized libraries)
- Geostatistical estimation (see companion resource estimation framework)
- 3D visualization (use PyVista, Blender, etc.)
- Database management (focus on computation)

---

## 2. FUNDAMENTAL OBJECTS

### 2.1 Mesh

```python
class Mesh:
    """Triangular mesh representing a 3D domain or surface"""
    
    vertices: np.ndarray    # (N, 3) float64 - vertex coordinates
    faces: np.ndarray       # (M, 3) int32 - triangle vertex indices
    metadata: dict          # Name, category, custom properties
    crs: str | None         # Coordinate reference system (EPSG or WKT)
    
    # Core methods
    def is_valid(self) -> bool:
        """Check mesh validity (closed, manifold, no self-intersections)"""
    
    def is_closed(self) -> bool:
        """Check if mesh is watertight"""
    
    def volume(self) -> float:
        """Calculate enclosed volume"""
    
    def transform(self, to_crs: str, z_offset: float = 0.0) -> 'Mesh':
        """Transform to different CRS with optional vertical offset"""
    
    def translate(self, offset: tuple[float, float, float]) -> 'Mesh':
        """Translate mesh by offset vector"""
    
    @property
    def __geo_interface__(self) -> dict:
        """GeoJSON-like representation for ecosystem compatibility"""
```

**Loading:**
```python
# From file (delegates to trimesh for standard formats)
mesh = gp.mesh.load('domain.obj')  # OBJ, STL, PLY, OFF, GLTF
mesh = gp.mesh.load('domain.msh')  # Seequent .msh (custom loader)
mesh = gp.mesh.load('domain.lfm')  # Leapfrog .lfm (custom loader)

# From trimesh
import trimesh
tm = trimesh.load('file.stl')
mesh = gp.Mesh.from_trimesh(tm)

# From arrays
mesh = gp.Mesh(
    vertices=np.array([[...], [...], ...]),
    faces=np.array([[0, 1, 2], [1, 2, 3], ...]),
    crs='EPSG:31983'
)
```

---

### 2.2 BlockModel

```python
class BlockModel:
    """Regular 3D grid of blocks"""
    
    origin: tuple[float, float, float]  # Lower-left-front corner
    size: tuple[float, float, float]    # Block dimensions (dx, dy, dz)
    extent: tuple[int, int, int]        # Grid size (nx, ny, nz)
    rotation: float = 0                 # Grid rotation (degrees)
    crs: str | None = None             # Coordinate reference system
    
    @property
    def n_blocks(self) -> int:
        """Total number of blocks"""
        return self.extent[0] * self.extent[1] * self.extent[2]
    
    @property
    def bounds(self) -> np.ndarray:
        """Model bounding box [(xmin, ymin, zmin), (xmax, ymax, zmax)]"""
    
    def block_center(self, i: int, j: int, k: int) -> np.ndarray:
        """Get center coordinates of block (i, j, k)"""
    
    def block_corners(self, i: int, j: int, k: int) -> np.ndarray:
        """Get 8 corner coordinates of block (i, j, k)"""
```

**Creation:**
```python
# Simple grid
blocks = gp.BlockModel(
    origin=(500000, 9200000, 100),
    size=(10, 10, 16),           # 10×10×16m blocks
    extent=(200, 300, 50),       # 200×300×50 blocks
    crs='EPSG:31983'
)

# Rotated grid
blocks = gp.BlockModel(
    origin=(500000, 9200000, 100),
    size=(10, 10, 16),
    extent=(200, 300, 50),
    rotation=45,                 # 45° rotation
    crs='EPSG:31983'
)
```

---

### 2.3 Polygon

```python
class Polygon:
    """2D or 3D polygon for selections and extrusions"""
    
    points: np.ndarray              # (N, 2) or (N, 3) float64
    holes: list[np.ndarray] | None  # Optional hole polygons
    crs: str | None = None
    
    @property
    def is_2d(self) -> bool:
        """Check if polygon is 2D"""
    
    @property
    def is_3d(self) -> bool:
        """Check if polygon is 3D"""
    
    def extrude(
        self, 
        distance: float | tuple[float, float] | None = None,
        direction: np.ndarray | None = None
    ) -> Mesh:
        """
        Extrude polygon to create 3D volume
        
        For 2D polygons:
            distance=(base, top): Vertical extrusion to elevation range
            direction=[dx, dy, dz]: Extrude along vector
        
        For 3D polygons:
            distance=value: Extrude along polygon normal
            direction=[dx, dy, dz]: Extrude along vector
        
        Examples:
            # 2D polygon, vertical extrusion
            poly_2d.extrude(distance=(0, 100))  # 0 to 100m elevation
            
            # 3D polygon (fault plane), extrude along normal
            fault_poly.extrude(distance=50)  # 50m thick fault zone
            
            # 3D polygon, extrude in specific direction
            poly_3d.extrude(direction=[10, 0, 5])  # Offset ribbon
        """
    
    @property
    def __geo_interface__(self) -> dict:
        """GeoJSON representation for Shapely/GeoPandas compatibility"""
    
    @classmethod
    def from_geo_interface(cls, geo: dict) -> 'Polygon':
        """Create from any __geo_interface__ object"""
```

**Creation:**
```python
# 2D polygon (footprint)
poly = gp.Polygon(
    points=np.array([[0, 0], [100, 0], [100, 50], [0, 50]]),
    crs='EPSG:31983'
)

# 2D polygon with holes
poly = gp.Polygon(
    points=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
    holes=[np.array([[20, 20], [80, 20], [80, 80], [20, 80]])],
    crs='EPSG:31983'
)

# 3D polygon (fault plane, vein trace)
poly = gp.Polygon(
    points=np.array([[x1, y1, z1], [x2, y2, z2], ...]),
    crs='EPSG:31983'
)

# From Shapely (if available)
import shapely.geometry
shapely_poly = shapely.geometry.Polygon([...])
poly = gp.Polygon.from_geo_interface(shapely_poly)
```

---

### 2.4 Result

```python
class Result:
    """Unified result from proportion calculation or point queries"""
    
    block_ids: np.ndarray       # Block/point indices
    proportions: np.ndarray     # Values [0, 1]
    metadata: dict              # Method, timing, statistics
    
    # Convenience properties
    @property
    def selected(self) -> np.ndarray:
        """Boolean mask (proportion > 0)"""
        return self.proportions > 0
    
    def threshold(self, value: float) -> np.ndarray:
        """Boolean mask at custom threshold"""
        return self.proportions > value
    
    # Fuzzy logic operations
    def union(self, other: 'Result') -> 'Result':
        """Fuzzy OR: max(A, B)"""
    
    def intersection(self, other: 'Result') -> 'Result':
        """Fuzzy AND: min(A, B)"""
    
    def difference(self, other: 'Result') -> 'Result':
        """Fuzzy DIFFERENCE: min(A, 1-B)"""
    
    def complement(self) -> 'Result':
        """Fuzzy NOT: 1-A"""
    
    # Operator overloading
    def __or__(self, other: 'Result') -> 'Result':
        """result_a | result_b → union"""
    
    def __and__(self, other: 'Result') -> 'Result':
        """result_a & result_b → intersection"""
    
    def __sub__(self, other: 'Result') -> 'Result':
        """result_a - result_b → difference"""
    
    def __invert__(self) -> 'Result':
        """~result → complement"""
    
    # Export methods
    def to_dataframe(self, format: str = 'polars') -> pl.DataFrame | pd.DataFrame:
        """Export to DataFrame (Polars default, Pandas optional)"""
    
    def to_array_3d(self, blocks: BlockModel) -> np.ndarray:
        """Export as 3D array matching block model shape"""
```

---

## 3. API DESIGN

### 3.1 Design Principles

1. **Progressive disclosure**: Simple things simple, complex things possible
2. **Sensible defaults**: Auto-detect best method, reasonable resolution
3. **Explicit is better than implicit**: Clear parameter names, no magic
4. **Fail loudly**: Validate inputs, give helpful error messages
5. **Performance transparency**: Report method used, timing info
6. **Universal inputs**: Accept meshes, polygons, points, DataFrames

---

### 3.2 Primary API

```python
import geoprops as gp

# ============================================================================
# BASIC API - Covers 90% of use cases
# ============================================================================

# Calculate mesh proportions in blocks
result = gp.calculate(blocks, mesh)

# Calculate polygon proportions (with extrusion)
result = gp.calculate(blocks, polygon, extrude=(0, 100))

# Query point containment (are points inside mesh/polygon?)
points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
result = gp.calculate(points, mesh)  # Returns 1.0 (inside) or 0.0 (outside)

# Works with DataFrames transparently
composites = pl.read_parquet('composites.parquet')
result = gp.calculate(composites[['X', 'Y', 'Z']], ore_mesh)
composites = composites.with_columns(
    pl.Series('in_ore', result.selected)
)

# Batch processing (multiple meshes)
results = gp.calculate_batch(blocks, meshes, n_workers=4)

# ============================================================================
# INTERMEDIATE API - More control
# ============================================================================

# Specify method explicitly
result = gp.calculate(
    blocks, 
    mesh, 
    method='gpu',              # 'auto', 'gpu', 'cpu'
    resolution=1.0,            # Grid resolution for GPU
    gpu_device=0               # Which GPU
)

# Polygon extrusion options
# Option 1: Inline extrusion
result = gp.calculate(blocks, polygon_2d, extrude=(base, top))

# Option 2: Pre-extrude polygon
polygon_3d = polygon_2d.extrude(distance=(0, 100))
result = gp.calculate(blocks, polygon_3d)

# Option 3: 3D polygon with normal extrusion
fault_zone = fault_plane.extrude(distance=50)  # 50m thick zone

# ============================================================================
# ADVANCED API - Full control
# ============================================================================

# Manual calculator selection
from geoprops.gpu import DepthPeelingCalculator

calculator = DepthPeelingCalculator(
    resolution=1.0,
    max_layers=10,
    device=0
)

with calculator:
    result = calculator.calculate(blocks, mesh)

# Factory pattern
calculator = gp.create_calculator(
    method='gpu',
    backend='depth_peeling',  # or 'compute_shader'
    **kwargs
)
```

---

### 3.3 Fuzzy Logic Operations

```python
# Calculate domain proportions
hg = gp.calculate(blocks, high_grade_mesh)
lg = gp.calculate(blocks, low_grade_mesh)
oxide = gp.calculate(blocks, oxide_mesh)
sulphide = gp.calculate(blocks, sulphide_mesh)

# Combine using fuzzy logic (NO mesh boolean operations!)
ore = hg | lg                       # Union: any ore
oxide_ore = ore & oxide             # Intersection: ore AND oxide
sulphide_ore = ore & sulphide       # Intersection: ore AND sulphide
waste = ~ore                        # Complement: NOT ore
transitional = ore & (~oxide) & (~sulphide)  # Neither oxide nor sulphide

# Export results
oxide_ore.to_dataframe().write_parquet('oxide_ore_proportions.parquet')

# More complex operations
from geoprops import operations

# Weighted blend
mixed = operations.blend(
    [hg, lg], 
    weights=[0.7, 0.3]
)

# Multi-way union
all_ore = operations.union(hg, lg, transitional)
```

---

### 3.4 Configuration API

```python
# Global configuration
gp.config.set_default_method('gpu')
gp.config.set_default_resolution(1.0)
gp.config.set_gpu_device(0)
gp.config.verbose = 2  # 0=silent, 1=warnings, 2=info, 3=debug
gp.config.show_progress = True  # tqdm progress bars

# Context manager for temporary config
with gp.config.temporary(method='cpu', verbose=3):
    result = gp.calculate(blocks, mesh)
```

---

## 4. INTERNAL ARCHITECTURE

### 4.1 Module Structure

```
geoprops/
│
├── __init__.py              # Public API exports
├── config.py                # Global configuration
├── version.py               # Version info
│
├── core/                    # CPU implementations
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── heightfield.py       # 2.5D height field method
│   ├── raycasting.py        # General ray casting method
│   └── utils.py             # CPU utilities
│
├── gpu/                     # GPU implementations
│   ├── __init__.py
│   ├── base.py              # GPU base classes
│   ├── depth_peeling.py     # Depth peeling renderer
│   ├── compute_shader.py    # Compute shader approach
│   ├── context.py           # OpenGL context management
│   └── utils.py             # GPU utilities
│
├── mesh/                    # Mesh handling
│   ├── __init__.py
│   ├── io.py                # Load/save (OBJ, STL, MSH, LFM, etc.)
│   ├── validation.py        # Mesh quality checks
│   ├── topology.py          # Classify 2.5D vs general
│   └── preprocessing.py     # Mesh cleanup/repair
│
├── blocks/                  # Block model utilities
│   ├── __init__.py
│   ├── model.py             # BlockModel class
│   ├── indexing.py          # Spatial indexing
│   └── io.py                # Block model I/O
│
├── polygons/                # Polygon operations
│   ├── __init__.py
│   ├── polygon.py           # Polygon class
│   ├── geometry.py          # Point-in-polygon, triangulation
│   └── extrusion.py         # 3D extrusion
│
├── operations/              # Fuzzy logic operations
│   ├── __init__.py
│   └── fuzzy.py             # Union, intersection, difference, blend
│
├── validation/              # Testing & validation
│   ├── __init__.py
│   ├── test_cases.py        # Known test cases
│   ├── benchmarks.py        # Performance benchmarks
│   └── comparison.py        # CPU vs GPU comparison
│
└── utils/                   # General utilities
    ├── __init__.py
    ├── logging.py           # Logging configuration
    ├── spatial.py           # CRS transforms (pyproj)
    └── profiling.py         # Performance profiling
```

---

### 4.2 Abstract Base Classes

```python
# geoprops/core/base.py

from abc import ABC, abstractmethod

class Calculator(ABC):
    """Base class for all proportion calculators"""
    
    @abstractmethod
    def calculate(
        self, 
        blocks: BlockModel, 
        geometry: Mesh | Polygon
    ) -> Result:
        """Calculate proportions"""
    
    @abstractmethod
    def can_handle(
        self, 
        blocks: BlockModel, 
        geometry: Mesh | Polygon
    ) -> bool:
        """Check if this calculator can handle these inputs"""
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Human-readable method name"""
    
    def validate_inputs(
        self, 
        blocks: BlockModel, 
        geometry: Mesh | Polygon
    ) -> None:
        """Validate inputs before calculation"""


class GPUCalculator(Calculator):
    """Base class for GPU calculators"""
    
    def __init__(self, device: int = 0, resolution: float = 1.0):
        self.device = device
        self.resolution = resolution
        self._context = None
    
    def __enter__(self):
        """Context manager for GPU resources"""
        self._initialize_gpu()
        return self
    
    def __exit__(self, *args):
        """Cleanup GPU resources"""
        self._cleanup_gpu()
```

---

### 4.3 Strategy Pattern for Method Selection

```python
# geoprops/core/strategy.py

class CalculationStrategy:
    """Selects and executes the best calculation method"""
    
    def __init__(self):
        self.calculators = self._register_calculators()
    
    def _register_calculators(self) -> list[Calculator]:
        """Register available calculators in priority order"""
        calculators = []
        
        # Try GPU methods first (if available)
        try:
            from geoprops.gpu import (
                ComputeShaderCalculator,
                DepthPeelingCalculator
            )
            calculators.extend([
                ComputeShaderCalculator(),  # Fastest
                DepthPeelingCalculator(),   # Still very fast
            ])
        except ImportError:
            pass  # GPU not available
        
        # CPU fallbacks
        from geoprops.core import (
            HeightFieldCalculator,
            RayCastingCalculator
        )
        calculators.extend([
            HeightFieldCalculator(),  # Fast for 2.5D
            RayCastingCalculator(),   # General purpose
        ])
        
        return calculators
    
    def calculate(
        self, 
        target: BlockModel | np.ndarray,
        geometry: Mesh | Polygon,
        method: str = 'auto',
        **kwargs
    ) -> Result:
        """Select calculator and compute"""
        
        # Handle DataFrames
        if isinstance(target, (pl.DataFrame, pd.DataFrame)):
            if hasattr(target, 'to_numpy'):
                points = target.to_numpy()  # Polars
            else:
                points = target.values  # Pandas
            target = points
        
        # Handle point queries
        if isinstance(target, np.ndarray):
            return self._query_containment(target, geometry)
        
        # Handle block model proportions
        calculator = self.select_calculator(target, geometry, method)
        
        with calculator:
            result = calculator.calculate(target, geometry, **kwargs)
        
        result.metadata['method'] = calculator.method_name
        return result
    
    def _query_containment(
        self, 
        points: np.ndarray, 
        geometry: Mesh | Polygon
    ) -> Result:
        """Check if points are inside mesh or polygon"""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be (N,3) array, got {points.shape}")
        
        if isinstance(geometry, Mesh):
            # Use trimesh's optimized contains()
            import trimesh
            mesh_tri = trimesh.Trimesh(
                vertices=geometry.vertices, 
                faces=geometry.faces
            )
            inside = mesh_tri.contains(points)
        
        elif isinstance(geometry, Polygon):
            if geometry.is_2d:
                # 2D point-in-polygon (ignore Z)
                inside = np.array([
                    point_in_polygon_2d(p[:2], geometry.points) 
                    for p in points
                ])
            else:
                # 3D polygon → project and test
                inside = np.array([
                    point_in_polygon_3d(p, geometry) 
                    for p in points
                ])
        
        return Result(
            block_ids=np.arange(len(points)).reshape(-1, 1),
            proportions=inside.astype(float),  # 1.0 or 0.0
            metadata={
                'method': 'point_containment',
                'n_points': len(points),
                'n_inside': inside.sum()
            }
        )
```

---

## 5. CALCULATION METHODS

### 5.1 Method Classification

| Method | Speed | Best For | Accuracy |
|--------|-------|----------|----------|
| **GPU Compute Shader** | ⚡⚡⚡ | Any mesh | Excellent |
| **GPU Depth Peeling** | ⚡⚡⚡ | General meshes | Excellent |
| **GPU Height Field** | ⚡⚡⚡ | 2.5D meshes | Excellent |
| **CPU Height Field** | ⚡ | 2.5D meshes | Excellent |
| **CPU Ray Casting** | ⚡ | General meshes | Excellent |

### 5.2 Automatic Selection Logic

```python
def select_calculator(blocks, geometry, method='auto'):
    if method != 'auto':
        return get_calculator_by_name(method)
    
    # 1. Try GPU Compute Shader (fastest, most general)
    if gpu_available() and gpu_has_memory(blocks, geometry):
        return ComputeShaderCalculator()
    
    # 2. Try GPU Depth Peeling (fast, works for anything)
    if gpu_available():
        return DepthPeelingCalculator()
    
    # 3. Classify mesh topology
    topology = classify_mesh_topology(geometry)
    
    # 4. Use appropriate CPU method
    if topology == '2.5D':
        return HeightFieldCalculator()  # Fast for topography
    else:
        return RayCastingCalculator()   # General purpose
```

### 5.3 Mesh Topology Classification

```python
def classify_mesh_topology(mesh, n_samples=10000):
    """
    Classify mesh as:
    - '2.5D': Single-valued height field (topography-like)
    - 'general': Multiple intersections (<10 avg)
    - 'complex': Many intersections (>10 avg)
    """
    # Sample rays from above
    # Count intersections per ray
    # Classify based on statistics
    
    if >98% have ≤1 intersection:
        return '2.5D'
    elif avg_intersections ≤ 10:
        return 'general'
    else:
        return 'complex'
```

---

## 6. FUZZY LOGIC OPERATIONS

### 6.1 Core Operations

Fuzzy logic operations allow combining domain proportions without mesh boolean operations:

```python
# Standard fuzzy set operations
union(A, B) = max(A, B)                    # OR
intersection(A, B) = min(A, B)             # AND
difference(A, B) = min(A, 1-B)             # A NOT B
complement(A) = 1 - A                      # NOT
```

### 6.2 Implementation

```python
class Result:
    def union(self, other):
        return Result(
            block_ids=self.block_ids,
            proportions=np.maximum(self.proportions, other.proportions)
        )
    
    def intersection(self, other):
        return Result(
            block_ids=self.block_ids,
            proportions=np.minimum(self.proportions, other.proportions)
        )
    
    def difference(self, other):
        return Result(
            block_ids=self.block_ids,
            proportions=np.minimum(self.proportions, 1.0 - other.proportions)
        )
    
    def complement(self):
        return Result(
            block_ids=self.block_ids,
            proportions=1.0 - self.proportions
        )
```

### 6.3 Why This is Powerful

**Problem**: Mesh boolean operations are notoriously buggy
- Self-intersections after union/difference
- Non-manifold edges
- Numerical instability
- Slow performance

**Solution**: Fuzzy operations on proportions
- Fast (just NumPy array operations)
- Numerically stable
- Handles overlapping domains naturally
- Preserves partial proportions

**Example Use Case:**
```python
# Define mineable ore without touching meshes
hg = gp.calculate(blocks, high_grade_mesh)
lg = gp.calculate(blocks, low_grade_mesh)
weathered = gp.calculate(blocks, weathered_mesh)
fault = gp.calculate(blocks, fault_zone_mesh)

# Complex domain logic
mineable_ore = ((hg | lg) - weathered) - fault

# Export for estimation
mineable_ore.to_dataframe().write_parquet('mineable_ore.parquet')
```

---

## 7. FILE FORMAT SUPPORT

### 7.1 Standard Formats (via trimesh)

- OBJ (Wavefront)
- STL (Stereolithography)
- PLY (Polygon File Format)
- OFF (Object File Format)
- GLTF/GLB (GL Transmission Format)

### 7.2 Resource Modeling Formats (custom loaders)

**Seequent .msh format:**
- Simple text-based format
- Header + vertex list + triangle list
- Custom loader in `geoprops/mesh/io.py`

**Leapfrog .lfm format:**
- Binary format
- Specific to Leapfrog Geo
- Custom loader in `geoprops/mesh/io.py`

### 7.3 Loading API

```python
# Auto-detect format
mesh = gp.mesh.load('domain.obj')
mesh = gp.mesh.load('domain.stl')
mesh = gp.mesh.load('domain.msh')  # Seequent
mesh = gp.mesh.load('domain.lfm')  # Leapfrog

# Explicit format
mesh = gp.mesh.load('domain.dat', format='msh')

# Save mesh
gp.mesh.save(mesh, 'output.obj')
```

---

## 8. CRS HANDLING

### 8.1 Strategy

**Horizontal (X, Y)**: Full CRS support via pyproj  
**Vertical (Z)**: Manual offsets (pragmatic approach)

**Rationale:**
- pyproj handles horizontal transformations excellently
- Vertical datums are messy, software-specific
- Most resource models use same vertical datum
- Manual offset covers 95% of real-world cases

### 8.2 Implementation

```python
from pyproj import Transformer

def transform_mesh(mesh, from_crs, to_crs, z_offset=0.0):
    """Transform mesh between CRS"""
    transformer = Transformer.from_crs(
        from_crs, to_crs, always_xy=True
    )
    
    # Transform X, Y
    x, y = transformer.transform(
        mesh.vertices[:, 0], 
        mesh.vertices[:, 1]
    )
    
    # Manual Z offset
    z = mesh.vertices[:, 2] + z_offset
    
    return Mesh(
        vertices=np.column_stack([x, y, z]),
        faces=mesh.faces,
        metadata=mesh.metadata.copy(),
        crs=to_crs
    )
```

### 8.3 Validation

```python
def validate_crs_match(blocks, mesh):
    """Warn if CRS mismatch"""
    if blocks.crs and mesh.crs and blocks.crs != mesh.crs:
        warnings.warn(
            f"CRS mismatch: blocks={blocks.crs}, mesh={mesh.crs}. "
            "Results may be incorrect. Use gp.transform() to convert."
        )
```

### 8.4 Usage

```python
# Transform mesh to match block model CRS
mesh_utm23 = gp.mesh.load('domain.obj')
mesh_utm23.crs = 'EPSG:31983'  # UTM Zone 23S

mesh_utm24 = mesh_utm23.transform(
    'EPSG:31984',      # UTM Zone 24S
    z_offset=10.0      # Optional vertical adjustment
)

# Or use standalone function
mesh_utm24 = gp.transform(
    mesh_utm23,
    from_crs='EPSG:31983',
    to_crs='EPSG:31984',
    z_offset=10.0
)
```

---

## 9. PERFORMANCE STRATEGY

### 9.1 Memory Management

```python
class MemoryManager:
    @staticmethod
    def estimate_gpu_memory(blocks, mesh, resolution):
        """Estimate GPU memory required (bytes)"""
        # Height field texture
        width = int((mesh.bounds[1, 0] - mesh.bounds[0, 0]) / resolution)
        height = int((mesh.bounds[1, 1] - mesh.bounds[0, 1]) / resolution)
        n_layers = 10
        
        texture_memory = width * height * n_layers * 4  # float32
        mesh_memory = mesh.vertices.nbytes + mesh.faces.nbytes
        block_memory = blocks.n_blocks * 4
        
        return texture_memory + mesh_memory + block_memory
    
    @staticmethod
    def check_gpu_memory(required, device=0):
        """Check if GPU has enough free memory"""
        try:
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(device)
            free, total = device.mem_get_info()
            return free > required * 1.2  # 20% safety margin
        except:
            return False
```

### 9.2 Batch Processing

```python
def calculate_batch(blocks, meshes, method='auto', n_workers=4):
    """Process multiple meshes efficiently"""
    
    strategy = CalculationStrategy()
    calculator = strategy.select_calculator(blocks, meshes[0], method)
    
    if isinstance(calculator, GPUCalculator):
        # GPU: Serial (GPU handles parallelism internally)
        results = {}
        with calculator:
            for mesh in meshes:
                result = calculator.calculate(blocks, mesh)
                results[mesh.metadata['name']] = result
        return results
    
    else:
        # CPU: Parallel processing
        from multiprocessing import Pool
        
        def process_one(mesh):
            calc = strategy.select_calculator(blocks, mesh, 'cpu')
            return mesh.metadata['name'], calc.calculate(blocks, mesh)
        
        with Pool(n_workers) as pool:
            results = dict(pool.map(process_one, meshes))
        
        return results
```

---

## 10. TESTING STRATEGY

### 10.1 Test Hierarchy

```
tests/
├── unit/                   # Individual component tests
│   ├── test_mesh.py
│   ├── test_blocks.py
│   ├── test_polygon.py
│   └── test_result.py
│
├── integration/            # End-to-end tests
│   ├── test_simple_cases.py
│   ├── test_cpu_methods.py
│   ├── test_gpu_methods.py
│   └── test_fuzzy_ops.py
│
├── validation/             # Known answer tests
│   ├── test_cube_in_grid.py
│   ├── test_sphere_volume.py
│   └── test_method_agreement.py
│
└── benchmarks/             # Performance tests
    ├── benchmark_cpu.py
    ├── benchmark_gpu.py
    └── benchmark_comparison.py
```

### 10.2 Key Test Cases

**Simple Geometric Tests:**
```python
def test_cube_in_grid():
    """Cube mesh in aligned grid - known answer"""
    mesh = create_cube_mesh(center=(50, 50, 50), size=20)
    blocks = BlockModel(origin=(0, 0, 0), size=(10, 10, 10), extent=(10, 10, 10))
    
    result = gp.calculate(blocks, mesh)
    
    # Cube should fully occupy 8 blocks (2×2×2)
    assert (result.proportions == 1.0).sum() == 8

def test_sphere_volume():
    """Sphere mesh - volume conservation"""
    radius = 10
    mesh = create_sphere_mesh(center=(50, 50, 50), radius=radius)
    blocks = BlockModel(origin=(0, 0, 0), size=(2, 2, 2), extent=(50, 50, 50))
    
    result = gp.calculate(blocks, mesh)
    
    # Sum of (proportion × block_volume) should equal sphere volume
    block_volume = 2 * 2 * 2
    total_volume = (result.proportions * block_volume).sum()
    expected_volume = (4/3) * np.pi * radius**3
    
    assert abs(total_volume - expected_volume) / expected_volume < 0.01
```

**Method Agreement Tests:**
```python
def test_cpu_gpu_agreement():
    """CPU and GPU methods should give same results"""
    mesh = load_test_mesh('complex_ore_body.obj')
    blocks = BlockModel(origin=(0, 0, 0), size=(10, 10, 10), extent=(20, 20, 20))
    
    result_cpu = gp.calculate(blocks, mesh, method='cpu')
    result_gpu = gp.calculate(blocks, mesh, method='gpu')
    
    # Results should be very close
    np.testing.assert_allclose(
        result_cpu.proportions, 
        result_gpu.proportions, 
        rtol=1e-5, atol=1e-6
    )
```

**Point Containment Tests:**
```python
def test_point_containment():
    """Test point-in-mesh queries"""
    mesh = create_sphere_mesh(center=(0, 0, 0), radius=10)
    
    # Points inside
    points_inside = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]])
    result = gp.calculate(points_inside, mesh)
    assert result.selected.all()
    
    # Points outside
    points_outside = np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
    result = gp.calculate(points_outside, mesh)
    assert not result.selected.any()
```

**Fuzzy Logic Tests:**
```python
def test_fuzzy_union():
    """Test fuzzy union operation"""
    a = Result(block_ids=np.array([[0,0,0]]), proportions=np.array([0.3]))
    b = Result(block_ids=np.array([[0,0,0]]), proportions=np.array([0.7]))
    
    c = a | b
    assert c.proportions[0] == 0.7  # max(0.3, 0.7)

def test_fuzzy_intersection():
    """Test fuzzy intersection operation"""
    a = Result(block_ids=np.array([[0,0,0]]), proportions=np.array([0.3]))
    b = Result(block_ids=np.array([[0,0,0]]), proportions=np.array([0.7]))
    
    c = a & b
    assert c.proportions[0] == 0.3  # min(0.3, 0.7)
```

---

## 11. TECHNOLOGY STACK

### 11.1 Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | ≥3.10 | Core language |
| **NumPy** | ≥1.24 | Numerical computing |
| **Polars** | ≥0.20 | DataFrame operations (default) |
| **Pandas** | ≥2.0 | DataFrame compatibility |
| **trimesh** | ≥4.0 | Mesh I/O and operations |
| **pyproj** | ≥3.5 | CRS transformations |

### 11.2 GPU Dependencies (optional)

| Library | Purpose |
|---------|---------|
| **moderngl** | OpenGL context and rendering |
| **PyOpenGL** | Fallback OpenGL bindings |

### 11.3 Development Dependencies

| Library | Purpose |
|---------|---------|
| **pytest** | Testing framework |
| **pytest-benchmark** | Performance benchmarking |
| **black** | Code formatting |
| **ruff** | Linting |
| **mypy** | Type checking |

---

## 12. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1)
- [ ] Project structure setup
- [ ] Core classes (Mesh, BlockModel, Polygon, Result)
- [ ] Configuration system
- [ ] Logging setup
- [ ] Basic I/O (OBJ loading via trimesh)

### Phase 2: CPU Implementation (Week 2)
- [ ] Point-in-polygon (custom, no Shapely)
- [ ] Point-in-mesh (via trimesh)
- [ ] CPU height field method (2.5D)
- [ ] CPU ray casting method (general)
- [ ] Mesh topology classifier

### Phase 3: GPU Implementation (Week 3-4)
- [ ] OpenGL context management
- [ ] GPU depth peeling renderer
- [ ] GPU compute shader approach
- [ ] GPU height field method
- [ ] Memory management

### Phase 4: Features (Week 5)
- [ ] Polygon extrusion (2D and 3D)
- [ ] Point containment queries
- [ ] Fuzzy logic operations
- [ ] Batch processing
- [ ] Custom mesh loaders (.msh, .lfm)

### Phase 5: CRS & Transforms (Week 6)
- [ ] pyproj integration
- [ ] CRS validation
- [ ] Mesh transformation
- [ ] `__geo_interface__` protocol

### Phase 6: Testing (Week 7)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Known answer validation
- [ ] CPU vs GPU comparison tests
- [ ] Point containment tests
- [ ] Fuzzy logic tests

### Phase 7: Optimization (Week 8)
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] Batch processing optimization
- [ ] Benchmarking

### Phase 8: Documentation (Week 9)
- [ ] API documentation
- [ ] Usage examples
- [ ] Tutorial notebooks
- [ ] Performance guide
- [ ] Contributing guide

### Phase 9: Packaging (Week 10)
- [ ] setup.py / pyproject.toml
- [ ] README with badges
- [ ] LICENSE
- [ ] CHANGELOG
- [ ] PyPI release

---

## APPENDIX A: Design Decisions

### Key Architectural Choices

1. **Strategy Pattern for Method Selection**
   - Why: Allows easy addition of new methods
   - Alternative: Factory pattern (less flexible)

2. **Context Managers for GPU Resources**
   - Why: Ensures cleanup, prevents leaks
   - Alternative: Manual init/cleanup (error-prone)

3. **Polars Primary, Pandas Compatible**
   - Why: Modern + fast, but ecosystem compatible
   - Alternative: Pandas only (slower, more memory)

4. **Custom Polygon Geometry (no Shapely)**
   - Why: Avoid messy dependencies, GPU-compatible
   - Alternative: Shapely (compilation issues, no GPU)

5. **Fuzzy Logic on Results**
   - Why: Sidesteps mesh boolean operation bugs
   - Alternative: Mesh CSG (buggy, slow, complex)

6. **Unified Result Object**
   - Why: Selection is just thresholded proportion
   - Alternative: Separate objects (redundant)

7. **Universal calculate() Function**
   - Why: Works with any input type (blocks/points × mesh/polygon)
   - Alternative: Separate functions per type (confusing)

8. **Point Queries for Containment**
   - Why: Check if points are inside geometry (common use case)
   - Alternative: Block assignment (less useful for resource modeling)

---

## APPENDIX B: Future Enhancements

### Post-v1.0 Features

1. **Irregular block models** (non-uniform blocks)
2. **Sub-blocking** (adaptive resolution)
3. **GPU kriging integration** (for resource estimation)
4. **Parallel CPU ray casting** (multiprocessing)
5. **Advanced mesh repair** (automatic fixing)
6. **Volume mesh support** (tetrahedra, not just surface)
7. **Implicit surface support** (SDFs, RBFs)

### Companion Library: fuzzyvolume

Separate library for:
- Signed Distance Field (SDF) operations
- Fuzzy set theory for geology
- Soft boundary modeling
- Uncertain domain boundaries

---

**END OF DESIGN DOCUMENT**
