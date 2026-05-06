"""Public advanced VTK-to-USD conversion layer for PhysioMotion4D.

This subpackage is a stable low-level API for advanced external users. Code in
PhysioMotion4D experiments, workflows, and CLIs should use ConvertVTKToUSD from
physiomotion4d.convert_vtk_to_usd instead of importing this package directly.

Provides:
- File facade: convert_vtk_file
- Data containers: MeshData, ConversionSettings, MaterialData, etc.
- VTK file readers (.vtk, .vtp, .vtu)
- USD primitive writers: UsdMeshConverter, MaterialManager
- Coordinate helpers (RAS to Y-up) and mesh splitting utilities
"""

from .converter import convert_vtk_file
from .data_structures import (
    ConversionSettings,
    DataType,
    GenericArray,
    MaterialData,
    MeshData,
    TimeStepData,
    VolumeData,
)
from .material_manager import MaterialManager
from .mesh_utils import (
    cell_type_name_for_vertex_count,
    split_mesh_data_by_cell_type,
    split_mesh_data_by_connectivity,
)
from .usd_mesh_converter import UsdMeshConverter
from .usd_utils import (
    compute_mesh_extent,
    create_primvar,
    ras_normals_to_usd,
    ras_points_to_usd,
    ras_to_usd,
    sanitize_primvar_name,
    triangulate_face,
)
from .vtk_reader import (
    LegacyVTKReader,
    PolyDataReader,
    UnstructuredGridReader,
    VTKReader,
    read_vtk_file,
    validate_time_series_topology,
)

__all__ = [
    # File facade
    "convert_vtk_file",
    # Data structures
    "ConversionSettings",
    "DataType",
    "GenericArray",
    "MaterialData",
    "MeshData",
    "TimeStepData",
    "VolumeData",
    # Managers
    "MaterialManager",
    "UsdMeshConverter",
    # Utilities
    "ras_to_usd",
    "ras_points_to_usd",
    "ras_normals_to_usd",
    "create_primvar",
    "sanitize_primvar_name",
    "triangulate_face",
    "compute_mesh_extent",
    # Mesh utils (cell type split)
    "cell_type_name_for_vertex_count",
    "split_mesh_data_by_cell_type",
    "split_mesh_data_by_connectivity",
    # Readers
    "VTKReader",
    "PolyDataReader",
    "LegacyVTKReader",
    "UnstructuredGridReader",
    "read_vtk_file",
    "validate_time_series_topology",
]

__version__ = "0.1.0"
