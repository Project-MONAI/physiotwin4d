"""USD utility functions for VTK to USD conversion.

Provides helper functions for coordinate conversion, primvar creation, and USD type mapping.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from .data_structures import DataType, GenericArray

logger = logging.getLogger(__name__)


def ras_to_usd(point: NDArray | tuple | list) -> Gf.Vec3f:
    """Convert RAS (Right-Anterior-Superior) coordinates to USD's right-handed Y-up system.

    VTK/Medical imaging typically uses RAS coordinate system:
    - R (Right): X-axis points to patient's right
    - A (Anterior): Y-axis points to patient's front
    - S (Superior): Z-axis points to patient's head

    USD uses right-handed Y-up:
    - X: right
    - Y: up
    - Z: back (toward camera)

    Conversion: USD(x, y, z) = RAS(x, z, -y) * 0.001  (mm → m)

    Args:
        point: Point in RAS coordinates [x, y, z] in millimeters

    Returns:
        Gf.Vec3f: Point in USD coordinates in meters
    """
    if isinstance(point, (tuple, list)):
        return Gf.Vec3f(
            float(point[0]) * 0.001,
            float(point[2]) * 0.001,
            float(-point[1]) * 0.001,
        )
    else:
        return Gf.Vec3f(
            float(point[0]) * 0.001,
            float(point[2]) * 0.001,
            float(-point[1]) * 0.001,
        )


def ras_points_to_usd(points: NDArray) -> Vt.Vec3fArray:
    """Convert array of RAS points (mm) to USD coordinates (m).

    Applies axis swap RAS → Y-up and scales millimeters to meters (* 0.001).

    Args:
        points: Array of points with shape (N, 3) in millimeters

    Returns:
        Vt.Vec3fArray: Points in USD Y-up coordinates in meters
    """
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

    # Vectorized: USD(x, y, z) = RAS(x, z, -y) * 0.001  (mm → m)
    usd_points = np.empty(points.shape, dtype=np.float32)
    usd_points[:, 0] = points[:, 0] * 0.001
    usd_points[:, 1] = points[:, 2] * 0.001
    usd_points[:, 2] = -points[:, 1] * 0.001

    return Vt.Vec3fArray.FromNumpy(usd_points)


def ras_normals_to_usd(normals: NDArray) -> Vt.Vec3fArray:
    """Convert array of RAS normals to USD Y-up coordinates.

    Applies only the axis swap — normals are unit direction vectors and must
    not be scaled by the mm→m factor.

    Args:
        normals: Array of normals with shape (N, 3)

    Returns:
        Vt.Vec3fArray: Normals in USD Y-up coordinates (unit length preserved)
    """
    if normals.shape[1] != 3:
        raise ValueError(f"Normals must have shape (N, 3), got {normals.shape}")

    usd_normals = np.empty(normals.shape, dtype=np.float32)
    usd_normals[:, 0] = normals[:, 0]
    usd_normals[:, 1] = normals[:, 2]
    usd_normals[:, 2] = -normals[:, 1]

    return Vt.Vec3fArray.FromNumpy(usd_normals)


def numpy_to_vt_array(array: NDArray, data_type: DataType) -> Any:
    """Convert numpy array to appropriate VtArray type.

    Args:
        array: Numpy array to convert
        data_type: Target data type

    Returns:
        Appropriate VtArray based on data_type and array shape
    """
    # Ensure contiguous array for efficient conversion
    array = np.ascontiguousarray(array)

    # Determine number of components
    if array.ndim == 1:
        num_components = 1
    elif array.ndim == 2:
        num_components = array.shape[1]
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    # Convert based on type and components
    if data_type in [DataType.FLOAT, DataType.DOUBLE]:
        array_f = array.astype(np.float32)
        if num_components == 1:
            return Vt.FloatArray.FromNumpy(array_f)
        elif num_components == 2:
            return Vt.Vec2fArray.FromNumpy(array_f)
        elif num_components == 3:
            return Vt.Vec3fArray.FromNumpy(array_f)
        elif num_components == 4:
            return Vt.Vec4fArray.FromNumpy(array_f)
        else:
            # Fallback: flatten to float array
            return Vt.FloatArray.FromNumpy(array_f.flatten())

    elif data_type in [DataType.INT, DataType.LONG]:
        array_i = array.astype(np.int32)
        if num_components == 1:
            return Vt.IntArray.FromNumpy(array_i)
        elif num_components == 2:
            return Vt.Vec2iArray.FromNumpy(array_i)
        elif num_components == 3:
            return Vt.Vec3iArray.FromNumpy(array_i)
        elif num_components == 4:
            return Vt.Vec4iArray.FromNumpy(array_i)
        else:
            return Vt.IntArray.FromNumpy(array_i.flatten())

    elif data_type in [DataType.UINT, DataType.ULONG]:
        array_ui = array.astype(np.uint32)
        if num_components == 1:
            return Vt.UIntArray.FromNumpy(array_ui)
        else:
            # No Vec types for uint, flatten
            return Vt.UIntArray.FromNumpy(array_ui.flatten())

    elif data_type in [DataType.UCHAR, DataType.CHAR]:
        array_uc = array.astype(np.uint8)
        return Vt.UCharArray.FromNumpy(array_uc.flatten())

    elif data_type in [DataType.SHORT, DataType.USHORT]:
        # Convert to int
        array_i = array.astype(np.int32)
        return Vt.IntArray.FromNumpy(array_i.flatten())

    else:
        # Fallback to float
        array_f = array.astype(np.float32)
        return Vt.FloatArray.FromNumpy(array_f.flatten())


def get_sdf_value_type(data_type: DataType, num_components: int) -> Sdf.ValueTypeName:
    """Get appropriate SDF value type for primvar creation.

    Args:
        data_type: Data type
        num_components: Number of components (1, 2, 3, or 4)

    Returns:
        Sdf.ValueTypeName: Appropriate USD type
    """
    if data_type in [DataType.FLOAT, DataType.DOUBLE]:
        if num_components == 1:
            return Sdf.ValueTypeNames.FloatArray
        elif num_components == 2:
            return Sdf.ValueTypeNames.Float2Array
        elif num_components == 3:
            return Sdf.ValueTypeNames.Float3Array
        elif num_components == 4:
            return Sdf.ValueTypeNames.Float4Array
        else:
            return Sdf.ValueTypeNames.FloatArray

    elif data_type in [DataType.INT, DataType.LONG]:
        if num_components == 1:
            return Sdf.ValueTypeNames.IntArray
        elif num_components == 2:
            return Sdf.ValueTypeNames.Int2Array
        elif num_components == 3:
            return Sdf.ValueTypeNames.Int3Array
        elif num_components == 4:
            return Sdf.ValueTypeNames.Int4Array
        else:
            return Sdf.ValueTypeNames.IntArray

    elif data_type in [DataType.UINT, DataType.ULONG]:
        return Sdf.ValueTypeNames.UIntArray

    elif data_type in [DataType.UCHAR, DataType.CHAR]:
        return Sdf.ValueTypeNames.UCharArray

    elif data_type in [DataType.SHORT, DataType.USHORT]:
        return Sdf.ValueTypeNames.IntArray

    else:
        return Sdf.ValueTypeNames.FloatArray


def sanitize_primvar_name(name: str) -> str:
    """Sanitize a name to be USD-compliant.

    USD attribute names must:
    - Start with a letter or underscore
    - Contain only letters, numbers, and underscores
    - Not contain dots, spaces, or special characters

    Args:
        name: Original name

    Returns:
        str: Sanitized name safe for USD
    """
    import re

    # Replace dots with underscores
    name = name.replace(".", "_")

    # Replace spaces and other special characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure it starts with a letter or underscore
    if name and name[0].isdigit():
        name = "_" + name

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Remove trailing underscores
    name = name.rstrip("_")

    return name


def create_primvar(
    geom: UsdGeom.Gprim,
    array: GenericArray,
    array_name_prefix: str = "",
    time_code: float | None = None,
) -> UsdGeom.Primvar | None:
    """Create a USD primvar from a GenericArray.

    Args:
        geom: USD geometry prim (Mesh, Points, etc.)
        array: GenericArray containing data
        array_name_prefix: Prefix for primvar name (e.g., "vtk_point_")
        time_code: Optional time code for time-varying data

    Returns:
        UsdGeom.Primvar: Created primvar, or None if validation failed
    """
    # Sanitize the array name to be USD-compliant
    sanitized_name = sanitize_primvar_name(array.name)
    primvar_name = f"{array_name_prefix}{sanitized_name}"

    # Log if name was changed
    if sanitized_name != array.name:
        logger.debug(f"Sanitized primvar name: '{array.name}' → '{sanitized_name}'")

    # Validate array size for meshes
    if isinstance(geom, UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(geom)

        # Check size matches expected count based on interpolation
        if array.interpolation == "vertex":
            # Get number of points
            points_attr = mesh.GetPointsAttr()
            if points_attr:
                points = points_attr.Get(
                    time_code if time_code is not None else Usd.TimeCode.Default()
                )
                if points and len(array.data) != len(points):
                    logger.warning(
                        f"Skipping primvar '{primvar_name}': size mismatch "
                        f"(got {len(array.data)}, expected {len(points)} vertices)"
                    )
                    return None

        elif array.interpolation == "uniform":
            # Get number of faces
            face_counts_attr = mesh.GetFaceVertexCountsAttr()
            if face_counts_attr:
                face_counts = face_counts_attr.Get(
                    time_code if time_code is not None else Usd.TimeCode.Default()
                )
                if face_counts and len(array.data) != len(face_counts):
                    logger.warning(
                        f"Skipping primvar '{primvar_name}': size mismatch "
                        f"(got {len(array.data)}, expected {len(face_counts)} faces)"
                    )
                    return None

    # Skip if array has no data
    if len(array.data) == 0:
        logger.debug(f"Skipping empty primvar '{primvar_name}'")
        return None

    # Get primvars API
    primvars_api = UsdGeom.PrimvarsAPI(geom)

    # Get appropriate USD type
    sdf_type = get_sdf_value_type(array.data_type, array.num_components)

    # Create primvar
    primvar = primvars_api.CreatePrimvar(primvar_name, sdf_type)

    # Set interpolation
    if array.interpolation == "vertex":
        primvar.SetInterpolation(UsdGeom.Tokens.vertex)
    elif array.interpolation == "uniform":
        primvar.SetInterpolation(UsdGeom.Tokens.uniform)
    elif array.interpolation == "constant":
        primvar.SetInterpolation(UsdGeom.Tokens.constant)
    else:
        primvar.SetInterpolation(UsdGeom.Tokens.vertex)

    # If this is a multi-component array that we're storing in a scalar array type
    # (e.g. FloatArray for >4 components), preserve the component grouping via elementSize.
    # This makes downstream tools (and USDTools.apply_colormap_from_primvar) able to reshape.
    if array.num_components > 1 and sdf_type in (
        Sdf.ValueTypeNames.FloatArray,
        Sdf.ValueTypeNames.IntArray,
        Sdf.ValueTypeNames.UIntArray,
        Sdf.ValueTypeNames.UCharArray,
    ):
        try:
            primvar.SetElementSize(int(array.num_components))
        except Exception:
            # Not fatal; continue without elementSize.
            pass

    # Convert data to VtArray
    vt_array = numpy_to_vt_array(array.data, array.data_type)

    # Set value (with or without time code)
    if time_code is not None:
        primvar.Set(vt_array, time_code)
    else:
        primvar.Set(vt_array)

    logger.debug(
        f"Created primvar '{primvar_name}' with {len(array.data)} elements, "
        f"{array.num_components} components, type {array.data_type.value}"
    )

    return primvar


def triangulate_face(face_counts: NDArray, face_indices: NDArray) -> tuple:
    """Triangulate polygonal faces.

    Converts quads and polygons to triangles using simple fan triangulation.

    Args:
        face_counts: Array of vertex counts per face
        face_indices: Array of vertex indices

    Returns:
        tuple: (triangulated_counts, triangulated_indices)
    """
    tri_counts: list[int] = []
    tri_indices: list[int] = []

    idx = 0
    for count in face_counts:
        if count == 3:
            # Already a triangle
            tri_counts.append(3)
            tri_indices.extend(face_indices[idx : idx + 3])
        elif count == 4:
            # Quad -> 2 triangles
            v0, v1, v2, v3 = face_indices[idx : idx + 4]
            tri_counts.extend([3, 3])
            tri_indices.extend([v0, v1, v2, v0, v2, v3])
        else:
            # Polygon -> fan triangulation
            v0 = face_indices[idx]
            for i in range(1, count - 1):
                tri_counts.append(3)
                tri_indices.extend(
                    [v0, face_indices[idx + i], face_indices[idx + i + 1]]
                )

        idx += count

    return np.array(tri_counts, dtype=np.int32), np.array(tri_indices, dtype=np.int32)


def compute_mesh_extent(points: Vt.Vec3fArray) -> Vt.Vec3fArray:
    """Compute bounding box extent for a mesh.

    Args:
        points: Array of points

    Returns:
        Vt.Vec3fArray: Extent as [min_point, max_point]
    """
    return UsdGeom.Mesh.ComputeExtent(points)
