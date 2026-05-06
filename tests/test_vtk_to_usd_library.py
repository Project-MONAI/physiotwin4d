#!/usr/bin/env python
"""
Tests for VTK-to-USD conversion via ConvertVTKToUSD.

Covers:
- VTK file reading (VTP, VTK, VTU formats) via internal vtk_to_usd helpers
- ConvertVTKToUSD.from_files() — single file, time series, settings
- Material and primvar preservation
- Data structure validation (GenericArray, MeshData, etc.)

Note: Tests marked requires_data need manually downloaded data:
- KCL-Heart-Model: data/KCL-Heart-Model/
- CHOP-Valve4D: data/CHOP-Valve4D/
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv
from pxr import UsdGeom, UsdShade

from physiomotion4d import ConvertVTKToUSD
from physiomotion4d.vtk_to_usd import (
    DataType,
    GenericArray,
    MeshData,
    read_vtk_file,
)


# Helper to get data paths
def get_data_dir() -> Path:
    """Get the data directory path."""
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    return project_root / "data"


def check_kcl_heart_data() -> bool:
    """Check if KCL Heart Model data is available."""
    data_dir = get_data_dir() / "KCL-Heart-Model"
    vtk_file = data_dir / "average_mesh.vtk"
    return vtk_file.exists()


def check_valve4d_data() -> bool:
    """Check if CHOP Valve4D data is available."""
    data_dir = get_data_dir() / "CHOP-Valve4D"
    alterra_dir = data_dir / "Alterra"
    return alterra_dir.exists() and any(alterra_dir.glob("*.vtk"))


def get_or_create_average_surface(test_directories: dict[str, Path]) -> Path:
    """
    Get or create average_surface.vtp from average_mesh.vtk.

    This function extracts the surface from the volumetric mesh and caches it
    in the test output directory for reuse across test runs.

    Args:
        test_directories: Dictionary with 'output' key pointing to test output directory

    Returns:
        Path to the average_surface.vtp file
    """
    output_dir = test_directories["output"] / "vtk_to_usd_library"
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_file = output_dir / "average_surface.vtp"

    # If surface file already exists, return it
    if surface_file.exists():
        print(f"\nUsing cached surface file: {surface_file}")
        return surface_file

    # Create surface from volumetric mesh
    data_dir = get_data_dir() / "KCL-Heart-Model"
    vtk_file = data_dir / "average_mesh.vtk"

    print(f"\nCreating surface from: {vtk_file}")

    # Load volumetric mesh
    vtk_mesh = pv.read(str(vtk_file))

    # Extract surface
    surface = vtk_mesh.extract_surface(algorithm="dataset_surface")

    # Save to output directory
    surface.save(str(surface_file))

    print(f"Created and saved surface: {surface_file}")
    print(f"  Points: {surface.n_points:,}")
    print(f"  Faces: {surface.n_faces_strict:,}")

    return surface_file


@pytest.fixture(scope="session")
def kcl_average_surface(test_directories: dict[str, Path]) -> Path:
    """
    Fixture providing the KCL average heart surface.

    Generates average_surface.vtp from average_mesh.vtk if needed,
    caching the result for subsequent test runs.

    Returns:
        Path to average_surface.vtp file
    """
    if not check_kcl_heart_data():
        pytest.skip("KCL-Heart-Model data not available (must be manually downloaded)")

    return get_or_create_average_surface(test_directories)


class TestGenericArray:
    """Test GenericArray data structure validation and reshaping."""

    def test_scalar_1d_array(self) -> None:
        """Test that 1D scalar arrays (num_components=1) are kept as-is."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        array = GenericArray(
            name="test_scalar",
            data=data,
            num_components=1,
            data_type=DataType.FLOAT,
        )
        assert array.data.ndim == 1
        assert len(array.data) == 4
        np.testing.assert_array_equal(array.data, data)

    def test_flat_multicomponent_array_reshape(self) -> None:
        """Test that flat 1D arrays with num_components>1 are reshaped to 2D."""
        # 12 values that should reshape to (4, 3)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
        array = GenericArray(
            name="test_vector",
            data=data,
            num_components=3,
            data_type=DataType.FLOAT,
        )
        assert array.data.ndim == 2
        assert array.data.shape == (4, 3)
        # Verify data is preserved correctly
        expected = data.reshape(-1, 3)
        np.testing.assert_array_equal(array.data, expected)

    def test_2d_array_valid(self) -> None:
        """Test that 2D arrays with correct shape are accepted."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
        array = GenericArray(
            name="test_vector",
            data=data,
            num_components=3,
            data_type=DataType.FLOAT,
        )
        assert array.data.ndim == 2
        assert array.data.shape == (4, 3)
        np.testing.assert_array_equal(array.data, data)

    def test_flat_array_not_divisible_raises_error(self) -> None:
        """Test that flat arrays with length not divisible by num_components raise error."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)  # 5 values, not divisible by 3
        with pytest.raises(ValueError, match="not divisible by num_components"):
            GenericArray(
                name="test_invalid",
                data=data,
                num_components=3,
                data_type=DataType.FLOAT,
            )

    def test_2d_array_wrong_shape_raises_error(self) -> None:
        """Test that 2D arrays with wrong shape raise error."""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # Shape (3, 2)
        with pytest.raises(ValueError, match="incompatible with num_components"):
            GenericArray(
                name="test_invalid",
                data=data,
                num_components=3,  # Expects shape[1] == 3, got 2
                data_type=DataType.FLOAT,
            )

    def test_3d_array_raises_error(self) -> None:
        """Test that 3D arrays are rejected."""
        data = np.ones((2, 3, 4), dtype=float)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            GenericArray(
                name="test_invalid",
                data=data,
                num_components=3,
                data_type=DataType.FLOAT,
            )

    def test_flat_array_large_components(self) -> None:
        """Test reshaping with large num_components (e.g., 9 for 3x3 tensors)."""
        # 18 values that should reshape to (2, 9)
        data = np.arange(18, dtype=float)
        array = GenericArray(
            name="test_tensor",
            data=data,
            num_components=9,
            data_type=DataType.FLOAT,
        )
        assert array.data.ndim == 2
        assert array.data.shape == (2, 9)
        np.testing.assert_array_equal(array.data, data.reshape(-1, 9))


class TestFromFilesValidation:
    """Synthetic tests for ConvertVTKToUSD.from_files() — no real data required.

    Covers:
    - Gap A: time_codes length and monotonicity validation
    - Gap B: _cached_mesh_data population and reuse in _convert_unified()
    """

    # ------------------------------------------------------------------
    # Gap A — time_codes validation
    # ------------------------------------------------------------------

    def test_time_codes_length_mismatch_raises(self, tmp_path: Path) -> None:
        """from_files() must reject time_codes whose length != len(vtk_files)."""
        sphere = pv.Sphere(theta_resolution=4, phi_resolution=4)
        f0 = tmp_path / "f0.vtp"
        f1 = tmp_path / "f1.vtp"
        sphere.save(str(f0))
        sphere.save(str(f1))
        with pytest.raises(ValueError, match="time_codes length"):
            ConvertVTKToUSD.from_files("X", [f0, f1], time_codes=[0.0])

    def test_time_codes_non_monotone_raises(self, tmp_path: Path) -> None:
        """from_files() must reject time_codes that decrease between frames."""
        sphere = pv.Sphere(theta_resolution=4, phi_resolution=4)
        f0 = tmp_path / "f0.vtp"
        f1 = tmp_path / "f1.vtp"
        sphere.save(str(f0))
        sphere.save(str(f1))
        with pytest.raises(ValueError, match="non-decreasing order"):
            ConvertVTKToUSD.from_files("X", [f0, f1], time_codes=[2.0, 1.0])

    def test_time_codes_equal_consecutive_is_valid(self, tmp_path: Path) -> None:
        """Equal consecutive time codes are non-decreasing and must not raise."""
        sphere = pv.Sphere(theta_resolution=4, phi_resolution=4)
        f0 = tmp_path / "f0.vtp"
        f1 = tmp_path / "f1.vtp"
        sphere.save(str(f0))
        sphere.save(str(f1))
        converter = ConvertVTKToUSD.from_files("X", [f0, f1], time_codes=[1.0, 1.0])
        assert converter._time_codes == [1.0, 1.0]

    # ------------------------------------------------------------------
    # Gap B — topology cache population and reuse
    # ------------------------------------------------------------------

    def test_from_files_populates_cached_mesh_data(self, tmp_path: Path) -> None:
        """from_files() with >1 frame must populate _cached_mesh_data."""
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        files = []
        for i in range(3):
            p = tmp_path / f"p{i}.vtp"
            plane.save(str(p))
            files.append(p)
        converter = ConvertVTKToUSD.from_files("Plane", files)
        assert converter._cached_mesh_data is not None
        assert len(converter._cached_mesh_data) == 3
        assert all(isinstance(m, MeshData) for m in converter._cached_mesh_data)

    def test_from_files_cache_reused_in_convert(self, tmp_path: Path) -> None:
        """_convert_unified() must not call _vtk_to_mesh_data() when cache is populated."""
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        files = []
        for i in range(3):
            p = tmp_path / f"p{i}.vtp"
            plane.save(str(p))
            files.append(p)
        converter = ConvertVTKToUSD.from_files("Plane", files)
        with patch.object(
            converter, "_vtk_to_mesh_data", wraps=converter._vtk_to_mesh_data
        ) as spy:
            stage = converter.convert(str(tmp_path / "out.usd"))
        assert spy.call_count == 0, (
            f"_vtk_to_mesh_data called {spy.call_count} time(s); cache should have been used"
        )
        assert stage.GetPrimAtPath("/World/Plane/Mesh").IsValid()

    def test_from_files_single_file_no_cache(self, tmp_path: Path) -> None:
        """A single-file converter must not populate _cached_mesh_data."""
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        f0 = tmp_path / "p0.vtp"
        plane.save(str(f0))
        converter = ConvertVTKToUSD.from_files("P", [f0])
        assert converter._cached_mesh_data is None

    def test_from_files_static_merge_no_cache(self, tmp_path: Path) -> None:
        """static_merge=True must not populate _cached_mesh_data."""
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        f0 = tmp_path / "p0.vtp"
        f1 = tmp_path / "p1.vtp"
        plane.save(str(f0))
        plane.save(str(f1))
        converter = ConvertVTKToUSD.from_files("P", [f0, f1], static_merge=True)
        assert converter._cached_mesh_data is None


@pytest.mark.requires_data
class TestVTKReader:
    """Test VTK file reading capabilities."""

    def test_read_vtp_file(self, kcl_average_surface: Path) -> None:
        """Test reading VTP (PolyData) files."""
        vtp_file = kcl_average_surface

        assert vtp_file.exists(), f"VTP file not found: {vtp_file}"

        # Read the file
        mesh_data = read_vtk_file(vtp_file)

        # Verify mesh data structure
        assert mesh_data is not None
        assert mesh_data.points is not None
        assert len(mesh_data.points) > 0
        assert mesh_data.face_vertex_counts is not None
        assert mesh_data.face_vertex_indices is not None

        print(f"\nRead VTP file: {vtp_file.name}")
        print(f"  Points: {len(mesh_data.points):,}")
        print(f"  Faces: {len(mesh_data.face_vertex_counts):,}")
        print(f"  Data arrays: {len(mesh_data.generic_arrays)}")

    def test_read_legacy_vtk_file(self) -> None:
        """Test reading legacy VTK files."""
        if not check_kcl_heart_data():
            pytest.skip(
                "KCL-Heart-Model data not available (must be manually downloaded)"
            )

        data_dir = get_data_dir() / "KCL-Heart-Model"
        vtk_file = data_dir / "average_mesh.vtk"

        assert vtk_file.exists(), f"VTK file not found: {vtk_file}"

        # Read the file with surface extraction
        mesh_data = read_vtk_file(vtk_file, extract_surface=True)

        # Verify mesh data structure
        assert mesh_data is not None
        assert mesh_data.points is not None
        assert len(mesh_data.points) > 0
        assert mesh_data.face_vertex_counts is not None
        assert mesh_data.face_vertex_indices is not None

        print(f"\nRead legacy VTK file: {vtk_file.name}")
        print(f"  Points: {len(mesh_data.points):,}")
        print(f"  Faces: {len(mesh_data.face_vertex_counts):,}")
        print(f"  Data arrays: {len(mesh_data.generic_arrays)}")

    def test_generic_arrays_preserved(self, kcl_average_surface: Path) -> None:
        """Test that generic data arrays are preserved during reading."""
        vtp_file = kcl_average_surface

        mesh_data = read_vtk_file(vtp_file)

        # Verify generic arrays
        assert len(mesh_data.generic_arrays) > 0, "No data arrays found"

        # Check array structure
        for array in mesh_data.generic_arrays:
            assert array.name is not None
            assert array.data is not None
            assert array.num_components > 0
            assert array.interpolation in ["vertex", "uniform", "constant"]

        print("\nGeneric arrays preserved:")
        for array in mesh_data.generic_arrays:
            print(
                f"  - {array.name}: {array.num_components} components, {len(array.data):,} values"
            )


@pytest.mark.requires_data
class TestVTKToUSDConversion:
    """Test VTK to USD conversion capabilities."""

    def test_single_file_conversion(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test converting a single VTK file to USD."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        vtp_file = kcl_average_surface
        output_usd = output_dir / "heart_surface.usd"

        stage = ConvertVTKToUSD.from_files(
            data_basename="HeartSurface",
            vtk_files=[vtp_file],
        ).convert(str(output_usd))

        assert output_usd.exists()
        assert stage is not None

        # No split: mesh lives at /World/HeartSurface/Mesh
        mesh_prim = stage.GetPrimAtPath("/World/HeartSurface/Mesh")
        assert mesh_prim.IsValid()
        assert mesh_prim.IsA(UsdGeom.Mesh)

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        assert len(points) > 0

        print("\nConverted single file to USD")
        print(f"  Input: {vtp_file.name}")
        print(f"  Output: {output_usd}")
        print(f"  Points: {len(points):,}")

    def test_conversion_with_material(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test conversion with a custom solid color material."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        vtp_file = kcl_average_surface
        output_usd = output_dir / "heart_with_material.usd"

        stage = ConvertVTKToUSD.from_files(
            data_basename="HeartSurface",
            vtk_files=[vtp_file],
            solid_color=(0.9, 0.3, 0.3),
        ).convert(str(output_usd))

        # Material for the no-split case is named "Mesh_material"
        material_prim = stage.GetPrimAtPath("/World/Looks/Mesh_material")
        assert material_prim.IsValid()
        assert material_prim.IsA(UsdShade.Material)

        # Verify material is bound to the mesh prim
        mesh_prim = stage.GetPrimAtPath("/World/HeartSurface/Mesh")
        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        bound_material = binding_api.ComputeBoundMaterial()[0]
        assert bound_material.GetPrim().IsValid()

        # Verify the shader's diffuseColor input carries the requested solid color
        shader_prim = stage.GetPrimAtPath("/World/Looks/Mesh_material/PreviewSurface")
        assert shader_prim.IsValid()
        shader = UsdShade.Shader(shader_prim)
        diffuse_value = shader.GetInput("diffuseColor").Get()
        assert diffuse_value is not None
        assert tuple(diffuse_value) == pytest.approx((0.9, 0.3, 0.3), abs=1e-5)

        print("\nConverted with custom solid color material")
        print("  Material path: /World/Looks/Mesh_material")

    def test_conversion_settings(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test that ConvertVTKToUSD applies correct default stage metadata."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        vtp_file = kcl_average_surface
        output_usd = output_dir / "heart_custom_settings.usd"

        stage = ConvertVTKToUSD.from_files(
            data_basename="Mesh",
            vtk_files=[vtp_file],
        ).convert(str(output_usd))

        # ConvertVTKToUSD always uses meters_per_unit=1.0 and Y-up
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y

        print("\nVerified stage metadata defaults")
        print(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
        print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")

    def test_primvar_preservation(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test that VTK data arrays are preserved as USD primvars."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        vtp_file = kcl_average_surface

        # Read source to count arrays
        mesh_data = read_vtk_file(vtp_file)
        array_names = [arr.name for arr in mesh_data.generic_arrays]

        output_usd = output_dir / "heart_with_primvars.usd"

        stage = ConvertVTKToUSD.from_files(
            data_basename="Mesh",
            vtk_files=[vtp_file],
        ).convert(str(output_usd))

        # No split: mesh at /World/Mesh/Mesh
        mesh_prim = stage.GetPrimAtPath("/World/Mesh/Mesh")
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        primvars = primvars_api.GetPrimvars()

        primvar_names = [pv.GetPrimvarName() for pv in primvars]
        assert len(primvar_names) > 0

        print("\nPrimvars preserved:")
        print(f"  Source arrays: {len(array_names)}")
        print(f"  USD primvars: {len(primvar_names)}")
        for name in primvar_names[:5]:
            print(f"    - {name}")


@pytest.mark.requires_data
class TestTimeSeriesConversion:
    """Test time-series conversion capabilities."""

    def test_time_series_conversion(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test converting multiple VTK files as a time series."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        vtp_file = kcl_average_surface

        # Use the same file three times to simulate a time series
        vtk_files = [vtp_file] * 3
        time_codes = [0.0, 1.0, 2.0]

        output_usd = output_dir / "heart_time_series.usd"

        stage = ConvertVTKToUSD.from_files(
            data_basename="Mesh",
            vtk_files=vtk_files,
            time_codes=time_codes,
        ).convert(str(output_usd))

        assert stage.GetStartTimeCode() == 0.0
        assert stage.GetEndTimeCode() == 2.0

        # No split: mesh at /World/Mesh/Mesh
        mesh_prim = stage.GetPrimAtPath("/World/Mesh/Mesh")
        mesh = UsdGeom.Mesh(mesh_prim)
        time_samples = mesh.GetPointsAttr().GetTimeSamples()
        assert len(time_samples) == 3
        assert time_samples == time_codes

        print("\nConverted time series")
        print(f"  Frames: {len(vtk_files)}")
        print(f"  Time codes: {time_codes}")
        print(
            f"  Stage time range: {stage.GetStartTimeCode()} - {stage.GetEndTimeCode()}"
        )


@pytest.mark.slow
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_end_to_end_conversion(
        self, test_directories: dict[str, Path], kcl_average_surface: Path
    ) -> None:
        """Test complete conversion workflow with all features."""
        output_dir = test_directories["output"] / "vtk_to_usd_library"
        output_dir.mkdir(parents=True, exist_ok=True)

        vtp_file = kcl_average_surface
        output_usd = output_dir / "heart_complete.usd"

        stage = ConvertVTKToUSD.from_files(
            data_basename="CardiacModel",
            vtk_files=[vtp_file],
            solid_color=(0.85, 0.2, 0.2),
            times_per_second=24.0,
        ).convert(str(output_usd))

        assert output_usd.exists()
        assert stage is not None

        # No split: mesh at /World/CardiacModel/Mesh
        mesh_prim = stage.GetPrimAtPath("/World/CardiacModel/Mesh")
        assert mesh_prim.IsValid()

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        assert len(points) > 0

        # Material is auto-named "Mesh_material"
        material_prim = stage.GetPrimAtPath("/World/Looks/Mesh_material")
        assert material_prim.IsValid()

        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        primvars = primvars_api.GetPrimvars()
        assert len(primvars) > 0

        print("\nEnd-to-end conversion complete")
        print(f"  Output: {output_usd}")
        print(f"  Size: {output_usd.stat().st_size / 1024:.1f} KB")
        print(f"  Points: {len(points):,}")
        print(f"  Primvars: {len(primvars)}")


class TestUnitScaling:
    """Verify that VTK mm coordinates are converted to USD meter coordinates."""

    def test_mm_to_m_point_scaling(self, tmp_path: Path) -> None:
        """Points written to USD must be 0.001× their original mm values."""
        # Sphere with radius=100 mm — vertices should be near ±100 in VTK.
        mesh = pv.Sphere(radius=100.0)
        output_usd = tmp_path / "sphere.usd"

        stage = ConvertVTKToUSD(
            data_basename="Sphere",
            input_polydata=[mesh],
        ).convert(str(output_usd))

        mesh_prim = stage.GetPrimAtPath("/World/Sphere/Mesh")
        assert mesh_prim.IsValid(), "Mesh prim not found at expected path"

        usd_mesh = UsdGeom.Mesh(mesh_prim)
        usd_points = usd_mesh.GetPointsAttr().Get()
        assert usd_points is not None and len(usd_points) > 0

        coords = np.array(usd_points)
        max_coord = float(np.abs(coords).max())

        # In meters a 100 mm sphere has vertices ≤ 0.1 m (plus floating-point headroom).
        assert max_coord < 0.15, (
            f"Max coordinate {max_coord:.4f} is not in meters. "
            "Expected < 0.15 m for a 100 mm radius sphere; "
            "got a value that looks like millimeters."
        )
        # Sanity-check it's not collapsed to near zero (e.g., double-scaling).
        assert max_coord > 0.05, (
            f"Max coordinate {max_coord:.6f} is unexpectedly small."
        )

    def test_normals_remain_unit_length(self, tmp_path: Path) -> None:
        """Normal vectors must not be scaled — they should remain unit length."""
        mesh = pv.Sphere(radius=100.0)
        mesh.compute_normals(inplace=True)
        output_usd = tmp_path / "sphere_normals.usd"

        stage = ConvertVTKToUSD(
            data_basename="Sphere",
            input_polydata=[mesh],
        ).convert(str(output_usd))

        mesh_prim = stage.GetPrimAtPath("/World/Sphere/Mesh")
        usd_mesh = UsdGeom.Mesh(mesh_prim)
        normals_attr = usd_mesh.GetNormalsAttr()

        if normals_attr is None or normals_attr.Get() is None:
            pytest.skip("No normals on this mesh")

        normals = np.array(normals_attr.Get())
        norms = np.linalg.norm(normals, axis=1)
        # Every normal should be ≈ 1.0 (unit vector), not 0.001.
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"Normals are not unit length after conversion. "
            f"Mean norm: {norms.mean():.6f}, expected ≈ 1.0"
        )

    def test_stage_meters_per_unit(self, tmp_path: Path) -> None:
        """Stage metersPerUnit metadata must be 1.0 (coordinates stored in meters)."""
        mesh = pv.Sphere(radius=100.0)
        output_usd = tmp_path / "sphere_meta.usd"

        stage = ConvertVTKToUSD(
            data_basename="Sphere",
            input_polydata=[mesh],
        ).convert(str(output_usd))

        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
