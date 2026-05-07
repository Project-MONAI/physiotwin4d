"""Public low-level file conversion facade for vtk_to_usd."""

from pathlib import Path
from typing import Optional, Union

from pxr import Usd, UsdGeom

from .data_structures import ConversionSettings, MaterialData
from .material_manager import MaterialManager
from .usd_mesh_converter import UsdMeshConverter
from .vtk_reader import read_vtk_file


def convert_vtk_file(
    vtk_file: Union[str, Path],
    output_usd_file: Union[str, Path],
    *,
    data_basename: Optional[str] = None,
    mesh_name: str = "Mesh",
    extract_surface: bool = True,
    settings: Optional[ConversionSettings] = None,
    material: Optional[MaterialData] = None,
) -> Usd.Stage:
    """Convert one VTK file to one USD stage.

    This is the stable low-level facade for advanced users who want the
    file-based vtk_to_usd conversion layer directly. In-repository workflows,
    experiments, and CLIs should use :class:`physiomotion4d.ConvertVTKToUSD`
    instead.

    Args:
        vtk_file: Input VTK file path (.vtk, .vtp, or .vtu).
        output_usd_file: Output USD file path.
        data_basename: Root prim name under /World. Defaults to the input stem.
        mesh_name: Mesh prim name under /World/{data_basename}.
        extract_surface: If True, extract surfaces from volumetric VTK datasets.
        settings: Optional conversion settings. Defaults to ConversionSettings().
        material: Optional material. Defaults to settings.default_color.

    Returns:
        Created USD stage.
    """
    input_path = Path(vtk_file)
    output_path = Path(output_usd_file)
    conversion_settings = settings or ConversionSettings()
    root_name = data_basename or input_path.stem

    if output_path.exists():
        output_path.unlink()

    mesh_data = read_vtk_file(input_path, extract_surface=extract_surface)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, conversion_settings.meters_per_unit)
    up_axis = (
        UsdGeom.Tokens.z
        if conversion_settings.up_axis.upper() == "Z"
        else UsdGeom.Tokens.y
    )
    UsdGeom.SetStageUpAxis(stage, up_axis)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    root_path = f"/World/{root_name}"
    UsdGeom.Xform.Define(stage, root_path)

    material_mgr = MaterialManager(stage)
    mat_data = material or MaterialData(
        name=f"{mesh_name}_material",
        diffuse_color=conversion_settings.default_color,
    )
    material_mgr.get_or_create_material(mat_data)
    mesh_data.material_id = mat_data.name

    mesh_converter = UsdMeshConverter(stage, conversion_settings, material_mgr)
    mesh_converter.create_mesh(mesh_data, f"{root_path}/{mesh_name}")

    stage.Save()
    return stage
