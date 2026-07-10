"""
VTK to USD conversion workflow and batch runner.

Implements the pipeline from the Convert_VTK_To_USD experiment notebooks:
take one or more meshes, optionally split by connectivity or cell type,
convert to USD, then apply a chosen appearance (solid color, anatomic material,
or colormap from a primvar with auto or specified intensity range).
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import pyvista as pv
import vtk

from .convert_vtk_to_usd import ConvertVTKToUSD
from .physiotwin4d_base import PhysioTwin4DBase
from .usd_anatomy_tools import USDAnatomyTools
from .usd_tools import USDTools

AppearanceKind = Literal["solid", "anatomy", "colormap"]


class WorkflowConvertVTKToUSD(PhysioTwin4DBase):
    """
    Workflow to convert one or more meshes to USD with configurable
    splitting and appearance (solid color, anatomic material, or colormap).
    """

    def __init__(
        self,
        input_meshes: Sequence[Union[pv.DataSet, vtk.vtkDataSet]],
        usd_project_name: str,
        output_directory: Union[str, Path],
        *,
        separate_by_connectivity: bool = True,
        separate_by_cell_type: bool = False,
        frames_per_second: float = 60.0,
        extract_surface: bool = True,
        static_merge: bool = False,
        time_codes: Optional[list[float]] = None,
        appearance: AppearanceKind = "solid",
        solid_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        anatomy_type: str = "heart",
        colormap_primvar: Optional[str] = None,
        colormap_name: str = "viridis",
        colormap_intensity_range: Optional[tuple[float, float]] = None,
        log_level: int | str = logging.INFO,
    ):
        """
        Initialize the VTK-to-USD workflow.

        Args:
            input_meshes: One or more PyVista/VTK meshes. A single mesh, or
                static_merge=True, produces a static scene; multiple meshes
                with static_merge=False (default) are treated as ordered
                time-series frames, in list order.
            usd_project_name: Project name; used as the root USD prim name
                (/World/{usd_project_name}) and the output filename
                ({usd_project_name}.usd).
            output_directory: Directory where the output USD file is written.
            separate_by_connectivity: If True, split mesh into separate objects by connectivity.
            separate_by_cell_type: If True, split mesh by cell type (triangle/quad/...).
                Cannot be True when separate_by_connectivity is True.
            frames_per_second: FPS for time-varying data.
            extract_surface: For volumetric meshes, extract surface before conversion.
            static_merge: If True, input_meshes is not a time series - each mesh is
                written as a separate static object with no time samples (see
                ConvertVTKToUSD).
            time_codes: Explicit time codes aligned to input_meshes, used when
                static_merge is False. If None, uses sequential integers [0, 1, 2, ...].
            appearance: "solid" | "anatomy" | "colormap".
            solid_color: RGB in [0,1] when appearance == "solid".
            anatomy_type: Anatomy material name when appearance == "anatomy"
                (e.g. heart, lung, bone, soft_tissue).
            colormap_primvar: Primvar name for coloring when appearance == "colormap"
                (e.g. vtk_point_stress_c0). If None, a candidate is auto-picked when possible.
            colormap_name: Matplotlib colormap name when appearance == "colormap".
            colormap_intensity_range: Optional (vmin, vmax) for colormap; None = auto from data.
            log_level: Logging level.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.input_meshes = list(input_meshes)
        self.usd_project_name = usd_project_name
        self.output_directory = Path(output_directory)
        self.separate_by_connectivity = separate_by_connectivity
        self.separate_by_cell_type = separate_by_cell_type
        self.frames_per_second = frames_per_second
        self.extract_surface = extract_surface
        self.static_merge = static_merge
        self.time_codes = time_codes
        self.appearance = appearance
        self.solid_color = solid_color
        self.anatomy_type = anatomy_type
        self.colormap_primvar = colormap_primvar
        self.colormap_name = colormap_name
        self.colormap_intensity_range = colormap_intensity_range

        if separate_by_connectivity and separate_by_cell_type:
            raise ValueError(
                "separate_by_connectivity and separate_by_cell_type cannot both be True"
            )

    def process(self) -> str:
        """
        Run the full workflow: convert meshes to USD, then apply the chosen appearance.

        Returns:
            Path to the created USD file (str).
        """
        self.log_section("VTK to USD conversion workflow")

        if not self.input_meshes:
            raise ValueError("input_meshes must not be empty")

        n_frames = len(self.input_meshes)
        time_codes = (
            None
            if self.static_merge
            else self.time_codes or [float(i) for i in range(n_frames)]
        )

        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_usd = self.output_directory / f"{self.usd_project_name}.usd"

        self.log_info("Input: %d mesh(es)", n_frames)
        if self.static_merge:
            self.log_info(
                "static_merge=True; outputting static scene (no time samples)"
            )
        self.log_info("Output: %s", output_usd)

        separate_by: Literal["none", "connectivity", "cell_type"] = (
            "connectivity"
            if self.separate_by_connectivity
            else "cell_type"
            if self.separate_by_cell_type
            else "none"
        )

        converter = ConvertVTKToUSD(
            data_basename=self.usd_project_name,
            input_polydata=self.input_meshes,
            convert_to_surface=self.extract_surface,
            separate_by=separate_by,
            frames_per_second=self.frames_per_second,
            solid_color=self.solid_color,
            static_merge=self.static_merge,
            time_codes=time_codes,
            log_level=self.log_level,
        )
        stage = converter.convert(str(output_usd))

        # Post-process: apply chosen appearance to all meshes under /World/{usd_project_name}
        usd_tools = USDTools(log_level=self.log_level)
        mesh_paths = usd_tools.list_mesh_paths_under(
            str(output_usd), parent_path=f"/World/{self.usd_project_name}"
        )
        if not mesh_paths:
            self.log_warning(
                "No mesh prims found under /World/%s", self.usd_project_name
            )
            return str(output_usd)

        # Static merge has no time samples; pass None so only default time is used
        appearance_time_codes = None if self.static_merge else time_codes

        self.log_info(
            "Applying appearance '%s' to %d mesh(es)", self.appearance, len(mesh_paths)
        )

        if self.appearance == "solid":
            for mesh_path in mesh_paths:
                usd_tools.set_solid_display_color(
                    str(output_usd),
                    mesh_path,
                    self.solid_color,
                    time_codes=appearance_time_codes,
                    bind_vertex_color_material=True,
                )

        elif self.appearance == "anatomy":
            anatomy_tools = USDAnatomyTools(stage, log_level=self.log_level)
            for mesh_path in mesh_paths:
                anatomy_tools.apply_anatomy_material_to_mesh(
                    mesh_path, self.anatomy_type
                )
            stage.Save()

        elif self.appearance == "colormap":
            primvar = self.colormap_primvar
            for mesh_path in mesh_paths:
                if primvar is None:
                    primvars = usd_tools.list_mesh_primvars(str(output_usd), mesh_path)
                    primvar = usd_tools.pick_color_primvar(primvars)
                if primvar is None:
                    self.log_warning(
                        "No color primvar found for %s; skip colormap", mesh_path
                    )
                    primvar = self.colormap_primvar
                    continue
                self.log_info(
                    "Applying colormap to %s from primvar %s", mesh_path, primvar
                )
                usd_tools.apply_colormap_from_primvar(
                    str(output_usd),
                    mesh_path,
                    primvar,
                    cmap=self.colormap_name,
                    intensity_range=self.colormap_intensity_range,
                    write_default_at_t0=True,
                    bind_vertex_color_material=True,
                )
                if self.colormap_primvar is None:
                    primvar = None  # next mesh: auto-pick again

        self.log_info("Workflow complete: %s", output_usd)
        return str(output_usd)
