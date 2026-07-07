"""
VTK to USD conversion workflow and batch runner.

Implements the pipeline from the Convert_VTK_To_USD experiment notebooks:
load one or more VTK files, optionally split by connectivity or cell type,
convert to USD, then apply a chosen appearance (solid color, anatomic material,
or colormap from a primvar with auto or specified intensity range).
"""

import logging
import re
from pathlib import Path
from typing import Literal

from .convert_vtk_to_usd import ConvertVTKToUSD
from .physiotwin4d_base import PhysioTwin4DBase
from .usd_anatomy_tools import USDAnatomyTools
from .usd_tools import USDTools

AppearanceKind = Literal["solid", "anatomy", "colormap"]


class WorkflowConvertVTKToUSD(PhysioTwin4DBase):
    """
    Workflow to convert one or more VTK files to USD with configurable
    splitting and appearance (solid color, anatomic material, or colormap).
    """

    def __init__(
        self,
        vtk_files: list[str | Path],
        output_usd: str | Path,
        *,
        separate_by_connectivity: bool = True,
        separate_by_cell_type: bool = False,
        mesh_name: str = "Mesh",
        times_per_second: float = 60.0,
        extract_surface: bool = True,
        time_series_pattern: str = r"\.t(\d+)\.(vtk|vtp|vtu)$",
        appearance: AppearanceKind = "solid",
        solid_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        anatomy_type: str = "heart",
        colormap_primvar: str | None = None,
        colormap_name: str = "viridis",
        colormap_intensity_range: tuple[float, float] | None = None,
        log_level: int | str = logging.INFO,
    ):
        """
        Initialize the VTK-to-USD workflow.

        Args:
            vtk_files: List of paths to VTK files (.vtk, .vtp, .vtu). One file = single frame;
                multiple files = time series (ordered by time_series_pattern).
            output_usd: Path to output USD file.
            separate_by_connectivity: If True, split mesh into separate objects by connectivity.
            separate_by_cell_type: If True, split mesh by cell type (triangle/quad/...).
                Cannot be True when separate_by_connectivity is True.
            mesh_name: Base name for the mesh (or first mesh when not splitting).
            times_per_second: FPS for time-varying data.
            extract_surface: For .vtu, extract surface before conversion.
            time_series_pattern: Regex to extract time index from filenames (one group).
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
        self.vtk_files = [Path(f) for f in vtk_files]
        self.output_usd = Path(output_usd)
        self.separate_by_connectivity = separate_by_connectivity
        self.separate_by_cell_type = separate_by_cell_type
        self.mesh_name = mesh_name
        self.times_per_second = times_per_second
        self.extract_surface = extract_surface
        self.time_series_pattern = time_series_pattern
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

    def discover_time_series(
        self,
        paths: list[Path],
        pattern: str = r"\.t(\d+)\.(vtk|vtp|vtu)$",
    ) -> tuple[list[tuple[int, Path]], bool]:
        """Discover and sort time-series VTK files by extracted time index.

        Args:
            paths: List of paths to VTK files
            pattern: Regex with one group for time step number (default matches .t123.vtk)

        Returns:
            (time_series, pattern_matched): Sorted list of (time_step, path) tuples, and
            a flag True only when every path matched the pattern (true time series).
            If any path does not match, time_series is [(0, p) for p in paths] and
            pattern_matched is False (static merge).
        """
        regex = re.compile(pattern, re.IGNORECASE)
        invalid_series = False
        parsed: list[tuple[int, Path]] = []
        for p in paths:
            match = regex.search(p.name)
            if match:
                parsed.append((int(match.group(1)), Path(p)))
            else:
                parsed.append((-1, Path(p)))
                invalid_series = True
        # Only treat as time series when all paths match; otherwise static merge
        if invalid_series:
            self.log_warning("Not a time series: %s", paths)
            time_series = [(0, p) for _, p in parsed]
            return time_series, False
        time_series = [(t, p) for t, p in parsed]
        time_series.sort(key=lambda x: (x[0], str(x[1])))
        return time_series, True

    def run(self) -> str:
        """
        Run the full workflow: convert VTK to USD, then apply the chosen appearance.

        Returns:
            Path to the created USD file (str).
        """
        self.log_section("VTK to USD conversion workflow")

        if not self.vtk_files:
            raise ValueError("vtk_files must not be empty")

        # Discover time series
        time_series, pattern_matched = self.discover_time_series(
            self.vtk_files, pattern=self.time_series_pattern
        )
        time_steps = [t for t, _ in time_series]
        time_codes = [float(t) for t in time_steps]
        paths_ordered = [p for _, p in time_series]
        n_frames = len(paths_ordered)

        # Multiple files but no pattern match: treat as static scene (no time samples)
        is_static_merge = n_frames > 1 and not pattern_matched

        self.log_info("Input: %d file(s), time steps: %s", n_frames, time_steps[:5])
        if n_frames > 5:
            self.log_info("  ... and %d more", n_frames - 5)
        if is_static_merge:
            self.log_info(
                "Filenames do not match time-series pattern; outputting static scene (no time samples)"
            )
        self.log_info("Output: %s", self.output_usd)

        separate_by: Literal["none", "connectivity", "cell_type"] = (
            "connectivity"
            if self.separate_by_connectivity
            else "cell_type"
            if self.separate_by_cell_type
            else "none"
        )

        converter = ConvertVTKToUSD.from_files(
            data_basename=self.mesh_name,
            vtk_files=paths_ordered,
            extract_surface=self.extract_surface,
            separate_by=separate_by,
            times_per_second=self.times_per_second,
            solid_color=self.solid_color,
            time_codes=time_codes if not is_static_merge else None,
            static_merge=is_static_merge,
            log_level=self.log_level,
        )
        stage = converter.convert(str(self.output_usd))

        # Post-process: apply chosen appearance to all meshes under /World/{mesh_name}
        usd_tools = USDTools(log_level=self.log_level)
        mesh_paths = usd_tools.list_mesh_paths_under(
            str(self.output_usd), parent_path=f"/World/{self.mesh_name}"
        )
        if not mesh_paths:
            self.log_warning("No mesh prims found under /World/%s", self.mesh_name)
            return str(self.output_usd)

        # Static merge has no time samples; pass None so only default time is used
        appearance_time_codes = None if is_static_merge else time_codes

        self.log_info(
            "Applying appearance '%s' to %d mesh(es)", self.appearance, len(mesh_paths)
        )

        if self.appearance == "solid":
            for mesh_path in mesh_paths:
                usd_tools.set_solid_display_color(
                    str(self.output_usd),
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
                    primvars = usd_tools.list_mesh_primvars(
                        str(self.output_usd), mesh_path
                    )
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
                    str(self.output_usd),
                    mesh_path,
                    primvar,
                    cmap=self.colormap_name,
                    intensity_range=self.colormap_intensity_range,
                    write_default_at_t0=True,
                    bind_vertex_color_material=True,
                )
                if self.colormap_primvar is None:
                    primvar = None  # next mesh: auto-pick again

        self.log_info("Workflow complete: %s", self.output_usd)
        return str(self.output_usd)
