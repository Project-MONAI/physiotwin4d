"""Tests for the appearance and object-naming behavior of the VTK-to-USD workflow.

Synthetic meshes only - no segmentation or image data required.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from pxr import Usd, UsdShade

from physiotwin4d import WorkflowConvertVTKToUSD


def _labeled_sphere(
    center: tuple[float, float, float],
    label_name: str,
    group: Optional[str] = None,
) -> pv.PolyData:
    """Return a sphere annotated the way WorkflowConvertImageToVTK annotates one."""
    surface = pv.Sphere(radius=1.0, center=center, theta_resolution=8, phi_resolution=8)
    surface.field_data["SegmentationLabelNames"] = np.array([label_name])
    if group is not None:
        surface.field_data["AnatomyGroup"] = np.array([group])
    return surface


def _bound_material_path(stage: Usd.Stage, mesh_path: str) -> str:
    prim = stage.GetPrimAtPath(mesh_path)
    assert prim.IsValid(), f"Missing prim: {mesh_path}"
    binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
    return str(binding.GetMaterialPath())


class TestAnatomyAppearance:
    """Per-structure materials must follow the structure names on the meshes."""

    def test_label_names_drive_prim_names_and_materials(self, tmp_path: Path) -> None:
        """Each labeled mesh becomes its own prim with its own anatomy material."""
        meshes = [
            _labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium"),
            _labeled_sphere((3.0, 0.0, 0.0), "highres_ventricle_left"),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        # separate_by_connectivity defaults to True, and each sphere is a
        # single connected component, so every object gains one "_object1" part.
        myocardium = _bound_material_path(
            stage, "/World/heart/highres_myocardium_object1"
        )
        ventricle = _bound_material_path(
            stage, "/World/heart/highres_ventricle_left_object1"
        )
        assert myocardium.endswith("OmniSurface_Myocardium")
        assert ventricle.endswith("OmniSurface_Ventricle_Left")

    def test_explicit_anatomy_type_overrides_names(self, tmp_path: Path) -> None:
        """A caller-supplied anatomy_type still paints every object the same."""
        meshes = [
            _labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium"),
            _labeled_sphere((3.0, 0.0, 0.0), "highres_ventricle_left"),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            anatomy_type="heart",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        for name in ("highres_myocardium", "highres_ventricle_left"):
            material = _bound_material_path(stage, f"/World/heart/{name}_object1")
            assert material.endswith("OmniSurface_Heart")

    def test_unmatched_name_falls_back_to_group(self, tmp_path: Path) -> None:
        """No material is named "rib_left_3", so its anatomy group decides."""
        meshes = [
            _labeled_sphere((0.0, 0.0, 0.0), "rib_left_3", group="bone"),
            _labeled_sphere((3.0, 0.0, 0.0), "vertebrae_T7", group="bone"),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="chest",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        for name in ("rib_left_3", "vertebrae_T7"):
            material = _bound_material_path(stage, f"/World/chest/{name}_object1")
            assert material.endswith("OmniSurface_Bone")

    def test_structure_name_wins_over_group(self, tmp_path: Path) -> None:
        """A structure with its own material must not collapse onto its group."""
        mesh = _labeled_sphere((0.0, 0.0, 0.0), "highres_myocardium", group="heart")

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[mesh],
            usd_project_name="heart",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        material = _bound_material_path(
            stage, "/World/heart/highres_myocardium_object1"
        )
        assert material.endswith("OmniSurface_Myocardium")

    def test_unmatched_name_falls_back_to_other(self, tmp_path: Path) -> None:
        """A mesh whose name matches no anatomy still gets a material."""
        mesh = _labeled_sphere((0.0, 0.0, 0.0), "calibration_phantom")

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=[mesh],
            usd_project_name="scan",
            output_directory=tmp_path,
            appearance="anatomy",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        material = _bound_material_path(
            stage, "/World/scan/calibration_phantom_object1"
        )
        assert material.endswith("OmniSurface_Other")

    def test_unlabeled_meshes_keep_positional_names(self, tmp_path: Path) -> None:
        """Without SegmentationLabelNames, naming stays {project}_{index}."""
        meshes = [
            pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8),
            pv.Sphere(
                radius=1.0, center=(3.0, 0.0, 0.0), theta_resolution=8, phi_resolution=8
            ),
        ]

        workflow = WorkflowConvertVTKToUSD(
            input_meshes=meshes,
            usd_project_name="scan",
            output_directory=tmp_path,
            appearance="anatomy",
            anatomy_type="heart",
            static_merge=True,
        )
        result = workflow.process()

        stage = Usd.Stage.Open(result["usd_file"])
        assert stage.GetPrimAtPath("/World/scan/scan_0_object1").IsValid()
        assert stage.GetPrimAtPath("/World/scan/scan_1_object1").IsValid()
