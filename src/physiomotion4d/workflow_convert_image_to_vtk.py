"""Workflow for segmenting a CT image and converting anatomy groups to VTK surfaces and meshes.

The workflow segments a 3D CT image using a chosen backend, then extracts one VTP
(surface) and one VTU (voxel mesh) per non-empty anatomy group.  Each output object
carries anatomy metadata and solid color from :class:`USDAnatomyTools` as field and
cell data so that downstream tools (PyVista, Paraview, USD pipeline) can use them
directly.

Typical usage::

    import itk
    from physiomotion4d import WorkflowConvertImageToVTK

    ct = itk.imread('chest_ct.nii.gz')
    workflow = WorkflowConvertImageToVTK(segmentation_method='ChestTotalSegmentator')
    result = workflow.run_workflow(ct, contrast_enhanced_study=True)

    # Combined single-file output (default)
    WorkflowConvertImageToVTK.save_combined_surface(result['surfaces'], './out', prefix='patient')
    WorkflowConvertImageToVTK.save_combined_mesh(result['meshes'], './out', prefix='patient')

    # Per-group split output
    WorkflowConvertImageToVTK.save_surfaces(result['surfaces'], './out', prefix='patient')
    WorkflowConvertImageToVTK.save_meshes(result['meshes'], './out', prefix='patient')
"""

import logging
import os
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.segment_anatomy_base import SegmentAnatomyBase
from physiomotion4d.usd_anatomy_tools import USDAnatomyTools

#: Ordered tuple of anatomy group names matching :meth:`SegmentAnatomyBase.segment` keys.
ANATOMY_GROUPS: tuple[str, ...] = (
    "heart",
    "lung",
    "major_vessels",
    "bone",
    "soft_tissue",
    "other",
    "contrast",
)

#: Supported segmentation backend identifiers.
SEGMENTATION_METHODS: tuple[str, ...] = (
    "ChestTotalSegmentator",
    "HeartSimpleware",
)


class WorkflowConvertImageToVTK(PhysioMotion4DBase):
    """Segment a CT image and produce per-anatomy-group VTK surfaces and meshes.

    **Segmentation backends**

    - ``'ChestTotalSegmentator'`` — :class:`SegmentChestTotalSegmentator`
      (CPU-capable, default).
    - ``'HeartSimpleware'`` — :class:`SegmentHeartSimpleware` (cardiac only;
      requires a Simpleware Medical installation).

    **Output anatomy groups**

    ``heart``, ``lung``, ``major_vessels``, ``bone``, ``soft_tissue``, ``other``,
    ``contrast``.  Groups that are empty after segmentation are silently skipped.

    **VTK object annotation**

    Each :class:`pyvista.PolyData` surface and :class:`pyvista.UnstructuredGrid` mesh
    returned by :meth:`run_workflow` carries:

    - ``field_data['AnatomyGroup']`` — anatomy group name, e.g. ``'heart'``.
    - ``field_data['SegmentationLabelNames']`` — individual structure names within the
      group (e.g. ``['left_ventricle', 'right_ventricle', …]``).
    - ``field_data['SegmentationLabelIds']`` — corresponding integer label IDs.
    - ``field_data['AnatomyColor']`` — RGB float color from :class:`USDAnatomyTools`.
    - ``cell_data['Color']`` — RGBA uint8 array (n_cells × 4) for direct VTK rendering.

    **I/O contract**

    :meth:`run_workflow` performs *no* file I/O.  Use the static helpers
    :meth:`save_surfaces`, :meth:`save_meshes`, :meth:`save_combined_surface`, and
    :meth:`save_combined_mesh` — or the CLI ``physiomotion4d-convert-image-to-vtk`` — to
    write results to disk.
    """

    #: Valid anatomy group names.
    ANATOMY_GROUPS: tuple[str, ...] = ANATOMY_GROUPS
    #: Valid segmentation method identifiers.
    SEGMENTATION_METHODS: tuple[str, ...] = SEGMENTATION_METHODS

    def __init__(
        self,
        segmentation_method: str = "ChestTotalSegmentator",
        log_level: int | str = logging.INFO,
    ) -> None:
        """Initialize the workflow.

        Args:
            segmentation_method: Segmentation backend to use.  One of
                ``'ChestTotalSegmentator'`` (default) or ``'HeartSimpleware'``.
            log_level: Logging level.  Default: ``logging.INFO``.

        Raises:
            ValueError: If *segmentation_method* is not one of
                :attr:`SEGMENTATION_METHODS`.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        if segmentation_method not in self.SEGMENTATION_METHODS:
            raise ValueError(
                f"Unknown segmentation_method '{segmentation_method}'. "
                f"Choose from: {', '.join(self.SEGMENTATION_METHODS)}"
            )

        self.segmentation_method_name: str = segmentation_method
        self._segmenter: Optional[SegmentAnatomyBase] = None
        self._contour_tools: ContourTools = ContourTools(log_level=log_level)

        # Build anatomy-group → RGB color from USDAnatomyTools.
        # USDAnatomyTools sets up its color dicts entirely in __init__ without
        # accessing the stage, so stage=None is safe for this lookup-only use.
        _anatomy_tools = USDAnatomyTools(stage=None, log_level=log_level)
        supported_types = set(_anatomy_tools.get_anatomy_types())
        self._anatomy_color_map: dict[str, tuple[float, float, float]] = {
            group: _anatomy_tools.get_anatomy_diffuse_color(group)
            for group in ANATOMY_GROUPS
            if group in supported_types
        }

    # ─────────────────────────── Internal helpers ──────────────────────────

    def _create_segmenter(self) -> SegmentAnatomyBase:
        """Instantiate the chosen segmentation backend (lazy import)."""
        if self.segmentation_method_name == "ChestTotalSegmentator":
            from physiomotion4d.segment_chest_total_segmentator import (
                SegmentChestTotalSegmentator,
            )

            return SegmentChestTotalSegmentator(log_level=self.log_level)
        if self.segmentation_method_name == "HeartSimpleware":
            from physiomotion4d.segment_heart_simpleware import SegmentHeartSimpleware

            segmenter = SegmentHeartSimpleware(log_level=self.log_level)
            segmenter.set_trim_branches(True)
            return segmenter
        raise ValueError(
            f"Unknown segmentation method: {self.segmentation_method_name}"
        )

    def _get_label_info_for_group(self, group: str) -> tuple[list[str], list[int]]:
        """Return ``(label_names, label_ids)`` for *group* from the active segmenter.

        Reads the segmenter's :class:`AnatomyTaxonomy`. Returns empty lists if
        the group is not present (e.g. HeartSimpleware does not register
        lung/bone).
        """
        assert self._segmenter is not None, (
            "_create_segmenter() must be called before _get_label_info_for_group()"
        )
        mask_ids = self._segmenter.taxonomy.labels_in_group(group)
        return list(mask_ids.values()), list(mask_ids.keys())

    @staticmethod
    def _annotate(
        vtk_obj: pv.DataSet,
        group: str,
        label_names: list[str],
        label_ids: list[int],
        color_rgb: tuple[float, float, float],
    ) -> None:
        """Attach anatomy metadata and solid RGBA color to a VTK object **in-place**.

        Sets:

        - ``field_data['AnatomyGroup']`` — group name.
        - ``field_data['SegmentationLabelNames']`` — individual label names.
        - ``field_data['SegmentationLabelIds']`` — integer label IDs (int32).
        - ``field_data['AnatomyColor']`` — RGB float32 color.
        - ``cell_data['Color']`` — RGBA uint8 solid color (n_cells × 4).
        """
        vtk_obj.field_data["AnatomyGroup"] = np.array([group])
        vtk_obj.field_data["SegmentationLabelNames"] = np.array(
            label_names if label_names else [group]
        )
        vtk_obj.field_data["SegmentationLabelIds"] = np.array(label_ids, dtype=np.int32)
        vtk_obj.field_data["AnatomyColor"] = np.array(color_rgb, dtype=np.float32)

        r, g, b = color_rgb
        rgba = np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        if vtk_obj.n_cells > 0:
            vtk_obj.cell_data["Color"] = np.tile(rgba, (vtk_obj.n_cells, 1))

    def _extract_surface(self, mask_image: Any) -> Optional[pv.PolyData]:
        """Extract a smoothed triangulated surface (VTP) from a binary mask image.

        Delegates to :meth:`ContourTools.extract_contours`.

        Returns:
            Smoothed :class:`pyvista.PolyData`, or ``None`` if the mask is empty.
        """
        arr = itk.GetArrayFromImage(mask_image)
        if int(arr.sum()) == 0:
            return None
        return self._contour_tools.extract_contours(mask_image)

    def _extract_mesh(self, mask_image: Any) -> Optional[pv.UnstructuredGrid]:
        """Extract a voxel-based volumetric mesh (VTU) from a binary mask image.

        Wraps the ITK image as a VTK ImageData and thresholds at 0.5 to obtain
        hexahedral voxel cells for non-zero voxels.

        Returns:
            :class:`pyvista.UnstructuredGrid` of labeled voxels, or ``None`` if empty.
        """
        arr = itk.GetArrayFromImage(mask_image)
        if int(arr.sum()) == 0:
            return None

        vtk_image = pv.wrap(itk.vtk_image_from_image(mask_image))
        if not isinstance(vtk_image, pv.ImageData):
            self.log_warning(
                "Expected pv.ImageData from vtk_image_from_image, got %s — skipping mesh",
                type(vtk_image).__name__,
            )
            return None

        thresholded = vtk_image.threshold(0.5)
        if isinstance(thresholded, pv.UnstructuredGrid):
            return thresholded
        return cast(pv.UnstructuredGrid, thresholded.cast_to_unstructured_grid())

    # ─────────────────────────── Main workflow ─────────────────────────────

    def run_workflow(
        self,
        input_image: Any,
        contrast_enhanced_study: bool = False,
        anatomy_groups: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Segment the CT image and extract per-anatomy-group VTK objects.

        Args:
            input_image: Input 3D CT image (``itk.Image``).
            contrast_enhanced_study: If ``True``, an additional connected-component
                pass identifies contrast-enhanced blood.  Default: ``False``.
            anatomy_groups: Subset of anatomy groups to process.  ``None`` (default)
                processes all non-empty groups.  Valid names: ``'heart'``,
                ``'lung'``, ``'major_vessels'``, ``'bone'``, ``'soft_tissue'``,
                ``'other'``, ``'contrast'``.

        Returns:
            ``dict`` with the following keys:

            - ``'surfaces'`` — ``dict[str, pv.PolyData]``: smoothed surface per group.
            - ``'meshes'`` — ``dict[str, pv.UnstructuredGrid]``: voxel mesh per group.
            - ``'labelmap'`` — ``itk.Image``: detailed per-structure segmentation
              labelmap from the segmenter.
            - ``'segmentation_masks'`` — ``dict[str, itk.Image]``: per-group binary
              masks used to produce the VTK objects.

        Raises:
            ValueError: If any name in *anatomy_groups* is invalid.
        """
        self.log_section("STARTING IMAGE TO VTK WORKFLOW")

        # Validate requested groups
        if anatomy_groups is not None:
            invalid = [g for g in anatomy_groups if g not in self.ANATOMY_GROUPS]
            if invalid:
                raise ValueError(
                    f"Unknown anatomy groups: {invalid}. "
                    f"Valid: {list(self.ANATOMY_GROUPS)}"
                )
            groups_to_process: list[str] = list(anatomy_groups)
        else:
            groups_to_process = list(self.ANATOMY_GROUPS)

        # Create and run segmenter
        self.log_info("Creating segmenter: %s", self.segmentation_method_name)
        self._segmenter = self._create_segmenter()

        self.log_section("Running segmentation")
        seg_result: dict[str, Any] = self._segmenter.segment(
            input_image, contrast_enhanced_study=contrast_enhanced_study
        )

        # Extract VTK objects per anatomy group
        self.log_section("Extracting VTK objects")
        surfaces: dict[str, pv.PolyData] = {}
        meshes: dict[str, pv.UnstructuredGrid] = {}
        seg_masks: dict[str, Any] = {}

        for group in groups_to_process:
            if group not in seg_result:
                self.log_warning(
                    "Group %s absent from segmentation result — skipping", group
                )
                continue

            mask_image = seg_result[group]
            if int(itk.GetArrayFromImage(mask_image).sum()) == 0:
                self.log_info("Group %s is empty — skipping", group)
                continue

            self.log_info("Processing anatomy group: %s", group)
            seg_masks[group] = mask_image

            label_names, label_ids = self._get_label_info_for_group(group)
            color = self._anatomy_color_map.get(group, (0.7, 0.7, 0.7))

            self.log_info("  Extracting surface for: %s", group)
            surface = self._extract_surface(mask_image)
            if surface is not None:
                self._annotate(surface, group, label_names, label_ids, color)
                surfaces[group] = surface

            self.log_info("  Extracting voxel mesh for: %s", group)
            mesh = self._extract_mesh(mask_image)
            if mesh is not None:
                self._annotate(mesh, group, label_names, label_ids, color)
                meshes[group] = mesh

        self.log_section("IMAGE TO VTK WORKFLOW COMPLETE")
        self.log_info("Surfaces extracted: %d", len(surfaces))
        self.log_info("Meshes extracted:   %d", len(meshes))

        return {
            "surfaces": surfaces,
            "meshes": meshes,
            "labelmap": seg_result["labelmap"],
            "segmentation_masks": seg_masks,
        }

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @staticmethod
    def save_surfaces(
        surfaces: dict[str, pv.PolyData],
        output_dir: str,
        prefix: str = "",
    ) -> dict[str, str]:
        """Save each group surface to its own VTP file.

        Args:
            surfaces: Mapping of anatomy group name → surface (from
                :meth:`run_workflow`).
            output_dir: Directory to write files into (created if absent).
            prefix: Optional filename prefix.  Each file is named
                ``{prefix}_{group}.vtp`` (or ``{group}.vtp`` when *prefix* is empty).

        Returns:
            Mapping of anatomy group name → absolute path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, str] = {}
        for name, surface in surfaces.items():
            stem = f"{prefix}_{name}" if prefix else name
            path = os.path.join(output_dir, f"{stem}.vtp")
            surface.save(path)
            saved[name] = path
        return saved

    @staticmethod
    def save_meshes(
        meshes: dict[str, pv.UnstructuredGrid],
        output_dir: str,
        prefix: str = "",
    ) -> dict[str, str]:
        """Save each group voxel mesh to its own VTU file.

        Args:
            meshes: Mapping of anatomy group name → mesh (from :meth:`run_workflow`).
            output_dir: Directory to write files into (created if absent).
            prefix: Optional filename prefix.  Each file is named
                ``{prefix}_{group}.vtu`` (or ``{group}.vtu`` when *prefix* is empty).

        Returns:
            Mapping of anatomy group name → absolute path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, str] = {}
        for name, mesh in meshes.items():
            stem = f"{prefix}_{name}" if prefix else name
            path = os.path.join(output_dir, f"{stem}.vtu")
            mesh.save(path)
            saved[name] = path
        return saved

    @staticmethod
    def save_combined_surface(
        surfaces: dict[str, pv.PolyData],
        output_dir: str,
        prefix: str = "",
    ) -> str:
        """Merge all group surfaces into a single VTP file.

        The merged mesh retains per-cell ``Color`` (RGBA uint8) from each group's
        annotation, enabling colour-by-anatomy rendering in Paraview, PyVista, etc.
        Per-object ``field_data`` is not preserved in the merged file.

        Args:
            surfaces: Mapping of anatomy group name → surface.
            output_dir: Directory to write the file into (created if absent).
            prefix: Optional filename prefix.  Output is ``{prefix}_surfaces.vtp``
                (or ``surfaces.vtp`` when *prefix* is empty).

        Returns:
            Absolute path to the saved VTP file.

        Raises:
            ValueError: If *surfaces* is empty.
        """
        if not surfaces:
            raise ValueError("No surfaces to save.")
        os.makedirs(output_dir, exist_ok=True)
        stem = f"{prefix}_surfaces" if prefix else "surfaces"
        output_file = os.path.join(output_dir, f"{stem}.vtp")
        merged = cast(
            pv.PolyData, pv.merge(list(surfaces.values()), merge_points=False)
        )
        merged.save(output_file)
        return output_file

    @staticmethod
    def save_combined_mesh(
        meshes: dict[str, pv.UnstructuredGrid],
        output_dir: str,
        prefix: str = "",
    ) -> str:
        """Merge all group meshes into a single VTU file.

        The merged mesh retains per-cell ``Color`` (RGBA uint8) from each group's
        annotation.  Per-object ``field_data`` is not preserved in the merged file.

        Args:
            meshes: Mapping of anatomy group name → voxel mesh.
            output_dir: Directory to write the file into (created if absent).
            prefix: Optional filename prefix.  Output is ``{prefix}_meshes.vtu``
                (or ``meshes.vtu`` when *prefix* is empty).

        Returns:
            Absolute path to the saved VTU file.

        Raises:
            ValueError: If *meshes* is empty.
        """
        if not meshes:
            raise ValueError("No meshes to save.")
        os.makedirs(output_dir, exist_ok=True)
        stem = f"{prefix}_meshes" if prefix else "meshes"
        output_file = os.path.join(output_dir, f"{stem}.vtu")
        merged = cast(
            pv.UnstructuredGrid, pv.merge(list(meshes.values()), merge_points=False)
        )
        merged.save(output_file)
        return output_file
