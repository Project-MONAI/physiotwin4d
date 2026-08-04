"""Movement interpretation of PhysicsNeMo mesh-stage predictions.

:class:`WorkflowInferMovement` wraps a
:class:`physiotwin4d.WorkflowInferPhysicsNeMo` whose targets are per-point
displacements from the subject's reference mesh, and turns those raw predictions
into geometry: deformed meshes (``reference + displacement``), error statistics
in millimetres, and rasterized deformation / surface-normal fields.

The generic workflow stays target-agnostic; everything that assumes "the target
is a 3-vector displacement in mm" lives here. Composition, not inheritance, so
one decoder serves both the MeshGraphNet and the MLP inference methods.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .physiotwin4d_base import PhysioTwin4DBase
from .workflow_infer_physicsnemo import WorkflowInferPhysicsNeMo


class WorkflowInferMovement(PhysioTwin4DBase):
    """Reconstruct geometry from displacement predictions.

    The displacements are added to the caller's reference mesh when one is
    available — the manifest's ``reference_mesh``, or the ``reference_mesh``
    argument of the single-subject methods — which keeps the result in that
    mesh's world frame. With no reference mesh they are added to the mesh
    reconstructed from the PCA coefficients alone, which stays in the model's
    PCA frame.

    Args:
        inference_workflow: A loaded :class:`WorkflowInferPhysicsNeMo` whose
            model predicts three-component displacements.
        log_level: Logging level. Default: ``logging.INFO``.

    Raises:
        ValueError: If the wrapped model does not predict exactly three
            components, which cannot be a displacement.
    """

    def __init__(
        self,
        inference_workflow: WorkflowInferPhysicsNeMo,
        log_level: int | str = logging.INFO,
    ) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        if inference_workflow.n_target != 3:
            raise ValueError(
                "WorkflowInferMovement needs a 3-component target, but the "
                f"model predicts {inference_workflow.n_target} components."
            )
        self.inference_workflow = inference_workflow
        self.model_directory = inference_workflow.model_directory

    def _reference_points(
        self, pca_coeffs: np.ndarray, reference_mesh: Optional[pv.DataSet]
    ) -> np.ndarray:
        """Return the points the displacements are added to.

        Raises:
            ValueError: If the supplied mesh has the wrong point count.
        """
        if reference_mesh is None:
            return self.inference_workflow.reference_points_from_coefficients(
                pca_coeffs
            )
        points = np.asarray(reference_mesh.points, dtype=np.float32)
        n_expected = self.inference_workflow.template_mesh.n_points
        if points.shape[0] != n_expected:
            raise ValueError(
                f"Reference mesh has {points.shape[0]} points, expected "
                f"{n_expected} (template topology)."
            )
        return points

    # ─────────────────────────── Public API ────────────────────────────────
    def process(
        self,
        subject_manifest: Path,
        stages: Optional[list[float]] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict a subject's deformed meshes from a manifest.

        When ``stages`` is ``None`` every phase in the manifest is predicted and
        compared against its ground truth (``reference + stored displacement``);
        when ``stages`` is given those arbitrary stages are predicted without
        comparison. The displacements are added to the manifest's
        ``reference_mesh`` points.

        Args:
            subject_manifest: Path to the subject manifest JSON.
            stages: Optional list of stages to predict.
            output_directory: Output directory; defaults to
                ``<model_directory>/<subject_id>``.

        Returns:
            Dict with ``subject_id``, ``predicted_surfaces`` (paths), and, in the
            phase mode, ``statistics``, ``statistics_file`` and ``rmse_surface``.
        """
        workflow = self.inference_workflow
        manifest = pnt.parse_manifest(subject_manifest)
        pca_coeffs = pnt.load_pca_coefficients(manifest.pca_coefficients)
        ref_mesh = cast(pv.DataSet, pv.read(str(manifest.reference_mesh)))
        ref_points = self._reference_points(pca_coeffs, ref_mesh)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / manifest.subject_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = manifest.subject_id
        self.log_section("INFER MOVEMENT [%s]", sid)

        suffix = ".vtp" if isinstance(ref_mesh, pv.PolyData) else ".vtu"
        requested = stages if stages is not None else [p.stage for p in manifest.phases]
        surfaces: list[Path] = []
        stats: list[dict] = []
        sq_err_sum = np.zeros(ref_points.shape[0], dtype=np.float64)

        for index, stage in enumerate(requested):
            pred_points = ref_points + workflow.predict(pca_coeffs, stage)
            pred_mesh = ref_mesh.copy(deep=True)
            pred_mesh.points = pred_points
            path = out_dir / f"{sid}_s{int(stage * 100):03d}_pred{suffix}"
            pred_mesh.save(str(path))
            surfaces.append(path)

            if stages is not None:
                self.log_info("stage %.3f -> %s", stage, path.name)
                continue

            actual = ref_points + pnt.load_target_array(
                manifest.phases[index].mesh, manifest.target_array
            )
            euclidean = np.linalg.norm(pred_points - actual, axis=1)
            sq_err_sum += euclidean.astype(np.float64) ** 2
            stats.append(self._error_row(sid, stage, pred_points, actual))
            self.log_info(
                "stage %.3f: mean=%.3f mm  max=%.3f mm",
                stage,
                stats[-1]["mean_error_mm"],
                stats[-1]["max_error_mm"],
            )

        result: dict[str, Any] = {"subject_id": sid, "predicted_surfaces": surfaces}
        if stages is None:
            point_rmse = np.sqrt(sq_err_sum / len(requested)).astype(np.float32)
            rmse_mesh = ref_mesh.copy(deep=True)
            rmse_mesh.point_data["RMSE_mm"] = point_rmse
            rmse_file = out_dir / f"{sid}_rmse{suffix}"
            rmse_mesh.save(str(rmse_file))

            stats_file = out_dir / "statistics_per_phase.csv"
            with stats_file.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
                writer.writeheader()
                writer.writerows(stats)
            result["rmse_surface"] = rmse_file
            result["statistics"] = stats
            result["statistics_file"] = stats_file
        return result

    def predict_single(
        self,
        shape_parameters: Path,
        stage: float,
        reference_mesh: Optional[Path] = None,
        ground_truth: Optional[Path] = None,
        output_directory: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Predict one subject at one stage without a manifest.

        Without ``reference_mesh`` the subject reference is reconstructed from
        the PCA shape parameters (``P = mean + Σ b_i·std_i·eigenvector_i``) in
        the SSM/PCA frame, so the prediction is self-consistent with no
        reference mesh file at all; with one, the prediction stays in that
        mesh's world frame.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stage: Target stage to predict.
            reference_mesh: The subject's reference mesh; omit to displace the
                PCA reconstruction instead.
            ground_truth: Optional mesh whose points are the true stage
                positions, for error reporting.
            output_directory: Output directory; defaults to
                ``<model_directory>/single_prediction``.

        Returns:
            Dict with ``predicted_surface`` (path), ``predicted_points``, and,
            when ``ground_truth`` is supplied, ``statistics``.
        """
        workflow = self.inference_workflow
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        ref_mesh = (
            cast(pv.DataSet, pv.read(str(reference_mesh)))
            if reference_mesh is not None
            else None
        )
        ref_points = self._reference_points(coeffs, ref_mesh)
        pred_points = ref_points + workflow.predict(coeffs, stage)

        out_dir = (
            Path(output_directory)
            if output_directory is not None
            else self.model_directory / "single_prediction"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        template = workflow.template_mesh
        pred_mesh = template.copy(deep=True)
        pred_mesh.points = pred_points
        suffix = ".vtp" if isinstance(template, pv.PolyData) else ".vtu"
        stem = Path(shape_parameters).stem
        path = out_dir / f"{stem}_pred_s{int(stage * 100):03d}{suffix}"
        pred_mesh.save(str(path))
        self.log_info("single prediction stage %.3f -> %s", stage, path.name)

        result: dict[str, Any] = {
            "predicted_surface": path,
            "predicted_points": pred_points,
        }
        if ground_truth is not None:
            actual = np.asarray(pv.read(str(ground_truth)).points, dtype=np.float32)
            result["statistics"] = self._error_row(stem, stage, pred_points, actual)
            self.log_info(
                "ground-truth error: mean=%.3f mm  max=%.3f mm",
                result["statistics"]["mean_error_mm"],
                result["statistics"]["max_error_mm"],
            )
        return result

    def create_deformation_field(
        self,
        shape_parameters: Path,
        stage: float,
        reference_image: itk.Image,
        output_directory: Optional[Path] = None,
        reference_mesh: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Rasterize the inferred deformation onto a reference image grid.

        Each mesh vertex is binned by its **reference (undeformed) position**
        into ``reference_image``'s voxel grid. Each voxel of the deformation
        field holds the mean network displacement ``(dx, dy, dz)`` of the
        vertices that fall in it; each voxel of the normal image holds the mean
        (renormalized) reference-surface normal of those vertices. Empty voxels
        are zero.

        The binning positions come from ``reference_mesh``, so a patient scan
        whose statistical-model fit applied a pose transform not captured by the
        shape coefficients is binned where it actually aligns with
        ``reference_image``. Omit it to bin at the PCA reconstruction instead,
        in the model's own frame. The network displacements themselves depend
        only on the coefficients and stage, not on the binning positions.

        Args:
            shape_parameters: JSON file with the subject PCA coefficient vector.
            stage: Target stage for the deformation.
            reference_image: The frame's image; defines the output grid geometry
                (size, spacing, origin, direction).
            output_directory: If given, the two images are written there as
                compressed ``.mha`` files.
            reference_mesh: Mesh whose points supply the binning positions and
                normals; omit to use the PCA reconstruction. Must share the
                template topology (same point count and ordering).

        Returns:
            Dict with ``deformation_field`` and ``normal_image`` (ITK vector
            images), ``deformed_surface`` (the stage mesh as ``pv.DataSet``)
            and, when written, their paths.
        """
        workflow = self.inference_workflow
        template = workflow.template_mesh
        coeffs = pnt.load_pca_coefficients(shape_parameters)
        patient_mesh = (
            cast(pv.DataSet, pv.read(str(reference_mesh)))
            if reference_mesh is not None
            else None
        )
        ref_points = self._reference_points(coeffs, patient_mesh)
        disps = workflow.predict(coeffs, stage)

        # Reference (undeformed) surface normals. Extraction drops the interior
        # points of a volumetric template, so the normals come back on a subset
        # of the points in a different order; ``vtkOriginalPointIds`` scatters
        # them back into ``ref_points`` order. Interior points keep a zero
        # normal, which contributes nothing to a voxel's mean.
        ref_mesh = template.copy(deep=True)
        ref_mesh.points = ref_points
        surface = ref_mesh.extract_surface(
            pass_pointid=True, algorithm="dataset_surface"
        ).compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True
        )
        surface_normals = np.asarray(surface.point_data["Normals"], dtype=np.float64)
        original_ids = np.asarray(
            surface.point_data["vtkOriginalPointIds"], dtype=np.intp
        )
        normals = np.zeros((ref_points.shape[0], 3), dtype=np.float64)
        normals[original_ids] = surface_normals

        size = itk.size(reference_image)  # x, y, z
        sx, sy, sz = int(size[0]), int(size[1]), int(size[2])
        disp_sum = np.zeros((sz, sy, sx, 3), dtype=np.float64)
        normal_sum = np.zeros((sz, sy, sx, 3), dtype=np.float64)
        count = np.zeros((sz, sy, sx), dtype=np.float64)

        for i in range(ref_points.shape[0]):
            point = [float(c) for c in ref_points[i]]
            idx = reference_image.TransformPhysicalPointToIndex(point)
            ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
            if 0 <= ix < sx and 0 <= iy < sy and 0 <= iz < sz:
                disp_sum[iz, iy, ix] += disps[i]
                normal_sum[iz, iy, ix] += normals[i]
                count[iz, iy, ix] += 1.0

        occupied = count > 0
        disp_field = np.zeros_like(disp_sum, dtype=np.float32)
        normal_field = np.zeros_like(normal_sum, dtype=np.float32)
        disp_field[occupied] = (disp_sum[occupied] / count[occupied, None]).astype(
            np.float32
        )
        mean_normal = normal_sum[occupied] / count[occupied, None]
        norm = np.linalg.norm(mean_normal, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        normal_field[occupied] = (mean_normal / norm).astype(np.float32)

        deformation_image = self._vector_image_like(disp_field, reference_image)
        normal_image = self._vector_image_like(normal_field, reference_image)
        self.log_info(
            "Deformation field: %d/%d voxels populated by %d vertices",
            int(occupied.sum()),
            sx * sy * sz,
            ref_points.shape[0],
        )

        # Deformed (stage) mesh: reference positions displaced by the network,
        # keeping the template topology.
        deformed_surface = template.copy(deep=True)
        deformed_surface.points = (ref_points + disps).astype(np.float32)

        result: dict[str, Any] = {
            "deformation_field": deformation_image,
            "normal_image": normal_image,
            "deformed_surface": deformed_surface,
        }
        if output_directory is not None:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".vtp" if isinstance(template, pv.PolyData) else ".vtu"
            field_path = out_dir / "deformation_field.mha"
            normal_path = out_dir / "surface_normal_field.mha"
            surface_path = out_dir / f"deformed_surface{suffix}"
            itk.imwrite(deformation_image, str(field_path), compression=True)
            itk.imwrite(normal_image, str(normal_path), compression=True)
            deformed_surface.save(str(surface_path))
            result["deformation_field_file"] = field_path
            result["normal_image_file"] = normal_path
            result["deformed_surface_file"] = surface_path
        return result

    @staticmethod
    def _vector_image_like(array: np.ndarray, reference_image: itk.Image) -> itk.Image:
        """Wrap a ``(z, y, x, 3)`` array as an ITK vector image on ``reference``'s grid."""
        image = itk.image_from_array(np.ascontiguousarray(array), is_vector=True)
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())
        return image

    @staticmethod
    def _error_row(
        subject_id: str, stage: float, pred: np.ndarray, actual: np.ndarray
    ) -> dict:
        """Per-phase error statistics between predicted and actual points."""
        errors = pred - actual
        euclidean = np.linalg.norm(errors, axis=1)
        return {
            "subject_id": subject_id,
            "stage": stage,
            "n_points": int(len(euclidean)),
            "mean_error_mm": float(euclidean.mean()),
            "median_error_mm": float(np.median(euclidean)),
            "max_error_mm": float(euclidean.max()),
            "rms_error_mm": float(np.sqrt(np.mean(euclidean**2))),
            "std_error_mm": float(euclidean.std()),
            "mean_abs_error_x_mm": float(np.abs(errors[:, 0]).mean()),
            "mean_abs_error_y_mm": float(np.abs(errors[:, 1]).mean()),
            "mean_abs_error_z_mm": float(np.abs(errors[:, 2]).mean()),
        }
