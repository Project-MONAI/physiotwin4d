"""
Tutorial 11: Combined Heart and Lung Motion

Purpose
-------
Build a single 4D animation of the DirLab ``Case1Pack`` thorax that combines
cardiac motion (a trained PhysicsNeMo MeshGraphNet applied to the patient heart)
with respiratory motion (the Tutorial 1 lung registration transforms).

The script runs in two stages:

1. **Cardiac motion.** Assemble the Tutorial 9 MeshGraphNet run directory, apply
   the model to the Case1Pack heart fit (Tutorial 5) at each cardiac stage, and
   rasterize each stage's displacement onto the Case1Pack grid as a deformation
   field (``WorkflowInferPhysicsNeMoMGN.create_deformation_field``). A
   beating-heart 4D USD of the heart surfaces is written for reference.

2. **Combined motion.** Build a per-cell-labeled thorax surface by contouring the
   Tutorial 2 patient labelmap, smooth each cardiac deformation field and apply it
   at the reference frame, then carry the cardiac-deformed surface through the
   respiratory cycle with the Tutorial 1 forward transforms. Breathing and the
   heartbeat run as independent rhythms (see Composition order). The 100 combined
   frames are written as VTP files and assembled into a single animated 4D USD
   that is split by anatomy label and painted with per-organ OmniSurface
   materials.

Anatomy materials
-----------------
The per-cell ``boundary_labels`` produced by contouring the labelmap propagate
unchanged through every warp (each frame is a deep copy with only its points
moved; label-preserving decimation is applied if enabled). ``ConvertVTKToUSD``,
given ``segmenter.taxonomy.all_labels()`` and the segmenter, splits each frame
into per-organ prims, and ``USDAnatomyTools.enhance_meshes`` then binds the
matching OmniSurface material (diffuse color, subsurface scattering, etc.).

Composition order
-----------------
Cardiac deformation is applied first, at the reference frame where the cardiac
fields are defined, giving one warped surface per (breath phase, cardiac stage).
Each rendered frame then bilinearly interpolates that precomputed grid: the
respiratory axis advances with the breath phase, while the cardiac axis advances
**independently and continuously** at ``cardiac_cycles_per_phase`` beats per
phase. A value below 1.0 therefore lets a single heartbeat carry across a phase
boundary (0.75 = each breath phase covers three-quarters of a beat). Both axes
wrap, so the sequence loops.

Inputs (hard-coded near the top; edit for your layout)
------
- MGN model run directory (Tutorial 9), epoch-300 checkpoint.
- PCA template volume (``pca-vol-kcl/pca_mean.vtu``).
- Case1Pack reference image (``data/DirLab-4DCT/Case1Pack_T70.mha``).
- Case1Pack heart fit (Tutorial 5): registered coefficients + volume mesh.
- Respiratory forward transforms (Tutorial 1 lung output).
- Segmented patient labelmap (Tutorial 2 lung output).

Outputs (under ``output/tutorial_11_heart_and_lung``)
-------
- ``deformation_field_s<sss>.mha`` / ``surface_normal_field_s<sss>.mha`` -
  per-stage cardiac fields on the Case1Pack grid.
- ``deformed_heart_surface_s<sss>.vtp`` + ``beating_heart.usd`` - the beating
  heart alone.
- ``combined_frame_<iii>.vtp`` (``000..099``) + ``heart_and_lung_motion.usd`` -
  the combined respiratory + cardiac 4D motion, painted with anatomy materials.

Prerequisites
-------------
Run Tutorials 1 (lung), 2 (lung), 5 (Case1Pack with the pca-vol-kcl model) and 9
(MGN) first. Requires the ``[physicsnemo]`` extra.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import cast

import itk
import numpy as np
import pyvista as pv

from physiotwin4d import (
    ContourTools,
    ConvertVTKToUSD,
    ImageTools,
    SegmentChestTotalSegmentator,
    TransformTools,
    USDAnatomyTools,
    WorkflowConvertVTKToUSD,
    WorkflowInferPhysicsNeMoMGN,
)
from physiotwin4d import physicsnemo_tools as pnt


def _ensure_mgn_inference_assets(
    model_dir: Path, epoch: int, pca_mean_volume: Path
) -> None:
    """Complete an interrupted MGN run directory so it can be loaded for inference.

    ``WorkflowInferPhysicsNeMoMGN`` expects a finalized run directory
    (``mgn_stage_model.pt``, ``pca_mean_surface.vtp`` and the shared graph
    tensors). A run that only holds epoch checkpoints is completed here by
    regenerating the missing assets deterministically:

    - ``mgn_stage_model.pt`` from the self-describing epoch checkpoint (it
      carries the normalization stats and architecture the loader reads);
    - ``pca_mean_surface.vtp`` and the shared MGN graph tensors from the PCA
      template volume, using the same steps the trainer used.

    All writes are idempotent (skipped when the target already exists).
    """
    import torch

    epoch_ckpt = model_dir / f"mgn_stage_model_epoch_{epoch:05d}.pt"
    if not epoch_ckpt.exists():
        raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_ckpt}")

    final_ckpt = model_dir / "mgn_stage_model.pt"
    if not final_ckpt.exists():
        shutil.copy2(epoch_ckpt, final_ckpt)

    surface_file = model_dir / "pca_mean_surface.vtp"
    if not surface_file.exists():
        volume = pv.read(str(pca_mean_volume))
        mean_surface = volume.extract_surface(algorithm="dataset_surface")
        mean_surface.save(str(surface_file))
    mean_surface = cast(pv.PolyData, pv.read(str(surface_file)))

    edge_index_file = model_dir / "shared_edge_index.pt"
    edge_feats_file = model_dir / "shared_edge_features.pt"
    if not edge_index_file.exists() or not edge_feats_file.exists():
        edge_index = pnt.mesh_to_edge_index(mean_surface)
        coords = np.asarray(mean_surface.points, dtype=np.float32)
        edge_feats = pnt.compute_edge_features(coords, edge_index)
        torch.save(edge_index, str(edge_index_file))
        torch.save(edge_feats, str(edge_feats_file))


def _smoothed_cardiac_transform(
    field: itk.Image, sigma_mm: float, transform_tools: TransformTools
) -> itk.DisplacementFieldTransform:
    """Wrap a cardiac deformation field as a Gaussian-smoothed field transform.

    The float vector ``field`` is converted to a double-precision vector field,
    wrapped as a ``DisplacementFieldTransform`` and Gaussian-smoothed by
    ``sigma_mm`` (physical millimeters). Smoothing spreads the thin surface-shell
    field into a continuous deformation (and attenuates its peak magnitude).
    """
    field_double = ImageTools().convert_array_to_image_of_vectors(
        itk.array_from_image(field), reference_image=field, ptype=itk.D
    )
    field_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    field_transform.SetDisplacementField(field_double)
    return transform_tools.smooth_transform(
        field_transform, sigma=sigma_mm, reference_image=field
    )


def _condition_surface(
    surface: pv.PolyData,
    decimation_reduction: float,
    smoothing_iterations: int,
) -> pv.PolyData:
    """Optionally decimate then smooth the model surface (no-op when disabled).

    Applied once to the labeled patient surface so every warped frame inherits
    the same resolution and smoothing. Decimation uses ``decimate_pro`` on a
    triangulated copy; because ``decimate_pro`` discards cell data, the per-cell
    ``boundary_labels`` (needed for anatomy splitting downstream) are transferred
    back onto the decimated cells from their nearest original cell so anatomy
    materials still apply. Smoothing uses non-shrinking Taubin smoothing, which
    only moves points and therefore preserves cells and their labels.
    """
    conditioned = surface
    if decimation_reduction > 0.0:
        original = conditioned
        conditioned = conditioned.triangulate().decimate_pro(decimation_reduction)
        if "boundary_labels" in original.cell_data:
            nearest = original.find_closest_cell(conditioned.cell_centers().points)
            conditioned.cell_data["boundary_labels"] = np.asarray(
                original.cell_data["boundary_labels"]
            )[nearest]
    if smoothing_iterations > 0:
        conditioned = conditioned.smooth_taubin(n_iter=smoothing_iterations)
    return conditioned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("tutorial_11_heart_and_lung_motion")

    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    # ---- Hard-coded inputs (edit for your layout) --------------------------
    # Cardiac MGN model (Tutorial 9): epoch-300 checkpoint run directory and the
    # PCA template volume the model was trained on.
    mgn_model_dir = tutorials_dir / "output" / "tutorial_09_byod_mgn_3"
    mgn_epoch = 1500
    pca_mean_volume = Path("D:/PhysioTwin4D/kcl-heart-pca/pca-vol-kcl/pca_mean.vtu")

    # Case1Pack reference image (defines the cardiac deformation-field grid).
    reference_image_file = repo_root / "data" / "DirLab-4DCT" / "Case1Pack_T70.mha"

    # 15-mode pca-vol-kcl fit of the Case1Pack heart (Tutorial 5, re-run).
    tutorial_05_dir = tutorials_dir / "output" / "tutorial_05_heart_to_lung"
    coefficients_file = (
        tutorial_05_dir / "tutorial_05_heart_to_lung_registered_coefficients.json"
    )
    registered_mesh_file = (
        tutorial_05_dir / "tutorial_05_heart_to_lung_template_mesh_registered.vtu"
    )

    # Respiratory forward transforms (Tutorial 1 lung) and the segmented patient
    # labelmap (Tutorial 2 lung). The labelmap yields a per-cell-labeled surface
    # so the final USD can be split by anatomy and painted with organ materials.
    respiratory_dir = tutorials_dir / "output" / "tutorial_01_lung"
    patient_labelmap_file = (
        tutorials_dir / "output" / "tutorial_02_lung" / "patient_labelmap.mha"
    )

    output_dir = tutorials_dir / "output" / "tutorial_11_heart_and_lung"

    # ---- Parameters --------------------------------------------------------
    # Cardiac stages sampled over one heartbeat (fraction of the RR interval).
    cardiac_stages = [round(0.1 * k, 2) for k in range(10)]
    # Fraction of a cardiac cycle advanced per respiratory phase. The heart is
    # decoupled from breathing, so a value < 1 means a single beat continues into
    # the next phase (1.0 locks exactly one full beat to each phase).
    cardiac_cycles_per_phase = 0.75
    # Gaussian sigma (mm) used to smooth the sparse cardiac deformation fields.
    smoothing_sigma_mm = 10.0
    # One-time conditioning of the patient surface, reused for every frame.
    surface_decimation_reduction = 0.0  # fraction of triangles to remove
    surface_smoothing_iterations = 0  # Taubin (non-shrinking) iterations
    # USD playback rate; 10 stages/second ~= one heartbeat per second.
    frames_per_second = 10.0

    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        reference_image_file,
        coefficients_file,
        registered_mesh_file,
        pca_mean_volume,
        patient_labelmap_file,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")

    forward_transform_files = sorted(respiratory_dir.glob("slice_*_all_forward.hdf"))
    if not forward_transform_files:
        raise FileNotFoundError(
            f"No respiratory forward transforms found in {respiratory_dir}. "
            "Run tutorial_01_lung_gated_ct_to_usd.py first."
        )

    transform_tools = TransformTools()

    # ========================================================================
    # Stage 1: cardiac motion - apply the MGN model to the Case1Pack heart.
    # ========================================================================
    _ensure_mgn_inference_assets(mgn_model_dir, mgn_epoch, pca_mean_volume)
    reference_image = itk.imread(str(reference_image_file))
    infer = WorkflowInferPhysicsNeMoMGN(model_directory=mgn_model_dir, epoch=mgn_epoch)

    cardiac_fields: list[itk.Image] = []
    heart_surfaces: list[pv.PolyData] = []
    for stage in cardiac_stages:
        result = infer.create_deformation_field(
            shape_parameters=coefficients_file,
            stage=float(stage),
            reference_image=reference_image,
            reference_surface=registered_mesh_file,
        )
        pct = int(round(stage * 100))
        itk.imwrite(
            result["deformation_field"],
            str(output_dir / f"deformation_field_s{pct:03d}.mha"),
            compression=True,
        )
        itk.imwrite(
            result["normal_image"],
            str(output_dir / f"surface_normal_field_s{pct:03d}.mha"),
            compression=True,
        )
        result["deformed_surface"].save(
            str(output_dir / f"deformed_heart_surface_s{pct:03d}.vtp")
        )
        cardiac_fields.append(result["deformation_field"])
        heart_surfaces.append(result["deformed_surface"])
        logger.info("cardiac stage %.2f -> deformation_field_s%03d.mha", stage, pct)

    # Beating-heart-only 4D USD (heart surfaces, one heartbeat per second).
    heart_usd = WorkflowConvertVTKToUSD(
        input_meshes=heart_surfaces,
        usd_project_name="beating_heart",
        output_directory=output_dir,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        frames_per_second=float(len(cardiac_stages)),
    )
    heart_usd.process()

    # ========================================================================
    # Stage 2: combine the cardiac fields with respiratory motion.
    # ========================================================================
    cardiac_transforms = [
        _smoothed_cardiac_transform(field, smoothing_sigma_mm, transform_tools)
        for field in cardiac_fields
    ]
    logger.info(
        "Smoothed %d cardiac deformation fields by %.1f mm",
        len(cardiac_transforms),
        smoothing_sigma_mm,
    )

    # Labeled contour surface: each cell carries `boundary_labels`, which survive
    # the warps (deep copies) and let ConvertVTKToUSD split the USD by anatomy so
    # USDAnatomyTools.enhance_meshes can bind per-organ OmniSurface materials.
    contour_tools = ContourTools()
    patient_labelmap = itk.imread(str(patient_labelmap_file))
    patient_surface = contour_tools.extract_contours(patient_labelmap)
    patient_surface = _condition_surface(
        patient_surface, surface_decimation_reduction, surface_smoothing_iterations
    )
    logger.info(
        "Patient surface: %d points, %d cells (decimation=%.2f, smoothing_iters=%d)",
        patient_surface.n_points,
        patient_surface.n_cells,
        surface_decimation_reduction,
        surface_smoothing_iterations,
    )

    # Cardiac motion at the reference frame, once per stage, reused across phases.
    cardiac_surfaces = [
        transform_tools.transform_pvcontour(patient_surface, cardiac_transform)
        for cardiac_transform in cardiac_transforms
    ]

    n_phases = len(forward_transform_files)
    n_stages = len(cardiac_surfaces)

    # Respiratory-warped vertex positions for every (phase, stage):
    # resp_points[phase][stage] = forward_phase(cardiac_surface[stage]).points.
    resp_points: list[list[np.ndarray]] = []
    for phase_idx, forward_file in enumerate(forward_transform_files):
        forward_transform = itk.transformread(str(forward_file))
        resp_points.append(
            [
                np.asarray(
                    transform_tools.transform_pvcontour(
                        cardiac_surface, forward_transform
                    ).points,
                    dtype=np.float32,
                )
                for cardiac_surface in cardiac_surfaces
            ]
        )
        logger.info("respiratory warp phase %d/%d done", phase_idx + 1, n_phases)

    for stale_frame in output_dir.glob("combined_frame_*.vtp"):
        stale_frame.unlink()

    # Render frames by bilinearly interpolating the precomputed (phase, stage)
    # warped-point grid: respiratory advances with the breath phase, while the
    # cardiac cycle advances continuously and independently at
    # ``cardiac_cycles_per_phase`` beats per phase, so a heartbeat carries across
    # phase boundaries. Both axes wrap, so the sequence loops.
    frames_per_phase = n_stages
    n_frames = n_phases * frames_per_phase
    combined_files: list[Path] = []
    usd_frames: list[pv.PolyData] = []
    for frame_idx in range(n_frames):
        # Respiratory position: current breath phase and fraction into it.
        phase_pos = frame_idx / frames_per_phase
        phase_idx = int(phase_pos)
        next_phase_idx = (phase_idx + 1) % n_phases
        resp_blend = phase_pos - phase_idx

        # Cardiac position: continuous across phases, wrapping within the cycle.
        card_pos = (phase_pos * cardiac_cycles_per_phase * n_stages) % n_stages
        stage_idx = int(card_pos)
        next_stage_idx = (stage_idx + 1) % n_stages
        card_blend = card_pos - stage_idx

        # Interpolate the cardiac cycle within each bounding phase, then between
        # the two phases.
        phase_a = (1.0 - card_blend) * resp_points[phase_idx][stage_idx] + (
            card_blend * resp_points[phase_idx][next_stage_idx]
        )
        phase_b = (1.0 - card_blend) * resp_points[next_phase_idx][stage_idx] + (
            card_blend * resp_points[next_phase_idx][next_stage_idx]
        )
        points = (1.0 - resp_blend) * phase_a + resp_blend * phase_b

        combined_surface = patient_surface.copy(deep=True)
        combined_surface.points = points

        frame_file = output_dir / f"combined_frame_{frame_idx:03d}.vtp"
        combined_surface.save(str(frame_file))
        combined_files.append(frame_file)
        usd_frames.append(combined_surface)

    del resp_points
    logger.info(
        "Wrote %d combined-motion surfaces to %s", len(combined_files), output_dir
    )

    # Assemble the ordered frames into a single animated 4D USD, split by anatomy
    # label (via the mesh `boundary_labels`) so each organ becomes its own prim
    # under /World/heart_and_lung_motion/{group}/{label}. enhance_meshes then
    # binds the per-organ OmniSurface materials (colors, subsurface scatter, ...).
    segmenter = SegmentChestTotalSegmentator()
    converter = ConvertVTKToUSD(
        "heart_and_lung_motion",
        usd_frames,
        segmenter.taxonomy.all_labels(),
        segmenter=segmenter,
        frames_per_second=frames_per_second,
        log_level=logging.INFO,
    )
    usd_file = output_dir / "heart_and_lung_motion.usd"
    stage = converter.convert(str(usd_file))
    USDAnatomyTools(stage).enhance_meshes(segmenter)
    stage.Save()
    logger.info("Wrote 4D USD with anatomy materials: %s", usd_file)

    tutorial_results = {
        "cardiac_field_count": len(cardiac_fields),
        "combined_surfaces": combined_files,
        "beating_heart_usd": str(output_dir / "beating_heart.usd"),
        "usd_file": str(usd_file),
    }
