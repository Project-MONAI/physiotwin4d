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
- ``combined_ct_<iii>.mha`` / ``combined_labelmap_<iii>.mha`` (``000..099``) - the
  original CT and labelmap warped by the same per-frame combined deformation
  (signed-short), so their anatomy tracks the displaced surfaces.

Prerequisites
-------------
Run Tutorials 1 (lung), 2 (lung), 5 (Case1Pack with the pca-vol-kcl model) and 9
(MGN) first. Requires the ``[physicsnemo]`` extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("tutorial_11_heart_and_lung_motion")

    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    # ---- Hard-coded inputs (edit for your layout) --------------------------
    # Cardiac MGN model (Tutorial 9): epoch-1500 checkpoint run directory and the
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
        transform_tools.smooth_deformation_field_transform(field, smoothing_sigma_mm)
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
    patient_surface = contour_tools.smooth_and_decimate_surface(
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

    forward_transforms = [
        itk.transformread(str(forward_file)) for forward_file in forward_transform_files
    ]

    # Respiratory-warped vertex positions for every (phase, stage):
    # resp_points[phase][stage] = forward_phase(cardiac_surface[stage]).points.
    resp_points: list[list[np.ndarray]] = []
    for phase_idx, forward_transform in enumerate(forward_transforms):
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

    for pattern in (
        "combined_frame_*.vtp",
        "combined_ct_*.mha",
        "combined_labelmap_*.mha",
    ):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    # Combined displacement field d_ps(x) = forward_p(cardiac_s(x)) - x sampled on
    # the reference grid, one per (phase, stage). Bilinearly blending these fields
    # per frame reproduces the surface point-blend exactly (the blend is affine and
    # the shared x cancels), so the warped CT/labelmap track the displaced surfaces.
    # ITK CompositeTransform applies its last-added transform first, so cardiac is
    # added last to act before respiratory. Fields are built lazily and evicted per
    # phase so only the two phases bracketing the current frame are ever held.
    image_tools = ImageTools()
    field_cache: dict[tuple[int, int], np.ndarray] = {}

    def combined_field(phase: int, stage: int) -> np.ndarray:
        cached = field_cache.get((phase, stage))
        if cached is not None:
            return cached
        forward = forward_transforms[phase]
        composite = itk.CompositeTransform[itk.D, 3].New()
        composite.AddTransform(
            forward[0] if isinstance(forward, (list, tuple)) else forward
        )
        composite.AddTransform(cardiac_transforms[stage])
        field = transform_tools.convert_transform_to_displacement_field(
            composite, reference_image, np_component_type=np.float32
        )
        arr: np.ndarray = itk.array_from_image(field)
        field_cache[(phase, stage)] = arr
        return arr

    def to_signed_short(image: itk.Image) -> itk.Image:
        caster = itk.CastImageFilter[type(image), itk.Image[itk.SS, 3]].New(Input=image)
        caster.Update()
        return caster.GetOutput()

    # Render frames by bilinearly interpolating the precomputed (phase, stage)
    # warped-point grid: respiratory advances with the breath phase, while the
    # cardiac cycle advances continuously and independently at
    # ``cardiac_cycles_per_phase`` beats per phase, so a heartbeat carries across
    # phase boundaries. Both axes wrap, so the sequence loops.
    frames_per_phase = n_stages
    n_frames = n_phases * frames_per_phase
    combined_files: list[Path] = []
    ct_files: list[Path] = []
    labelmap_files: list[Path] = []
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

        # Warp the original CT and labelmap by the same combined deformation, using
        # the field bilinearly blended over the same four (phase, stage) corners.
        field_a = (1.0 - card_blend) * combined_field(phase_idx, stage_idx) + (
            card_blend * combined_field(phase_idx, next_stage_idx)
        )
        field_b = (1.0 - card_blend) * combined_field(next_phase_idx, stage_idx) + (
            card_blend * combined_field(next_phase_idx, next_stage_idx)
        )
        field_arr = (1.0 - resp_blend) * field_a + resp_blend * field_b
        field_img = image_tools.convert_array_to_image_of_vectors(
            field_arr, ptype=itk.D, reference_image=reference_image
        )
        field_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        field_transform.SetDisplacementField(field_img)

        warped_ct = transform_tools.transform_image(
            reference_image, field_transform, reference_image, "linear"
        )
        warped_labelmap = transform_tools.transform_image(
            patient_labelmap, field_transform, reference_image, "nearest"
        )

        ct_file = output_dir / f"combined_ct_{frame_idx:03d}.mha"
        labelmap_file = output_dir / f"combined_labelmap_{frame_idx:03d}.mha"
        itk.imwrite(warped_ct, str(ct_file), compression=True)
        itk.imwrite(
            to_signed_short(warped_labelmap), str(labelmap_file), compression=True
        )
        ct_files.append(ct_file)
        labelmap_files.append(labelmap_file)

        # Keep only the two phases bracketing the current frame in the field cache.
        for key in [k for k in field_cache if k[0] not in (phase_idx, next_phase_idx)]:
            del field_cache[key]

    del resp_points
    logger.info(
        "Wrote %d combined-motion surfaces, CT and labelmap volumes to %s",
        len(combined_files),
        output_dir,
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
        "combined_ct_volumes": ct_files,
        "combined_labelmap_volumes": labelmap_files,
        "beating_heart_usd": str(output_dir / "beating_heart.usd"),
        "usd_file": str(usd_file),
    }
