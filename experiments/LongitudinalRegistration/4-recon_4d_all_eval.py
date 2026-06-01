"""Compare longitudinal cardiac CT registration methods with landmarks.

The experiment registers each gated time-point image to the high-resolution
reference image for the same subject.  Input images are 3D CT volumes in LPS
world space.  Landmarks are CSV rows with physical LPS coordinates
``Name,X,Y,Z`` in millimeters.

Two accuracy modes are written:
1. Direct landmarks: reference landmarks are transformed into each time-point
   image space with the inverse registration transform and compared to the
   precomputed time-point landmarks.
2. Re-segmented landmarks: the reference image is warped into each time-point
   image space, re-segmented with Simpleware, and the newly extracted landmarks
   are compared to the precomputed time-point landmarks.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import itk
import numpy as np

from physiomotion4d import (
    RegisterTimeSeriesImages,
    SegmentHeartSimpleware,
    TransformTools,
)

DEFAULT_REF_DIR = Path("d:/PhysioMotion4D/duke_data/ref_images")
DEFAULT_TIMEPOINT_BASE_DIR = Path("d:/PhysioMotion4D/duke_data/gated_nii")
DEFAULT_SEGMENTATION_BASE_DIR = Path("d:/PhysioMotion4D/duke_data/simple_ascardio")
DEFAULT_OUTPUT_DIR = Path("d:/PhysioMotion4D/duke_data/longitudinal_registration")
DEFAULT_EXCLUDE_TOKENS = ("nop", "dia", "sys", "_ref")
DEFAULT_SEGMENTATION_DIR = "results-labelmaps_and_landmarks"
DEFAULT_METHODS = ("ANTS", "greedy", "icon_default", "ants_icon_default")
TIMEPOINT_RE = re.compile(r"_g(?P<timepoint>[0-9]{3})")


Landmarks = dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class MethodSpec:
    """Registration method plus optional ICON checkpoint."""

    output_name: str
    registration_method: str
    icon_weights_path: Optional[Path] = None


@dataclass(frozen=True)
class ImageArtifacts:
    """Input files associated with one image volume."""

    image_file: Path
    landmark_file: Optional[Path]
    labelmap_file: Optional[Path]
    timepoint: str


def nii_stem(path: Path) -> str:
    """Return a stable stem for ``.nii.gz`` or single-suffix files."""
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def timepoint_from_name(path: Path) -> str:
    """Extract the gated time-point tag from a filename."""
    match = TIMEPOINT_RE.search(path.name)
    if match:
        return match.group("timepoint")
    return nii_stem(path)


def first_existing(paths: list[Path]) -> Optional[Path]:
    """Return the first existing path from a candidate list."""
    for path in paths:
        if path.exists():
            return path
    return None


def landmark_candidates(
    image_file: Path,
    segmentation_dir: str,
    artifact_dir: Optional[Path],
) -> list[Path]:
    """Return likely landmark CSV paths for an image."""
    stem = nii_stem(image_file)
    parent = image_file.parent
    seg_parent = parent / segmentation_dir
    candidates = [
        parent / f"{stem}_landmark.csv",
        parent / f"{stem}_landmarks.csv",
        seg_parent / f"{stem}_landmark.csv",
        seg_parent / f"{stem}_landmarks.csv",
    ]
    if artifact_dir is not None:
        candidates = [
            artifact_dir / f"{stem}_landmark.csv",
            artifact_dir / f"{stem}_landmarks.csv",
            *candidates,
        ]
    return candidates


def labelmap_candidates(
    image_file: Path,
    segmentation_dir: str,
    artifact_dir: Optional[Path],
) -> list[Path]:
    """Return likely labelmap paths for an image."""
    stem = nii_stem(image_file)
    parent = image_file.parent
    seg_parent = parent / segmentation_dir
    candidates = [
        parent / f"{stem}_labelmap.nii.gz",
        seg_parent / f"{stem}_labelmap.nii.gz",
    ]
    if artifact_dir is not None:
        candidates = [artifact_dir / f"{stem}_labelmap.nii.gz", *candidates]
    return candidates


def image_artifacts(
    image_file: Path,
    segmentation_dir: str,
    artifact_dir: Optional[Path] = None,
) -> ImageArtifacts:
    """Find landmarks and labelmaps associated with one image."""
    return ImageArtifacts(
        image_file=image_file,
        landmark_file=first_existing(
            landmark_candidates(image_file, segmentation_dir, artifact_dir)
        ),
        labelmap_file=first_existing(
            labelmap_candidates(image_file, segmentation_dir, artifact_dir)
        ),
        timepoint=timepoint_from_name(image_file),
    )


def read_landmarks(path: Path) -> Landmarks:
    """Read physical LPS landmarks from ``Name,X,Y,Z`` CSV."""
    landmarks: Landmarks = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            landmarks[row["Name"]] = (
                float(row["X"]),
                float(row["Y"]),
                float(row["Z"]),
            )
    return landmarks


def write_landmarks(path: Path, landmarks: Landmarks) -> None:
    """Write physical LPS landmarks to ``Name,X,Y,Z`` CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Name", "X", "Y", "Z"])
        for name, coords in sorted(landmarks.items()):
            writer.writerow([name, coords[0], coords[1], coords[2]])


def transform_landmarks(landmarks: Landmarks, transform: itk.Transform) -> Landmarks:
    """Apply an ITK physical-space transform to landmark coordinates."""
    transformed: Landmarks = {}
    for name, point in landmarks.items():
        transformed_point = transform.TransformPoint(point)
        transformed[name] = (
            float(transformed_point[0]),
            float(transformed_point[1]),
            float(transformed_point[2]),
        )
    return transformed


def landmark_errors(source: Landmarks, target: Landmarks) -> dict[str, float]:
    """Return per-landmark Euclidean errors in millimeters."""
    errors: dict[str, float] = {}
    for name in sorted(source.keys() & target.keys()):
        source_point = np.asarray(source[name], dtype=np.float64)
        target_point = np.asarray(target[name], dtype=np.float64)
        errors[name] = float(np.linalg.norm(source_point - target_point))
    return errors


def summarize_errors(errors: dict[str, float], prefix: str) -> dict[str, object]:
    """Summarize landmark errors for one comparison mode."""
    if not errors:
        return {
            f"{prefix}_landmarks": 0,
            f"{prefix}_mean_mm": "",
            f"{prefix}_median_mm": "",
            f"{prefix}_max_mm": "",
        }
    values = np.asarray(list(errors.values()), dtype=np.float64)
    return {
        f"{prefix}_landmarks": len(errors),
        f"{prefix}_mean_mm": float(np.mean(values)),
        f"{prefix}_median_mm": float(np.median(values)),
        f"{prefix}_max_mm": float(np.max(values)),
    }


def write_error_details(
    path: Path,
    subject_id: str,
    method_name: str,
    timepoint: str,
    mode: str,
    errors: dict[str, float],
) -> None:
    """Append per-landmark errors to the detail CSV."""
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        fieldnames = ["subject_id", "method", "timepoint", "mode", "name", "error_mm"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for name, error in sorted(errors.items()):
            writer.writerow(
                {
                    "subject_id": subject_id,
                    "method": method_name,
                    "timepoint": timepoint,
                    "mode": mode,
                    "name": name,
                    "error_mm": error,
                }
            )


def dice_by_label(
    labelmap_a: itk.Image,
    labelmap_b: itk.Image,
) -> dict[int, float]:
    """Compute Dice scores for labels present in either 3D labelmap."""
    arr_a = itk.array_from_image(labelmap_a)
    arr_b = itk.array_from_image(labelmap_b)
    if arr_a.shape != arr_b.shape:
        return {}
    labels = sorted(set(np.unique(arr_a)).union(set(np.unique(arr_b))) - {0})
    scores: dict[int, float] = {}
    for label in labels:
        mask_a = arr_a == label
        mask_b = arr_b == label
        denom = int(mask_a.sum() + mask_b.sum())
        if denom > 0:
            scores[int(label)] = float(
                2.0 * np.logical_and(mask_a, mask_b).sum() / denom
            )
    return scores


def summarize_dice(scores: dict[int, float]) -> dict[str, object]:
    """Summarize per-label Dice scores."""
    if not scores:
        return {"dice_labels": 0, "dice_mean": "", "dice_min": ""}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    return {
        "dice_labels": len(scores),
        "dice_mean": float(np.mean(values)),
        "dice_min": float(np.min(values)),
    }


def discover_subjects(
    reference_dir: Path,
    timepoint_base_dir: Path,
    reference_pattern: str,
    timepoint_pattern: str,
    exclude_tokens: tuple[str, ...],
    segmentation_dir: str,
    segmentation_base_dir: Optional[Path],
) -> list[tuple[str, ImageArtifacts, list[ImageArtifacts]]]:
    """Discover reference and time-point files for each subject."""
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference image directory not found: {reference_dir}")
    if not timepoint_base_dir.exists():
        raise FileNotFoundError(
            f"Time-point image base directory not found: {timepoint_base_dir}"
        )

    subjects: list[tuple[str, ImageArtifacts, list[ImageArtifacts]]] = []
    for reference_file in sorted(reference_dir.glob(reference_pattern)):
        subject_id = reference_file.name[:6]
        source_dir = timepoint_base_dir / subject_id
        if not source_dir.exists():
            raise FileNotFoundError(
                f"No time-point directory for {subject_id}: {source_dir}"
            )
        artifact_dir = None
        if segmentation_base_dir is not None:
            candidate_dir = segmentation_base_dir / subject_id
            if candidate_dir.exists():
                artifact_dir = candidate_dir

        reference_in_source = source_dir / reference_file.name
        reference_artifacts = image_artifacts(
            reference_in_source if reference_in_source.exists() else reference_file,
            segmentation_dir,
            artifact_dir,
        )

        timepoint_files = [
            path
            for path in sorted(source_dir.glob(timepoint_pattern))
            if not any(token in path.name for token in exclude_tokens)
        ]
        timepoints = [
            image_artifacts(path, segmentation_dir, artifact_dir)
            for path in timepoint_files
            if path.is_file()
        ]
        subjects.append((subject_id, reference_artifacts, timepoints))
    return subjects


def build_method_specs(
    method_names: list[str],
    finetuned_weights_path: Optional[Path],
) -> list[MethodSpec]:
    """Map output method labels to registrar methods and optional weights."""
    specs: list[MethodSpec] = []
    for method_name in method_names:
        if method_name == "ANTS":
            specs.append(MethodSpec(method_name, "ANTS"))
        elif method_name == "greedy":
            specs.append(MethodSpec(method_name, "greedy"))
        elif method_name == "icon_default":
            specs.append(MethodSpec(method_name, "ICON"))
        elif method_name == "ants_icon_default":
            specs.append(MethodSpec(method_name, "ANTS_ICON"))
        elif method_name == "greedy_icon_default":
            specs.append(MethodSpec(method_name, "greedy_ICON"))
        elif method_name == "icon_finetuned":
            specs.append(MethodSpec(method_name, "ICON", finetuned_weights_path))
        elif method_name == "ants_icon_finetuned":
            specs.append(MethodSpec(method_name, "ANTS_ICON", finetuned_weights_path))
        elif method_name == "greedy_icon_finetuned":
            specs.append(MethodSpec(method_name, "greedy_ICON", finetuned_weights_path))
        else:
            raise ValueError(f"Unknown method: {method_name}")

    for spec in specs:
        if "finetuned" in spec.output_name and spec.icon_weights_path is None:
            raise ValueError(f"{spec.output_name} requires --finetuned-weights-path")
    return specs


def configure_registrar(
    method_spec: MethodSpec,
    fixed_image: itk.Image,
    fixed_labelmap: Optional[itk.Image],
    ants_iterations: list[int],
    greedy_iterations: list[int],
    icon_iterations: int,
) -> RegisterTimeSeriesImages:
    """Create and configure the time-series registrar."""
    registrar = RegisterTimeSeriesImages(
        registration_method=method_spec.registration_method
    )
    registrar.set_modality("ct")
    registrar.set_fixed_image(fixed_image)
    registrar.set_fixed_labelmap(fixed_labelmap)
    registrar.set_number_of_iterations_ANTS(ants_iterations)
    registrar.set_number_of_iterations_greedy(greedy_iterations)
    registrar.set_number_of_iterations_ICON(icon_iterations)
    if method_spec.icon_weights_path is not None:
        registrar.registrar_ICON.set_weights_path(str(method_spec.icon_weights_path))
    return registrar


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Write experiment summary rows."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_method_for_subject(
    subject_id: str,
    reference_artifacts: ImageArtifacts,
    timepoint_artifacts: list[ImageArtifacts],
    method_spec: MethodSpec,
    output_dir: Path,
    run_resegmentation: bool,
    ants_iterations: list[int],
    greedy_iterations: list[int],
    icon_iterations: int,
    error_detail_file: Path,
) -> list[dict[str, object]]:
    """Run one registration method for one subject and return summary rows."""
    if reference_artifacts.landmark_file is None:
        raise FileNotFoundError(
            f"Missing reference landmarks for {reference_artifacts.image_file}"
        )

    fixed_image = itk.imread(str(reference_artifacts.image_file), pixel_type=itk.F)
    fixed_labelmap = None
    if reference_artifacts.labelmap_file is not None:
        fixed_labelmap = itk.imread(str(reference_artifacts.labelmap_file))

    moving_images = [
        itk.imread(str(artifacts.image_file), pixel_type=itk.F)
        for artifacts in timepoint_artifacts
    ]
    moving_labelmaps = None
    if all(artifacts.labelmap_file is not None for artifacts in timepoint_artifacts):
        moving_labelmaps = [
            itk.imread(str(artifacts.labelmap_file))
            for artifacts in timepoint_artifacts
        ]

    registrar = configure_registrar(
        method_spec,
        fixed_image,
        fixed_labelmap,
        ants_iterations,
        greedy_iterations,
        icon_iterations,
    )

    result = registrar.register_time_series(
        moving_images=moving_images,
        moving_labelmaps=moving_labelmaps,
        reference_frame=0,
        register_reference=True,
        prior_weight=0.0,
    )

    reference_landmarks = read_landmarks(reference_artifacts.landmark_file)
    transform_tools = TransformTools()
    segmenter = SegmentHeartSimpleware() if run_resegmentation else None
    subject_method_dir = output_dir / method_spec.output_name / subject_id
    subject_method_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, artifacts in enumerate(timepoint_artifacts):
        timepoint_dir = subject_method_dir / artifacts.timepoint
        timepoint_dir.mkdir(parents=True, exist_ok=True)

        forward_transform = result["forward_transforms"][index]
        inverse_transform = result["inverse_transforms"][index]
        loss = result["losses"][index]

        forward_file = timepoint_dir / "time_to_reference.hdf"
        inverse_file = timepoint_dir / "reference_to_time.hdf"
        itk.transformwrite(forward_transform, str(forward_file), compression=True)
        itk.transformwrite(inverse_transform, str(inverse_file), compression=True)

        moving_to_reference = transform_tools.transform_image(
            moving_images[index],
            forward_transform,
            fixed_image,
        )
        moving_to_reference_file = timepoint_dir / "time_to_reference.mha"
        itk.imwrite(
            moving_to_reference, str(moving_to_reference_file), compression=True
        )

        reference_to_time = transform_tools.transform_image(
            fixed_image,
            inverse_transform,
            moving_images[index],
        )
        reference_to_time_file = timepoint_dir / "reference_to_time.mha"
        itk.imwrite(reference_to_time, str(reference_to_time_file), compression=True)

        row: dict[str, object] = {
            "subject_id": subject_id,
            "method": method_spec.output_name,
            "timepoint": artifacts.timepoint,
            "moving_image": str(artifacts.image_file),
            "forward_transform": str(forward_file),
            "inverse_transform": str(inverse_file),
            "loss": float(loss),
        }

        if artifacts.landmark_file is not None:
            timepoint_landmarks = read_landmarks(artifacts.landmark_file)
            # Warp the reference landmarks into the timepoint (moving) space to
            # compare against this timepoint's landmarks. Warping reference ->
            # time POINTS uses forward_transform (the fixed -> moving point map),
            # which is the opposite of the reference_to_time IMAGE above (images
            # pull back, points push forward). See
            # docs/developer/transform_conventions.
            direct_landmarks = transform_landmarks(
                reference_landmarks,
                forward_transform,
            )
            direct_errors = landmark_errors(direct_landmarks, timepoint_landmarks)
            write_error_details(
                error_detail_file,
                subject_id,
                method_spec.output_name,
                artifacts.timepoint,
                "direct",
                direct_errors,
            )
            row.update(summarize_errors(direct_errors, "direct"))
        else:
            row.update(summarize_errors({}, "direct"))

        if run_resegmentation and segmenter is not None:
            segmentation = segmenter.segment(
                reference_to_time,
                contrast_enhanced_study=False,
            )
            warped_labelmap = segmentation["labelmap"]
            warped_labelmap_file = timepoint_dir / "reference_to_time_labelmap.nii.gz"
            itk.imwrite(warped_labelmap, str(warped_labelmap_file), compression=True)
            reseg_landmarks = segmenter.get_landmarks()
            reseg_landmark_file = timepoint_dir / "reference_to_time_landmark.csv"
            write_landmarks(reseg_landmark_file, reseg_landmarks)
            row["resegmented_labelmap"] = str(warped_labelmap_file)
            row["resegmented_landmarks"] = str(reseg_landmark_file)

            if artifacts.landmark_file is not None:
                timepoint_landmarks = read_landmarks(artifacts.landmark_file)
                reseg_errors = landmark_errors(reseg_landmarks, timepoint_landmarks)
                write_error_details(
                    error_detail_file,
                    subject_id,
                    method_spec.output_name,
                    artifacts.timepoint,
                    "resegmented",
                    reseg_errors,
                )
                row.update(summarize_errors(reseg_errors, "resegmented"))
            else:
                row.update(summarize_errors({}, "resegmented"))

            if artifacts.labelmap_file is not None:
                timepoint_labelmap = itk.imread(str(artifacts.labelmap_file))
                row.update(
                    summarize_dice(dice_by_label(warped_labelmap, timepoint_labelmap))
                )
            else:
                row.update(summarize_dice({}))
        else:
            row["resegmented_labelmap"] = ""
            row["resegmented_landmarks"] = ""
            row.update(summarize_errors({}, "resegmented"))
            row.update(summarize_dice({}))

        rows.append(row)

    return rows


def parse_iterations(value: str) -> list[int]:
    """Parse comma-separated multi-resolution iteration counts."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    """Run the longitudinal registration comparison experiment."""
    parser = argparse.ArgumentParser(
        description="Compare ANTS, Greedy, and ICON longitudinal registration."
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument(
        "--timepoint-base-dir",
        type=Path,
        default=DEFAULT_TIMEPOINT_BASE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--segmentation-base-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_BASE_DIR,
        help="Directory with per-subject precomputed *_labelmap and *_landmark files.",
    )
    parser.add_argument("--reference-pattern", default="pm00*.nii.gz")
    parser.add_argument("--timepoint-pattern", default="*.nii.gz")
    parser.add_argument("--segmentation-dir", default=DEFAULT_SEGMENTATION_DIR)
    parser.add_argument(
        "--exclude-token",
        action="append",
        default=list(DEFAULT_EXCLUDE_TOKENS),
        help="Filename token to exclude from time-point inputs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to run. Defaults include finetuned methods when weights are set.",
    )
    parser.add_argument("--finetuned-weights-path", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--ANTS-iterations", default="30,15,7,3")
    parser.add_argument("--greedy-iterations", default="30,15,7,3")
    parser.add_argument("--ICON-iterations", type=int, default=20)
    parser.add_argument(
        "--skip-resegmentation",
        action="store_true",
        help="Skip Simpleware re-segmentation mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovered files and planned methods without registration.",
    )
    args = parser.parse_args()

    method_names = args.methods
    if method_names is None:
        method_names = list(DEFAULT_METHODS)
        method_names.append("greedy_icon_default")
        if args.finetuned_weights_path is not None:
            method_names.extend(
                [
                    "icon_finetuned",
                    "ants_icon_finetuned",
                    "greedy_icon_finetuned",
                ]
            )

    method_specs = build_method_specs(method_names, args.finetuned_weights_path)
    subjects = discover_subjects(
        args.reference_dir,
        args.timepoint_base_dir,
        args.reference_pattern,
        args.timepoint_pattern,
        tuple(args.exclude_token),
        args.segmentation_dir,
        args.segmentation_base_dir,
    )
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    if args.dry_run:
        for subject_id, reference_artifacts, timepoint_artifacts in subjects:
            if args.max_timepoints is not None:
                timepoint_artifacts = timepoint_artifacts[: args.max_timepoints]
            missing_landmarks = sum(
                artifacts.landmark_file is None for artifacts in timepoint_artifacts
            )
            missing_labelmaps = sum(
                artifacts.labelmap_file is None for artifacts in timepoint_artifacts
            )
            print(
                f"{subject_id}: {len(timepoint_artifacts)} time points, "
                f"reference_landmarks={reference_artifacts.landmark_file is not None}, "
                f"reference_labelmap={reference_artifacts.labelmap_file is not None}, "
                f"missing_time_landmarks={missing_landmarks}, "
                f"missing_time_labelmaps={missing_labelmaps}"
            )
        print("Methods: " + ", ".join(spec.output_name for spec in method_specs))
        return 0

    summary_rows: list[dict[str, object]] = []
    detail_file = args.output_dir / "landmark_errors_by_point.csv"
    if detail_file.exists():
        detail_file.unlink()

    for subject_id, reference_artifacts, timepoint_artifacts in subjects:
        if args.max_timepoints is not None:
            timepoint_artifacts = timepoint_artifacts[: args.max_timepoints]
        if not timepoint_artifacts:
            raise ValueError(f"No time-point images found for {subject_id}")
        print(
            f"Running {subject_id}: {len(timepoint_artifacts)} time points, "
            f"{len(method_specs)} methods"
        )
        for method_spec in method_specs:
            print(f"  Method: {method_spec.output_name}")
            rows = run_method_for_subject(
                subject_id,
                reference_artifacts,
                timepoint_artifacts,
                method_spec,
                args.output_dir,
                not args.skip_resegmentation,
                parse_iterations(args.ants_iterations),
                parse_iterations(args.greedy_iterations),
                args.icon_iterations,
                detail_file,
            )
            summary_rows.extend(rows)
            write_summary(args.output_dir / "registration_summary.csv", summary_rows)

    print(f"Wrote summary: {args.output_dir / 'registration_summary.csv'}")
    print(f"Wrote landmark details: {detail_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
