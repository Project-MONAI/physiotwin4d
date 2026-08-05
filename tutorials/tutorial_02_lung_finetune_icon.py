"""
Tutorial 2: Finetune uniGradICON on DIR-Lab 4D CT

Purpose
-------
Finetune uniGradICON on every DIR-Lab 4D CT case except Case 1, then register
``Case1Pack_T00.mha`` (moving) to ``Case1Pack_T50.mha`` (fixed) three ways:
``RegisterImagesGreedy`` alone, deformable, with its default iteration
schedule, and ``RegisterImagesICON`` with the stock uniGradICON weights and
with the finetuned weights.  Case 1 is never seen during finetuning, so it is a held-out
evaluation pair.

Accuracy is measured two ways.  The primary metric is target registration
error: DIR-Lab ships 300 expert landmarks for the extreme phases (T00 and T50)
of every case, so each fixed-image landmark is mapped through the registration
transform and compared, in millimeters, against its moving-image counterpart.
The secondary metric is label overlap: ``SegmentNVSegmentCTMRI`` segments the
fixed and moving images once each, and the moving labelmap is warped onto the
fixed grid by every transform, so the Dice scores reflect the transform rather
than segmentation variability on re-segmented warped volumes.  The moving image
and labelmap resampled onto the fixed grid without registration supply the
"before registration" reference row for both metrics.

Reported per method: the mean, standard deviation, 95th percentile and maximum
landmark error in millimeters; the mean, 5th percentile, median, 95th
percentile, minimum and maximum of the per-class Dice scores; the number of
mislabeled voxels; and the wall-clock registration time.

Finetuning artifacts (dataset JSON, YAML config, checkpoint tree) are written
under ``tutorials/network_weights/icon_dirlab_4dct``.  The final checkpoint is
``tutorials/network_weights/icon_dirlab_4dct/icon_dirlab_4dct_model/checkpoints/
network_weights_final.trch``, the path returned by
``WorkflowFinetuneICONRegistration.expected_weights_path()``.  That directory is
deleted at the start of every run, so each run finetunes from scratch; see the
comment above the ``shutil.rmtree`` call for how to reuse a previous run.

Data Required
-------------
Full data: ``data/DirLab-4DCT`` (all 10 cases, converted to HU ``.mha`` by
``data/DirLab-4DCT/fix_downloaded_data.py``), including the raw
``downloaded_data/Case1Pack/ExtremePhases`` landmark files
Test data: ``data/test/DirLab-4DCT``
"""

# Imports
from __future__ import annotations

import csv
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np

from physiotwin4d import (
    PhysioTwin4DBase,
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesICON,
    SegmentNVSegmentCTMRI,
    TestTools,
    TransformTools,
    WorkflowFinetuneICONRegistration,
)

# Only run if this script is not imported as a module

# unigradicon finetuning is launched as a subprocess and torch spawns worker
# processes; on Windows the spawn start method re-imports this script in each
# child, so all top-level work stays under the __name__ == "__main__" guard.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_02_lung_finetune_icon"

    output_dir = tutorials_dir / "output" / "tutorial_02_lung"
    baselines_dir = repo_root / "tests" / "baselines"

    # The workflow writes its dataset JSON, YAML config, and checkpoint tree
    # under ``weights_dir / finetune_name``.
    weights_dir = tutorials_dir / "network_weights"
    finetune_name = "icon_dirlab_4dct"

    run_finetuning = True

    test_mode = TestTools.running_as_test()
    if test_mode:
        data_dir = repo_root / "data" / "test" / "DirLab-4DCT"
        number_of_iterations_greedy: Optional[list[int]] = [1, 0]
        epochs = 1
    else:
        data_dir = repo_root / "data" / "DirLab-4DCT"
        number_of_iterations_greedy = [60, 30, 20]  # Greedy defaults
        # 90 training frames at batch_size 4 is 22 optimizer steps per epoch, so
        # 100 epochs is ~2200 steps at a 5e-5 learning rate.  Far fewer than
        # that leaves the finetuned weights statistically indistinguishable
        # from the stock weights they started from.
        epochs = 100

    log_level = logging.INFO
    reporter = PhysioTwin4DBase(class_name=class_name, log_level=log_level)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Held-out evaluation pair (Case 1 is excluded from finetuning).  T00 and
    # T50 are the extreme inhale/exhale phases, the only pair DIR-Lab supplies
    # expert landmarks for.
    fixed_file = data_dir / "Case1Pack_T50.mha"
    moving_file = data_dir / "Case1Pack_T00.mha"
    landmark_dir = data_dir / "downloaded_data" / "Case1Pack" / "ExtremePhases"
    fixed_landmark_file = landmark_dir / "Case1_300_T50_xyz.txt"
    moving_landmark_file = landmark_dir / "Case1_300_T00_xyz.txt"
    missing = [
        str(p)
        for p in (fixed_file, moving_file, fixed_landmark_file, moving_landmark_file)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing DirLab phase images or landmarks: {missing}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )

    # Finetuning cohort: every case except Case1Pack.  ``Case10Pack_*`` is kept
    # because only the exact ``Case1Pack_`` prefix is excluded.
    training_files = sorted(
        path
        for path in data_dir.glob("Case*_T??.mha")
        if not path.name.startswith("Case1Pack_")
    )
    subject_image_files: dict[str, list[str]] = {}
    for path in training_files:
        subject_image_files.setdefault(path.name.split("_")[0], []).append(str(path))
    if not subject_image_files:
        raise FileNotFoundError(
            f"No non-Case1 DirLab phase images found under {data_dir}.\n"
            "See data/DirLab-4DCT/README.md for download instructions."
        )
    reporter.log_info(
        "Finetuning cohort: %d cases, %d frames",
        len(subject_image_files),
        len(training_files),
    )

    # Always finetune from scratch.  uniGradICON refuses to overwrite an
    # existing experiment directory: it appends "-N" to the name instead
    # (``icon_dirlab_4dct_model-5``, ...), while expected_weights_path() keeps
    # pointing at the original, never-written path.  Deleting the tree up front
    # keeps the two in agreement.
    #
    # To reuse a previous run instead, delete the shutil.rmtree call below and
    # guard the process() call:
    #     weights_path = workflow.expected_weights_path()
    #     if not weights_path.exists():
    #         weights_path = workflow.process()
    experiment_dir = weights_dir / finetune_name
    if run_finetuning:
        if experiment_dir.exists():
            reporter.log_info(
                "Removing previous finetuning outputs: %s", experiment_dir
            )
            shutil.rmtree(experiment_dir)

        # DIR-Lab ships no segmentations, so no labelmaps or masks are supplied and
        # the Dice loss must be disabled: uniGradICON requires a ``segmentation``
        # field on every dataset entry when dice_loss_weight > 0.
        #
        # lncc_sigma matches the sigma RegisterImagesICON uses at inference, so
        # finetuning optimizes the similarity this comparison scores.
        workflow = WorkflowFinetuneICONRegistration(
            subject_image_files=list(subject_image_files.values()),
            output_dir=weights_dir,
            finetune_name=finetune_name,
            subject_ids=list(subject_image_files.keys()),
            epochs=epochs,
            dice_loss_weight=0.0,
            lncc_sigma=5,
            log_level=log_level,
        )
        weights_path = workflow.process()
    else:
        weights_path = (
            Path(__file__).resolve().parent
            / "network_weights/icon_dirlab_4dct/icon_dirlab_4dct_model/checkpoints/network_weights_final.trch"
        )

    # Registration comparison
    fixed_image = itk.imread(str(fixed_file), pixel_type=itk.F)
    moving_image = itk.imread(str(moving_file), pixel_type=itk.F)
    transform_tools = TransformTools()

    def read_landmarks(landmark_file: Path, image: itk.Image) -> np.ndarray:
        """Read a DIR-Lab landmark file as an (N, 3) array of world points.

        Each line holds one 1-based voxel index as ``x y z``.
        """
        indices = np.loadtxt(landmark_file, dtype=int) - 1
        return np.array(
            [
                image.TransformIndexToPhysicalPoint([int(v) for v in index])
                for index in indices
            ]
        )

    fixed_landmarks = read_landmarks(fixed_landmark_file, fixed_image)
    moving_landmarks = read_landmarks(moving_landmark_file, moving_image)

    def landmark_metrics(errors_mm: np.ndarray) -> dict[str, Any]:
        """Summarize per-landmark target registration errors, in millimeters."""
        return {
            "tre_mean": float(errors_mm.mean()),
            "tre_std": float(errors_mm.std()),
            "tre_p95": float(np.percentile(errors_mm, 95)),
            "tre_max": float(errors_mm.max()),
        }

    def landmark_errors(transform: itk.Transform) -> np.ndarray:
        """Distance from each mapped fixed landmark to its moving counterpart.

        ``forward_transform`` is the resampling transform: it maps points on the
        fixed grid back into moving space, which is the direction the landmark
        correspondences are defined in.
        """
        mapped = np.array(
            [transform.TransformPoint(tuple(point)) for point in fixed_landmarks]
        )
        return np.asarray(np.linalg.norm(mapped - moving_landmarks, axis=1))

    # Each image is segmented once and the moving labelmap is warped by every
    # transform, so Dice reflects the transform rather than what the segmenter
    # does differently on each interpolated volume.
    segmenter = SegmentNVSegmentCTMRI(log_level=log_level)
    fixed_labelmap = segmenter.segment(fixed_image)["labelmap"]
    moving_labelmap = segmenter.segment(moving_image)["labelmap"]
    fixed_labels = itk.array_from_image(fixed_labelmap)

    def overlap_metrics(labelmap: itk.Image) -> dict[str, Any]:
        """Per-class Dice summary against the fixed labelmap.

        Classes are the union of the two labelmaps' non-zero ids, so a class
        found in only one of them scores 0 rather than being dropped.
        """
        labels = itk.array_from_image(labelmap)
        classes = np.union1d(np.unique(fixed_labels), np.unique(labels))
        classes = classes[classes != 0]
        dice = np.array(
            [
                2.0
                * np.count_nonzero((fixed_labels == c) & (labels == c))
                / (np.count_nonzero(fixed_labels == c) + np.count_nonzero(labels == c))
                for c in classes
            ]
        )
        return {
            "n_classes": int(dice.size),
            "dice_mean": float(dice.mean()),
            "dice_p5": float(np.percentile(dice, 5)),
            "dice_median": float(np.median(dice)),
            "dice_p95": float(np.percentile(dice, 95)),
            "dice_min": float(dice.min()),
            "dice_max": float(dice.max()),
            "mislabeled_voxels": int(np.count_nonzero(fixed_labels != labels)),
        }

    # Reference row: the moving image and its labelmap on the fixed grid,
    # unregistered.
    unregistered_image = itk.resample_image_filter(
        moving_image,
        ReferenceImage=fixed_image,
        UseReferenceImage=True,
    )
    unregistered_labelmap = itk.resample_image_filter(
        moving_labelmap,
        Interpolator=itk.NearestNeighborInterpolateImageFunction.New(moving_labelmap),
        ReferenceImage=fixed_image,
        UseReferenceImage=True,
    )

    registered_images: dict[str, itk.Image] = {"unregistered": unregistered_image}
    labelmaps: dict[str, itk.Image] = {"unregistered": unregistered_labelmap}
    rows: list[dict[str, Any]] = [
        {
            "method": "unregistered",
            "weights": "-",
            "registration_time_s": None,
            "loss": None,
            **landmark_metrics(
                np.linalg.norm(fixed_landmarks - moving_landmarks, axis=1)
            ),
            **overlap_metrics(unregistered_labelmap),
        }
    ]
    for method_name, method_weights in (
        ("greedy", None),
        ("icon_stock", None),
        ("icon_finetuned", weights_path),
    ):
        registrar: RegisterImagesBase
        if method_name == "greedy":
            registrar = RegisterImagesGreedy(log_level=log_level)
            registrar.set_transform_type("Deformable")
            if number_of_iterations_greedy is not None:
                registrar.set_number_of_iterations(number_of_iterations_greedy)
        else:
            registrar = RegisterImagesICON(log_level=log_level)
            # None, not 0: icon_registration rejects 0 and takes None to mean
            # "no test-time finetuning steps", so the comparison reflects what
            # each set of weights predicts rather than per-pair optimization.
            registrar.set_number_of_iterations(None)
            registrar.set_mass_preservation(True)  # For non-contrast CT
            if method_weights is not None:
                registrar.set_weights_path(str(method_weights))
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)

        start_time = time.perf_counter()
        result = registrar.register(moving_image)
        elapsed_s = time.perf_counter() - start_time

        registered_images[method_name] = transform_tools.transform_image(
            moving_image, result["forward_transform"], fixed_image
        )
        labelmaps[method_name] = transform_tools.transform_image(
            moving_labelmap,
            result["forward_transform"],
            fixed_image,
            interpolation_method="nearest",
        )
        rows.append(
            {
                "method": method_name,
                "weights": str(method_weights) if method_weights else "-",
                "registration_time_s": elapsed_s,
                "loss": float(result["loss"]),
                **landmark_metrics(landmark_errors(result["forward_transform"])),
                **overlap_metrics(labelmaps[method_name]),
            }
        )

    # Result saving
    itk.imwrite(
        fixed_labelmap, str(output_dir / "fixed_labelmap.mha"), compression=True
    )
    for method_name, image in registered_images.items():
        itk.imwrite(
            image,
            str(output_dir / f"registered_{method_name}.mha"),
            compression=True,
        )
    for method_name, labelmap in labelmaps.items():
        itk.imwrite(
            labelmap,
            str(output_dir / f"labelmap_{method_name}.mha"),
            compression=True,
        )

    summary_file = output_dir / "registration_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Reporting
    reporter.log_info(
        "Case1Pack_T00 -> Case1Pack_T50, error at %d expert landmarks, mm",
        len(fixed_landmarks),
    )
    reporter.log_info(
        "  %-13s %7s %7s %7s %7s %9s", "method", "mean", "std", "p95", "max", "time_s"
    )
    for row in rows:
        elapsed = row["registration_time_s"]
        reporter.log_info(
            "  %-13s %7.2f %7.2f %7.2f %7.2f %9s",
            row["method"],
            row["tre_mean"],
            row["tre_std"],
            row["tre_p95"],
            row["tre_max"],
            "-" if elapsed is None else f"{float(elapsed):.1f}",
        )

    reporter.log_info("Per-class Dice of the warped moving labelmap against the fixed")
    reporter.log_info(
        "  %-13s %7s %7s %7s %7s %7s %7s %7s %12s",
        "method",
        "classes",
        "mean",
        "p5",
        "median",
        "p95",
        "min",
        "max",
        "mislabeled",
    )
    for row in rows:
        reporter.log_info(
            "  %-13s %7d %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %12d",
            row["method"],
            row["n_classes"],
            row["dice_mean"],
            row["dice_p5"],
            row["dice_median"],
            row["dice_p95"],
            row["dice_min"],
            row["dice_max"],
            row["mislabeled_voxels"],
        )
    reporter.log_info("Wrote summary: %s", summary_file)

    # Testing
    tt = TestTools(
        class_name=class_name,
        results_dir=output_dir,
        baselines_dir=baselines_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = [
        tt.save_screenshot_image_slice(
            fixed_image,
            "fixed_frame.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
        )
    ]
    for method_name, image in registered_images.items():
        screenshots.append(
            tt.save_screenshot_image_slice(
                image,
                f"registered_{method_name}.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
            )
        )

    tutorial_results = {
        "weights_path": weights_path,
        "registration_metrics": rows,
        "labelmaps": labelmaps,
        "summary_file": summary_file,
        "registered_images": registered_images,
        "screenshots": screenshots,
    }
