"""
Tutorial 2: Finetune uniGradICON on DIR-Lab 4D CT

Purpose
-------
Finetune uniGradICON on every DIR-Lab 4D CT case except Case 1, then register
``Case1Pack_T30.mha`` (moving) to ``Case1Pack_T70.mha`` (fixed) three ways:
``RegisterImagesGreedy`` alone with its default settings, and
``RegisterImagesGreedyICON`` with the stock uniGradICON weights and with the
finetuned weights.  Case 1 is never seen during finetuning, so it is a held-out
evaluation pair.

Accuracy is measured by label overlap.  ``SegmentNVSegmentCTMRI`` segments the
fixed image, and segments each registered moving image after it is warped onto
the fixed grid; every labelmap is then compared against the fixed one.
Reported per method: the mean, 5th percentile, median, 95th percentile,
minimum and maximum of the per-class Dice scores, the number of mislabeled
voxels, and the wall-clock registration time.  The unregistered moving image,
resampled onto the fixed grid and segmented the same way, supplies the "before
registration" reference row.

Note that segmenting each registered image separately means the scores include
segmentation variability on the warped volumes, not the geometric error of the
transform alone.  It also costs one GPU segmentation per method.

Finetuning artifacts (dataset JSON, YAML config, checkpoint tree) are written
under ``tutorials/network_weights/icon_dirlab_4dct``.  The final checkpoint is
``tutorials/network_weights/icon_dirlab_4dct/icon_dirlab_4dct_model/checkpoints/
Finetune_multi_final.trch``, the path returned by
``WorkflowFinetuneICONRegistration.expected_weights_path()``.  That directory is
deleted at the start of every run, so each run finetunes from scratch; see the
comment above the ``shutil.rmtree`` call for how to reuse a previous run.

Data Required
-------------
Full data: ``data/DirLab-4DCT`` (all 10 cases, converted to HU ``.mha`` by
``data/DirLab-4DCT/fix_downloaded_data.py``)
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
    RegisterImagesGreedyICON,
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

    test_mode = TestTools.running_as_test()
    if test_mode:
        data_dir = repo_root / "data" / "test" / "DirLab-4DCT"
        number_of_iterations_greedy: Optional[list[int]] = [1, 0]
        epochs = 5
    else:
        data_dir = repo_root / "data" / "DirLab-4DCT"
        number_of_iterations_greedy = None  # Greedy defaults
        epochs = 100

    log_level = logging.INFO
    reporter = PhysioTwin4DBase(class_name=class_name, log_level=log_level)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Held-out evaluation pair (Case 1 is excluded from finetuning)
    fixed_file = data_dir / "Case1Pack_T70.mha"
    moving_file = data_dir / "Case1Pack_T30.mha"
    missing = [str(p) for p in (fixed_file, moving_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing DirLab phase images: {missing}.\n"
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
    if experiment_dir.exists():
        reporter.log_info("Removing previous finetuning outputs: %s", experiment_dir)
        shutil.rmtree(experiment_dir)

    # DIR-Lab ships no segmentations, so no labelmaps or masks are supplied and
    # the Dice loss must be disabled: uniGradICON requires a ``segmentation``
    # field on every dataset entry when dice_loss_weight > 0.
    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=list(subject_image_files.values()),
        output_dir=weights_dir,
        finetune_name=finetune_name,
        subject_ids=list(subject_image_files.keys()),
        epochs=epochs,
        dice_loss_weight=0.0,
        log_level=log_level,
    )
    weights_path = workflow.process()

    # Registration comparison
    fixed_image = itk.imread(str(fixed_file), pixel_type=itk.F)
    moving_image = itk.imread(str(moving_file), pixel_type=itk.F)
    transform_tools = TransformTools()

    # Every image is segmented independently: the fixed image once, then each
    # registered image after warping.  The Dice scores therefore include
    # whatever the segmenter does differently on each warped volume, not only
    # the geometric error of the transform.
    segmenter = SegmentNVSegmentCTMRI(log_level=log_level)
    fixed_labelmap = segmenter.segment(fixed_image)["labelmap"]
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

    # Reference row: the moving image on the fixed grid, unregistered.
    unregistered_image = itk.resample_image_filter(
        moving_image,
        ReferenceImage=fixed_image,
        UseReferenceImage=True,
    )

    registered_images: dict[str, itk.Image] = {}
    labelmaps: dict[str, itk.Image] = {
        "unregistered": segmenter.segment(unregistered_image)["labelmap"]
    }
    rows: list[dict[str, Any]] = [
        {
            "method": "unregistered",
            "weights": "-",
            "registration_time_s": None,
            "loss": None,
            **overlap_metrics(labelmaps["unregistered"]),
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
            if number_of_iterations_greedy is not None:
                registrar.set_number_of_iterations(number_of_iterations_greedy)
        else:
            registrar = RegisterImagesGreedyICON(log_level=log_level)
            if number_of_iterations_greedy is not None:
                registrar.greedy.set_number_of_iterations(number_of_iterations_greedy)
            # None, not 0: icon_registration rejects 0 and takes None to mean
            # "no test-time finetuning steps", so the comparison reflects what
            # each set of weights predicts rather than per-pair optimization.
            registrar.icon.set_number_of_iterations(None)
            registrar.icon.set_mass_preservation(True)  # For non-contrast CT
            if method_weights is not None:
                registrar.icon.set_weights_path(str(method_weights))
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)

        start_time = time.perf_counter()
        result = registrar.register(moving_image)
        elapsed_s = time.perf_counter() - start_time

        registered = transform_tools.transform_image(
            moving_image, result["forward_transform"], fixed_image
        )
        registered_images[method_name] = registered
        labelmaps[method_name] = segmenter.segment(registered)["labelmap"]
        rows.append(
            {
                "method": method_name,
                "weights": str(method_weights) if method_weights else "-",
                "registration_time_s": elapsed_s,
                "loss": float(result["loss"]),
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
        "Case1Pack_T30 -> Case1Pack_T70, per-class Dice against the fixed labelmap"
    )
    reporter.log_info(
        "  %-13s %7s %7s %7s %7s %7s %7s %7s %12s %9s",
        "method",
        "classes",
        "mean",
        "p5",
        "median",
        "p95",
        "min",
        "max",
        "mislabeled",
        "time_s",
    )
    for row in rows:
        elapsed = row["registration_time_s"]
        reporter.log_info(
            "  %-13s %7d %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %12d %9s",
            row["method"],
            row["n_classes"],
            row["dice_mean"],
            row["dice_p5"],
            row["dice_median"],
            row["dice_p95"],
            row["dice_min"],
            row["dice_max"],
            row["mislabeled_voxels"],
            "-" if elapsed is None else f"{float(elapsed):.1f}",
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
