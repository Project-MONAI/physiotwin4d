"""
Tutorial 7: Finetune uniGradICON on DIR-Lab 4D CT

Purpose
-------
Finetune uniGradICON on every DIR-Lab 4D CT case except Case 1, then register
``Case1Pack_T30.mha`` (moving) to ``Case1Pack_T70.mha`` (fixed) with
``RegisterImagesGreedyICON`` twice: once with the stock uniGradICON weights and
once with the finetuned weights.  Case 1 is never seen during finetuning, so
it is a held-out evaluation pair.

Each registration reports the intensity RMSE (Hounsfield units) between the
fixed image and the registered moving image, plus the wall-clock registration
time.  The pre-registration RMSE is reported as a reference point.

Finetuned weights are written to
``tutorials/network_weights/icon_dirlab_4dct``.  An existing checkpoint there is
reused, so re-running the tutorial skips finetuning.

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
import time
from pathlib import Path
from typing import Optional

import itk
import numpy as np

from physiotwin4d import (
    RegisterImagesGreedyICON,
    TestTools,
    TransformTools,
    WorkflowFinetuneICONRegistration,
)
from physiotwin4d.physiotwin4d_base import PhysioTwin4DBase

# Only run if this script is not imported as a module

# unigradicon finetuning is launched as a subprocess and torch spawns worker
# processes; on Windows the spawn start method re-imports this script in each
# child, so all top-level work stays under the __name__ == "__main__" guard.
if __name__ == "__main__":
    # Data directory specification
    repo_root = Path(__file__).resolve().parent.parent
    tutorials_dir = Path(__file__).resolve().parent

    class_name = "tutorial_07_finetune_icon_dirlab"

    output_dir = tutorials_dir / "output" / "tutorial_07_lung"
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

    # Finetuning. DIR-Lab ships no segmentations, so no labelmaps or masks are
    # supplied and the Dice loss must be disabled: uniGradICON requires a
    # ``segmentation`` field on every dataset entry when dice_loss_weight > 0.
    workflow = WorkflowFinetuneICONRegistration(
        subject_image_files=list(subject_image_files.values()),
        output_dir=weights_dir,
        finetune_name=finetune_name,
        subject_ids=list(subject_image_files.keys()),
        epochs=epochs,
        dice_loss_weight=0.0,
        log_level=log_level,
    )
    weights_path = workflow.expected_weights_path()
    if weights_path.exists():
        reporter.log_info("Reusing existing finetuned weights: %s", weights_path)
    else:
        weights_path = workflow.run_finetuning()

    # Registration with and without the finetuned weights
    fixed_image = itk.imread(str(fixed_file), pixel_type=itk.F)
    moving_image = itk.imread(str(moving_file), pixel_type=itk.F)
    transform_tools = TransformTools()

    fixed_array = itk.array_from_image(fixed_image).astype(np.float64)

    def rmse_to_fixed(image: itk.Image) -> float:
        """RMSE in HU between ``image`` and the fixed image, on the fixed grid."""
        array = itk.array_from_image(image).astype(np.float64)
        return float(np.sqrt(np.mean((array - fixed_array) ** 2)))

    initial_rmse = rmse_to_fixed(
        itk.resample_image_filter(
            moving_image,
            ReferenceImage=fixed_image,
            UseReferenceImage=True,
        )
    )

    registered_images: dict[str, itk.Image] = {}
    rows: list[dict[str, object]] = []
    for method_name, method_weights in (
        ("default", None),
        ("finetuned", weights_path),
    ):
        registrar = RegisterImagesGreedyICON(log_level=log_level)
        if number_of_iterations_greedy is not None:
            registrar.greedy.set_number_of_iterations(number_of_iterations_greedy)
        registrar.icon.set_number_of_iterations(0)
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
        rows.append(
            {
                "method": method_name,
                "weights": str(method_weights) if method_weights else "unigradicon",
                "rmse_hu": rmse_to_fixed(registered),
                "registration_time_s": elapsed_s,
                "loss": float(result["loss"]),
            }
        )

    # Result saving
    for method_name, image in registered_images.items():
        itk.imwrite(
            image,
            str(output_dir / f"registered_{method_name}.mha"),
            compression=True,
        )

    summary_file = output_dir / "registration_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Reporting
    reporter.log_info(
        "Case1Pack_T30 -> Case1Pack_T70, RMSE before registration: %.3f HU",
        initial_rmse,
    )
    for row in rows:
        reporter.log_info(
            "  %-10s RMSE: %8.3f HU   time: %7.1f s   loss: %.6f",
            row["method"],
            row["rmse_hu"],
            row["registration_time_s"],
            row["loss"],
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
        "initial_rmse_hu": initial_rmse,
        "registration_metrics": rows,
        "summary_file": summary_file,
        "registered_images": registered_images,
        "screenshots": screenshots,
    }
