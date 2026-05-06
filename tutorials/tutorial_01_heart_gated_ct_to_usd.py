"""
Tutorial 1: Heart-Gated CT to Animated USD

Purpose
-------
Convert a 4D cardiac CT scan (multiple gated time frames) into an animated USD
model suitable for visualization in NVIDIA Omniverse. The workflow segments the
heart and surrounding anatomy from a reference frame, registers all other frames
to that reference using deep learning or classical registration, and assembles
the resulting time-varying surface meshes into a single USD file with anatomical
materials applied.

Inputs
------
- A 4D NRRD sequence file (``*.seq.nrrd``) **or** a list of 3D CT volumes
  (``*.mha`` / ``*.nrrd``) representing successive cardiac phases.
  Expected location: ``data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd``
- Optional: a reference frame image to fix the cardiac phase used as the
  segmentation source.

Outputs
-------
- ``output_dir/cardiac_model.dynamic_anatomy_painted.usd`` - animated USD with
  anatomy materials
- ``output_dir/<phase>_*.vtp`` - per-frame surface meshes (VTK PolyData)
- Screenshots (PNG) for documentation and regression testing:
  - ``reference_frame_axial.png`` - axial slice of the reference CT frame
  - ``segmentation_overlay.png`` - segmentation mask overlaid on reference
  - ``contours_3d.png`` - 3-D isometric view of the reference-frame contours

Strengths
---------
- Single call (``WorkflowConvertHeartGatedCTToUSD.process()``) runs the full pipeline.
- Supports both GPU-accelerated ICON registration and CPU-capable ANTs registration.
- Automatically detects contrast enhancement and adjusts segmentation thresholds.
- Output is Omniverse-ready with anatomical materials (USDAnatomyTools).

Weaknesses / Limitations
------------------------
- Requires a GPU for ICON registration (``registration_method='icon'``); use
  ``registration_method='ants'`` for CPU-only environments (about 10x slower).
- Segmentation quality depends on TotalSegmentator's training distribution;
  unusual pathologies or pediatric anatomy may degrade results.
- Large 4D datasets (>20 phases, high resolution) can require 32 GB+ RAM.

Classes Used
------------
- WorkflowConvertHeartGatedCTToUSD (workflow_convert_heart_gated_ct_to_usd.py):
    Orchestrates the full pipeline: 4D NRRD -> segmentation -> registration ->
    contour extraction -> USD export.
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation of 117 anatomical structures (used internally).
- RegisterImagesICON / RegisterImagesANTs (register_images_icon.py / _ants.py):
    Frame-to-frame image registration (used internally).
- ContourTools (contour_tools.py):
    Extracts and transforms surface meshes from segmentation masks (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Applies clinical material colours to USD prims (used internally).

CLI Equivalent
--------------
The same main outputs (without screenshots) can be produced via the CLI::

    physiomotion4d-heart-gated-ct \\
        data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd \\
        --contrast \\
        --project-name cardiac_model \\
        --registration-method ants \\
        --registration-iterations 1 \\
        --output-dir ./output/tutorial_01

See ``src/physiomotion4d/cli/convert_heart_gated_ct_to_usd.py`` for full CLI
documentation.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/Slicer-Heart-CT/Slicer-Heart-CT
This script expects the data to already exist at
``data/Slicer-Heart-CT/TruncalValve_4DCT.seq.nrrd``. Run the repository data
download notebook or download the file manually before running this tutorial.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import itk

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_convert_heart_gated_ct_to_usd import (
    WorkflowConvertHeartGatedCTToUSD,
)


def run_tutorial(
    data_dir: Path,
    output_dir: Path,
    *,
    registration_method: str = "ants",
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Run Tutorial 1: Heart-Gated CT to Animated USD.

    Args:
        data_dir: Root of the ``data/`` directory (see data/README.md).
        output_dir: Directory to write outputs and screenshots.
        registration_method: ``'ants'`` (CPU-capable, default) or ``'icon'`` (GPU).
        log_level: Python logging level.

    Returns:
        dict with keys:

        - ``'usd_file'`` (str): path to the final painted USD.
        - ``'screenshots'`` (list[Path]): paths to saved PNG screenshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    nrrd_file = data_dir / "Slicer-Heart-CT" / "TruncalValve_4DCT.seq.nrrd"
    if not nrrd_file.exists():
        raise FileNotFoundError(
            f"Slicer-Heart-CT data not found: {nrrd_file}\n"
            "See data/README.md for download instructions."
        )

    workflow = WorkflowConvertHeartGatedCTToUSD(
        input_filenames=[str(nrrd_file)],
        contrast_enhanced=True,
        output_directory=str(output_dir),
        project_name="cardiac_model",
        registration_method=registration_method,
        number_of_registration_iterations=1,
        log_level=log_level,
    )

    usd_file = output_dir / workflow.process()

    # Screenshots
    tt = TestTools(
        results_dir=output_dir,
        baselines_dir=output_dir / "baselines",
        class_name="tutorial_01",
        log_level=log_level,
    )

    screenshots: list[Path] = []

    # Reference frame: the workflow caches 3D frames in output_dir
    ref_frames = sorted(output_dir.glob("slice_???.mha"))
    if ref_frames:
        ref_image = itk.imread(str(ref_frames[0]))
        screenshots.append(
            tt.save_screenshot_image_slice(
                ref_image,
                "reference_frame_axial.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
                vmin=-200,
                vmax=600,
            )
        )

        # Segmentation overlay: look for cached labelmap
        label_files = sorted(output_dir.glob("slice_???_labelmap*.mha"))
        overlay = itk.imread(str(label_files[0])) if label_files else None
        screenshots.append(
            tt.save_screenshot_image_slice(
                ref_image,
                "segmentation_overlay.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
                vmin=-200,
                vmax=600,
                overlay_mask=overlay,
            )
        )

    # 3-D contour view: any .vtp produced by the workflow
    vtp_files = sorted(output_dir.glob("*.vtp"))
    if vtp_files:
        import pyvista as pv

        merged = pv.read(str(vtp_files[0]))
        screenshots.append(
            tt.save_screenshot_mesh(
                merged,
                "contours_3d.png",
                camera_position="iso",
                color="tomato",
                opacity=0.85,
            )
        )

    return {"usd_file": str(usd_file), "screenshots": screenshots}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "tutorial_01",
        help="Output directory (default: ./output/tutorial_01)",
    )
    parser.add_argument(
        "--registration-method",
        default="ants",
        choices=["ants", "icon"],
        help="Registration method: ants (CPU) or icon (GPU). Default: ants",
    )
    args = parser.parse_args()

    results = run_tutorial(
        args.data_dir,
        args.output_dir,
        registration_method=args.registration_method,
    )
    print(f"USD file: {results['usd_file']}")
    print(f"Screenshots: {[str(p) for p in results['screenshots']]}")
