"""
Tutorial 6: Reconstruct High-Resolution 4D CT

Purpose
-------
Reconstruct a high-resolution dynamic 4D CT volume from a time series of
lower-resolution or sparse CT frames. The workflow registers each time frame
to a high-resolution reference image, producing a sequence of reconstructed
volumes that share the spatial resolution of the reference. This is useful for
respiratory-gated lung CT (DirLab-4DCT) where breath-hold reference scans are
available alongside lower-quality respiratory-phase images.

Inputs
------
- A list of 3D CT images (``itk.Image``): the time series to reconstruct.
  Expected location: ``data/DirLab-4DCT/Case1/`` (T00-T90 phases).
- A high-resolution fixed reference image (``itk.Image``):
  the target space for reconstruction.
  Expected location: ``data/DirLab-4DCT/Case1/`` (any phase used as reference).

Note
----
The DirLab-4DCT sample data used by this tutorial does not include a separate
high-resolution breath-hold reference image. For demonstration and regression
testing, the tutorial uses one available DirLab respiratory phase as the fixed
reference, so the reconstructed outputs inherit that phase image's resolution
rather than true higher-resolution reference spacing.

Outputs
-------
- ``output_dir/reconstructed_frame_<N>.mha`` - one reconstructed 3D image per frame
- Screenshots (PNG):
  - ``reference_frame.png`` - axial slice of the high-resolution reference image
  - ``reconstructed_frame.png`` - axial slice of the first reconstructed frame

Strengths
---------
- Bidirectional propagation of registration from the reference frame reduces
  accumulated error for frames far from the reference.
- Temporal smoothing via ``prior_weight`` parameter reduces frame-to-frame jitter.
- Supports ``'ants'``, ``'icon'``, and ``'ants_icon'`` (two-stage) registration.

Weaknesses / Limitations
------------------------
- Requires the DirLab-4DCT dataset (manual download; see data/README.md).
- ICON registration (default part of ``'ants_icon'``) requires a GPU.
- Reconstruction quality is bounded by the accuracy of the registration; large
  respiratory excursion between phases can cause residual artefacts.
- Runtime is proportional to the number of frames times registration cost.

Classes Used
------------
- WorkflowReconstructHighres4DCT (workflow_reconstruct_highres_4d_ct.py):
    Registers each time frame to the fixed reference and reconstructs volumes.
- RegisterTimeSeriesImages (register_time_series_images.py):
    Chains frame-to-frame registration with optional temporal smoothing (used
    internally).
- RegisterImagesANTs / RegisterImagesICON (register_images_ants.py / _icon.py):
    Individual frame registration backends (used internally).

CLI Equivalent
--------------
The same main outputs (without screenshots) can be produced via the CLI::

    physiomotion4d-reconstruct-highres-4d-ct \\
        --time-series-dir data/DirLab-4DCT/Case1 \\
        --fixed-image data/DirLab-4DCT/Case1/case1_T00.mhd \\
        --registration-method ants \\
        --output-dir ./output/tutorial_06

See ``src/physiomotion4d/cli/reconstruct_highres_4d_ct.py`` for full CLI
documentation.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: DirLab 4D-CT - https://www.dir-lab.com/ReferenceData.html
Manual download required. Place files under ``data/DirLab-4DCT/`` as described
in data/README.md.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import itk

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_reconstruct_highres_4d_ct import (
    WorkflowReconstructHighres4DCT,
)


def run_tutorial(
    data_dir: Path,
    output_dir: Path,
    *,
    case: int = 1,
    max_frames: int = 4,
    registration_method: str = "ants",
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Run Tutorial 6: Reconstruct High-Resolution 4D CT.

    Args:
        data_dir: Root of the ``data/`` directory (see data/README.md).
        output_dir: Directory to write outputs and screenshots.
        case: DirLab case number (1-10). Default: 1.
        max_frames: Maximum number of time frames to reconstruct (for speed).
        registration_method: ``'ants'`` (CPU-capable) or ``'icon'`` (GPU) or
            ``'ants_icon'`` (two-stage). Default: ``'ants'``.
        log_level: Python logging level.

    Returns:
        dict with keys:

        - ``'reconstructed_images'`` (list[itk.Image]): reconstructed volumes.
        - ``'reconstructed_files'`` (list[Path]): saved ``.mha`` paths.
        - ``'screenshots'`` (list[Path]): paths to saved PNG screenshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dirlab_dir = data_dir / "DirLab-4DCT"
    case_dir = dirlab_dir / f"Case{case}"

    # Discover phase images (MetaImage .mhd or .mha)
    phase_files = sorted(list(case_dir.glob("*.mhd")) + list(case_dir.glob("*.mha")))
    if not phase_files:
        phase_files = sorted(
            list(dirlab_dir.glob(f"Case{case}Pack_T*.mhd"))
            + list(dirlab_dir.glob(f"Case{case}Pack_T*.mha"))
        )
    if not phase_files:
        raise FileNotFoundError(
            f"No .mhd / .mha files found for DirLab-4DCT case {case} under "
            f"{case_dir} or {dirlab_dir}.\n"
            "See data/README.md for manual download instructions."
        )

    phase_files = phase_files[:max_frames]
    time_series = [itk.imread(str(f)) for f in phase_files]
    fixed_image = time_series[0]  # use first phase as high-res reference

    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=time_series,
        fixed_image=fixed_image,
        reference_frame=0,
        registration_method=registration_method,
        log_level=log_level,
    )
    workflow.set_modality("ct")
    result = workflow.run_workflow()

    reconstructed: list[itk.Image] = result["reconstructed_images"]
    reconstructed_files: list[Path] = []
    for i, vol in enumerate(reconstructed):
        out_path = output_dir / f"reconstructed_frame_{i:03d}.mha"
        itk.imwrite(vol, str(out_path), compression=True)
        reconstructed_files.append(out_path)

    # Screenshots
    tt = TestTools(
        results_dir=output_dir,
        baselines_dir=output_dir / "baselines",
        class_name="tutorial_06",
        results_output_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []

    screenshots.append(
        tt.save_screenshot_image_slice(
            fixed_image,
            "reference_frame.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
        )
    )

    if reconstructed:
        screenshots.append(
            tt.save_screenshot_image_slice(
                reconstructed[0],
                "reconstructed_frame.png",
                axis=0,
                slice_fraction=0.5,
                colormap="gray",
            )
        )

    return {
        "reconstructed_images": reconstructed,
        "reconstructed_files": reconstructed_files,
        "screenshots": screenshots,
    }


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
        default=Path("output") / "tutorial_06",
        help="Output directory (default: ./output/tutorial_06)",
    )
    parser.add_argument(
        "--case",
        type=int,
        default=1,
        choices=list(range(1, 11)),
        help="DirLab case number 1-10 (default: 1)",
    )
    parser.add_argument(
        "--registration-method",
        default="ants",
        choices=["ants", "icon", "ants_icon"],
        help="Registration method (default: ants)",
    )
    args = parser.parse_args()

    results = run_tutorial(
        args.data_dir,
        args.output_dir,
        case=args.case,
        registration_method=args.registration_method,
    )
    print(f"Reconstructed frames: {[str(p) for p in results['reconstructed_files']]}")
    print(f"Screenshots:          {[str(p) for p in results['screenshots']]}")
