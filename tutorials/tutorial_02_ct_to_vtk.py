"""
Tutorial 2: CT Segmentation to VTK Surfaces

Purpose
-------
Segment a 3D CT image into anatomical groups (heart, lungs, vessels, bone,
soft tissue) and export per-group VTK surface and mesh files. Each output mesh
is annotated with anatomy metadata and colour so it can be used directly in
PyVista, ParaView, or the downstream USD pipeline (Tutorial 5).

Inputs
------
- A single 3D CT image in any ITK-readable format (NIfTI, MHA, NRRD, etc.).
  This tutorial uses one time frame from the Slicer-Heart-CT dataset.
  Expected location: ``data/Slicer-Heart-CT/`` (any ``slice_???.mha`` frame).

Outputs
-------
- ``output_dir/patient_surfaces.vtp`` - all anatomy surfaces in one file
- ``output_dir/patient_meshes.vtu`` - all voxel meshes in one file
- Screenshots (PNG):
  - ``segmentation_overlay.png`` - segmentation mask overlaid on axial CT slice
  - ``vtk_surfaces.png`` - 3-D isometric view of the combined surface

Strengths
---------
- One call to ``WorkflowConvertCTToVTK.run_workflow()`` handles segmentation and
  mesh extraction for all anatomy groups in a single pass.
- Each output mesh carries field data (group name, label IDs, colour) for
  downstream tools.
- Combined-file output (default) produces a single VTP/VTU rather than one file
  per group, simplifying downstream handling.

Weaknesses / Limitations
------------------------
- TotalSegmentator requires ~8 GB GPU VRAM for full segmentation; CPU fallback
  is available but much slower (~30 min per volume).
- Small or unusual anatomical structures (e.g., pediatric heart) may be partially
  missed by the default TotalSegmentator model.
- Output mesh resolution is governed by the input CT voxel size; coarse scans
  yield coarser meshes.

Classes Used
------------
- WorkflowConvertCTToVTK (workflow_convert_ct_to_vtk.py):
    Segments a CT image and extracts per-anatomy-group VTK surfaces and meshes.
- SegmentChestTotalSegmentator (segment_chest_total_segmentator.py):
    Deep-learning segmentation backend (used internally).
- ContourTools (contour_tools.py):
    Mesh extraction via marching cubes (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Provides anatomy group colours for mesh annotation (used internally).

CLI Equivalent
--------------
The same main outputs (without screenshots) can be produced via the CLI::

    physiomotion4d-convert-ct-to-vtk \\
        --input-image data/Slicer-Heart-CT/slice_000.mha \\
        --output-dir ./output/tutorial_02 \\
        --output-prefix patient \\
        --contrast

See ``src/physiomotion4d/cli/convert_ct_to_vtk.py`` for full CLI documentation.

Data Required
-------------
See data/README.md for download instructions and dataset licensing.
Dataset: Slicer-Heart-CT - https://github.com/Slicer-Heart-CT/Slicer-Heart-CT
Auto-download: the conftest fixture downloads
``data/test/TruncalValve_4DCT.seq.nrrd`` and extracts frames as
``data/test/slice_???.mha``. For this tutorial the full dataset is at
``data/Slicer-Heart-CT/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import itk
import pyvista as pv

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_convert_ct_to_vtk import WorkflowConvertCTToVTK


def run_tutorial(
    data_dir: Path,
    output_dir: Path,
    *,
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Run Tutorial 2: CT Segmentation to VTK Surfaces.

    Args:
        data_dir: Root of the ``data/`` directory (see data/README.md).
        output_dir: Directory to write outputs and screenshots.
        log_level: Python logging level.

    Returns:
        dict with keys:

        - ``'result'`` (dict): workflow result dict with ``'surfaces'`` and
          ``'meshes'`` entries (pv.PolyData / pv.UnstructuredGrid per group).
        - ``'surface_file'`` (Path): path to the combined ``.vtp`` file.
        - ``'mesh_file'`` (Path): path to the combined ``.vtu`` file.
        - ``'screenshots'`` (list[Path]): paths to saved PNG screenshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer full dataset; fall back to test cache
    candidates = list((data_dir / "Slicer-Heart-CT").glob("slice_???.mha"))
    if not candidates:
        candidates = list((data_dir / "test").glob("slice_???_sml.mha"))
    if not candidates:
        raise FileNotFoundError(
            "No CT frame found under data/Slicer-Heart-CT/ or data/test/.\n"
            "See data/README.md for download instructions."
        )
    candidates.sort()
    ct_file = candidates[0]

    ct_image = itk.imread(str(ct_file))

    workflow = WorkflowConvertCTToVTK(
        segmentation_method="total_segmentator",
        log_level=log_level,
    )
    result = workflow.run_workflow(
        input_image=ct_image,
        contrast_enhanced_study=True,
    )

    surface_file = output_dir / "patient_surfaces.vtp"
    mesh_file = output_dir / "patient_meshes.vtu"
    WorkflowConvertCTToVTK.save_combined_surface(
        result["surfaces"], str(output_dir), prefix="patient"
    )
    WorkflowConvertCTToVTK.save_combined_mesh(
        result["meshes"], str(output_dir), prefix="patient"
    )

    # Screenshots
    tt = TestTools(
        results_dir=output_dir,
        baselines_dir=output_dir / "baselines",
        class_name="tutorial_02",
        results_output_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []

    # Segmentation overlay on axial CT slice
    labelmap = result.get("labelmap")
    screenshots.append(
        tt.save_screenshot_image_slice(
            ct_image,
            "segmentation_overlay.png",
            axis=0,
            slice_fraction=0.5,
            colormap="gray",
            vmin=-200,
            vmax=600,
            overlay_mask=labelmap,
        )
    )

    # 3-D view of combined surface
    surfaces = [s for s in result["surfaces"].values() if s is not None]
    if surfaces:
        combined = pv.merge(surfaces) if len(surfaces) > 1 else surfaces[0]
        screenshots.append(
            tt.save_screenshot_mesh(
                combined,
                "vtk_surfaces.png",
                camera_position="iso",
                color="lightblue",
                opacity=0.85,
            )
        )

    return {
        "result": result,
        "surface_file": surface_file,
        "mesh_file": mesh_file,
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
        default=Path("output") / "tutorial_02",
        help="Output directory (default: ./output/tutorial_02)",
    )
    args = parser.parse_args()

    results = run_tutorial(args.data_dir, args.output_dir)
    print(f"Surface file: {results['surface_file']}")
    print(f"Mesh file:    {results['mesh_file']}")
    print(f"Screenshots:  {[str(p) for p in results['screenshots']]}")
