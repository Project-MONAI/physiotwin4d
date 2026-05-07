"""
Tutorial 5: VTK Surface Series to Animated USD

Purpose
-------
Convert one or more VTK surface files into a USD file suitable for NVIDIA
Omniverse. Supports a single static mesh, a time-series (animated) set of
meshes, or a mesh with scalar data visualised via a colormap. This tutorial
uses the surface files produced by Tutorial 2 (CT Segmentation to VTK) as
input, but any VTK/VTP/VTU files will work.

Inputs
------
- One or more VTK-compatible surface files (``.vtp`` / ``.vtk`` / ``.vtu``).
  This tutorial looks for ``output/tutorial_02/patient_surfaces.vtp`` (output
  of Tutorial 2). If that file does not exist, it falls back to any ``.vtp``
  under ``data/``.

Outputs
-------
- ``output_dir/surfaces.usd`` - USD file with anatomy materials applied
- Screenshots (PNG):
  - ``usd_mesh_rendering.png`` - PyVista off-screen render of the mesh

Strengths
---------
- Supports time-varying USD for animated sequences (one VTK file per frame).
- Three appearance modes: ``solid`` (flat colour), ``anatomy`` (clinical
  material by anatomy type), and ``colormap`` (scalar field visualisation).
- Coordinate system is automatically converted from RAS to Omniverse Y-up.

Weaknesses / Limitations
------------------------
- Requires Tutorial 2 output (or any VTK file) as input; not standalone.
- Time-series ordering relies on a filename regex pattern (``\.t\d+\.vtp$``);
  non-conforming filenames are treated as static single-frame input.
- USD materials use UsdPreviewSurface; advanced Omniverse MDL materials require
  additional post-processing.

Classes Used
------------
- WorkflowConvertVTKToUSD (workflow_convert_vtk_to_usd.py):
    Loads VTK files, splits meshes, converts to USD, and applies appearance.
- ConvertVTKToUSD (convert_vtk_to_usd.py):
    High-level PyVista-to-USD converter with colormap support (used internally).
- USDAnatomyTools (usd_anatomy_tools.py):
    Applies clinical material colours to USD prims (used internally).

CLI Equivalent
--------------
The same main outputs (without screenshots) can be produced via the CLI::

    physiomotion4d-convert-vtk-to-usd \\
        --input output/tutorial_02/patient_surfaces.vtp \\
        --output-usd ./output/tutorial_05/surfaces.usd \\
        --appearance anatomy \\
        --anatomy-type heart

See ``src/physiomotion4d/cli/convert_vtk_to_usd.py`` for full CLI documentation.

Data Required
-------------
This tutorial uses the output of Tutorial 2 (``output/tutorial_02/patient_surfaces.vtp``).
Run Tutorial 2 first, or provide any VTK surface file via the ``--vtk-file`` flag.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import pyvista as pv

from physiomotion4d.test_tools import TestTools
from physiomotion4d.workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD


def run_tutorial(
    data_dir: Path,
    output_dir: Path,
    *,
    vtk_file: Optional[Path] = None,
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Run Tutorial 5: VTK Surface Series to Animated USD.

    Args:
        data_dir: Root of the ``data/`` directory (see data/README.md).
        output_dir: Directory to write outputs and screenshots.
        vtk_file: Explicit path to a VTK file; overrides auto-discovery.
        log_level: Python logging level.

    Returns:
        dict with keys:

        - ``'usd_file'`` (str): path to the output USD file.
        - ``'screenshots'`` (list[Path]): paths to saved PNG screenshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if vtk_file is None:
        # Prefer Tutorial 2 output
        project_root = data_dir.parent
        candidate = project_root / "output" / "tutorial_02" / "patient_surfaces.vtp"
        if candidate.exists():
            vtk_file = candidate
        else:
            # Fall back to any .vtp under data/
            found = list(data_dir.rglob("*.vtp"))
            if not found:
                raise FileNotFoundError(
                    "No VTK file found. Run Tutorial 2 first, or specify "
                    "--vtk-file <path>."
                )
            vtk_file = found[0]

    output_usd = output_dir / "surfaces.usd"

    workflow = WorkflowConvertVTKToUSD(
        vtk_files=[vtk_file],
        output_usd=output_usd,
        appearance="anatomy",
        anatomy_type="heart",
        separate_by_connectivity=True,
        log_level=log_level,
    )
    usd_path = workflow.run()

    # Screenshots
    tt = TestTools(
        results_dir=output_dir,
        baselines_dir=output_dir / "baselines",
        class_name="tutorial_05",
        results_output_dir=output_dir,
        log_level=log_level,
    )

    screenshots: list[Path] = []

    mesh = pv.read(str(vtk_file))
    screenshots.append(
        tt.save_screenshot_mesh(
            mesh,
            "usd_mesh_rendering.png",
            camera_position="iso",
            color="lightcoral",
            opacity=0.9,
        )
    )

    return {"usd_file": usd_path, "screenshots": screenshots}


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
        default=Path("output") / "tutorial_05",
        help="Output directory (default: ./output/tutorial_05)",
    )
    parser.add_argument(
        "--vtk-file",
        type=Path,
        default=None,
        help="Explicit VTK input file (default: auto-discover from Tutorial 2 output)",
    )
    args = parser.parse_args()

    results = run_tutorial(args.data_dir, args.output_dir, vtk_file=args.vtk_file)
    print(f"USD file:    {results['usd_file']}")
    print(f"Screenshots: {[str(p) for p in results['screenshots']]}")
