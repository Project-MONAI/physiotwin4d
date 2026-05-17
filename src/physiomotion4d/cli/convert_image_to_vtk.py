#!/usr/bin/env python
"""Command-line interface for the image-to-VTK segmentation workflow.

Segments a 3D image using a chosen backend and writes per-anatomy-group VTP
surfaces and VTU voxel meshes annotated with anatomy labels and colors.
"""

import argparse
import os
import sys
import traceback

ANATOMY_GROUPS = (
    "heart",
    "lung",
    "major_vessels",
    "bone",
    "soft_tissue",
    "other",
    "contrast",
)

SEGMENTATION_METHODS = ("ChestTotalSegmentator", "HeartSimpleware")


def main() -> int:
    """CLI entry point for image to VTK conversion."""
    parser = argparse.ArgumentParser(
        description="Segment a 3D image and export anatomy groups as VTK surfaces and meshes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Anatomy groups
--------------
  heart, lung, major_vessels, bone, soft_tissue, other, contrast
  (empty groups are skipped automatically)

Output files — combined mode (default)
---------------------------------------
  {prefix}_surfaces.vtp   all surfaces merged into one file
  {prefix}_meshes.vtu     all voxel meshes merged into one file

Output files — split mode (--split-files)
------------------------------------------
  {prefix}_{group}.vtp    one surface per anatomy group
  {prefix}_{group}.vtu    one voxel mesh per anatomy group

Examples
--------
  # Segment with TotalSegmentator, combined output
  %(prog)s \\
    --input-image chest_ct.nii.gz \\
    --output-dir ./results

  # Simpleware heart-only, cardiac anatomy groups, combined output
  %(prog)s \\
    --input-image chest_ct.nii.gz \\
    --segmentation-method HeartSimpleware \\
    --anatomy-groups heart major_vessels \\
    --output-dir ./results \\
    --output-prefix patient01

  # Also save the ITK segmentation labelmap
  %(prog)s \\
    --input-image chest_ct.nii.gz \\
    --output-dir ./results \\
    --save-labelmap
        """,
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--input-image",
        required=True,
        help="Path to the input 3D image (.nii.gz, .nrrd, .mha, …).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files (created if absent).",
    )

    # ── Segmentation ──────────────────────────────────────────────────────
    parser.add_argument(
        "--segmentation-method",
        default="ChestTotalSegmentator",
        choices=list(SEGMENTATION_METHODS),
        help=(
            "Segmentation backend.  ChestTotalSegmentator (default) | HeartSimpleware"
        ),
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        default=False,
        help="Enable contrast-enhanced blood segmentation (default: disabled).",
    )
    parser.add_argument(
        "--anatomy-groups",
        nargs="+",
        metavar="GROUP",
        choices=list(ANATOMY_GROUPS),
        default=None,
        help=(
            "Anatomy groups to extract.  Default: all non-empty groups.  "
            "Choices: " + " ".join(ANATOMY_GROUPS)
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Filename prefix for output files (default: no prefix).",
    )
    parser.add_argument(
        "--split-files",
        action="store_true",
        default=False,
        help=(
            "Write one VTP and one VTU file per anatomy group instead of "
            "merging all groups into a single VTP and VTU (default: combined)."
        ),
    )
    parser.add_argument(
        "--save-labelmap",
        action="store_true",
        default=False,
        help="Also save the detailed per-structure segmentation labelmap as a NIfTI file.",
    )

    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.exists(args.input_image):
        print(f"Error: input image not found: {args.input_image}")
        return 1

    # ── Load image ─────────────────────────────────────────────────────────
    print(f"Loading input image: {args.input_image}")
    try:
        import itk

        input_image = itk.imread(args.input_image)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Error loading image: {exc}")
        traceback.print_exc()
        return 1

    # ── Run workflow ────────────────────────────────────────────────────────
    print(f"Segmentation method : {args.segmentation_method}")
    print(f"Contrast enhanced   : {args.contrast}")
    print(f"Anatomy groups      : {args.anatomy_groups or 'all'}")
    print("=" * 70)

    try:
        from physiomotion4d import WorkflowConvertImageToVTK

        workflow = WorkflowConvertImageToVTK(
            segmentation_method=args.segmentation_method,
        )
        result = workflow.run_workflow(
            input_image=input_image,
            contrast_enhanced_study=args.contrast,
            anatomy_groups=args.anatomy_groups,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error during workflow: {exc}")
        traceback.print_exc()
        return 1

    surfaces = result["surfaces"]
    meshes = result["meshes"]

    if not surfaces and not meshes:
        print("No anatomy groups produced any output.  Check the input image.")
        return 1

    # ── Save results ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.output_prefix

    try:
        if args.split_files:
            # One file per anatomy group
            if surfaces:
                saved_surfaces = WorkflowConvertImageToVTK.save_surfaces(
                    surfaces, args.output_dir, prefix=prefix
                )
                for group, path in saved_surfaces.items():
                    print(f"  Surface  [{group:15s}] → {path}")
            if meshes:
                saved_meshes = WorkflowConvertImageToVTK.save_meshes(
                    meshes, args.output_dir, prefix=prefix
                )
                for group, path in saved_meshes.items():
                    print(f"  Mesh     [{group:15s}] → {path}")
        else:
            # Combined single-file output
            if surfaces:
                surface_file = WorkflowConvertImageToVTK.save_combined_surface(
                    surfaces, args.output_dir, prefix=prefix
                )
                print(f"  Combined surface → {surface_file}")
            if meshes:
                mesh_file = WorkflowConvertImageToVTK.save_combined_mesh(
                    meshes, args.output_dir, prefix=prefix
                )
                print(f"  Combined mesh    → {mesh_file}")

        if args.save_labelmap:
            labelmap = result["labelmap"]
            stem = f"{prefix}_labelmap" if prefix else "labelmap"
            labelmap_file = os.path.join(args.output_dir, f"{stem}.nii.gz")
            itk.imwrite(labelmap, labelmap_file)
            print(f"  Labelmap         → {labelmap_file}")

    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error saving results: {exc}")
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Conversion completed successfully.")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
