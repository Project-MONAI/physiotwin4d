#!/usr/bin/env python
"""
Command-line interface for the Image-to-USD workflow.

This script provides a CLI to process 4D CT images through the complete workflow,
generating dynamic USD models suitable for visualization in NVIDIA Omniverse.
"""

import argparse
import os
import sys


def main() -> int:
    """Command-line interface for the Image-to-USD workflow."""
    parser = argparse.ArgumentParser(
        description="Process 4D CT images to dynamic USD models for Omniverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single 4D NRRD file
  %(prog)s input_4d.nrrd --contrast --output-dir ./results

  # Process multiple 3D NRRD files as time series
  %(prog)s frame_*.nrrd --output-dir ./results --project-name cardiac

  # Specify reference image and registration iterations
  %(prog)s input.nrrd --reference-image ref.mha --registration-iterations 50

  # Use ANTs registration instead of ICON
  %(prog)s input.nrrd --contrast --registration-method ANTS

  # Use the cardiac-only Simpleware segmentation backend
  %(prog)s input.nrrd --segmentation-method HeartSimpleware
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help=(
            "Input image source(s): a single 4D file (NRRD/NIfTI/MHA/...), "
            "a directory containing a DICOM series (3D or 4D), or a list of "
            "3D files representing a time series."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--project-name",
        default="cardiac_model",
        help="Project name for USD organization (default: cardiac_model)",
    )
    parser.add_argument(
        "--contrast", action="store_true", help="Indicate if study is contrast enhanced"
    )
    parser.add_argument(
        "--reference-image",
        help="Path to reference image file (default: uses 70%% time point)",
    )
    parser.add_argument(
        "--registration-iterations",
        type=int,
        default=1,
        help="Number of registration iterations (default: 1)",
    )
    parser.add_argument(
        "--segmentation-method",
        choices=["ChestTotalSegmentator", "HeartSimpleware"],
        default="ChestTotalSegmentator",
        help=(
            "Segmentation backend to use: ChestTotalSegmentator (default) "
            "or HeartSimpleware."
        ),
    )
    parser.add_argument(
        "--registration-method",
        choices=["ANTS", "ICON"],
        default="ICON",
        help="Registration method to use: ANTS or ICON (default: ICON)",
    )

    args = parser.parse_args()

    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return 1

    # Initialize processor
    print("Initializing Image-to-USD processor...")
    try:
        from physiomotion4d import WorkflowConvertImageToUSD

        processor = WorkflowConvertImageToUSD(
            input_filenames=args.input_files,
            contrast_enhanced=args.contrast,
            output_directory=args.output_dir,
            project_name=args.project_name,
            reference_image_filename=args.reference_image,
            number_of_registration_iterations=args.registration_iterations,
            segmentation_method=args.segmentation_method,
            registration_method=args.registration_method,
        )
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        return 1

    try:
        # Execute complete workflow
        print("\nStarting Image-to-USD processing pipeline...")
        print("=" * 60)
        processor.process()

        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print(f"\nOutput files created in: {args.output_dir}")
        print(f"  - {args.project_name}.dynamic_painted.usd")
        print(f"  - {args.project_name}.static_painted.usd")
        print(f"  - {args.project_name}.all_painted.usd")
        print("\nYou can now open these files in NVIDIA Omniverse.")

        return 0

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
