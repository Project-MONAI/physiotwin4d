#!/usr/bin/env python
"""Command-line interface for downloading PhysioTwin4D example data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..data_download_tools import DataDownloadTools

SLICER_HEART_CT = "Slicer-Heart-CT"


def main(argv: Optional[list[str]] = None) -> int:
    """Download a supported PhysioTwin4D example dataset."""
    parser = argparse.ArgumentParser(
        description="Download PhysioTwin4D example data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s
  %(prog)s {SLICER_HEART_CT} --directory data/Slicer-Heart-CT
        """,
    )
    parser.add_argument(
        "data_name",
        nargs="?",
        choices=[SLICER_HEART_CT],
        default=SLICER_HEART_CT,
        help=f"Dataset to download (default: {SLICER_HEART_CT})",
    )
    parser.add_argument(
        "--directory",
        default=f"data/{SLICER_HEART_CT}",
        help=f"Directory where data will be stored (default: data/{SLICER_HEART_CT})",
    )

    args = parser.parse_args(argv)
    output_dir = Path(args.directory)

    if args.data_name == SLICER_HEART_CT:
        data_file = DataDownloadTools.DownloadSlicerHeartCTData(output_dir)
        print(f"Downloaded {SLICER_HEART_CT} to: {data_file}")
        return 0

    parser.error(f"Unsupported dataset: {args.data_name}")


if __name__ == "__main__":
    sys.exit(main())
