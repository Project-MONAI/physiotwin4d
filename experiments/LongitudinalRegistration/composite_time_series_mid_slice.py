"""Create a mid-slice composite volume from a directory of MHA images."""

import argparse
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import itk
import numpy as np

DEFAULT_IMAGE_REGEX = r"^pm00.*_init\.mha$"
OUTPUT_FILENAME = "composite.mha"


def select_directory() -> Optional[Path]:
    """Open a directory chooser and return the selected directory."""
    try:
        root = tk.Tk()
    except tk.TclError:
        return None
    try:
        root.withdraw()
        root.update()
        selected_dir = filedialog.askdirectory(
            title="Select directory containing time-series MHA images"
        )
    finally:
        root.destroy()
    if not selected_dir:
        return None
    return Path(selected_dir)


def find_image_files(input_dir: Path, image_regex: str) -> list[Path]:
    """Return sorted image paths whose filename matches ``image_regex``."""
    pattern = re.compile(image_regex)
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and pattern.fullmatch(path.name)
    )


def extract_middle_slice(image_path: Path) -> tuple[np.ndarray, itk.Image]:
    """Read ``image_path`` and return its middle Z slice and ITK image."""
    image = itk.imread(str(image_path))
    image_array = itk.array_from_image(image)
    if image_array.ndim != 3:
        raise ValueError(
            f"Expected 3D image at {image_path}, got array shape {image_array.shape}"
        )
    middle_slice_index = image_array.shape[0] // 2
    return image_array[middle_slice_index, :, :], image


def create_composite_volume(image_files: list[Path]) -> itk.Image:
    """Stack middle Z slices from 3D ITK images into a composite volume."""
    if not image_files:
        raise ValueError("No input images were provided")

    middle_slices: list[np.ndarray] = []
    first_image: Optional[itk.Image] = None
    expected_shape: Optional[tuple[int, ...]] = None
    for image_path in image_files:
        middle_slice, image = extract_middle_slice(image_path)
        if first_image is None:
            first_image = image
            expected_shape = middle_slice.shape
        elif middle_slice.shape != expected_shape:
            raise ValueError(
                f"Middle slice shape mismatch for {image_path}: "
                f"expected {expected_shape}, got {middle_slice.shape}"
            )
        middle_slices.append(middle_slice)

    composite_array = np.stack(middle_slices, axis=0)
    composite_image = itk.image_from_array(composite_array)
    if first_image is None:
        raise ValueError("No input images were read")

    input_spacing = first_image.GetSpacing()
    composite_image.SetSpacing((float(input_spacing[0]), float(input_spacing[1]), 1.0))
    return composite_image


def adjacent_slice_rmse(composite_array: np.ndarray) -> list[float]:
    """Return RMSE values between each adjacent slice in ``composite_array``."""
    if composite_array.ndim != 3:
        raise ValueError(
            f"Expected 3D composite array, got shape {composite_array.shape}"
        )

    rmse_values: list[float] = []
    for slice_index in range(composite_array.shape[0] - 1):
        difference = composite_array[slice_index + 1].astype(
            np.float64
        ) - composite_array[slice_index].astype(np.float64)
        rmse_values.append(float(np.sqrt(np.mean(difference**2))))
    return rmse_values


def print_adjacent_slice_rmse(composite_array: np.ndarray) -> None:
    """Print per-pair and total adjacent-slice RMSE values."""
    rmse_values = adjacent_slice_rmse(composite_array)
    for slice_index, rmse_value in enumerate(rmse_values):
        print(f"RMSE slice {slice_index} to {slice_index + 1}: {rmse_value:.6g}")
    print(f"Total adjacent-slice RMSE: {sum(rmse_values):.6g}")


def write_composite(input_dir: Path, image_regex: str) -> Path:
    """Create ``composite.mha`` in ``input_dir`` from matching image files."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    image_files = find_image_files(input_dir, image_regex)
    if not image_files:
        raise FileNotFoundError(
            f"No image files in {input_dir} matched regex {image_regex!r}"
        )

    composite_image = create_composite_volume(image_files)
    print_adjacent_slice_rmse(itk.array_from_image(composite_image))
    output_path = input_dir / OUTPUT_FILENAME
    itk.imwrite(composite_image, str(output_path), compression=True)
    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    """Run the mid-slice composite volume command."""
    parser = argparse.ArgumentParser(
        description=(
            "Create composite.mha from middle Z slices of images matching a regex."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory containing input images. Opens a dialog when omitted.",
    )
    parser.add_argument(
        "--regex",
        default=DEFAULT_IMAGE_REGEX,
        help=f"Filename regex to match input images. Default: {DEFAULT_IMAGE_REGEX}",
    )
    args = parser.parse_args(argv)

    input_dir = args.directory
    if input_dir is None:
        input_dir = select_directory()
    if input_dir is None:
        print("No directory selected")
        return 1

    output_path = write_composite(input_dir, args.regex)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
