#!/usr/bin/env python
"""
Test for converting a 4D image to a 3D time series using ITK readers.

This test depends on test_download_heart_data and replicates the functionality
from cell 3 of the notebook Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb.
"""

from pathlib import Path

import pytest

from physiomotion4d.convert_image_4d_to_3d import ConvertImage4DTo3D


@pytest.mark.requires_data
class TestConvertImage4DTo3D:
    """Test suite for converting a 4D image to a 3D time series."""

    def test_convert_4d_to_3d(
        self,
        download_test_data: Path,
        test_directories: dict[str, Path],
    ) -> None:
        """Test conversion of 4D image to 3D time series."""
        output_dir = test_directories["output"] / "convert_image_4d_to_3d"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_4d_file = download_test_data

        print("\nConverting 4D image to 3D time series...")
        conv = ConvertImage4DTo3D()
        conv.load_image_4d(str(input_4d_file))
        conv.save_3d_images(output_dir, "slice")

        slice_007 = output_dir / "slice_007.mha"
        assert slice_007.exists(), f"Expected slice file not created: {slice_007}"

        slice_files = list(output_dir.glob("slice_*.mha"))
        print(f"Created {len(slice_files)} slice files")
        assert len(slice_files) > 0, "No slice files were created"

    def test_slice_files_created(
        self,
        download_test_data: Path,
        test_directories: dict[str, Path],
    ) -> None:
        """Test that all expected slice files are present after conversion."""
        output_dir = test_directories["output"] / "convert_image_4d_to_3d"
        output_dir.mkdir(parents=True, exist_ok=True)

        conv = ConvertImage4DTo3D()
        conv.load_image_4d(str(download_test_data))
        conv.save_3d_images(output_dir, "slice")

        slice_files = list(output_dir.glob("slice_*.mha"))
        assert len(slice_files) > 10, (
            f"Expected more than 10 slice files, found {len(slice_files)}"
        )

        slice_007 = output_dir / "slice_007.mha"
        assert slice_007.exists(), "Expected slice_007.mha not found"

        print(f"\nFound {len(slice_files)} slice files")

    def test_load_image_4d(self, download_test_data: Path) -> None:
        """Test loading a 4D image."""
        input_4d_file = download_test_data

        conv = ConvertImage4DTo3D()
        conv.load_image_4d(str(input_4d_file))

        assert conv.get_number_of_3d_images() > 0, "No time points found in 4D image"

        print(f"\nLoaded 4D image with {conv.get_number_of_3d_images()} time points")

    def test_save_3d_images(
        self,
        download_test_data: Path,
        test_directories: dict[str, Path],
    ) -> None:
        """Test saving 3D images from a 4D source."""
        output_dir = test_directories["output"] / "convert_image_4d_to_3d"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_4d_file = download_test_data

        conv = ConvertImage4DTo3D()
        conv.load_image_4d(str(input_4d_file))

        num_time_points = conv.get_number_of_3d_images()

        conv.save_3d_images(output_dir, "test_slice")

        test_slice_files = list(output_dir.glob("test_slice_*.mha"))
        assert len(test_slice_files) > 0, "No test slice files were created"
        assert len(test_slice_files) == num_time_points, (
            f"Expected {num_time_points} files, found {len(test_slice_files)}"
        )

        print(f"\nSaved {len(test_slice_files)} 3D images")

        for test_file in test_slice_files:
            test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
