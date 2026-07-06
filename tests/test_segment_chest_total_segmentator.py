#!/usr/bin/env python
"""
Test for chest CT segmentation using TotalSegmentator.

This test depends on test_convert_image_4d_to_3d and tests segmentation
functionality on two time points from the converted 3D data.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestSegmentChestTotalSegmentator:
    """Test suite for TotalSegmentator chest CT segmentation."""

    def test_segmenter_initialization(
        self, segmenter_total_segmentator: SegmentChestTotalSegmentator
    ) -> None:
        """Test that SegmentChestTotalSegmentator initializes correctly."""
        assert segmenter_total_segmentator is not None, "Segmenter not initialized"
        assert segmenter_total_segmentator.target_spacing == 1.5, (
            "Target spacing not set correctly"
        )

        # Check that anatomical structure ID mappings are defined via the
        # shared taxonomy.
        taxonomy = segmenter_total_segmentator.taxonomy
        assert len(taxonomy.labels_in_group("heart")) > 0, "Heart mask IDs not defined"
        assert len(taxonomy.labels_in_group("major_vessels")) > 0, (
            "Major vessels mask IDs not defined"
        )
        assert len(taxonomy.labels_in_group("lung")) > 0, "Lung mask IDs not defined"
        assert len(taxonomy.labels_in_group("bone")) > 0, "Bone mask IDs not defined"
        assert len(taxonomy.labels_in_group("soft_tissue")) > 0, (
            "Soft tissue mask IDs not defined"
        )

        print("\nSegmenter initialized with correct parameters")
        print(f"  Heart structures: {len(taxonomy.labels_in_group('heart'))}")
        print(f"  Major vessels: {len(taxonomy.labels_in_group('major_vessels'))}")
        print(f"  Lung structures: {len(taxonomy.labels_in_group('lung'))}")
        print(f"  Bone structures: {len(taxonomy.labels_in_group('bone'))}")
        print(
            f"  Soft tissue structures: {len(taxonomy.labels_in_group('soft_tissue'))}"
        )

    def test_segment_single_image(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on a single time point."""
        output_dir = test_directories["output"]

        # Test on first time point only
        input_image = test_images[0]

        print("\nSegmenting time point 0...")
        print(f"  Input image size: {itk.size(input_image)}")

        # Run segmentation
        result = segmenter_total_segmentator.segment(input_image)

        # Verify result is a dictionary with expected keys
        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = [
            "labelmap",
            "lung",
            "heart",
            "major_vessels",
            "bone",
            "soft_tissue",
            "other",
            "contrast",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"

        # Verify labelmap properties
        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        # Check that labels are present
        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"

        print("Segmentation complete for time point 0")
        print(f"  Labelmap size: {itk.size(labelmap)}")
        print(f"  Unique labels: {len(unique_labels)}")

        # Save results
        seg_output_dir = output_dir / "segmentation_total_segmentator"
        seg_output_dir.mkdir(exist_ok=True)

        itk.imwrite(
            labelmap, str(seg_output_dir / "slice_000_labelmap.mha"), compression=True
        )
        print(f"  Saved labelmap to: {seg_output_dir / 'slice_000_labelmap.mha'}")

    def test_segment_multiple_images(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on two time points."""
        output_dir = test_directories["output"]
        seg_output_dir = output_dir / "segmentation_total_segmentator"
        seg_output_dir.mkdir(exist_ok=True)

        results = []
        for i, input_image in enumerate(test_images[0:2]):
            print(f"\nSegmenting time point {i}...")

            result = segmenter_total_segmentator.segment(input_image)
            results.append(result)

            # Save labelmap for each time point
            labelmap = result["labelmap"]
            output_file = seg_output_dir / f"slice_{i:03d}_labelmap.mha"
            itk.imwrite(labelmap, str(output_file), compression=True)

            print(f"Time point {i} complete")
            print(f"  Saved to: {output_file}")

        assert len(results) == 2, "Expected 2 segmentation results"
        print(f"\nSuccessfully segmented {len(results)} time points")

    def test_anatomy_group_masks(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test that anatomy group masks are created correctly."""
        input_image = test_images[0]

        # Run segmentation
        result = segmenter_total_segmentator.segment(input_image)

        # Check each anatomy group mask
        anatomy_groups = [
            "lung",
            "heart",
            "major_vessels",
            "bone",
            "soft_tissue",
            "other",
        ]

        for group in anatomy_groups:
            mask = result[group]
            assert mask is not None, f"{group} mask is None"

            # Check that mask is binary
            mask_arr = itk.array_from_image(mask)
            unique_values = np.unique(mask_arr)
            assert len(unique_values) <= 2, f"{group} mask should be binary"
            assert 0 in unique_values, f"{group} mask should contain background"

            # Check that mask has same size as input
            assert itk.size(mask) == itk.size(input_image), (
                f"{group} mask size mismatch"
            )

        print("\nAll anatomy group masks created correctly")
        for group in anatomy_groups:
            mask_arr = itk.array_from_image(result[group])
            num_voxels = np.sum(mask_arr > 0)
            print(f"  {group}: {num_voxels} voxels")

    def test_contrast_detection(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test contrast detection functionality."""
        input_image = test_images[0]

        # Test without contrast
        segmenter_total_segmentator.set_contrast_enhanced_study(False)
        result_no_contrast = segmenter_total_segmentator.segment(input_image)
        contrast_mask_no = result_no_contrast["contrast"]

        # Test with contrast flag
        segmenter_total_segmentator.set_contrast_enhanced_study(True)
        result_with_contrast = segmenter_total_segmentator.segment(input_image)
        contrast_mask_yes = result_with_contrast["contrast"]
        segmenter_total_segmentator.set_contrast_enhanced_study(False)

        # Both should return valid masks
        assert contrast_mask_no is not None, "Contrast mask (no flag) is None"
        assert contrast_mask_yes is not None, "Contrast mask (with flag) is None"

        print("\nContrast detection tested")

        contrast_arr_no = itk.array_from_image(contrast_mask_no)
        contrast_arr_yes = itk.array_from_image(contrast_mask_yes)
        print(f"  Without contrast flag: {np.sum(contrast_arr_no > 0)} voxels")
        print(f"  With contrast flag: {np.sum(contrast_arr_yes > 0)} voxels")

    def test_preprocessing(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test preprocessing functionality."""
        input_image = test_images[0]

        # Get original properties
        original_spacing = itk.spacing(input_image)

        # Preprocessing is done internally by segment(), not exposed as public method
        # Just verify that segment() works (which includes preprocessing)
        result = segmenter_total_segmentator.segment(input_image)

        # Check that segmentation was successful (which means preprocessing worked)
        assert result is not None, "Segmentation result is None"
        assert "labelmap" in result, "Labelmap not in result"

        print("\nPreprocessing tested (via successful segmentation)")
        print(f"  Original image spacing: {original_spacing}")

    def test_postprocessing(
        self,
        segmenter_total_segmentator: SegmentChestTotalSegmentator,
        test_images: list[Any],
    ) -> None:
        """Test postprocessing functionality."""
        input_image = test_images[0]

        # Run full segmentation to get labelmap
        result = segmenter_total_segmentator.segment(input_image)
        labelmap = result["labelmap"]

        # Postprocessing is part of segment(), verify output is properly sized
        assert itk.size(labelmap) == itk.size(input_image), (
            "Postprocessing failed: size mismatch"
        )

        # Check that labelmap has been resampled to original spacing
        original_spacing = itk.spacing(input_image)
        labelmap_spacing = itk.spacing(labelmap)

        # Spacing should match (within floating point tolerance)
        for i in range(3):
            assert abs(labelmap_spacing[i] - original_spacing[i]) < 0.01, (
                f"Spacing mismatch at dimension {i}"
            )

        print("\nPostprocessing tested")
        print(f"  Original spacing: {original_spacing}")
        print(f"  Labelmap spacing: {labelmap_spacing}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
