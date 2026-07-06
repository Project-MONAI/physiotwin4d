#!/usr/bin/env python
"""
Tests for heart CT segmentation using SegmentHeartSimpleware (Simpleware Medical ASCardio).

Uses the same input data as experiments/Heart-Simpleware_Segmentation:
  data/CHOP-Valve4D/CT/RVOT28-Dias.nii.gz

Requires Synopsys Simpleware Medical with ASCardio and the test data to run
full segmentation tests. Initialization and path tests run without Simpleware.
"""

import os
from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from physiomotion4d.segment_heart_simpleware import SegmentHeartSimpleware


def _simpleware_available(segmenter: SegmentHeartSimpleware) -> bool:
    """Return True if Simpleware Medical executable and script exist."""
    return os.path.exists(segmenter.simpleware_exe_path) and os.path.exists(
        segmenter.simpleware_script_path
    )


@pytest.mark.requires_gpu
@pytest.mark.requires_simpleware
@pytest.mark.slow
class TestSegmentHeartSimpleware:
    """Test suite for SegmentHeartSimpleware (Simpleware Medical ASCardio)."""

    def test_segmenter_initialization(
        self, segmenter_simpleware: SegmentHeartSimpleware
    ) -> None:
        """Test that SegmentHeartSimpleware initializes correctly."""
        seg = segmenter_simpleware
        assert seg is not None, "Segmenter not initialized"
        assert seg.target_spacing == 1.0, (
            "Target spacing should be 1.0 mm for Simpleware"
        )

        taxonomy = seg.taxonomy
        assert len(taxonomy.labels_in_group("heart")) > 0, "Heart mask IDs not defined"
        assert len(taxonomy.labels_in_group("major_vessels")) > 0, (
            "Major vessels mask IDs not defined"
        )
        # ASCardio does not segment lung or bone — those groups are never
        # registered, so labels_in_group returns an empty dict for them.
        # soft_tissue still contains the base-class placeholder (id 133).
        assert taxonomy.labels_in_group("lung") == {}, "ASCardio does not segment lungs"
        assert taxonomy.labels_in_group("bone") == {}, "ASCardio does not segment bone"
        assert taxonomy.labels_in_group("soft_tissue") == {133: "soft_tissue"}, (
            "Only the base-class soft_tissue placeholder should be present"
        )

        assert seg.simpleware_exe_path is not None, "Simpleware executable path not set"
        assert seg.simpleware_script_path is not None, "Simpleware script path not set"
        assert "SimplewareScript_heart_segmentation" in seg.simpleware_script_path

        print("\nSegmenter initialized with correct parameters")
        print(f"  Target spacing: {seg.target_spacing} mm")
        print(f"  Heart structures: {len(taxonomy.labels_in_group('heart'))}")
        print(f"  Major vessels: {len(taxonomy.labels_in_group('major_vessels'))}")

    def test_set_simpleware_executable_path(
        self, segmenter_simpleware: SegmentHeartSimpleware
    ) -> None:
        """Test setting custom Simpleware executable path."""
        seg = segmenter_simpleware
        original = seg.simpleware_exe_path
        custom = "D:/Custom/ConsoleSimplewareMedical.exe"
        seg.set_simpleware_executable_path(custom)
        assert seg.simpleware_exe_path == custom
        seg.set_simpleware_executable_path(original)
        assert seg.simpleware_exe_path == original
        print("\nset_simpleware_executable_path works correctly")

    def test_segment_single_image(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test segmentation on a cardiac CT time point."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip(
                "Simpleware Medical not found (executable or script). "
                "Install Simpleware Medical with ASCardio to run this test."
            )

        output_dir = test_directories["output"]
        input_image = test_images[3]

        print("\nSegmenting cardiac CT...")
        print(f"  Image size: {itk.size(input_image)}")

        result = segmenter_simpleware.segment(input_image)

        assert isinstance(result, dict), "Result should be a dictionary"
        # The Simpleware segmenter only registers the groups it actually
        # populates: heart + major_vessels (subclass) and soft_tissue +
        # contrast (inherited base-class placeholders). lung and bone are
        # NOT in the result because ASCardio does not segment them; callers
        # that need those groups must check membership first.
        expected_keys = [
            "labelmap",
            "heart",
            "major_vessels",
            "soft_tissue",
            "contrast",
            "other",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in result"
            assert result[key] is not None, f"Result['{key}'] is None"
        assert "lung" not in result, (
            "ASCardio does not segment lung; key must be absent"
        )
        assert "bone" not in result, (
            "ASCardio does not segment bone; key must be absent"
        )

        labelmap = result["labelmap"]
        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"

        labelmap_arr = itk.array_from_image(labelmap)
        unique_labels = np.unique(labelmap_arr)
        assert len(unique_labels) > 1, "Labelmap should contain multiple labels"

        print("Segmentation complete")
        print(f"  Unique labels: {len(unique_labels)}")

        seg_output_dir = output_dir / "segmentation_simpleware"
        seg_output_dir.mkdir(exist_ok=True)
        itk.imwrite(
            labelmap,
            str(seg_output_dir / "heart_labelmap_simpleware.nii.gz"),
            compression=True,
        )
        print(f"  Saved to: {seg_output_dir / 'heart_labelmap_simpleware.nii.gz'}")

    def test_anatomy_group_masks(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
    ) -> None:
        """Test that anatomy group masks are created (heart, vessels, etc.)."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = test_images[3]
        result = segmenter_simpleware.segment(input_image)

        # Only assert on groups Simpleware/ASCardio actually populates.
        anatomy_groups = [
            "heart",
            "major_vessels",
            "soft_tissue",
            "other",
        ]
        for group in anatomy_groups:
            assert group in result, f"{group} mask should be present"
            mask = result[group]
            assert mask is not None, f"{group} mask is None"
            mask_arr = itk.array_from_image(mask)
            unique_values = np.unique(mask_arr)
            assert len(unique_values) <= 2, f"{group} mask should be binary"
            assert 0 in unique_values or mask_arr.size == 0
            assert itk.size(mask) == itk.size(input_image), (
                f"{group} mask size mismatch"
            )

        heart_arr = itk.array_from_image(result["heart"])
        vessels_arr = itk.array_from_image(result["major_vessels"])
        print("\nAll anatomy group masks created correctly")
        print(f"  heart: {np.sum(heart_arr > 0)} voxels")
        print(f"  major_vessels: {np.sum(vessels_arr > 0)} voxels")

    def test_contrast_detection(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
    ) -> None:
        """Test contrast mask is returned (base class behavior)."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = test_images[3]
        result = segmenter_simpleware.segment(input_image)
        contrast_mask = result["contrast"]
        assert contrast_mask is not None
        assert itk.size(contrast_mask) == itk.size(input_image)
        print("\nContrast mask returned")

    def test_postprocessing(
        self,
        segmenter_simpleware: SegmentHeartSimpleware,
        test_images: list[Any],
    ) -> None:
        """Test that output labelmap matches input size and spacing."""
        if not _simpleware_available(segmenter_simpleware):
            pytest.skip("Simpleware Medical not found. Install to run this test.")

        input_image = test_images[3]
        result = segmenter_simpleware.segment(input_image)
        labelmap = result["labelmap"]

        assert itk.size(labelmap) == itk.size(input_image), "Labelmap size mismatch"
        original_spacing = itk.spacing(input_image)
        labelmap_spacing = itk.spacing(labelmap)
        for i in range(3):
            assert abs(labelmap_spacing[i] - original_spacing[i]) < 0.01, (
                f"Spacing mismatch at dimension {i}"
            )
        print("\nPostprocessing: labelmap size and spacing match input")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
