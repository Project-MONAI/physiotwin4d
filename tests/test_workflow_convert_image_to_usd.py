"""Tests for the image-to-USD workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import itk
import pytest
from pxr import Usd, UsdGeom

from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.workflow_convert_image_to_usd import WorkflowConvertImageToUSD


@pytest.mark.requires_gpu
@pytest.mark.slow
def test_workflow_convert_image_to_usd_default_operation(
    test_images: list[Any],
    tmp_path: Path,
) -> None:
    """Convert one real Slicer-Heart frame to USD.

    Input frame shape is the downloaded/resampled slicer_heart_small 3D image
    with axes (X, Y, Z) in LPS world frame.
    """
    reference_image = test_images[0]
    workflow = WorkflowConvertImageToUSD(
        time_series_images=[reference_image],
        reference_image=reference_image,
        usd_project_name="slicer_heart_small",
        output_directory=str(tmp_path),
        log_level=logging.CRITICAL,
    )

    assert isinstance(workflow.segmenter, SegmentChestTotalSegmentator)
    assert workflow.segmenter.contrast_threshold == 500
    assert workflow.segmenter.contrast_enhanced_study
    assert isinstance(workflow.registrar, RegisterImagesICON)
    workflow.registrar.set_number_of_iterations(2)

    result_filename = workflow.process()

    assert result_filename == "slicer_heart_small.all_painted.usd"
    assert workflow.reference_segmentation is not None
    assert "all" in workflow.reference_contours
    assert len(workflow.transformed_contours["all"]) == 1
    assert len(workflow.registration_results) == 1
    assert "all" in workflow.registration_results[0]
    assert workflow.registration_results[0]["all"]["forward_transform"] is not None
    assert workflow.registration_results[0]["all"]["inverse_transform"] is not None

    reference_labelmap = cast(
        itk.Image,
        workflow.reference_segmentation["labelmap"],
    )
    assert itk.size(reference_labelmap) == itk.size(reference_image)

    expected_outputs = [
        "reference_labelmap.mha",
        "slice_000_labelmap.mha",
        "slicer_heart_small.all.usd",
        "slicer_heart_small.all_painted.usd",
    ]
    for output_name in expected_outputs:
        output_path = tmp_path / output_name
        assert output_path.exists(), f"Missing workflow output: {output_path}"
        assert output_path.stat().st_size > 0, f"Empty workflow output: {output_path}"

    stage = Usd.Stage.Open(str(tmp_path / "slicer_heart_small.all_painted.usd"))
    assert stage is not None
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert stage.GetPrimAtPath("/World").IsValid()
    assert stage.GetPrimAtPath("/World/slicer_heart_small").IsValid()
