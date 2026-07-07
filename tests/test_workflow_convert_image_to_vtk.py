"""Tests for the image-to-VTK workflow's instance-based segmentation_method API."""

from __future__ import annotations

from typing import Any

import pytest

from physiotwin4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiotwin4d.segment_heart_simpleware import SegmentHeartSimpleware
from physiotwin4d.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK


def test_default_segmentation_method_is_chest_total_segmentator() -> None:
    """Omitting segmentation_method defaults to SegmentChestTotalSegmentator,
    matching this workflow's historical string default."""
    workflow = WorkflowConvertImageToVTK()
    assert isinstance(workflow._segmenter, SegmentChestTotalSegmentator)


def test_segmentation_method_rejects_wrong_type() -> None:
    """A non-SegmentAnatomyBase segmentation_method raises TypeError."""
    invalid_method: Any = "ChestTotalSegmentator"
    with pytest.raises(TypeError, match="segmentation_method must be"):
        WorkflowConvertImageToVTK(segmentation_method=invalid_method)


def test_caller_supplied_instance_is_used_as_is() -> None:
    """A caller-supplied segmenter instance is stored unmodified."""
    segmenter = SegmentHeartSimpleware()
    workflow = WorkflowConvertImageToVTK(segmentation_method=segmenter)
    assert workflow._segmenter is segmenter
