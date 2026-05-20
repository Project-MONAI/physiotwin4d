"""Tests for the image-to-USD workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.workflow_convert_image_to_usd import WorkflowConvertImageToUSD
import physiomotion4d.workflow_convert_image_to_usd as workflow_module


def test_create_usd_files_passes_times_per_second(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Workflow forwards FPS to VTK-to-USD for shape (X, Y, Z, T) outputs."""
    times_per_second_values: list[float] = []

    class FakeStage:
        def Export(self, _output_filename: str) -> None:
            return None

    class FakeConvertVTKToUSD:
        def __init__(
            self,
            _project_name: str,
            _input_polydata: list[Any],
            _mask_ids: dict[int, str],
            *,
            times_per_second: float,
            **_kwargs: Any,
        ) -> None:
            times_per_second_values.append(times_per_second)

        def convert(self, _usd_file: str) -> FakeStage:
            return FakeStage()

    class FakeUSDAnatomyTools:
        def __init__(self, _stage: FakeStage) -> None:
            return None

        def enhance_meshes(self, _segmenter: Any) -> None:
            return None

    class FakeTaxonomy:
        def all_labels(self) -> dict[int, str]:
            return {1: "heart"}

    class FakeSegmenter:
        taxonomy = FakeTaxonomy()

    monkeypatch.setattr(workflow_module, "ConvertVTKToUSD", FakeConvertVTKToUSD)
    monkeypatch.setattr(workflow_module, "USDAnatomyTools", FakeUSDAnatomyTools)

    workflow = WorkflowConvertImageToUSD.__new__(WorkflowConvertImageToUSD)
    PhysioMotion4DBase.__init__(
        workflow,
        class_name=WorkflowConvertImageToUSD.__name__,
        log_level=logging.CRITICAL,
    )
    workflow.project_name = "patient"
    workflow.output_directory = str(tmp_path)
    workflow.segmenter = FakeSegmenter()
    workflow.times_per_second = 12.5
    workflow._transformed_contours = {
        "all": [],
        "dynamic": [],
        "static": [],
    }

    workflow._create_usd_files()

    assert times_per_second_values == [12.5, 12.5, 12.5]
