"""Synthetic tests for model-to-patient workflow helpers."""

from __future__ import annotations

from typing import Any

import itk
import numpy as np
import pyvista as pv

from physiomotion4d.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)


def test_auto_generate_mask_accumulates_multilabel_models(
    monkeypatch: Any,
) -> None:
    """Multi-model masks accumulate label IDs instead of overwriting prior labels."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    model = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )
    workflow.patient_image = image
    masks = [
        np.pad(np.ones((1, 1, 1), dtype=np.uint8), ((0, 2), (0, 2), (0, 2))),
        np.pad(np.ones((1, 1, 1), dtype=np.uint8), ((2, 0), (2, 0), (2, 0))),
    ]

    def fake_create_mask_from_mesh(*_args: object, **_kwargs: object) -> Any:
        mask = itk.image_from_array(masks.pop(0))
        mask.CopyInformation(workflow.patient_image)
        return mask

    monkeypatch.setattr(
        workflow.contour_tools,
        "create_mask_from_mesh",
        fake_create_mask_from_mesh,
    )

    output = workflow._auto_generate_mask([model, model], dilate_mm=0.0)
    output_arr = itk.GetArrayFromImage(output)

    assert output_arr[0, 0, 0] == 1
    assert output_arr[2, 2, 2] == 2
    assert sorted(np.unique(output_arr).tolist()) == [0, 1, 2]


def test_transform_model_applies_staged_transform() -> None:
    """Transform helper updates mesh points with image shape (Z, Y, X) = (3, 3, 3)."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    model = pv.PolyData(points)
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )

    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetIdentity()
    transform.SetTranslation((1.0, 2.0, 3.0))
    workflow.icp_forward_point_transform = transform
    workflow.pca_coefficients = None
    workflow.use_m2m_registration = False
    workflow.use_m2i_registration = False

    output = workflow.transform_model()

    assert output is not None
    np.testing.assert_allclose(output.points, points + np.array([1.0, 2.0, 3.0]))
