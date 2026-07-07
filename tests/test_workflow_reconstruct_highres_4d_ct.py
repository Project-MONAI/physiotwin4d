"""Tests for the high-resolution 4D CT reconstruction workflow's
instance-based registration_method API."""

from __future__ import annotations

import numpy as np
import itk
import pytest

from physiotwin4d.register_images_base import RegisterImagesBase
from physiotwin4d.register_images_greedy_icon import RegisterImagesGreedyICON
from physiotwin4d.register_images_icon import RegisterImagesICON
from physiotwin4d.workflow_reconstruct_highres_4d_ct import (
    WorkflowReconstructHighres4DCT,
)


def _small_image() -> itk.Image:
    """A tiny synthetic image, shape (Z, Y, X) = (3, 3, 3)."""
    return itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))


def test_default_registration_method_is_greedy_icon() -> None:
    """Omitting registration_method defaults to RegisterImagesGreedyICON,
    matching this workflow's historical 'Greedy_ICON' string default."""
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=[_small_image(), _small_image()],
        fixed_image=_small_image(),
    )
    assert isinstance(workflow.registrar.registrar, RegisterImagesGreedyICON)


def test_registration_method_rejects_wrong_type() -> None:
    """A non-RegisterImagesBase registration_method raises TypeError."""
    with pytest.raises(TypeError, match="registration_method must be"):
        WorkflowReconstructHighres4DCT(
            time_series_images=[_small_image(), _small_image()],
            fixed_image=_small_image(),
            registration_method="ICON",  # type: ignore[arg-type]
        )


def test_caller_supplied_instance_is_used_as_is() -> None:
    """A caller-supplied registration_method instance is stored unmodified."""
    registrar: RegisterImagesBase = RegisterImagesICON()
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=[_small_image(), _small_image()],
        fixed_image=_small_image(),
        registration_method=registrar,
    )
    assert workflow.registrar.registrar is registrar
