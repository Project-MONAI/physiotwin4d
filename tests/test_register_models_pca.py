"""Synthetic tests for PCA model registration helpers."""

from __future__ import annotations

from typing import Any, cast

import itk
import numpy as np
import pytest
import pyvista as pv

from physiotwin4d.register_models_pca import RegisterModelsPCA


def _make_registrar() -> RegisterModelsPCA:
    """Create a small PCA registrar with a three-point template surface."""
    template_model = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    pca_eigenvectors = np.zeros((1, template_model.n_points * 3), dtype=np.float64)
    pca_std_deviations = np.ones(1, dtype=np.float64)
    fixed_distance_map = itk.image_from_array(np.zeros((4, 4, 4), dtype=np.float32))
    return RegisterModelsPCA(
        pca_template_model=template_model,
        pca_eigenvectors=pca_eigenvectors,
        pca_std_deviations=pca_std_deviations,
        pca_number_of_modes=1,
        fixed_distance_map=fixed_distance_map,
    )


def test_itk_template_points_are_distinct_objects() -> None:
    """Cached ITK points are distinct per template vertex."""
    registrar = _make_registrar()

    points = registrar._pca_template_model_points_itk
    assert points is not None
    assert len({id(point) for point in points}) == len(points)
    assert [float(points[0][0]), float(points[1][0]), float(points[2][1])] == [
        0.0,
        1.0,
        1.0,
    ]


def test_set_fixed_model_requires_reference_image() -> None:
    """set_fixed_model fails clearly when reference_image is None."""
    registrar = _make_registrar()

    with pytest.raises(ValueError, match="reference_image must not be None"):
        registrar.set_fixed_model(
            cast(pv.UnstructuredGrid, registrar.pca_template_model), None
        )


def test_transform_template_model_applies_post_pca_transform_after_deformation() -> (
    None
):
    """Post-PCA transform is applied after PCA deformation."""
    registrar = _make_registrar()
    registrar.registered_model_pca_coefficients = np.array([1.0], dtype=np.float64)
    registrar.registered_model_pca_deformation = np.tile(
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        (registrar.pca_template_model.n_points, 1),
    )
    transform = itk.ScaleTransform[itk.D, 3].New()
    transform.SetScale([2.0, 2.0, 2.0])
    registrar.post_pca_transform = transform

    registered_model: Any = registrar.transform_template_model()

    assert np.allclose(registered_model.points[0], [2.0, 0.0, 0.0])
    assert np.allclose(registered_model.points[1], [4.0, 0.0, 0.0])
