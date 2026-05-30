"""
Labelmap Tools for PhysioMotion4D

This module provides the :class:`LabelmapTools` class with the definitive
utility for turning a multi-label (or binary) segmentation labelmap into a
binary registration mask, optionally excluding specific labels and dilating
the result by a physical radius in millimeters.
"""

import logging
from typing import Optional

import itk
import numpy as np

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase


class LabelmapTools(PhysioMotion4DBase):
    """
    Utilities for converting segmentation labelmaps into registration masks.

    A labelmap is an ``itk.Image`` of integer labels where ``0`` is background
    and each positive value identifies an anatomical structure. A registration
    mask is a binary ``itk.Image`` where every foreground voxel is ``1``. This
    class centralizes the labelmap-to-mask conversion so that thresholding,
    label exclusion, and physically isotropic dilation are performed
    identically everywhere in the platform.

    Example:
        >>> tools = LabelmapTools()
        >>> # Binary mask of every labeled voxel, dilated 5 mm
        >>> mask = tools.convert_labelmap_to_mask(labelmap, dilation_in_mm=5.0)
        >>> # Exclude the table/background labels 8 and 9 before masking
        >>> mask = tools.convert_labelmap_to_mask(
        ...     labelmap, dilation_in_mm=5.0, labels_to_exclude=[8, 9]
        ... )
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize LabelmapTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def convert_labelmap_to_mask(
        self,
        labelmap: itk.Image,
        dilation_in_mm: float = 0.0,
        labels_to_exclude: Optional[list[int]] = None,
    ) -> itk.Image:
        """Convert a labelmap into a binary registration mask.

        Any voxel whose label is in ``labels_to_exclude`` is set to background
        first; every remaining non-zero voxel becomes foreground (``1``). The
        binary mask is then dilated by ``dilation_in_mm`` millimeters of
        physical radius. The radius is converted into per-axis voxel counts
        from the labelmap's spacing so the dilation is physically isotropic
        even on anisotropic grids; each per-axis count is clamped to at least
        1 voxel when ``dilation_in_mm > 0``.

        Axis ordering: the labelmap is a scalar 3D ``itk.Image`` in ITK
        world-axis order (X, Y, Z). All thresholding is performed on the numpy
        view (Z, Y, X) and written back through ``CopyInformation``, so origin,
        spacing, and direction are preserved.

        Args:
            labelmap: Multi-label or binary ``itk.Image``. Any non-zero voxel
                that is not excluded is treated as foreground.
            dilation_in_mm: Physical radius of the binary dilation in
                millimeters. Pass ``0`` (or negative) to skip dilation and
                return the raw thresholded mask. Default 0.0.
            labels_to_exclude: Optional list of integer label values to force
                to background before thresholding. When ``None`` (the default)
                no labels are excluded.

        Returns:
            ``itk.Image[itk.UC, 3]`` binary mask in the same physical space as
            ``labelmap`` (origin, spacing, direction copied from the input).
        """
        arr = itk.array_from_image(labelmap)
        if labels_to_exclude:
            arr = np.where(np.isin(arr, labels_to_exclude), 0, arr)
        mask_arr = (arr > 0).astype(np.uint8)
        mask = itk.image_from_array(mask_arr)
        mask.CopyInformation(labelmap)

        if dilation_in_mm <= 0:
            return mask

        spacing = labelmap.GetSpacing()
        radius = itk.Size[3]()
        for i in range(3):
            radius[i] = max(1, int(round(dilation_in_mm / float(spacing[i]))))
        structuring_element = itk.FlatStructuringElement[3].Ball(radius)
        return itk.binary_dilate_image_filter(
            mask, kernel=structuring_element, foreground_value=1
        )
