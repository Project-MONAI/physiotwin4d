"""Base class for segmenting anatomy in CT images.

This module provides the SegmentAnatomyBase class that serves as a foundation
for implementing different anatomy CT segmentation algorithms. It handles common
preprocessing, postprocessing, and anatomical structure organization tasks.
"""

import logging
from typing import Any

import itk
import numpy as np
from itk import TubeTK as tube

from physiomotion4d.anatomy_taxonomy import AnatomyTaxonomy
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase


class SegmentAnatomyBase(PhysioMotion4DBase):
    """Base class for anatomy segmentation that provides common functionality for
    segmenting anatomy in CT images.

    This class implements preprocessing, postprocessing, and mask creation
    methods that are shared across different anatomy segmentation
    implementations. It owns an :class:`AnatomyTaxonomy` instance that
    captures the group→organ structure (e.g. ``heart`` contains
    ``atrial_appendage_left`` at id 61); subclasses populate it via
    ``self.taxonomy.add_organ(...)`` and call
    :meth:`_finalize_other_group` once they're done.

    Extensibility
    -------------
    Each segmenter is free to define its own group names — the taxonomy does
    not hard-code a fixed set. A new subclass adds groups by calling
    ``self.taxonomy.add_organ(group_name, label_id, organ_name)`` for each
    organ; the group is created lazily on first use. To assign a custom
    OmniSurface look to a new group, register it in
    :data:`physiomotion4d.usd_anatomy_tools.DEFAULT_RENDER_PARAMS` (see that
    module's docstring). Groups without a registered look fall back to the
    ``"other"`` entry, so they still render.

    Attributes:
        target_spacing (float): Target isotropic spacing for resampling.
        rescale_intensity_range (bool): Whether to rescale intensity values.
        contrast_threshold (int): Threshold for contrast agent detection.
        taxonomy (AnatomyTaxonomy): Group→organ mapping shared with
            :class:`physiomotion4d.USDAnatomyTools`.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the SegmentAnatomyBase class.

        Sets up default parameters for image preprocessing and seeds the
        anatomy taxonomy with the two base-class default organs (contrast
        and soft_tissue). Subclasses should:

        1. Add their organ groups via ``self.taxonomy.add_organ(...)``.
        2. Call :meth:`_finalize_other_group` to fill in unclaimed ids.

        Args:
            log_level: Logging level (default: logging.INFO).
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.target_spacing: float = 0.0

        self.rescale_intensity_range: bool = False
        self.input_intensity_scale_range: list[int] = [0, 4096]
        self.output_intensity_scale_range: list[int] = [-1024, 3071]
        self.output_intensity_clip_range: list[int] = [-1024, 3071]

        self.contrast_threshold: int = 700

        # Single source of truth for the anatomy hierarchy. Subclasses
        # populate this; USDAnatomyTools and ConvertVTKToUSD consume it.
        self.taxonomy = AnatomyTaxonomy()
        # Base-class default labels that downstream code relies on existing.
        # Subclasses can override by adding the same id under a different
        # group, or leave these in place.
        self.taxonomy.add_organ("contrast", 135, "contrast")
        self.taxonomy.add_organ("soft_tissue", 133, "soft_tissue")

    def _finalize_other_group(self) -> None:
        """Fill the ``other`` group with any unclaimed ids in [1, 256).

        Subclasses call this at the end of ``__init__`` once they have
        populated their specific groups. The consolidated all-labels view is
        available via ``self.taxonomy.all_labels()``.
        """
        self.taxonomy.fill_other_group()

    def label_to_type(self, label_name: str) -> str:
        """Return the anatomy group ('heart', 'lung', etc.) for a label name.

        Used by :class:`physiomotion4d.ConvertVTKToUSD` to group label-mode
        mesh prims under per-type Xforms (e.g.
        ``/World/{basename}/heart/{label_name}``). Delegates to the taxonomy.

        Args:
            label_name: Organ name (a value in the taxonomy's group organ dicts).

        Returns:
            The anatomy group name. Falls back to ``"other"`` for any label
            the segmenter doesn't recognize.
        """
        return self.taxonomy.group_for_label(label_name)

    def set_target_spacing(self, target_spacing: float) -> None:
        """Set the target isotropic spacing for image resampling.

        Args:
            target_spacing (float): Target spacing in millimeters for all three
                spatial dimensions. Set to 0.0 to disable resampling.

        Example:
            >>> segmenter.set_target_spacing(1.0)  # 1mm isotropic spacing
        """
        self.target_spacing = target_spacing

    def preprocess_input(
        self,
        input_image: Any,
    ) -> Any:
        """Preprocess the input image for segmentation.

        Performs image preprocessing including resampling to isotropic spacing
        and optional intensity rescaling. The preprocessing ensures consistent
        image characteristics for reliable segmentation.

        Args:
            input_image (itk.image): The input 3D CT image to preprocess

        Returns:
            itk.image: The preprocessed image with isotropic spacing and
                optionally rescaled intensities

        Raises:
            AssertionError: If the input image is not 3D
            ValueError: If intensity rescaling parameters are invalid

        Example:
            >>> preprocessed = segmenter.preprocess_input(ct_image)
        """

        # Check the input image
        assert len(input_image.GetSpacing()) == 3, "The input image must be 3D"

        rescale_image = False
        results_image = None
        if self.target_spacing > 0.0:
            if (
                input_image.GetSpacing()[0] != self.target_spacing
                or input_image.GetSpacing()[1] != self.target_spacing
                or input_image.GetSpacing()[2] != self.target_spacing
            ):
                rescale_image = True
            else:
                isotropy = (
                    (input_image.GetSpacing()[1] / input_image.GetSpacing()[0])
                    + (input_image.GetSpacing()[2] / input_image.GetSpacing()[0])
                ) / 2
                if isotropy < 0.9 or isotropy > 1.1:
                    rescale_image = True
                    self.target_spacing = (
                        input_image.GetSpacing()[0]
                        + input_image.GetSpacing()[1]
                        + input_image.GetSpacing()[2]
                    ) / 3
                    self.log_info(
                        "Resampling to %.3f isotropic spacing", self.target_spacing
                    )
        if rescale_image:
            self.log_warning("The input image should have isotropic spacing")
            self.log_info("Input image has spacing: %s", str(input_image.GetSpacing()))
            self.log_info("Resampling to isotropic: %.3f", self.target_spacing)
            interpolator = itk.LinearInterpolateImageFunction.New(input_image)
            results_image = itk.resample_image_filter(
                input_image,
                interpolator=interpolator,
                output_spacing=[
                    self.target_spacing,
                    self.target_spacing,
                    self.target_spacing,
                ],
                size=[
                    int(
                        input_image.GetLargestPossibleRegion().GetSize()[0]
                        * input_image.GetSpacing()[0]
                        / self.target_spacing
                    ),
                    int(
                        input_image.GetLargestPossibleRegion().GetSize()[1]
                        * input_image.GetSpacing()[1]
                        / self.target_spacing
                    ),
                    int(
                        input_image.GetLargestPossibleRegion().GetSize()[2]
                        * input_image.GetSpacing()[2]
                        / self.target_spacing
                    ),
                ],
                output_origin=input_image.GetOrigin(),
                output_direction=input_image.GetDirection(),
            )
        else:
            results_image_arr = itk.GetArrayFromImage(input_image)
            results_image = itk.GetImageFromArray(results_image_arr)
            results_image.CopyInformation(input_image)

        results_image_arr = itk.GetArrayFromImage(results_image).astype(np.float32)
        minv = results_image_arr.min()
        maxv = results_image_arr.max()
        if self.rescale_intensity_range:
            self.log_info("Rescaling intensity range...")
            if (
                self.input_intensity_scale_range is None
                or self.output_intensity_scale_range is None
                or self.output_intensity_clip_range is None
            ):
                raise ValueError(
                    "output_intensity_scale_range must be set if input_intensity_scale_range is set"
                )
            minv = self.input_intensity_scale_range[0]
            maxv = self.input_intensity_scale_range[1]
            output_minv = self.output_intensity_scale_range[0]
            output_maxv = self.output_intensity_scale_range[1]
            results_image_arr = (results_image_arr - minv) / (maxv - minv) * (
                output_maxv - output_minv
            ) + output_minv
            results_image_arr = np.clip(
                results_image_arr,
                self.output_intensity_clip_range[0],
                self.output_intensity_clip_range[1],
            )

            new_results_image = itk.GetImageFromArray(results_image_arr)
            new_results_image.CopyInformation(results_image)
            results_image = new_results_image

        return results_image

    def postprocess_labelmap(
        self,
        labelmap_image: itk.image,
        input_image: itk.image,
    ) -> itk.image:
        """
        Resample the labelmap to match the input image spacing.

        Ensures the segmentation labelmap has the same spatial properties
        as the original input image by resampling using label-specific
        interpolation that preserves discrete label values.

        Args:
            labelmap_image (itk.image): The segmentation labelmap to resample
            input_image (itk.image): The original input image providing
                target spacing and geometry

        Returns:
            itk.image: The resampled labelmap matching input image properties

        Example:
            >>> final_labels = segmenter.postprocess_labelmap(labels, original_image)
        """
        input_spacing = np.array(input_image.GetSpacing())
        label_spacing = np.array(labelmap_image.GetSpacing())
        results_image = None
        if any(input_spacing != label_spacing):
            interpolator = itk.LabelImageGaussianInterpolateImageFunction.New(
                labelmap_image
            )
            results_image = itk.resample_image_filter(
                labelmap_image,
                interpolator=interpolator,
                ReferenceImage=input_image,
                UseReferenceImage=True,
            )
            labelmap_arr = itk.GetArrayFromImage(labelmap_image)
            results_arr = itk.GetArrayFromImage(results_image)
            new_results_arr = results_arr.copy()
            if results_arr[0, :, :].sum() == 0 and labelmap_arr[0, :, :].sum() > 0:
                sumi = 1
                sum = new_results_arr[sumi, :, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[sumi, :, :].sum()
                for i in range(sumi):
                    new_results_arr[i, :, :] = new_results_arr[sumi, :, :]
            if results_arr[-1, :, :].sum() == 0 and labelmap_arr[-1, :, :].sum() > 0:
                sumi = 2
                sum = new_results_arr[-sumi, :, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[-sumi, :, :].sum()
                for i in range(1, sumi):
                    new_results_arr[-i, :, :] = new_results_arr[-sumi, :, :]
            if results_arr[:, 0, :].sum() == 0 and labelmap_arr[:, 0, :].sum() > 0:
                sumi = 1
                sum = new_results_arr[:, sumi, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, sumi, :].sum()
                for i in range(sumi):
                    new_results_arr[:, i, :] = new_results_arr[:, sumi, :]
            if results_arr[:, -1, :].sum() == 0 and labelmap_arr[:, -1, :].sum() > 0:
                sumi = 2
                sum = new_results_arr[:, -sumi, :].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, -sumi, :].sum()
                for i in range(1, sumi):
                    new_results_arr[:, -i, :] = new_results_arr[:, -sumi, :]
            if results_arr[:, :, 0].sum() == 0 and labelmap_arr[:, :, 0].sum() > 0:
                sumi = 1
                sum = new_results_arr[:, :, sumi].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, :, sumi].sum()
                for i in range(sumi):
                    new_results_arr[:, :, i] = new_results_arr[:, :, sumi]
            if results_arr[:, :, -1].sum() == 0 and labelmap_arr[:, :, -1].sum() > 0:
                sumi = 2
                sum = new_results_arr[:, :, -sumi].sum()
                while sum == 0:
                    sumi += 1
                    sum = new_results_arr[:, :, -sumi].sum()
                for i in range(1, sumi):
                    new_results_arr[:, :, -i] = new_results_arr[:, :, -sumi]
            results_image = itk.GetImageFromArray(new_results_arr)
            results_image.CopyInformation(input_image)
        else:
            results_image_arr = itk.GetArrayFromImage(labelmap_image)
            results_image = itk.GetImageFromArray(results_image_arr)
            results_image.CopyInformation(labelmap_image)

        return results_image

    def segment_connected_component(
        self,
        preprocessed_image: itk.image,
        labelmap_image: itk.image,
        lower_threshold: int,
        upper_threshold: int,
        labelmap_ids: None | list[int] = None,
        mask_id: int = 0,
        use_mid_slice: bool = True,
        hole_fill: int = 2,
    ) -> itk.image:
        """
        Segment connected components based on intensity thresholding.

        Identifies connected regions within intensity thresholds and existing
        anatomical masks, then selects the largest component. This is useful
        for segmenting structures like contrast-enhanced blood or specific
        tissue types.

        Args:
            preprocessed_image (itk.image): The preprocessed input image
            labelmap_image (itk.image): Existing labelmap to constrain search
            lower_threshold (int): Lower intensity threshold
            upper_threshold (int): Upper intensity threshold
            labelmap_ids (None | list[int]): List of label IDs to search within.
                If None, searches within all existing labels
            mask_id (int): ID to assign to the segmented component
            use_mid_slice (bool): If True, find largest component in middle
                slice only; if False, use entire 3D volume
            hole_fill (int): Number of pixels to dilate/erode for hole filling

        Returns:
            itk.image: Updated labelmap with new component labeled as mask_id

        Example:
            >>> # Segment contrast-enhanced blood
            >>> updated_labels = segmenter.segment_connected_component(
            ...     preprocessed_image, labels, 700, 4000, mask_id=135
            ... )
        """
        thresh_image = itk.binary_threshold_image_filter(
            Input=preprocessed_image,
            LowerThreshold=lower_threshold,
            UpperThreshold=upper_threshold,
            InsideValue=1,
            OutsideValue=0,
        )
        thresh_arr = itk.GetArrayFromImage(thresh_image).astype(np.int16)
        thresh_image = itk.GetImageFromArray(thresh_arr)
        thresh_image.CopyInformation(preprocessed_image)

        label_arr = itk.GetArrayFromImage(labelmap_image)
        if labelmap_ids is None:
            labelmap_ids = list(self.taxonomy.all_labels().keys())
        label_arr = np.isin(label_arr, labelmap_ids)
        label_image = itk.GetImageFromArray(label_arr.astype(np.int16))
        label_image.CopyInformation(labelmap_image)

        connected_component_image = itk.connected_component_image_filter(
            Input=thresh_image,
            MaskImage=label_image,
        )

        connected_component_arr = itk.GetArrayFromImage(connected_component_image)
        if use_mid_slice:
            mid_slice = (
                connected_component_image.GetLargestPossibleRegion().GetSize()[2] // 2
            )
            tmp_connected_component_arr = connected_component_arr[:, :, mid_slice]
            ids = np.unique(tmp_connected_component_arr)
            if len(ids[ids != 0]) > 0:
                connected_component_arr = tmp_connected_component_arr

        ids = np.unique(connected_component_arr)
        ids = ids[ids != 0]
        if ids.size == 0:
            self.log_debug(
                "segment_connected_component: no connected components found "
                "in threshold [%d, %d]; returning labelmap unchanged",
                lower_threshold,
                upper_threshold,
            )
            return labelmap_image
        component_sums = [np.sum(connected_component_arr == id) for id in ids]
        largest_id = ids[np.argmax(component_sums)]
        connected_component_image = itk.binary_threshold_image_filter(
            Input=connected_component_image,
            LowerThreshold=int(largest_id),
            UpperThreshold=int(largest_id),
            InsideValue=1,
            OutsideValue=0,
        )
        imMath = tube.ImageMath.New(connected_component_image)
        imMath.Dilate(hole_fill, 1, 0)
        imMath.Erode(hole_fill, 1, 0)
        connected_component_image = imMath.GetOutputUChar()

        labelmap_arr = itk.GetArrayFromImage(labelmap_image)
        connected_component_arr = itk.GetArrayFromImage(connected_component_image)
        connected_component_mask = connected_component_arr > 0
        mask = label_arr & connected_component_mask
        labelmap_arr = np.where(mask, mask_id, labelmap_arr)
        results_image = itk.GetImageFromArray(labelmap_arr.astype(np.uint8))
        results_image.CopyInformation(preprocessed_image)

        return results_image

    def segment_contrast_agent(
        self, preprocessed_image: itk.image, labelmap_image: itk.image
    ) -> itk.image:
        """
        Include contrast-enhanced blood in the labelmap.

        Segments high-intensity regions corresponding to contrast-enhanced
        blood vessels and cardiac chambers. Uses connected component analysis
        focused on the middle slice where the heart is typically located.

        Args:
            preprocessed_image (itk.image): The preprocessed CT image
            labelmap_image (itk.image): Existing segmentation labelmap

        Returns:
            itk.image: Updated labelmap with contrast-enhanced regions labeled

        Note:
            Assumes the mid-z slice of the data contains the heart.

        Example:
            >>> contrast_labels = segmenter.segment_contrast_agent(preprocessed_image, base_labels)
        """
        thorasic_ids = (
            list(self.taxonomy.labels_in_group("heart").keys())
            + list(self.taxonomy.labels_in_group("lung").keys())
            + list(self.taxonomy.labels_in_group("major_vessels").keys())
            + [0]
        )
        contrast_ids = list(self.taxonomy.labels_in_group("contrast").keys())
        if len(contrast_ids) == 0:
            self.log_warning("No contrast-enhanced regions found in the labelmap")
            return labelmap_image

        results_image = self.segment_connected_component(
            preprocessed_image,
            labelmap_image,
            lower_threshold=self.contrast_threshold,
            upper_threshold=4000,
            labelmap_ids=thorasic_ids,
            mask_id=contrast_ids[-1],
            use_mid_slice=True,
            hole_fill=3,
        )

        return results_image

    def create_anatomy_group_masks(
        self, labelmap_image: itk.image
    ) -> dict[str, itk.image]:
        """
        Create binary masks for different anatomical groups from the labelmap.

        Generates separate binary masks for major anatomical systems by
        grouping related anatomical structures from the detailed labelmap.
        This is useful for motion analysis and visualization.

        Args:
            labelmap_image (itk.image): The detailed segmentation labelmap

        Returns:
            dict[str, itk.image]: Dictionary of binary masks keyed by group
                name. Exactly one entry per group registered in
                :attr:`taxonomy` (plus ``"other"``). The returned key set
                is segmenter-specific — callers that need a particular
                group should check membership (``"lung" in masks``) rather
                than assume a fixed schema.

        Example:
            >>> masks = segmenter.create_anatomy_group_masks(labelmap)
            >>> if "lung" in masks:
            ...     lung_mask = masks["lung"]
        """
        labelmap_arr = itk.GetArrayFromImage(labelmap_image)
        other_mask_arr = np.where(labelmap_arr > 0, 1, 0)

        masks: dict[str, itk.image] = {}
        for group_name in self.taxonomy.group_names():
            if group_name == AnatomyTaxonomy.OTHER_GROUP:
                continue
            group_ids = list(self.taxonomy.labels_in_group(group_name).keys())
            group_mask_arr = np.isin(labelmap_arr, group_ids).astype(np.uint8)
            other_mask_arr = np.where(group_mask_arr > 0, 0, other_mask_arr)
            group_mask = itk.GetImageFromArray(group_mask_arr)
            group_mask.CopyInformation(labelmap_image)
            masks[group_name] = group_mask

        other_mask = itk.GetImageFromArray(other_mask_arr.astype(np.uint8))
        other_mask.CopyInformation(labelmap_image)
        masks[AnatomyTaxonomy.OTHER_GROUP] = other_mask

        return masks

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Abstract method for image segmentation - must be implemented by subclasses.

        This method should contain the core segmentation algorithm specific to
        each implementation (e.g., TotalSegmentator).

        Args:
            preprocessed_image (itk.image): The preprocessed input image

        Returns:
            itk.image: The segmentation labelmap

        Raises:
            NotImplementedError: If called on the base class

        Note:
            This method must be implemented by subclasses to provide the
            specific segmentation algorithm.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def segment(
        self,
        input_image: itk.image,
        contrast_enhanced_study: bool = False,
    ) -> dict[str, itk.image]:
        """
        Perform complete chest CT segmentation.

        This is the main segmentation method that coordinates preprocessing,
        segmentation, contrast agent detection (if applicable), postprocessing,
        and anatomical group mask creation.

        Args:
            input_image (itk.image): The input 3D CT image to segment
            contrast_enhanced_study (bool): Whether the study uses contrast
                enhancement. If True, performs additional contrast agent
                segmentation to identify enhanced blood vessels

        Returns:
            dict[str, itk.image]: Dictionary containing:
                - "labelmap": Detailed segmentation labelmap
                - "lung": Binary mask of pulmonary structures
                - "heart": Binary mask of cardiac structures
                - "major_vessels": Binary mask of major blood vessels
                - "bone": Binary mask of skeletal structures
                - "soft_tissue": Binary mask of soft tissue organs
                - "other": Binary mask of remaining structures
                - "contrast": Binary mask of contrast-enhanced regions

        Example:
            >>> result = segmenter.segment(ct_image, contrast_enhanced_study=True)
            >>> labelmap = result['labelmap']
            >>> heart_mask = result['heart']
        """
        preprocessed_image = self.preprocess_input(input_image)

        labelmap_image = self.segmentation_method(preprocessed_image)

        labelmap_image = self.postprocess_labelmap(labelmap_image, input_image)

        if contrast_enhanced_study:
            labelmap_image = self.segment_contrast_agent(input_image, labelmap_image)

        masks = self.create_anatomy_group_masks(labelmap_image)

        labelmap_image = itk.GetImageFromArray(
            itk.GetArrayFromImage(labelmap_image).astype(np.uint8)
        )
        labelmap_image.CopyInformation(input_image)

        return {"labelmap": labelmap_image, **masks}
