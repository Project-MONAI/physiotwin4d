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

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase


class SegmentAnatomyBase(PhysioMotion4DBase):
    """Base class for anatomy segmentation that provides common functionality for
    segmenting anatomy in CT images.

    This class implements preprocessing, postprocessing, and mask creation
    methods that are shared across different anatomy segmentation
    implementations. It defines anatomical structure mappings and provides
    utilities for image preprocessing, intensity rescaling, and mask generation.

    The class maintains dictionaries of anatomical structure IDs for different
    organ systems (heart, lungs, bones, vessels, etc.) and provides methods
    to create binary masks for specific anatomical groups.

    Attributes:
        target_spacing (float): Target isotropic spacing for resampling (default 1.5mm)
        rescale_intensity_range (bool): Whether to rescale intensity values
        contrast_threshold (int): Threshold for contrast agent detection (default 700)
        all_mask_ids (dict): Dictionary mapping all anatomical structure IDs to names
        heart_mask_ids (dict): Dictionary of cardiac structure IDs
        lung_mask_ids (dict): Dictionary of pulmonary structure IDs
        bone_mask_ids (dict): Dictionary of skeletal structure IDs
        major_vessels_mask_ids (dict): Dictionary of major vessel IDs
        contrast_mask_ids (dict): Dictionary of contrast-enhanced region IDs
        soft_tissue_mask_ids (dict): Dictionary of soft tissue structure IDs
        other_mask_ids (dict): Dictionary of remaining structure IDs
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the SegmentAnatomyBase class.

        Sets up default parameters for image preprocessing and anatomical
        structure ID mappings. Subclasses should call this constructor and
        then override the mask ID dictionaries with their specific mappings.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.target_spacing: float = 0.0

        self.rescale_intensity_range: bool = False
        self.input_intensity_scale_range: list[int] = [0, 4096]
        self.output_intensity_scale_range: list[int] = [-1024, 3071]
        self.output_intensity_clip_range: list[int] = [-1024, 3071]

        self.contrast_threshold: int = 700

        self.all_mask_ids: dict[int, str] = {}
        self.heart_mask_ids: dict[int, str] = {}
        self.major_vessels_mask_ids: dict[int, str] = {}
        self.lung_mask_ids: dict[int, str] = {}
        self.bone_mask_ids: dict[int, str] = {}
        self.contrast_mask_ids: dict[int, str] = {135: "contrast"}
        self.soft_tissue_mask_ids: dict[int, str] = {133: "soft_tissue"}
        self.other_mask_ids: dict[int, str] = {}

        # Subclasses should call this function to complete the mask ID setup
        # self.set_other_and_all_mask_ids()

    def set_other_and_all_mask_ids(self) -> None:
        """Set the other mask IDs and consolidate all mask ID dictionaries.

        Creates the 'other' category for any anatomical structures not classified
        into the specific organ systems, then combines all individual mask ID
        dictionaries into the master all_mask_ids dictionary.

        This method should be called after setting up the individual organ
        system mask ID dictionaries in subclasses.
        """
        self.other_mask_ids = {id: f"other_{id}" for id in range(1, 256)}
        for id in self.contrast_mask_ids.keys():
            self.other_mask_ids.pop(id)
        for id in self.soft_tissue_mask_ids.keys():
            self.other_mask_ids.pop(id)
        for id in self.heart_mask_ids.keys():
            self.other_mask_ids.pop(id)
        for id in self.major_vessels_mask_ids.keys():
            self.other_mask_ids.pop(id)
        for id in self.lung_mask_ids.keys():
            self.other_mask_ids.pop(id)
        for id in self.bone_mask_ids.keys():
            self.other_mask_ids.pop(id)
        self.all_mask_ids = {
            **self.contrast_mask_ids,
            **self.soft_tissue_mask_ids,
            **self.heart_mask_ids,
            **self.major_vessels_mask_ids,
            **self.lung_mask_ids,
            **self.bone_mask_ids,
            **self.other_mask_ids,
        }

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
            labelmap_ids = list(self.all_mask_ids.keys())
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
            list(self.heart_mask_ids.keys())
            + list(self.lung_mask_ids.keys())
            + list(self.major_vessels_mask_ids.keys())
            + [0]
        )
        results_image = self.segment_connected_component(
            preprocessed_image,
            labelmap_image,
            lower_threshold=self.contrast_threshold,
            upper_threshold=4000,
            labelmap_ids=thorasic_ids,
            mask_id=list(self.contrast_mask_ids.keys())[-1],
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
            dict[str, itk.image]: Dictionary of binary masks with keys:
                - "lung": Pulmonary structures (lungs, trachea, airways)
                - "heart": Cardiac structures (heart chambers, valves)
                - "major_vessels": Large blood vessels (aorta, vena cava)
                - "bone": Skeletal structures (ribs, vertebrae, sternum)
                - "soft_tissue": Soft tissue organs (liver, kidneys, etc.)
                - "other": Remaining anatomical structures
                - "contrast": Contrast-enhanced regions

        Example:
            >>> masks = segmenter.create_anatomy_group_masks(labelmap)
            >>> lung_mask = masks['lung']
            >>> heart_mask = masks['heart']
        """
        labelmap_arr = itk.GetArrayFromImage(labelmap_image)
        other_mask_arr = np.where(labelmap_arr > 0, 1, 0)

        lung_mask_arr = np.isin(labelmap_arr, list(self.lung_mask_ids.keys())).astype(
            np.uint8
        )
        other_mask_arr = np.where(lung_mask_arr > 0, 0, other_mask_arr)

        heart_mask_arr = np.isin(labelmap_arr, list(self.heart_mask_ids.keys())).astype(
            np.uint8
        )
        other_mask_arr = np.where(heart_mask_arr > 0, 0, other_mask_arr)

        major_vessels_mask_arr = np.isin(
            labelmap_arr, list(self.major_vessels_mask_ids.keys())
        ).astype(np.uint8)
        other_mask_arr = np.where(major_vessels_mask_arr > 0, 0, other_mask_arr)

        bone_mask_arr = np.isin(labelmap_arr, list(self.bone_mask_ids.keys())).astype(
            np.uint8
        )
        other_mask_arr = np.where(bone_mask_arr > 0, 0, other_mask_arr)

        soft_tissue_mask_arr = np.isin(
            labelmap_arr, list(self.soft_tissue_mask_ids.keys())
        ).astype(np.uint8)
        other_mask_arr = np.where(soft_tissue_mask_arr > 0, 0, other_mask_arr)

        contrast_mask_arr = np.isin(
            labelmap_arr, list(self.contrast_mask_ids.keys())
        ).astype(np.uint8)
        other_mask_arr = np.where(contrast_mask_arr > 0, 0, other_mask_arr)

        lung_mask = itk.GetImageFromArray(lung_mask_arr)
        lung_mask.CopyInformation(labelmap_image)
        heart_mask = itk.GetImageFromArray(heart_mask_arr)
        heart_mask.CopyInformation(labelmap_image)
        major_vessels_mask = itk.GetImageFromArray(major_vessels_mask_arr)
        major_vessels_mask.CopyInformation(labelmap_image)
        bone_mask = itk.GetImageFromArray(bone_mask_arr)
        bone_mask.CopyInformation(labelmap_image)
        soft_tissue_mask = itk.GetImageFromArray(soft_tissue_mask_arr)
        soft_tissue_mask.CopyInformation(labelmap_image)
        contrast_mask = itk.GetImageFromArray(contrast_mask_arr)
        contrast_mask.CopyInformation(labelmap_image)
        other_mask = itk.GetImageFromArray(other_mask_arr.astype(np.uint8))
        other_mask.CopyInformation(labelmap_image)

        return {
            "lung": lung_mask,
            "heart": heart_mask,
            "major_vessels": major_vessels_mask,
            "bone": bone_mask,
            "soft_tissue": soft_tissue_mask,
            "other": other_mask,
            "contrast": contrast_mask,
        }

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

    def dilate_mask(self, mask: itk.image, dilation: int) -> itk.image:
        """
        Dilate a binary mask using morphological operations.

        Expands the mask regions by the specified number of pixels to create
        larger regions of interest. Useful for creating candidate regions or
        ensuring complete coverage of anatomical structures.

        Args:
            mask (itk.image): The binary mask to dilate
            dilation (int): Number of pixels to dilate in each direction

        Returns:
            itk.image: The dilated binary mask

        Example:
            >>> dilated_heart = segmenter.dilate_mask(heart_mask, 5)
        """
        imMath = tube.ImageMath.New(mask)
        imMath.Dilate(dilation, 1, 0)
        dilated_mask = imMath.GetOutputUChar()
        return dilated_mask

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

        return {
            "labelmap": labelmap_image,
            "lung": masks["lung"],
            "heart": masks["heart"],
            "major_vessels": masks["major_vessels"],
            "bone": masks["bone"],
            "soft_tissue": masks["soft_tissue"],
            "other": masks["other"],
            "contrast": masks["contrast"],
        }
