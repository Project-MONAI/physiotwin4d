"""
Heart-gated CT processor implementing the complete 4D CT to USD workflow.

This module implements the complete pipeline for processing 4D cardiac CT images
as demonstrated in the Heart-GatedCT experiment notebooks.
"""

import logging
import os
from typing import Optional

import itk
import numpy as np
import pyvista as pv

from physiomotion4d import ConvertVTKToUSD
from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.transform_tools import TransformTools
from physiomotion4d.usd_anatomy_tools import USDAnatomyTools


class WorkflowConvertHeartGatedCTToUSD(PhysioMotion4DBase):
    """
    Complete workflow for Heart-gated CT images to dynamic USD models.

    This class implements the full workflow from 4D CT images to painted USD files
    suitable for visualization in NVIDIA Omniverse.
    """

    def __init__(
        self,
        input_filenames: list,
        contrast_enhanced: bool,
        output_directory: str,
        project_name: str,
        reference_image_filename: Optional[str] = None,
        number_of_registration_iterations: Optional[int] = 1,
        registration_method: str = "icon",
        log_level: int | str = logging.INFO,
    ):
        """
        Initialize the Heart-gated CT to USD workflow.

        Args:
            input_filenames (List): List of paths to the 3D NRRD files containing cardiac CT data.
                If there is only one file, it will be used as the 4D NRRD file.
            contrast_enhanced (bool): Whether the study uses contrast enhancement
            output_directory (str): Directory path where output files will be stored
            project_name (str): Project name for USD file organization
            reference_image_filename (Optional[str]): Path to reference image file
            number_of_registration_iterations (Optional[int]): Number of registration iterations
            registration_method (str): Registration method to use: 'ants' or 'icon' (default: 'icon')
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.input_filenames = input_filenames
        self.contrast_enhanced = contrast_enhanced
        self.output_directory = output_directory
        self.project_name = project_name
        self.reference_image_filename = reference_image_filename
        self.number_of_registration_iterations = number_of_registration_iterations

        # Validate registration method
        if registration_method not in ["ants", "icon"]:
            raise ValueError(
                f"Invalid registration_method '{registration_method}'. "
                "Must be 'ants' or 'icon'."
            )
        self.registration_method = registration_method

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Initialize processing components
        self.converter = ConvertNRRD4DTo3D(log_level=log_level)
        self.segmenter = SegmentChestTotalSegmentator(log_level=log_level)
        self.segmenter.contrast_threshold = 500

        # Initialize registration method
        self.registrar: RegisterImagesBase
        if self.registration_method == "ants":
            self.log_info("Initializing ANTs registration...")
            ants_registrar = RegisterImagesANTs(log_level=log_level)
            ants_registrar.set_modality("ct")
            ants_registrar.set_transform_type("SyN")
            if (
                number_of_registration_iterations is not None
                and number_of_registration_iterations > 0
            ):
                ants_registrar.set_number_of_iterations(
                    [
                        number_of_registration_iterations,
                        number_of_registration_iterations // 2,
                        0,
                    ]
                )
            self.registrar = ants_registrar
        else:  # icon (default)
            self.log_info("Initializing ICON registration...")
            icon_registrar = RegisterImagesICON(log_level=log_level)
            icon_registrar.set_modality("ct")
            if (
                number_of_registration_iterations is not None
                and number_of_registration_iterations > 0
            ):
                icon_registrar.set_number_of_iterations(
                    number_of_registration_iterations
                )
            self.registrar = icon_registrar

        self.registrar.set_mask_dilation(5)
        self.contour_tools = ContourTools()

        # Data storage for processing pipeline
        self._num_time_points = 0
        self._time_series_images: list[itk.Image] = []
        self._fixed_image: Optional[itk.Image] = None
        self._fixed_segmentation: Optional[dict[str, itk.Image]] = None
        self._time_series_transforms: list[itk.Transform] = []
        self._reference_contours: dict[str, pv.PolyData] = {}

    def process(self) -> str:
        """
        Execute the complete workflow from 4D CT to dynamic USD models.

        Returns:
            str: Path to the final dynamic anatomy USD file
        """
        self.log_section("Heart-gated CT Processing Pipeline")

        # Load and convert data
        self._load_time_series()

        # Segment and register all frames
        self._segment_and_register_frames()

        # Generate reference contours
        self._generate_reference_contours()

        # Transform contours for each time point
        self._transform_all_contours()

        # Create USD files
        self._create_usd_files()

        self.log_info("Processing pipeline completed successfully")
        return f"{self.project_name}.dynamic_anatomy_painted.usd"

    def _load_time_series(self) -> None:
        """Load and convert 4D data to time series images."""
        self.log_info("Loading time series data...")

        if len(self.input_filenames) == 1:
            self.converter.load_nrrd_4d(self.input_filenames[0])
            self.converter.save_3d_images(
                os.path.join(
                    self.output_directory, os.path.basename(self.input_filenames[0])
                )
            )
        else:
            self.log_info("Loading %d 3D NRRD files", len(self.input_filenames))
            self.converter.load_nrrd_3d(self.input_filenames)

        self._num_time_points = self.converter.get_number_of_3d_images()
        if self._num_time_points <= 0:
            raise ValueError("No time-series images were produced from input data")

        # Load all time series images into memory
        for i in range(self._num_time_points):
            self._time_series_images.append(self.converter.get_3d_image(i))
        if not self._time_series_images:
            raise ValueError("No time-series images were loaded from input data")

        # Load reference image
        if self.reference_image_filename:
            self._fixed_image = itk.imread(self.reference_image_filename)
        else:
            # Use 70% frame as reference if none specified.
            reference_frame = int(self._num_time_points * 0.7)
            self._fixed_image = self._time_series_images[reference_frame]
            itk.imwrite(
                self._fixed_image,
                os.path.join(self.output_directory, "fixed_image.mha"),
                compression=True,
            )

        self.log_info("Loaded %d time points", self._num_time_points)

    def _segment_and_register_frames(self) -> None:
        """Segment each frame and register to reference image."""
        self.log_info("Segmenting and registering frames...")

        # Segment reference image
        self.log_info("Segmenting reference image...")
        assert self._fixed_image is not None, "Fixed image must be set"
        self._fixed_segmentation = self.segmenter.segment(
            self._fixed_image, contrast_enhanced_study=self.contrast_enhanced
        )

        # Create combined masks for registration
        assert self._fixed_segmentation is not None, "Fixed segmentation must be set"
        labelmap_mask = self._fixed_segmentation["labelmap"]
        lung_mask = self._fixed_segmentation["lung"]
        heart_mask = self._fixed_segmentation["heart"]
        major_vessels_mask = self._fixed_segmentation["major_vessels"]
        bone_mask = self._fixed_segmentation["bone"]
        other_mask = self._fixed_segmentation["other"]
        contrast_mask = self._fixed_segmentation["contrast"]
        itk.imwrite(
            labelmap_mask,
            os.path.join(self.output_directory, "fixed_image_mask.mha"),
            compression=True,
        )

        # Create masks for different anatomy types
        heart_arr = itk.GetArrayFromImage(heart_mask)
        contrast_arr = itk.GetArrayFromImage(contrast_mask)
        major_vessels_arr = itk.GetArrayFromImage(major_vessels_mask)
        fixed_dynamic_mask = itk.GetImageFromArray(
            heart_arr + contrast_arr + major_vessels_arr
        )
        fixed_dynamic_mask.CopyInformation(self._fixed_image)

        lung_arr = itk.GetArrayFromImage(lung_mask)
        bone_arr = itk.GetArrayFromImage(bone_mask)
        other_arr = itk.GetArrayFromImage(other_mask)
        fixed_static_mask = itk.GetImageFromArray(lung_arr + bone_arr + other_arr)
        fixed_static_mask.CopyInformation(self._fixed_image)

        # Set up registrar with fixed image
        self.registrar.set_fixed_image(self._fixed_image)
        if self.registration_method == "icon" and isinstance(
            self.registrar, RegisterImagesICON
        ):
            if self.contrast_enhanced:
                self.registrar.set_mass_preservation(False)
            else:
                self.registrar.set_mass_preservation(True)

        # Process each time point
        self._time_series_transforms = []
        for i in range(self._num_time_points):
            self.log_progress(i + 1, self._num_time_points, prefix="Processing frames")

            moving_image = self._time_series_images[i]

            # Register without mask first
            self.registrar.set_fixed_mask(None)
            result_all = self.registrar.register(moving_image)
            inverse_transform_all = result_all["inverse_transform"]
            forward_transform_all = result_all["forward_transform"]
            itk.transformwrite(
                inverse_transform_all,
                os.path.join(self.output_directory, f"slice_{i:03d}_all_AB.hdf"),
                compression=True,
            )
            itk.transformwrite(
                forward_transform_all,
                os.path.join(self.output_directory, f"slice_{i:03d}_all_BA.hdf"),
                compression=True,
            )

            # Estimate the moving dynamic mask by the inverse transform of the fixed dynamic mask
            moving_dynamic_mask = TransformTools().transform_image(
                fixed_dynamic_mask, inverse_transform_all, moving_image, "nearest"
            )
            itk.imwrite(
                moving_dynamic_mask,
                os.path.join(self.output_directory, f"slice_{i:03d}_dynamic_mask.mha"),
                compression=True,
            )
            self.registrar.set_fixed_mask(fixed_dynamic_mask)
            result_dynamic = self.registrar.register(moving_image, moving_dynamic_mask)
            inverse_transform_dynamic = result_dynamic["inverse_transform"]
            forward_transform_dynamic = result_dynamic["forward_transform"]

            # Estimate the moving static mask by the inverse transform of the fixed static mask
            moving_static_mask = TransformTools().transform_image(
                fixed_static_mask, inverse_transform_all, moving_image, "nearest"
            )
            itk.imwrite(
                moving_static_mask,
                os.path.join(self.output_directory, f"slice_{i:03d}_static_mask.mha"),
                compression=True,
            )
            self.registrar.set_fixed_mask(fixed_static_mask)
            result_static = self.registrar.register(moving_image, moving_static_mask)
            inverse_transform_static = result_static["inverse_transform"]
            forward_transform_static = result_static["forward_transform"]

            # Store transforms
            transforms = {
                "dynamic": {
                    "inverse_transform": inverse_transform_dynamic,
                    "forward_transform": forward_transform_dynamic,
                },
                "static": {
                    "inverse_transform": inverse_transform_static,
                    "forward_transform": forward_transform_static,
                },
                "all": {
                    "inverse_transform": inverse_transform_all,
                    "forward_transform": forward_transform_all,
                },
            }
            itk.transformwrite(
                inverse_transform_dynamic,
                os.path.join(self.output_directory, f"slice_{i:03d}_dynamic_AB.hdf"),
                compression=True,
            )
            itk.transformwrite(
                forward_transform_dynamic,
                os.path.join(self.output_directory, f"slice_{i:03d}_dynamic_BA.hdf"),
                compression=True,
            )
            itk.transformwrite(
                inverse_transform_static,
                os.path.join(self.output_directory, f"slice_{i:03d}_static_AB.hdf"),
                compression=True,
            )
            itk.transformwrite(
                forward_transform_static,
                os.path.join(self.output_directory, f"slice_{i:03d}_static_BA.hdf"),
                compression=True,
            )
            self._time_series_transforms.append(transforms)

    def _generate_reference_contours(self) -> None:
        """Generate contour meshes from reference segmentation."""
        self.log_info("Generating reference contours...")

        assert self._fixed_segmentation is not None, "Fixed segmentation must be set"
        labelmap_image = self._fixed_segmentation["labelmap"]
        lung_mask = self._fixed_segmentation["lung"]
        heart_mask = self._fixed_segmentation["heart"]
        major_vessels_mask = self._fixed_segmentation["major_vessels"]
        bone_mask = self._fixed_segmentation["bone"]
        soft_tissue_mask = self._fixed_segmentation["soft_tissue"]
        other_mask = self._fixed_segmentation["other"]
        contrast_mask = self._fixed_segmentation["contrast"]

        # Generate all anatomy contours
        all_contours = self.contour_tools.extract_contours(labelmap_image)

        # Generate dynamic anatomy contours
        label_arr = itk.array_from_image(labelmap_image)
        heart_arr = itk.array_from_image(heart_mask)
        contrast_arr = itk.array_from_image(contrast_mask)
        major_vessels_arr = itk.array_from_image(major_vessels_mask)

        dynamic_anatomy_arr = np.maximum(heart_arr, contrast_arr)
        dynamic_anatomy_arr = np.maximum(dynamic_anatomy_arr, major_vessels_arr)
        dynamic_anatomy_arr = np.where(dynamic_anatomy_arr, label_arr, 0)
        dynamic_anatomy_image = itk.image_from_array(
            dynamic_anatomy_arr.astype(np.int16)
        )
        dynamic_anatomy_image.CopyInformation(labelmap_image)

        dynamic_contours = self.contour_tools.extract_contours(dynamic_anatomy_image)

        # Generate static anatomy contours
        lung_arr = itk.array_from_image(lung_mask)
        bone_arr = itk.array_from_image(bone_mask)
        soft_tissue_arr = itk.array_from_image(soft_tissue_mask)
        other_arr = itk.array_from_image(other_mask)

        static_anatomy_arr = lung_arr + bone_arr + soft_tissue_arr + other_arr
        static_anatomy_arr = np.where(static_anatomy_arr, label_arr, 0)
        static_anatomy_image = itk.image_from_array(static_anatomy_arr.astype(np.int16))
        static_anatomy_image.CopyInformation(labelmap_image)

        static_contours = self.contour_tools.extract_contours(static_anatomy_image)

        # Store reference contours
        self._reference_contours = {
            "all": all_contours,
            "dynamic": dynamic_contours,
            "static": static_contours,
        }

    def _transform_all_contours(self) -> None:
        """Transform contours for all time points using registration transforms."""
        self.log_info("Transforming contours for all time points...")

        self._transformed_contours: dict[str, list[pv.PolyData]] = {
            "all": [],
            "dynamic": [],
            "static": [],
        }

        for i in range(self._num_time_points):
            self.log_progress(
                i + 1, self._num_time_points, prefix="Transforming contours"
            )

            frame_contours = {}
            for anatomy_type in ["all", "dynamic", "static"]:
                # Get the forward transform for this anatomy type and frame
                forward_transform = self._time_series_transforms[i][anatomy_type][
                    "forward_transform"
                ]

                # Transform the reference contours
                transformed_contours = self.contour_tools.transform_contours(
                    self._reference_contours[anatomy_type],
                    forward_transform,
                    with_deformation_magnitude=False,
                )

                frame_contours[anatomy_type] = transformed_contours
                self._transformed_contours[anatomy_type].append(transformed_contours)

    def _create_usd_files(self) -> None:
        """Create painted USD files for all anatomy types."""
        self.log_info("Creating USD files...")

        # Create USD for each anatomy type
        for anatomy_type in ["all", "dynamic", "static"]:
            self.log_info("Creating %s anatomy USD...", anatomy_type)

            # Convert VTK contours to USD
            converter = ConvertVTKToUSD(
                self.project_name,
                self._transformed_contours[anatomy_type],
                self.segmenter.all_mask_ids,
                log_level=self.log_level,
            )
            usd_file = os.path.join(
                self.output_directory, f"{self.project_name}.{anatomy_type}.usd"
            )
            stage = converter.convert(usd_file)

            # Paint the USD file
            self.log_info("Painting %s anatomy USD...", anatomy_type)
            output_filename = os.path.join(
                self.output_directory, f"{self.project_name}.{anatomy_type}_painted.usd"
            )
            if os.path.exists(output_filename):
                os.remove(output_filename)
            painter = USDAnatomyTools(stage)
            painter.enhance_meshes(self.segmenter)
            stage.Export(output_filename)
