"""High-resolution 4D CT reconstruction workflow.

This module provides the WorkflowReconstructHighres4DCT class for reconstructing
a high-resolution 4D CT time series from lower-resolution time-series images and
a single high-resolution reference image.

The workflow uses ANTS+ICON combined registration to:
1. Register each time-series image to the high-resolution reference
2. Apply inverse transforms to reconstruct high-resolution time series
3. Optionally upsample to the reference image resolution

This is particularly useful for cardiac CT where you have:
- Multiple low-resolution time-series images capturing cardiac motion
- One high-resolution static image for anatomical detail
- Goal: Combine both to create high-resolution dynamic images

Key Features:
    - Sequential time-series registration using RegisterTimeSeriesImages
    - Combined ANTS+ICON registration for optimal results
    - Bidirectional registration from reference frame
    - Optional temporal smoothing with prior transforms
    - High-resolution reconstruction with optional upsampling
    - No I/O operations (data passed in-memory)
"""

import logging
from typing import Optional

import itk

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_time_series_images import RegisterTimeSeriesImages


class WorkflowReconstructHighres4DCT(PhysioMotion4DBase):
    """Reconstruct high-resolution 4D CT from time series and reference image.

    This class implements a workflow for reconstructing high-resolution dynamic
    CT images by registering low-resolution time-series images to a high-resolution
    reference image using combined ANTS+ICON registration.

    **Registration Pipeline:**
        1. **Time Series Registration**: Register each time-series image to the
           high-resolution reference using RegisterTimeSeriesImages with ANTS_ICON method
        2. **Reconstruction**: Apply inverse transforms to reconstruct high-resolution
           time series
        3. **Optional Upsampling**: Resample to isotropic high resolution

    **Input Requirements:**
        - time_series_images: Ordered list of 3D images (typically lower resolution)
        - fixed_image: High-resolution reference image
        - All images should be in the same anatomical coordinate system

    Attributes:
        time_series_images (list[itk.Image]): Ordered list of time-series images
        fixed_image (itk.Image): High-resolution reference image
        reference_frame (int): Index of reference frame in time series
        register_reference (bool): Whether to register reference frame
        prior_weight (float): Weight for temporal smoothing (0.0-1.0)
        upsample_to_fixed_resolution (bool): Whether to upsample reconstruction
        registration_method (str): Registration method ('ANTS', 'ICON', or 'ANTS_ICON')
        number_of_iterations: Iterations for registration
        registrar (RegisterTimeSeriesImages): Internal registration object
        forward_transforms (list[itk.Transform]): one per frame; each warps its
            moving image onto the fixed grid
        inverse_transforms (list[itk.Transform]): one per frame; each warps the
            fixed image onto that frame's moving grid (used for reconstruction)
        losses (list[float]): Registration loss values
        reconstructed_images (list[itk.Image]): Reconstructed high-resolution images

    Example:
        >>> # Initialize workflow with data
        >>> workflow = WorkflowReconstructHighres4DCT(
        ...     time_series_images=lowres_images,
        ...     fixed_image=highres_reference,
        ...     reference_frame=3,
        ...     registration_method='ANTS_ICON',
        ... )
        >>>
        >>> # Configure registration parameters
        >>> workflow.set_number_of_iterations_ANTS([30, 15, 7])
        >>> workflow.set_number_of_iterations_ICON(20)
        >>> workflow.set_prior_weight(0.5)
        >>>
        >>> # Run complete workflow
        >>> result = workflow.run_workflow(upsample_to_fixed_resolution=True)
        >>>
        >>> # Access results
        >>> reconstructed = result['reconstructed_images']
        >>> transforms = result['forward_transforms']
        >>> losses = result['losses']
    """

    def __init__(
        self,
        time_series_images: list[itk.Image],
        fixed_image: itk.Image,
        reference_frame: int = 0,
        register_reference: bool = False,
        registration_method: str = "ANTS_ICON",
        log_level: int | str = logging.INFO,
    ):
        """Initialize the high-resolution 4D CT reconstruction workflow.

        Args:
            time_series_images (list[itk.Image]): Ordered list of 3D time-series images
                to be registered and reconstructed
            fixed_image (itk.Image): High-resolution 3D reference image
            reference_frame (int, optional): Index of the reference frame in the
                time series. Registration proceeds bidirectionally from this frame.
                Default: 0
            register_reference (bool, optional): If True, register the reference frame
                to the fixed image. If False, use identity transform for reference.
                Default: False
            registration_method (str, optional): Registration method to use.
                Options: 'ANTS', 'ICON', or 'ANTS_ICON'. Default: 'ANTS_ICON'
            log_level: Logging level (logging.DEBUG, logging.INFO, etc.).
                Default: logging.INFO

        Raises:
            ValueError: If time_series_images is empty
            ValueError: If reference_frame is out of range
            ValueError: If registration_method is invalid
        """
        # Initialize base class with logging
        super().__init__(
            class_name="WorkflowReconstructHighres4DCT", log_level=log_level
        )

        # Validate inputs
        if not time_series_images:
            raise ValueError("time_series_images cannot be empty")

        if reference_frame < 0 or reference_frame >= len(time_series_images):
            raise ValueError(
                f"reference_frame {reference_frame} out of range "
                f"[0, {len(time_series_images) - 1}]"
            )

        if registration_method not in ["ANTS", "ICON", "ANTS_ICON"]:
            raise ValueError(
                f"registration_method must be 'ANTS', 'ICON', or 'ANTS_ICON', "
                f"got '{registration_method}'"
            )

        # Store input data
        self.time_series_images = time_series_images
        self.fixed_image = fixed_image
        self.reference_frame = reference_frame
        self.register_reference = register_reference
        self.registration_method = registration_method

        # Initialize parameters with defaults
        self.prior_weight: float = 0.0
        self.upsample_to_fixed_resolution: bool = False
        self.modality: str = "ct"
        self.mask_dilation_mm: float = 0.0
        self.fixed_mask: Optional[itk.Image] = None
        self.moving_masks: Optional[list[Optional[itk.Image]]] = None

        # Set default number of iterations based on registration method
        self.number_of_iterations_ANTS: list[int] = [30, 15, 7, 3]
        self.number_of_iterations_ICON: int = 20

        # Initialize registrar
        self.registrar = RegisterTimeSeriesImages(
            registration_method=registration_method, log_level=log_level
        )

        # Results storage
        self.forward_transforms: Optional[list[itk.Transform]] = None
        self.inverse_transforms: Optional[list[itk.Transform]] = None
        self.losses: Optional[list[float]] = None
        self.reconstructed_images: Optional[list[itk.Image]] = None

    def set_number_of_iterations_ANTS(
        self, number_of_iterations_ANTS: list[int]
    ) -> None:
        """Set the number of iterations for ANTs registration.

        Args:
            number_of_iterations_ANTS: List of iterations for ANTs multi-resolution
                (e.g., [30, 15, 7, 3] for four resolution levels)
        """
        self.number_of_iterations_ANTS = number_of_iterations_ANTS

    def set_number_of_iterations_ICON(self, number_of_iterations_ICON: int) -> None:
        """Set the number of iterations for ICON registration.

        Args:
            number_of_iterations_ICON: Number of fine-tuning steps for ICON
        """
        self.number_of_iterations_ICON = number_of_iterations_ICON

    def set_prior_weight(self, prior_weight: float) -> None:
        """Set the weight for temporal smoothing with prior transforms.

        Args:
            prior_weight (float): Weight (0.0 to 1.0) for using the prior image's
                transform to initialize the next registration. 0.0 means no prior
                information is used (each registration starts from identity).
                Higher values provide more temporal smoothness but may propagate errors.

        Raises:
            ValueError: If prior_weight not in [0.0, 1.0]
        """
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError(f"prior_weight must be in [0.0, 1.0], got {prior_weight}")
        self.prior_weight = prior_weight

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        Args:
            modality (str): The imaging modality (e.g., 'ct', 'mri')
        """
        self.modality = modality

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        Args:
            mask_dilation_mm (float): The dilation in millimeters
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        Args:
            fixed_mask (itk.Image): Binary mask defining ROI in fixed image
        """
        self.fixed_mask = fixed_mask

    def set_moving_masks(
        self, moving_masks: Optional[list[Optional[itk.Image]]]
    ) -> None:
        """Set binary masks for the moving images.

        Args:
            moving_masks (list[itk.Image] | None): List of binary masks,
                one for each moving image. If None, no masks are used.
                Must have same length as time_series_images if provided.

        Raises:
            ValueError: If moving_masks length doesn't match time_series_images
        """
        if moving_masks is not None and len(moving_masks) != len(
            self.time_series_images
        ):
            raise ValueError(
                f"moving_masks length ({len(moving_masks)}) must match "
                f"time_series_images length ({len(self.time_series_images)})"
            )
        self.moving_masks = moving_masks

    def register_time_series(self) -> dict:
        """Register time series images to the fixed image.

        Performs sequential registration of all time-series images to the
        high-resolution reference image using the configured parameters.

        Returns:
            dict: Dictionary containing:
                - 'forward_transforms' (list[itk.Transform]): one per frame;
                  each warps its moving image onto the fixed grid
                - 'inverse_transforms' (list[itk.Transform]): one per frame;
                  each warps the fixed image onto that frame's moving grid
                  (see docs/developer/transform_conventions)
                - 'losses' (list[float]): Registration loss value for each image

        Raises:
            RuntimeError: If registration fails
        """
        self.log_section(
            "Stage 1: Time Series Registration (RegisterTimeSeriesImages)", width=70
        )

        # Configure registrar
        self.registrar.set_fixed_image(self.fixed_image)
        self.registrar.set_modality(self.modality)
        self.registrar.set_mask_dilation(self.mask_dilation_mm)
        self.registrar.set_number_of_iterations_ANTS(self.number_of_iterations_ANTS)
        self.registrar.set_number_of_iterations_ICON(self.number_of_iterations_ICON)
        self.registrar.set_fixed_mask(self.fixed_mask)

        self.log_info(f"Registration method: {self.registration_method}")
        self.log_info(f"Number of time points: {len(self.time_series_images)}")
        self.log_info(f"Reference frame: {self.reference_frame}")
        self.log_info(f"Register reference: {self.register_reference}")
        self.log_info(f"Prior weight: {self.prior_weight}")
        self.log_info(f"Number of iterations (ANTs): {self.number_of_iterations_ANTS}")
        self.log_info(f"Number of iterations (ICON): {self.number_of_iterations_ICON}")

        # Perform registration
        result = self.registrar.register_time_series(
            moving_images=self.time_series_images,
            moving_masks=self.moving_masks,
            reference_frame=self.reference_frame,
            register_reference=self.register_reference,
            prior_weight=self.prior_weight,
        )

        # Store results
        self.forward_transforms = result["forward_transforms"]
        self.inverse_transforms = result["inverse_transforms"]
        self.losses = result["losses"]

        self.log_info("Stage 1 complete: Time series registration finished.")
        self.log_info(f"  Average loss: {sum(self.losses) / len(self.losses):.6f}")
        self.log_info(f"  Min loss: {min(self.losses):.6f}")
        self.log_info(f"  Max loss: {max(self.losses):.6f}")

        return {
            "forward_transforms": self.forward_transforms,
            "inverse_transforms": self.inverse_transforms,
            "losses": self.losses,
        }

    def reconstruct_time_series(
        self, upsample_to_fixed_resolution: bool = False
    ) -> dict:
        """Reconstruct high-resolution time series using inverse transforms.

        Applies the inverse transforms from registration to reconstruct each
        time-series image in the high-resolution fixed image space.

        Args:
            upsample_to_fixed_resolution (bool, optional): If True, reconstructed
                images will be upsampled to isotropic resolution (mean of fixed
                image's X and Y spacing) while maintaining their original origin
                and direction. Default: False

        Returns:
            dict: Dictionary containing:
                - 'reconstructed_images' (list[itk.Image]): Reconstructed high-resolution
                  time-series images

        Raises:
            RuntimeError: If reconstruction fails
            ValueError: If inverse_transforms is not set (call register_time_series first)
        """
        if self.inverse_transforms is None:
            raise ValueError(
                "inverse_transforms not set. Call register_time_series() first."
            )

        self.log_section(
            "Stage 2: High-Resolution Time Series Reconstruction", width=70
        )

        self.log_info(f"Upsampling to fixed resolution: {upsample_to_fixed_resolution}")

        # Reconstruct time series
        self.reconstructed_images = self.registrar.reconstruct_time_series(
            moving_images=self.time_series_images,
            inverse_transforms=self.inverse_transforms,
            upsample_to_fixed_resolution=upsample_to_fixed_resolution,
        )

        self.log_info("Stage 2 complete: Time series reconstruction finished.")
        self.log_info(f"  Reconstructed {len(self.reconstructed_images)} images")

        # Log image properties for first reconstructed image
        if self.reconstructed_images:
            img = self.reconstructed_images[0]
            self.log_info(f"  Reconstructed image size: {itk.size(img)}")
            self.log_info(f"  Reconstructed image spacing: {itk.spacing(img)}")

        return {"reconstructed_images": self.reconstructed_images}

    def run_workflow(self, upsample_to_fixed_resolution: bool = False) -> dict:
        """Execute the complete high-resolution 4D CT reconstruction workflow.

        Runs the full pipeline:
        1. Register time series to high-resolution reference
        2. Reconstruct high-resolution time series using inverse transforms

        Args:
            upsample_to_fixed_resolution (bool, optional): If True, reconstructed
                images will be upsampled to isotropic high resolution.
                Default: False

        Returns:
            dict: Dictionary containing all results:
                - 'forward_transforms' (list[itk.Transform]): Registration transforms
                - 'inverse_transforms' (list[itk.Transform]): Inverse transforms
                - 'losses' (list[float]): Registration loss values
                - 'reconstructed_images' (list[itk.Image]): Reconstructed high-res images

        Raises:
            RuntimeError: If any stage of the workflow fails
        """
        self.log_section(
            "STARTING HIGH-RESOLUTION 4D CT RECONSTRUCTION WORKFLOW", width=70
        )

        self.log_info("Input configuration:")
        self.log_info(f"  Number of time points: {len(self.time_series_images)}")
        self.log_info(f"  Registration method: {self.registration_method}")
        self.log_info(f"  Reference frame: {self.reference_frame}")
        self.log_info(f"  Prior weight: {self.prior_weight}")
        self.log_info(f"  Upsample reconstruction: {upsample_to_fixed_resolution}")

        # Stage 1: Register time series
        _ = self.register_time_series()

        # Stage 2: Reconstruct high-resolution time series
        _ = self.reconstruct_time_series(
            upsample_to_fixed_resolution=upsample_to_fixed_resolution
        )

        self.log_section("RECONSTRUCTION WORKFLOW COMPLETE", width=70)
        assert self.reconstructed_images is not None, "Reconstructed images must be set"
        self.log_info(
            f"Successfully reconstructed {len(self.reconstructed_images)} "
            "high-resolution images"
        )

        return {
            "forward_transforms": self.forward_transforms,
            "inverse_transforms": self.inverse_transforms,
            "losses": self.losses,
            "reconstructed_images": self.reconstructed_images,
        }
