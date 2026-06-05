"""Time series image registration implementation.

This module provides the RegisterTimeSeriesImages class for registering an ordered
sequence of images (time series) to a fixed image. It supports ANTs, Greedy,
ICON, and combined ANTs/Greedy initialization followed by ICON refinement.

The class is particularly useful for 4D medical imaging applications such as cardiac
CT where sequential frames need to be registered to a common frame.
"""

import logging
from typing import Optional, Union, cast

import itk

from physiomotion4d.register_images_ants import RegisterImagesANTS
from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.register_images_greedy import RegisterImagesGreedy
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.transform_tools import TransformTools

REGISTRATION_METHODS: list[str] = [
    "ANTS",
    "greedy",
    "ICON",
    "ANTS_ICON",
    "greedy_ICON",
]


class RegisterTimeSeriesImages(RegisterImagesBase):
    """Register a time series of images to a fixed image.

    This class extends RegisterImagesBase to provide sequential registration of
    multiple images (time series) to a fixed image. It supports ANTs, Greedy,
    ICON, and combined ANTs/Greedy initialization followed by ICON refinement.
    It can propagate information from prior registrations to initialize
    subsequent ones.

    The registration proceeds in two passes from a reference frame:
    1. Forward pass: from reference_frame to the end of the series
    2. Backward pass: from reference_frame-1 to the beginning

    This bidirectional approach helps maintain temporal coherence in the
    registration results.

    Key features:
    - Sequential registration of ordered image lists
    - Support for ANTs, Greedy, ICON, ANTs+ICON, and Greedy+ICON backends
    - Optional use of prior transforms to initialize next registration
    - Configurable starting point in the time series
    - Returns all transforms and loss values for the entire series

    Attributes:
        registration_method_name (str): Registration method in use ('ANTS',
            'greedy', 'ICON', 'ANTS_ICON', or 'greedy_ICON').
        registrar_ANTS (RegisterImagesANTS): Internal ANTs registrar.
        registrar_greedy (RegisterImagesGreedy): Internal Greedy registrar.
        registrar_ICON (RegisterImagesICON): Internal ICON registrar (also used
            as the refinement stage for 'ANTS_ICON' and 'greedy_ICON').
        transform_tools (TransformTools): Utility for transform operations.

    Example:
        >>> # Register a cardiac CT time series
        >>> registrar = RegisterTimeSeriesImages(registration_method='ANTS')
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(fixed_image)
        >>> registrar.set_number_of_iterations_ANTS([40, 20, 10])
        >>>
        >>> # Register all time points to fixed image
        >>> result = registrar.register_time_series(
        ...     moving_images=time_series_images,
        ...     reference_frame=5,  # Start from middle of cardiac cycle
        ...     register_reference=True,
        ...     prior_weight=0.5,
        ... )
        >>>
        >>> forward_tfms = result['forward_transforms']  # warp moving images -> fixed grid
        >>> inverse_tfms = result['inverse_transforms']  # warp fixed image -> moving grids
        >>> losses = result['losses']
        >>>
        >>> # Reconstruct time series with optional upsampling
        >>> reconstructed = registrar.reconstruct_time_series(
        ...     moving_images=time_series_images,
        ...     inverse_transforms=inverse_tfms,
        ...     upsample_to_fixed_resolution=True,
        ... )
    """

    def __init__(
        self, registration_method: str = "ANTS", log_level: int | str = logging.INFO
    ) -> None:
        """Initialize the time series image registration class.

        Args:
            registration_method (str): Registration method to use.
                Options: 'ANTS', 'greedy', 'ICON', 'ANTS_ICON', or
                'greedy_ICON'. Default: 'ANTS'
            log_level: Logging level (default: logging.INFO)

        Raises:
            ValueError: If registration_method is not supported.
        """
        super().__init__(log_level=log_level)

        self.registration_method_name: str = registration_method

        self.registrar_ANTS = RegisterImagesANTS(log_level=log_level)
        self.registrar_greedy = RegisterImagesGreedy(log_level=log_level)
        self.registrar_ICON = RegisterImagesICON(log_level=log_level)

        # Set default iterations based on registration method
        self.number_of_iterations_ANTS: list[int] = [40, 20, 10]
        self.number_of_iterations_greedy: list[int] = [40, 20, 10]
        self.number_of_iterations_ICON: int = 50

        if self.registration_method_name not in REGISTRATION_METHODS:
            raise ValueError(
                "registration_method must be one of "
                f"{REGISTRATION_METHODS}, got '{registration_method}'"
            )

        self.transform_tools: TransformTools = TransformTools()

        self.smooth_prior_transform_sigma: float = 0.5

    def set_number_of_iterations_ANTS(
        self, number_of_iterations_ANTS: list[int]
    ) -> None:
        """Set the number of iterations for ANTs registration.

        Args:
            number_of_iterations_ANTS: List of iterations for ANTs multi-resolution
                (e.g., [40, 20, 10] for three resolution levels)
        """
        self.number_of_iterations_ANTS = number_of_iterations_ANTS

    def set_number_of_iterations_ICON(self, number_of_iterations_ICON: int) -> None:
        """Set the number of iterations for ICON registration.

        Args:
            number_of_iterations_ICON: Number of fine-tuning steps for ICON
        """
        self.number_of_iterations_ICON = number_of_iterations_ICON

    def set_number_of_iterations_greedy(
        self, number_of_iterations_greedy: list[int]
    ) -> None:
        """Set the number of iterations for Greedy registration.

        Args:
            number_of_iterations_greedy: List of iterations per Greedy
                resolution level, for example ``[40, 20, 10]``.
        """
        self.number_of_iterations_greedy = number_of_iterations_greedy

    def set_smooth_prior_transform_sigma(
        self, smooth_prior_transform_sigma: float
    ) -> None:
        """Set the sigma for smoothing the prior transform.

        Args:
            smooth_prior_transform_sigma (float): Sigma for smoothing the prior transform.
        """
        self.smooth_prior_transform_sigma = smooth_prior_transform_sigma

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        This passes through to the underlying registration method.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        This passes through to the underlying registration method.

        Args:
            modality (str): The imaging modality (e.g., 'ct', 'mri')
        """
        self.modality = modality

    def set_fixed_image(self, fixed_image: itk.Image) -> None:
        """Set the fixed image for registration.

        All moving images in the time series will be registered to this
        fixed image.

        Args:
            fixed_image (itk.Image): The 3D fixed image
        """
        self.fixed_image = fixed_image

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        This passes through to the underlying registration method.

        Args:
            fixed_mask (itk.Image): Binary mask defining ROI
        """
        self.fixed_mask = fixed_mask

    def set_fixed_labelmap(self, fixed_labelmap: Optional[itk.Image]) -> None:
        """Set a labelmap for the fixed image region of interest.

        This passes through to the underlying registration method.

        Args:
            fixed_labelmap (Optional[itk.Image]): Labelmap defining ROI
        """
        self.fixed_labelmap = fixed_labelmap

    def register_time_series(
        self,
        moving_images: list[itk.Image],
        moving_masks: Optional[list[Optional[itk.Image]]] = None,
        moving_labelmaps: Optional[list[Optional[itk.Image]]] = None,
        reference_frame: int = 0,
        register_reference: bool = True,
        prior_weight: float = 0.0,
    ) -> dict[str, list[itk.Transform] | list[float]]:
        """Register a time series of images to the fixed image.

        This method registers an ordered sequence of images to a common fixed
        frame. Registration proceeds bidirectionally from a reference frame:
        forward to the end and backward to the beginning.

        For each image after the reference image, the method can optionally use
        the transform from the previous image to initialize the registration,
        which can improve convergence and temporal coherence.

        Args:
            moving_images (list[itk.Image]): List of 3D images to register
            moving_masks (list[itk.Image], optional): List of binary masks,
                one for each moving image. If None, no masks are used. If provided,
                must have the same length as moving_images. Default: None
            moving_labelmaps (list[itk.Image], optional): Per-frame multi-label
                segmentations, one for each moving image. If None, no labelmaps are
                used. If provided, must have the same length as moving_images. Default: None
            reference_frame (int, optional): Index of the reference image to register first.
                Registration proceeds forward from this index to the end, then
                backward from this index to the beginning. Default: 0
            register_reference (bool, optional): If True, register the
                reference image to the fixed image. If False, use identity transform
                for the reference image. Default: True
            prior_weight (float, optional):
                Weight (0.0 to 1.0) for using the prior image's transform to
                initialize the next registration. 0.0 means no prior information
                is used (each registration starts from identity). Higher values
                provide more temporal smoothness but may propagate errors.
                Default: 0.0

        Returns:
            dict: Dictionary containing results:
                - "forward_transforms" (list[itk.Transform]): one per image;
                  each warps its moving image onto the fixed grid (warping
                  moving points/landmarks into fixed space uses the matching
                  inverse transform instead -- see
                  docs/developer/transform_conventions)
                - "inverse_transforms" (list[itk.Transform]): one per image;
                  each warps the fixed image onto that moving image's grid
                  (used by reconstruct_time_series)
                - "losses" (list[float]): Registration loss value for each image

        Raises:
            ValueError: If fixed_image is not set
            ValueError: If reference_frame is out of range
            ValueError: If prior_weight not in [0, 1]
            ValueError: If moving_masks length doesn't match moving_images length

        Note:
            The method compares registration with identity initialization versus
            prior transform initialization and selects the result with lower loss.
            This helps prevent error propagation in the temporal sequence.

            The fixed image mask can be set using set_fixed_mask() before
            calling this method.

        Example:
            >>> registrar = RegisterTimeSeriesImages(registration_method='ANTS')
            >>> registrar.set_fixed_image(fixed_image)
            >>> registrar.set_fixed_mask(fixed_mask)  # Optional
            >>> registrar.set_number_of_iterations_ANTS([30, 15, 5])
            >>>
            >>> # Use new intuitive parameter names
            >>> result = registrar.register_time_series(
            ...     moving_images=image_list,
            ...     moving_masks=mask_list,  # Optional
            ...     moving_labelmaps=labelmap_list,  # Optional
            ...     reference_frame=5,
            ...     register_reference=True,
            ...     prior_weight=0.5,
            ... )
            >>>
            >>> # Access results using new intuitive names
            >>> for i, (forward_tfm, loss) in enumerate(
            ...     zip(result['forward_transforms'], result['losses'])
            ... ):
            ...     # Apply forward transform to align moving image i to fixed
            ...     registered = transform_tools.transform_image(
            ...         moving_images[i], forward_tfm, fixed_image
            ...     )
        """
        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before registering time series")

        if self.registration_method_name == "ANTS":
            self.registrar_ANTS.set_fixed_image(self.fixed_image)
            self.registrar_ANTS.set_modality(self.modality)
            self.registrar_ANTS.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_ANTS.set_number_of_iterations(self.number_of_iterations_ANTS)
            self.registrar_ANTS.set_fixed_mask(self.fixed_mask)
            self.registrar_ANTS.set_fixed_labelmap(self.fixed_labelmap)
        elif self.registration_method_name == "greedy":
            self.registrar_greedy.set_fixed_image(self.fixed_image)
            self.registrar_greedy.set_modality(self.modality)
            self.registrar_greedy.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_greedy.set_number_of_iterations(
                self.number_of_iterations_greedy
            )
            self.registrar_greedy.set_fixed_mask(self.fixed_mask)
            self.registrar_greedy.set_fixed_labelmap(self.fixed_labelmap)
        elif self.registration_method_name == "ICON":
            self.registrar_ICON.set_fixed_image(self.fixed_image)
            self.registrar_ICON.set_modality(self.modality)
            self.registrar_ICON.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_ICON.set_number_of_iterations(self.number_of_iterations_ICON)
            self.registrar_ICON.set_fixed_mask(self.fixed_mask)
            self.registrar_ICON.set_fixed_labelmap(self.fixed_labelmap)
        elif self.registration_method_name in ["ANTS_ICON", "greedy_ICON"]:
            if self.registration_method_name == "ANTS_ICON":
                self.registrar_ANTS.set_fixed_image(self.fixed_image)
                self.registrar_ANTS.set_modality(self.modality)
                self.registrar_ANTS.set_mask_dilation(self.mask_dilation_mm)
                self.registrar_ANTS.set_number_of_iterations(
                    self.number_of_iterations_ANTS
                )
                self.registrar_ANTS.set_fixed_mask(self.fixed_mask)
                self.registrar_ANTS.set_fixed_labelmap(self.fixed_labelmap)
            else:
                self.registrar_greedy.set_fixed_image(self.fixed_image)
                self.registrar_greedy.set_modality(self.modality)
                self.registrar_greedy.set_mask_dilation(self.mask_dilation_mm)
                self.registrar_greedy.set_number_of_iterations(
                    self.number_of_iterations_greedy
                )
                self.registrar_greedy.set_fixed_mask(self.fixed_mask)
                self.registrar_greedy.set_fixed_labelmap(self.fixed_labelmap)
            self.registrar_ICON.set_fixed_image(self.fixed_image)
            self.registrar_ICON.set_modality(self.modality)
            self.registrar_ICON.set_mask_dilation(self.mask_dilation_mm)
            self.registrar_ICON.set_number_of_iterations(self.number_of_iterations_ICON)
            self.registrar_ICON.set_fixed_mask(self.fixed_mask)
            self.registrar_ICON.set_fixed_labelmap(self.fixed_labelmap)

        num_images = len(moving_images)

        if reference_frame < 0 or reference_frame >= num_images:
            raise ValueError(
                f"reference_frame {reference_frame} out of range [0, {num_images - 1}]"
            )

        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("prior_weight must be in [0.0, 1.0]")

        if moving_masks is not None and len(moving_masks) != num_images:
            raise ValueError(
                f"moving_masks length ({len(moving_masks)}) must match "
                f"moving_images length ({num_images})"
            )

        if moving_labelmaps is not None and len(moving_labelmaps) != num_images:
            raise ValueError(
                f"moving_labelmaps length ({len(moving_labelmaps)}) must match "
                f"moving_images length ({num_images})"
            )

        # Initialize result lists
        forward_transforms: list[Optional[itk.Transform]] = [None] * num_images
        inverse_transforms: list[Optional[itk.Transform]] = [None] * num_images
        losses = [0.0] * num_images

        # Create identity transform for fixed image
        identity_tfm = itk.IdentityTransform[itk.D, 3].New()
        identity_tfm = (
            self.transform_tools.convert_transform_to_displacement_field_transform(
                identity_tfm, self.fixed_image
            )
        )

        # Register the reference frame image
        if register_reference:
            reference_mask = (
                moving_masks[reference_frame] if moving_masks is not None else None
            )
            reference_labelmap = (
                moving_labelmaps[reference_frame]
                if moving_labelmaps is not None
                else None
            )
            if self.registration_method_name == "ANTS":
                result = self.registrar_ANTS.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                    moving_labelmap=reference_labelmap,
                )
            elif self.registration_method_name == "greedy":
                result = self.registrar_greedy.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                    moving_labelmap=reference_labelmap,
                )
            elif self.registration_method_name == "ICON":
                result = self.registrar_ICON.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                    moving_labelmap=reference_labelmap,
                )
            elif self.registration_method_name in ["ANTS_ICON", "greedy_ICON"]:
                registrar_initial = (
                    self.registrar_ANTS
                    if self.registration_method_name == "ANTS_ICON"
                    else self.registrar_greedy
                )
                result = registrar_initial.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                    moving_labelmap=reference_labelmap,
                )
                forward_initial = result["forward_transform"]
                result = self.registrar_ICON.register(
                    moving_images[reference_frame],
                    moving_mask=reference_mask,
                    moving_labelmap=reference_labelmap,
                    initial_forward_transform=forward_initial,
                )
            else:
                raise ValueError(
                    f"Invalid registration method: {self.registration_method_name}"
                )
            forward_transform = result["forward_transform"]
            inverse_transform = result["inverse_transform"]
            loss = result["loss"]
        else:
            # Use identity transform for reference frame
            forward_transform = identity_tfm
            inverse_transform = identity_tfm
            loss = 0.0

        forward_transforms[reference_frame] = forward_transform
        inverse_transforms[reference_frame] = inverse_transform
        losses[reference_frame] = loss

        # Compute prior transform for reference frame if needed
        prior_forward_ref = None
        if prior_weight > 0.0:
            prior_forward_ref = (
                self.transform_tools.combine_displacement_field_transforms(
                    identity_tfm,
                    forward_transform,
                    self.fixed_image,
                    tfm1_weight=1.0,
                    tfm2_weight=prior_weight,
                    tfm1_blur_sigma=0.0,
                    tfm2_blur_sigma=0.5,
                    mode="add",
                )
            )

        # Register forward and backward from reference frame
        for step, start_idx, end_idx in [
            (1, reference_frame + 1, num_images),  # Forward pass
            (-1, reference_frame - 1, -1),  # Backward pass
        ]:
            prior_forward = prior_forward_ref

            for img_idx in range(start_idx, end_idx, step):
                moving_image = moving_images[img_idx]
                moving_mask = (
                    moving_masks[img_idx] if moving_masks is not None else None
                )
                moving_labelmap = (
                    moving_labelmaps[img_idx] if moving_labelmaps is not None else None
                )

                # Try registration with identity initialization
                if self.registration_method_name == "ANTS":
                    result_init_identity = self.registrar_ANTS.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                        moving_labelmap=moving_labelmap,
                    )
                elif self.registration_method_name == "greedy":
                    result_init_identity = self.registrar_greedy.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                        moving_labelmap=moving_labelmap,
                    )
                elif self.registration_method_name == "ICON":
                    result_init_identity = self.registrar_ICON.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                        moving_labelmap=moving_labelmap,
                    )
                elif self.registration_method_name in ["ANTS_ICON", "greedy_ICON"]:
                    registrar_initial = (
                        self.registrar_ANTS
                        if self.registration_method_name == "ANTS_ICON"
                        else self.registrar_greedy
                    )
                    result_init_identity = registrar_initial.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                        moving_labelmap=moving_labelmap,
                    )
                    forward_initial = result_init_identity["forward_transform"]
                    result_init_identity = self.registrar_ICON.register(
                        moving_image=moving_image,
                        moving_mask=moving_mask,
                        moving_labelmap=moving_labelmap,
                        initial_forward_transform=forward_initial,
                    )
                else:
                    raise ValueError(
                        f"Invalid registration method: {self.registration_method_name}"
                    )
                forward_init_identity = result_init_identity["forward_transform"]
                inverse_init_identity = result_init_identity["inverse_transform"]
                loss_init_identity = result_init_identity["loss"]

                # Select best result based on prior usage
                if prior_weight > 0.0:
                    # Try with prior transform initialization
                    if self.registration_method_name == "ANTS":
                        result_init_prior = self.registrar_ANTS.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            moving_labelmap=moving_labelmap,
                            initial_forward_transform=prior_forward,
                        )
                    elif self.registration_method_name == "greedy":
                        result_init_prior = self.registrar_greedy.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            moving_labelmap=moving_labelmap,
                            initial_forward_transform=prior_forward,
                        )
                    elif self.registration_method_name == "ICON":
                        result_init_prior = self.registrar_ICON.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            moving_labelmap=moving_labelmap,
                            initial_forward_transform=prior_forward,
                        )
                    elif self.registration_method_name in ["ANTS_ICON", "greedy_ICON"]:
                        registrar_initial = (
                            self.registrar_ANTS
                            if self.registration_method_name == "ANTS_ICON"
                            else self.registrar_greedy
                        )
                        result_init_prior = registrar_initial.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            moving_labelmap=moving_labelmap,
                            initial_forward_transform=prior_forward,
                        )
                        forward_initial = result_init_prior["forward_transform"]
                        result_init_prior = self.registrar_ICON.register(
                            moving_image=moving_image,
                            moving_mask=moving_mask,
                            moving_labelmap=moving_labelmap,
                            initial_forward_transform=forward_initial,
                        )
                    else:
                        raise ValueError(
                            f"Invalid registration method: {self.registration_method_name}"
                        )
                    forward_init_prior = result_init_prior["forward_transform"]
                    inverse_init_prior = result_init_prior["inverse_transform"]
                    loss_init_prior = result_init_prior["loss"]

                    # Select result with lower loss
                    if loss_init_identity < loss_init_prior:
                        # Identity initialization was better
                        prior_forward = identity_tfm
                        forward_transform = forward_init_identity
                        inverse_transform = inverse_init_identity
                        loss = loss_init_identity
                    else:
                        # Prior initialization was better
                        forward_transform = forward_init_prior
                        inverse_transform = inverse_init_prior
                        loss = loss_init_prior

                    # Update prior for next iteration
                    prior_forward = (
                        self.transform_tools.combine_displacement_field_transforms(
                            identity_tfm,
                            forward_transform,
                            self.fixed_image,
                            tfm1_weight=1.0,
                            tfm2_weight=prior_weight,
                            tfm1_blur_sigma=0.0,
                            tfm2_blur_sigma=self.smooth_prior_transform_sigma,
                            mode="add",
                        )
                    )
                else:
                    # No prior usage, just use identity result
                    forward_transform = forward_init_identity
                    inverse_transform = inverse_init_identity
                    loss = loss_init_identity

                # Store results
                forward_transforms[img_idx] = forward_transform
                inverse_transforms[img_idx] = inverse_transform
                losses[img_idx] = loss

        assert all(t is not None for t in forward_transforms)
        assert all(t is not None for t in inverse_transforms)
        return {
            "forward_transforms": [t for t in forward_transforms if t is not None],
            "inverse_transforms": [t for t in inverse_transforms if t is not None],
            "losses": losses,
        }

    def reconstruct_time_series(
        self,
        moving_images: list[itk.Image],
        inverse_transforms: list[itk.Transform],
        upsample_to_fixed_resolution: bool = False,
    ) -> list[itk.Image]:
        """Reconstruct time series images using inverse transforms.

        This method applies the inverse transforms to reconstruct each moving image
        in the fixed image space. If upsample_to_fixed_resolution is enabled,
        the reconstructed images will use isotropic spacing (mean of fixed image's
        X and Y spacing) while maintaining each moving image's original origin and direction.

        Args:
            moving_images (list[itk.Image]): List of moving images to reconstruct
            inverse_transforms (list[itk.Transform]): List of inverse transforms
                (one per moving image), each used to warp the fixed image onto
                that moving image's grid
            upsample_to_fixed_resolution (bool, optional): If True, reconstructed
                images will be upsampled to isotropic resolution (mean of fixed image's
                X and Y spacing) while maintaining their original origin and direction.
                Default: False

        Returns:
            list[itk.Image]: List of reconstructed images in fixed image space

        Raises:
            ValueError: If fixed_image is not set
            ValueError: If lengths of moving_images and inverse_transforms don't match

        Example:
            >>> registrar = RegisterTimeSeriesImages(registration_method='ANTS')
            >>> registrar.set_fixed_image(fixed_image)
            >>>
            >>> result = registrar.register_time_series(
            ...     moving_images=time_series_images,
            ...     reference_frame=0,
            ... )
            >>>
            >>> reconstructed_images = registrar.reconstruct_time_series(
            ...     moving_images=time_series_images,
            ...     inverse_transforms=result['inverse_transforms'],
            ...     upsample_to_fixed_resolution=True,
            ... )
        """
        if self.fixed_image is None:
            raise ValueError(
                "Fixed image must be set before reconstructing time series"
            )

        if len(moving_images) != len(inverse_transforms):
            raise ValueError(
                f"Number of moving images ({len(moving_images)}) must match "
                f"number of inverse transforms ({len(inverse_transforms)})"
            )

        reconstructed_images: list[itk.Image] = []

        for moving_image, inverse_transform in zip(moving_images, inverse_transforms):
            if upsample_to_fixed_resolution:
                # Create a reference image with isotropic spacing (mean of fixed image's
                # X and Y spacing) and moving image's origin and direction
                reference_image = self._create_upsampled_reference(
                    moving_image, self.fixed_image
                )
            else:
                # Use fixed image as reference
                reference_image = moving_image

            # Transform the moving image to the reference space
            reconstructed = self.transform_tools.transform_image(
                self.fixed_image, inverse_transform, reference_image
            )
            reconstructed_images.append(reconstructed)

        return reconstructed_images

    def _create_upsampled_reference(
        self, moving_image: itk.Image, fixed_image: itk.Image
    ) -> itk.Image:
        """Create a reference image with isotropic spacing and moving image origin/direction.

        The spacing is calculated as the mean of the fixed image's X and Y spacing,
        applied to all three dimensions (X, Y, Z) for isotropic resolution.

        Args:
            moving_image (itk.Image): Image providing origin and direction
            fixed_image (itk.Image): Image providing spacing for X and Y dimensions

        Returns:
            itk.Image: Reference image with isotropic spacing and moving image's
                origin and direction
        """
        # Get properties from both images
        moving_origin = moving_image.GetOrigin()
        moving_direction = moving_image.GetDirection()
        moving_spacing = moving_image.GetSpacing()
        moving_size = moving_image.GetLargestPossibleRegion().GetSize()

        fixed_spacing = fixed_image.GetSpacing()

        # Calculate mean of X and Y spacing for isotropic resolution
        mean_xy_spacing = (fixed_spacing[0] + fixed_spacing[1]) / 2.0

        # Create ITK Vector for spacing
        isotropic_spacing = itk.Vector[itk.D, 3]()
        isotropic_spacing[0] = mean_xy_spacing
        isotropic_spacing[1] = mean_xy_spacing
        isotropic_spacing[2] = mean_xy_spacing

        # Calculate new size to cover the same physical extent with isotropic spacing
        new_size = itk.Size[3]()
        for i in range(3):
            new_size[i] = int(
                round((moving_size[i] * moving_spacing[i]) / isotropic_spacing[i])
            )

        # Create reference image with combined properties
        ImageType = type(moving_image)
        reference_image = ImageType.New()
        reference_image.SetOrigin(moving_origin)
        reference_image.SetDirection(moving_direction)
        reference_image.SetSpacing(isotropic_spacing)

        region = itk.ImageRegion[3]()
        region.SetSize(new_size)
        reference_image.SetRegions(region)
        reference_image.Allocate()

        return reference_image

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
        initial_forward_transform: Optional[itk.Transform] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Registration method required by RegisterImagesBase.

        This method is not typically called directly. Use register_time_series()
        instead for time series registration.

        Args:
            moving_image (itk.Image): Image to register
            moving_mask (itk.Image, optional): Binary mask
            moving_image_pre (itk.Image, optional): Preprocessed image
            initial_forward_transform (itk.Transform, optional): Initial transform

        Returns:
            dict: Registration result with forward_transform, inverse_transform, and loss
        """
        if self.registration_method_name == "ANTS":
            res = self.registrar_ANTS.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=initial_forward_transform,
            )
            return {
                "forward_transform": cast(itk.Transform, res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, res["inverse_transform"]),
                "loss": float(cast(float, res["loss"])),
            }
        if self.registration_method_name == "greedy":
            res = self.registrar_greedy.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=initial_forward_transform,
            )
            return {
                "forward_transform": cast(itk.Transform, res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, res["inverse_transform"]),
                "loss": float(cast(float, res["loss"])),
            }
        if self.registration_method_name == "ICON":
            res = self.registrar_ICON.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=initial_forward_transform,
            )
            return {
                "forward_transform": cast(itk.Transform, res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, res["inverse_transform"]),
                "loss": float(cast(float, res["loss"])),
            }
        if self.registration_method_name in ["ANTS_ICON", "greedy_ICON"]:
            registrar_initial = (
                self.registrar_ANTS
                if self.registration_method_name == "ANTS_ICON"
                else self.registrar_greedy
            )
            initial_res = registrar_initial.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=initial_forward_transform,
            )
            forward_initial = initial_res["forward_transform"]
            icon_res = self.registrar_ICON.registration_method(
                moving_image=moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
                moving_image_pre=moving_image_pre,
                initial_forward_transform=forward_initial,
            )
            return {
                "forward_transform": cast(itk.Transform, icon_res["forward_transform"]),
                "inverse_transform": cast(itk.Transform, icon_res["inverse_transform"]),
                "loss": float(cast(float, icon_res["loss"])),
            }
        raise ValueError(
            f"Invalid registration method: {self.registration_method_name}"
        )
