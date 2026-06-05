"""Icon-based image registration implementation.

This module provides the RegisterImagesICON class, a concrete implementation of
RegisterImagesBase that uses the Icon (Inverse Consistent Image Registration)
algorithm with deep learning models. It supports both masked and unmasked
registration for aligning medical images, particularly useful for 4D cardiac CT registration.

The module uses the unigradicon package which provides GPU-accelerated
deformable registration with mass preservation constraints.
"""

import logging
from typing import Optional, Union

import icon_registration as icon
import icon_registration.itk_wrapper
import itk
import numpy as np
import torch
import torch.nn.functional as F
from unigradicon import get_multigradicon, get_unigradicon
from unigradicon import preprocess as unigradicon_preprocess

from physiomotion4d.register_images_base import RegisterImagesBase
from physiomotion4d.transform_tools import TransformTools

DEFAULT_FINETUNE_LEARNING_RATE = 2e-5


class RegisterImagesICON(RegisterImagesBase):
    """ICON-based deformable image registration implementation.

    This class extends RegisterImagesBase to provide GPU-accelerated deformable
    image registration using the ICON (Inverse Consistent Image Registration)
    algorithm implemented with deep learning models. It supports both full image
    registration and mask-constrained registration for specific anatomical regions.

    The ICON algorithm ensures inverse consistency, meaning the forward and
    backward transformations are true inverses of each other. This is important
    for maintaining spatial relationships and avoiding registration artifacts.

    ICON-specific features:
    - GPU acceleration using UniGradIcon framework
    - Mass preservation
    - LNCC (Local Normalized Cross Correlation) similarity metric
    - Inverse consistent transformations
    - Fine-tuning with 50 optimization steps per registration

    Inherits from RegisterImagesBase:
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Standardized registration interface

    Attributes:
        net (unigradicon model): The ICON deep learning registration network

    Example:
        >>> registrar = RegisterImagesICON()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> forward_transform = result['forward_transform']
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the ICON image registration class.

        Calls the parent RegisterImagesBase constructor to set up common parameters.
        The ICON deep learning network is initialized lazily on first use to avoid
        unnecessary GPU memory allocation.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

        self.net = None
        self.number_of_iterations: int = 50
        self.use_multi_modality: bool = False
        self.use_mass_preservation: bool = False
        self.weights_path: Optional[str] = None

    def set_weights_path(self, weights_path: str) -> None:
        """Set a custom weights file for the uniGradICON network.

        Use this to load a fine-tuned checkpoint instead of the default
        pretrained weights. Clears any previously loaded network so the new
        weights are applied on the next call to register().

        Also, use this to specify the path to store the downloaded weights.  The
        file must not exist for the weights to be downloaded correctly.  Typical
        suffix is ".trch".

        Args:
            weights_path: Path to a uniGradICON checkpoint, e.g.
                "results/duke_4d_finetune/checkpoints/network_weights_100"
        """
        self.weights_path = weights_path
        self.net = None  # force reload on next register() call

    def set_number_of_iterations(self, number_of_iterations: int) -> None:
        """Set the number of iterations for ICON registration.

        Args:
            number_of_iterations: Number of fine-tuning steps for ICON registration
        """
        self.number_of_iterations = number_of_iterations

    def set_multi_modality(self, enable: bool) -> None:
        """Enable or disable multi-modality registration.

        Multi-modality registration is useful when aligning images from different
        imaging modalities (e.g., CT to MRI). Enabling this option adjusts the
        registration parameters to better handle differences in intensity
        distributions and contrast between modalities.

        Args:
            enable (bool): True to enable multi-modality registration, False to disable

        Example:
            >>> registrar.set_multi_modality(True)  # Enable for CT to MRI
            >>> registrar.set_multi_modality(False)  # Disable for CT to CT
        """
        self.use_multi_modality = enable

    def set_mass_preservation(self, enable: bool) -> None:
        """Enable or disable mass preservation constraint.

        Mass preservation is particularly useful for CT images where the
        intensity values correspond to physical tissue densities. Enabling
        this constraint helps maintain realistic intensity distributions
        during registration.

        Args:
            enable (bool): True to enable mass preservation, False to disable

        Example:
            >>> registrar.set_mass_preservation(True)  # Enable for CT
            >>> registrar.set_mass_preservation(False)  # Disable for MRI
        """
        self.use_mass_preservation = enable

    def preprocess(self, image: itk.Image, modality: str = "ct") -> itk.Image:
        """Preprocess the image for ICON registration.

        Applies modality-specific preprocessing steps to prepare the image
        for registration. This may include intensity normalization, bias
        field correction, and resampling.

        Args:
            image (itk.image): The input 3D image to preprocess
            modality (str): The imaging modality ('ct', 'mri', etc.)

        Returns:
            itk.image: The preprocessed image

        Example:
            >>> preprocessed_image = registrar.preprocess(raw_image, modality='ct')
        """
        # Placeholder implementation - override in subclass if needed
        return unigradicon_preprocess(image, modality=modality)

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
        initial_forward_transform: Optional[itk.Transform] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register moving image to fixed image using ICON registration algorithm.

        Implementation of the abstract register() method from RegisterImagesBase.
        Performs deformable registration to align the moving image with the
        fixed image using the ICON algorithm. The method automatically handles
        preprocessing, network initialization, and applies the computed transformation.

        Args:
            moving_image (itk.image): The 3D image to be registered/aligned
            moving_mask (itk.image, optional): Binary mask defining the
                region of interest in the moving image. If provided along with
                fixed_mask, enables mask-constrained registration
            moving_image_pre (itk.image, optional): Pre-processed moving image.
                If None, preprocessing is performed automatically
            initial_forward_transform (itk.Transform, optional): Initial transformation from moving
                to fixed. If provided, it is used to transform the moving image before
                registration.

        Returns:
            dict: Dictionary containing:
                - "forward_transform": Warps the moving image onto the fixed
                  grid (warping moving points/landmarks into fixed space uses
                  "inverse_transform" instead -- image and point warps use
                  opposite transforms; see
                  docs/developer/transform_conventions)
                - "inverse_transform": Warps the fixed image onto the moving grid
                - "loss": Loss value from the registration

        Note:
            The transformations are inverse consistent, meaning
            forward_transform is approximately inverse(inverse_transform).
            Use forward_transform to warp the moving image onto the fixed grid,
            and inverse_transform to warp the fixed image onto the moving grid.
            Point/landmark warps use the opposite transform from image warps
            (see docs/developer/transform_conventions).

        Implementation details:
            - Uses UniGradIcon with LNCC loss function
            - Optionally applies mass preservation
            - Performs 50 fine-tuning steps per registration
            - Supports both masked and unmasked registration modes

        Example:
            >>> # Basic registration
            >>> result = registrar.register(moving_image)
            >>> forward_transform = result['forward_transform']
            >>> inverse_transform = result['inverse_transform']
            >>>
            >>> # Masked registration for cardiac structures
            >>> registrar.set_fixed_mask(heart_mask_fixed)
            >>> result = registrar.register(moving_image, moving_mask=heart_mask_moving)
        """

        tfm_tools = TransformTools()

        if moving_image_pre is None:
            moving_image_pre = self.preprocess(moving_image, self.modality)

        new_moving_image_pre = moving_image_pre
        if initial_forward_transform is not None:
            new_moving_image_pre = tfm_tools.transform_image(
                moving_image_pre,
                initial_forward_transform,
                self.fixed_image,
            )

        # Prefer labelmap over binary mask when both sides have a labelmap.
        use_labelmaps = moving_labelmap is not None and self.fixed_labelmap is not None
        moving_effective_mask = moving_labelmap if use_labelmaps else moving_mask
        fixed_effective_mask = self.fixed_labelmap if use_labelmaps else self.fixed_mask

        self._ensure_net()

        inverse_transform = None
        forward_transform = None
        loss_artifacts = None
        if fixed_effective_mask is not None and moving_effective_mask is not None:
            inverse_transform, forward_transform, loss_artifacts = (
                icon_registration.itk_wrapper.register_pair_with_mask(
                    self.net,
                    self.fixed_image_pre,
                    new_moving_image_pre,
                    fixed_effective_mask,
                    moving_effective_mask,
                    finetune_steps=self.number_of_iterations,
                    return_artifacts=True,
                )
            )
        else:
            inverse_transform, forward_transform, loss_artifacts = (
                icon_registration.itk_wrapper.register_pair(
                    self.net,
                    self.fixed_image_pre,
                    new_moving_image_pre,
                    finetune_steps=self.number_of_iterations,
                    return_artifacts=True,
                )
            )

        loss = loss_artifacts[0]

        if initial_forward_transform is not None:
            forward_transform = tfm_tools.combine_displacement_field_transforms(
                initial_forward_transform,
                forward_transform,
                self.fixed_image,
                tfm1_weight=1.0,
                tfm2_weight=1.0,
                mode="compose",
            )

            dftfm = tfm_tools.convert_transform_to_displacement_field_transform(
                forward_transform,
                self.fixed_image,
            )
            inverse_transform = tfm_tools.invert_displacement_field_transform(dftfm)

        return {
            "forward_transform": forward_transform,
            "inverse_transform": inverse_transform,
            "loss": loss,
        }

    def _ensure_net(self) -> None:
        """Lazily instantiate the ICON network using current configuration.

        Honors set_weights_path() if a custom checkpoint has been requested,
        otherwise loads the default UniGradICON / MultiGradICON pretrained
        weights.
        """
        if self.net is not None:
            return
        if self.use_multi_modality:
            self.net = get_multigradicon(
                loss_fn=icon.LNCC(sigma=5),
                apply_intensity_conservation_loss=self.use_mass_preservation,
                weights_location=self.weights_path,
            )
        else:
            self.net = get_unigradicon(
                loss_fn=icon.LNCC(sigma=5),
                apply_intensity_conservation_loss=self.use_mass_preservation,
                weights_location=self.weights_path,
            )

    def _image_to_resized_tensor(
        self, image: itk.Image, shape: torch.Size
    ) -> torch.Tensor:
        """Convert an itk image to a torch tensor resized to the net's input grid.

        Mirrors the trilinear preprocessing path used by
        ``icon_registration.itk_wrapper.register_pair`` exactly.

        Axis ordering:
            - Input ``image`` is a scalar (single-channel) 3D ``itk.Image`` with
              ITK world-axis order (X, Y, Z). ``np.array(image)`` returns a
              C-contiguous array with axes **reversed** to (Z, Y, X) — ITK's
              standard numpy view.
            - ``torch.Tensor(arr)`` casts to ``float32`` (PyTorch's
              ``FloatTensor`` constructor) regardless of the source dtype.
            - Indexing with ``[None, None]`` prepends batch and channel
              singleton axes, producing shape ``(1, 1, Z, Y, X)``. This is
              PyTorch's NCDHW layout where ``D=Z``, ``H=Y``, ``W=X``.
            - ``shape`` is ``self.net.identity_map.shape`` (5D, NCDHW); the
              target spatial size is ``shape[2:] = (D_out, H_out, W_out)``.
            - Return shape: ``(1, 1, D_out, H_out, W_out)``, float32,
              C-contiguous on ``icon.config.device``.

        Notes:
            - Single-channel scalar inputs only. Vector/multi-channel
              ``itk.Image`` would yield ``(Z, Y, X, C)`` from ``np.array`` and
              break the assumed NCDHW layout — not supported here, matching
              ICON's own preprocessing.
            - No explicit time axis: 4D series must be split into 3D
              timepoints by the caller; pairs are processed one volume at a
              time.
            - No transpose is performed; the (Z, Y, X) numpy ordering is
              consumed directly as (D, H, W). Voxel values, not world
              coordinates, drive the trilinear resample.
        """
        arr = np.array(image)
        tensor = torch.Tensor(arr).to(icon.config.device)[None, None]
        return F.interpolate(
            tensor, size=shape[2:], mode="trilinear", align_corners=False
        )

    def _mask_to_resized_tensor(
        self, mask: itk.Image, shape: torch.Size
    ) -> torch.Tensor:
        """Convert an itk mask image to a torch tensor resized via nearest-neighbor.

        Mirrors the mask preprocessing used by
        ``icon_registration.itk_wrapper.register_pair_with_mask`` exactly.

        Axis ordering:
            - Input ``mask`` is a scalar (single-channel) 3D ``itk.Image``
              (typically ``uint8``/short labels) with ITK world-axis order
              (X, Y, Z). ``np.array(mask)`` returns a C-contiguous array with
              axes **reversed** to (Z, Y, X).
            - ``torch.Tensor(arr)`` casts to ``float32`` (label values become
              integral-valued floats; nearest-neighbor resampling preserves
              them).
            - ``[None, None]`` prepends batch and channel singletons →
              ``(1, 1, Z, Y, X)`` in NCDHW (``D=Z``, ``H=Y``, ``W=X``).
            - Target spatial size is ``shape[2:] = (D_out, H_out, W_out)``
              from ``self.net.identity_map.shape``.
            - Return shape: ``(1, 1, D_out, H_out, W_out)``, float32,
              C-contiguous on ``icon.config.device``.

        Notes:
            - Single-channel mask inputs only; multi-label masks are encoded
              as scalar integer values, not channels. Vector ``itk.Image``
              inputs are not supported.
            - No time axis: per-volume 3D processing.
            - Resampling uses ``mode='nearest'`` (no ``align_corners``) so
              label identities are preserved.
        """
        arr = np.array(mask)
        tensor = torch.Tensor(arr).to(icon.config.device)[None, None]
        return F.interpolate(tensor, size=shape[2:], mode="nearest")
