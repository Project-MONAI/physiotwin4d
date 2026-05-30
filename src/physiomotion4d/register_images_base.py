"""Base class for image registration algorithms.

This module provides the RegisterImagesBase class that serves as a foundation
for implementing different image registration algorithms. It defines a common
interface and shared functionality for deformable image registration, particularly
designed for medical imaging applications such as 4D cardiac CT registration.

The base class handles common operations including:
- Fixed and moving image management
- Binary mask processing and dilation
- Modality-specific parameter settings
- Standardized registration interface

Concrete implementations should inherit from RegisterImagesBase and implement
the register() method with their specific algorithm (e.g., Icon, ANTs, etc.).
"""

import logging
from typing import Any, Optional, Union

import itk

from physiomotion4d.labelmap_tools import LabelmapTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.transform_tools import TransformTools


class RegisterImagesBase(PhysioMotion4DBase):
    """Base class for deformable image registration algorithms.

    This class provides a common interface and shared functionality for
    implementing different image registration algorithms. It handles standard
    operations like image and mask management, preprocessing, and parameter
    configuration that are common across registration methods.

    The base class is designed to support various registration algorithms
    including deep learning-based methods (Icon, UniGradIcon) and traditional
    methods (ANTs, ITK). Concrete implementations should inherit from this
    class and implement the register() method.

    Key features:
    - Standardized interface for different registration algorithms
    - Fixed and moving image management
    - Binary mask processing with optional dilation
    - Modality-specific parameter configuration
    - Support for region-of-interest registration

    Attributes:
        net (object): Algorithm-specific network or registration object
        modality (str): Image modality ('ct', 'mri', etc.) for parameter optimization
        fixed_image (itk.image): The target/reference image
        fixed_image_pre (itk.image): Preprocessed fixed image
        fixed_mask (itk.image): Binary mask for fixed image ROI
        mask_dilation_mm (float): Mask dilation amount in millimeters

    Example:
        >>> class MyRegistration(RegisterImagesBase):
        ...     def registration_method(self, moving_image, **kwargs):
        ...         # Implement specific registration algorithm
        ...         return {
        ...             'forward_transform': tfm_forward,  # warps moving image -> fixed grid
        ...             'inverse_transform': tfm_inverse,  # warps fixed image -> moving grid
        ...             'loss': 0.0,
        ...         }
        >>>
        >>> registrar = MyRegistration()
        >>> registrar.set_modality('ct')
        >>> registrar.set_fixed_image(reference_image)
        >>> result = registrar.register(moving_image)
        >>> forward_tfm = result['forward_transform']  # warps moving image -> fixed grid
        >>> inverse_tfm = result['inverse_transform']  # warps fixed image -> moving grid
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the base image registration class.

        Sets up the common registration parameters with default values. Algorithm-specific
        components (like neural networks or optimization objects) should be initialized
        in the concrete implementation to avoid unnecessary resource allocation.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.labelmap_tools = LabelmapTools(log_level=log_level)

        self.net: Any = None

        self.modality: str = "ct"

        self.fixed_image: Optional[itk.Image] = None
        self.fixed_image_pre: Optional[itk.Image] = None
        self.fixed_mask: Optional[itk.Image] = None
        self.fixed_labelmap: Optional[itk.Image] = None

        self.moving_image: Optional[itk.Image] = None
        self.moving_image_pre: Optional[itk.Image] = None
        self.moving_mask: Optional[itk.Image] = None
        self.moving_labelmap: Optional[itk.Image] = None

        self.mask_dilation_mm: float = 5.0

        self.forward_transform: Optional[itk.Transform] = None
        self.inverse_transform: Optional[itk.Transform] = None
        self.loss: Optional[float] = None
        self.moving_image_registered: Optional[itk.Image] = None

    def set_modality(self, modality: str) -> None:
        """Set the imaging modality for registration optimization.

        Different imaging modalities benefit from different registration
        parameters. CT images.

        Args:
            modality (str): The imaging modality.
                Supported values: 'ct', 'mri'

        Example:
            >>> registrar.set_modality('ct')
            >>> registrar.set_modality('mri')
        """
        self.modality = modality

    def set_fixed_image(self, fixed_image: itk.Image) -> None:
        """Set the fixed/target image for registration.

        The fixed image serves as the reference coordinate system to which
        all moving images will be aligned. Setting a new fixed image clears
        any preprocessed data to ensure consistency.

        Args:
            fixed_image (itk.image): The 3D reference image that serves as
                the target for registration

        Example:
            >>> registrar.set_fixed_image(reference_frame)
        """
        self.fixed_image = fixed_image
        self.fixed_image_pre = None
        self.forward_transform = None
        self.inverse_transform = None
        self.loss = None
        self.moving_image_registered = None

    def set_mask_dilation(self, mask_dilation_mm: float) -> None:
        """Set the dilation of the fixed and moving image masks.

        Args:
            mask_dilation_mm (float): The dilation in millimeters.
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_fixed_mask(self, fixed_mask: Optional[itk.Image]) -> None:
        """Set a binary mask for the fixed image region of interest.

        The mask constrains registration to focus on specific anatomical
        regions, improving accuracy and reducing computation time. The mask
        is automatically converted to binary format. If mask_dilation_mm is set,
        the mask is dilated by the specified amount.

        Args:
            fixed_mask (itk.image): Binary or label mask defining the
                region of interest in the fixed image. Non-zero values are
                treated as foreground

        Example:
            >>> # Use heart mask to focus registration on cardiac structures
            >>> registrar.set_fixed_mask(heart_mask)
        """
        self.fixed_image_pre = None
        self.forward_transform = None
        self.inverse_transform = None
        self.loss = None
        self.moving_image_registered = None

        if fixed_mask is None:
            self.fixed_mask = None
            return

        if self.fixed_image is None:
            raise ValueError("Fixed image must be set before setting a fixed mask.")

        self.fixed_mask = self.labelmap_tools.convert_labelmap_to_mask(
            fixed_mask, dilation_in_mm=self.mask_dilation_mm
        )
        self.fixed_mask.CopyInformation(self.fixed_image)

    def set_fixed_labelmap(self, fixed_labelmap: Optional[itk.Image]) -> None:
        """Set the fixed image labelmap (multi-label segmentation).

        Args:
            fixed_labelmap (itk.Image, optional): Multi-label segmentation
                co-registered with the fixed image, or None to clear.
        """
        self.fixed_labelmap = fixed_labelmap
        self.forward_transform = None
        self.inverse_transform = None
        self.loss = None
        self.moving_image_registered = None

    def preprocess(self, image: itk.Image, modality: str = "ct") -> itk.Image:
        """Preprocess the image based on modality-specific requirements.

        This method applies preprocessing steps such as intensity normalization,
        histogram equalization, or noise reduction tailored to the specified
        imaging modality. Preprocessing enhances image quality and improves
        registration accuracy.

        Args:
            image (itk.image): The 3D image to preprocess
            modality (str): The imaging modality ('ct', 'mri', etc.)

        Returns:
            itk.image: The preprocessed image

        Example:
            >>> preprocessed_image = registrar.preprocess(raw_image, modality='ct')
        """
        # Placeholder implementation - override in subclass if needed
        return image

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
        initial_forward_transform: Optional[itk.Transform] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Main registration method to align moving image to fixed image.

        This method serves as the primary interface for performing image
        registration. It takes a moving image and optional mask and
        preprocessed image, and returns the forward and backward transformations.

        Note: This is an internal method that should be implemented by subclasses.
        The public API is register() which wraps this method.

        Args:
            moving_image (itk.image): The 3D image to be registered to the fixed image
            moving_mask (itk.image, optional): Binary mask for moving image ROI
            moving_labelmap (itk.image, optional): Multi-label segmentation for the moving image
            moving_image_pre (itk.image, optional): Preprocessed moving image
            initial_forward_transform (itk.Transform, optional): Initial transformation from moving to fixed

        Returns:
            dict: Dictionary containing:
                - "forward_transform": Warps the moving image onto the fixed
                  grid. Warping moving points/landmarks into fixed space uses
                  "inverse_transform" instead (see register() and
                  docs/developer/transform_conventions).
                - "inverse_transform": Warps the fixed image onto the moving grid
                - "loss": Registration loss/metric value

        Raises:
            ValueError: If fixed image is not set
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def register(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
        initial_forward_transform: Optional[itk.Transform] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register a moving image to the fixed image.

        This is the main registration method that must be implemented by
        concrete subclasses. It should align the moving image to the fixed
        image using the specific algorithm implemented by the subclass.

        Args:
            moving_image (itk.image): The 3D image to be registered to the fixed image
            moving_mask (itk.image, optional): Binary mask for moving image ROI
            moving_labelmap (itk.image, optional): Multi-label segmentation for the moving image
            moving_image_pre (itk.image, optional): Preprocessed moving image
            initial_forward_transform (itk.Transform, optional): Initial transformation from moving to fixed

        Returns:
            dict: Dictionary containing transformation results:
                - "forward_transform": Warps the moving IMAGE onto the fixed
                  grid, i.e. transform_image(moving, forward_transform, fixed).
                - "inverse_transform": Warps the fixed IMAGE onto the moving
                  grid, i.e. transform_image(fixed, inverse_transform, moving).
                - "loss": Registration loss/metric value

        Note:
            Image warps and point/landmark warps use OPPOSITE members of the
            transform pair, because ITK image resampling pulls back (it maps a
            fixed-grid sample to the moving image) while point transforms push
            forward (they map a point to its corresponding location):

            - Warp the moving image into fixed space  -> forward_transform
            - Warp moving points/landmarks into fixed  -> inverse_transform
            - Warp the fixed image into moving space   -> inverse_transform
            - Warp fixed points/landmarks into moving   -> forward_transform

            See docs/developer/transform_conventions for the full discussion.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        self.moving_image_registered = None
        self.forward_transform = None
        self.inverse_transform = None
        self.loss = None

        if self.fixed_image_pre is None:
            self.fixed_image_pre = self.preprocess(
                self.fixed_image,
                modality=self.modality,
            )

        if moving_image_pre is None:
            moving_image_pre = self.preprocess(
                moving_image,
                modality=self.modality,
            )

        new_moving_mask = moving_mask
        if moving_mask is not None:
            new_moving_mask = self.labelmap_tools.convert_labelmap_to_mask(
                moving_mask, dilation_in_mm=self.mask_dilation_mm
            )
            new_moving_mask.CopyInformation(moving_image)

        self.moving_image = moving_image
        self.moving_image_pre = moving_image_pre
        self.moving_mask = new_moving_mask
        self.moving_labelmap = moving_labelmap

        result = self.registration_method(
            moving_image,
            moving_mask=new_moving_mask,
            moving_labelmap=moving_labelmap,
            moving_image_pre=moving_image_pre,
            initial_forward_transform=initial_forward_transform,
        )

        self.forward_transform = result["forward_transform"]
        self.inverse_transform = result["inverse_transform"]
        self.loss = result["loss"]

        return {
            "forward_transform": self.forward_transform,
            "inverse_transform": self.inverse_transform,
            "loss": self.loss,
        }

    def get_registered_image(self) -> itk.Image:
        """Get the registered image.

        Returns:
            itk.Image: The registered image
        """
        if self.moving_image_registered is None:
            TfmTools = TransformTools()
            self.moving_image_registered = TfmTools.transform_image(
                self.moving_image, self.forward_transform, self.fixed_image
            )
        return self.moving_image_registered
