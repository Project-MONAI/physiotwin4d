"""Mask-based model-to-model registration for anatomical models.

This module provides the RegisterModelsDistanceMaps class for aligning anatomical
models using mask-based deformable registration. The workflow includes:
1. Generate binary masks from moving and fixed models
2. Generate ROI masks with dilation
4. Progressive registration stages:
   - rigid: ANTs rigid registration
   - affine: ANTs rigid → affine registration
   - deformable: ANTs rigid → affine → deformable (SyN) registration
5. Optional ICON refinement at end

The registration is particularly useful for aligning anatomical models where
shape differences require deformable transformations beyond rigid/affine ICP.

Key Features:
    - Automatic mask generation from PyVista models
    - Multi-stage ANTs registration (rigid/affine/deformable)
    - Optional ICON deep learning refinement
    - Automatic transform composition
    - Support for PyVista models

Example:
    >>> import itk
    >>> import pyvista as pv
    >>> from physiomotion4d import RegisterModelsDistanceMaps
    >>>
    >>> # Load models and reference image
    >>> moving_model = pv.read('generic_model.vtu').extract_surface(algorithm="dataset_surface")
    >>> fixed_model = pv.read('patient_surface.stl')
    >>> reference_image = itk.imread('patient_ct.nii.gz')
    >>>
    >>> # Run deformable registration with ICON refinement
    >>> registrar = RegisterModelsDistanceMaps(
    ...     moving_model=moving_model,
    ...     fixed_model=fixed_model,
    ...     reference_image=reference_image,
    ...     roi_dilation_mm=20,
    ... )
    >>> result = registrar.register(mode='deformable', use_ICON=True, icon_iterations=50)
    >>>
    >>> # Access results
    >>> aligned_model = result['registered_model']
    >>> forward_transform = result['forward_transform']  # warps moving image -> fixed grid
"""

import logging
from typing import Optional

import itk
import pyvista as pv

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.labelmap_tools import LabelmapTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_images_ants import RegisterImagesANTS
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.transform_tools import TransformTools


class RegisterModelsDistanceMaps(PhysioMotion4DBase):
    """Register anatomical models using mask-based deformable registration.

    This class provides mask-based alignment of 3D surface models with support for
    rigid, affine, and deformable transformation modes. The registration pipeline
    generates masks from models, applies optional dilation, and uses ANTs for
    progressive multi-stage registration with optional ICON refinement.

    **Registration Pipelines:**
        - **None mode**: No ANTs registration
        - **Rigid mode**: ANTs rigid registration
        - **Affine mode**: ANTs rigid → affine registration
        - **Deformable mode**: ANTs rigid → affine → deformable (SyN) registration
        - **Optional**: ICON deep learning refinement after any mode

    **Transform Convention:**
        These are the underlying image-registration (ANTs/ICON) transforms, so
        they follow the image convention (see
        docs/developer/transform_conventions):

        - forward_transform: warps the moving image/mask onto the fixed grid.
          Warping the moving MODEL points/landmarks onto the fixed model uses
          inverse_transform instead (image and point warps use opposite
          transforms).
        - inverse_transform: warps the fixed image/mask onto the moving grid.

    Attributes:
        moving_model (pv.PolyData): Surface model to be aligned
        fixed_model (pv.PolyData): Target surface model
        reference_image (itk.Image): Reference image for coordinate frame
        roi_dilation_mm (float): Dilation amount in mm for ROI mask
        transform_tools (TransformTools): Transform utility instance
        contour_tools (ContourTools): Model utility instance
        registrar_ANTS (RegisterImagesANTS): ANTs registration instance
        registrar_ICON (RegisterImagesICON): ICON registration instance
        forward_transform (itk.CompositeTransform): Optimized moving→fixed transform
        inverse_transform (itk.CompositeTransform): Optimized fixed→moving transform
        registered_model (pv.PolyData): Aligned moving model

    Example:
        >>> # Initialize with models and reference image
        >>> registrar = RegisterModelsDistanceMaps(
        ...     moving_model=model_surface,
        ...     fixed_model=patient_surface,
        ...     reference_image=patient_ct,
        ...     roi_dilation_mm=20,
        ... )
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(mode='rigid')
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(mode='affine')
        >>>
        >>> # Or run deformable with ICON refinement
        >>> result = registrar.register(
        ...     mode='deformable', use_ANTS=False, use_ICON=True, icon_iterations=50
        ... )
        >>>
        >>> # Get aligned model and transforms
        >>> aligned_model = result['registered_model']
        >>> forward_transform = result['forward_transform']
    """

    def __init__(
        self,
        moving_model: pv.PolyData,
        fixed_model: pv.PolyData,
        reference_image: itk.Image,
        roi_dilation_mm: float = 20,
        log_level: int | str = logging.INFO,
    ):
        """Initialize mask-based model registration.

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            fixed_model: PyVista target surface model
            reference_image: ITK image providing coordinate frame (origin, spacing, direction)
                for mask generation. Typically the patient CT/MRI image.
            roi_dilation_mm: Dilation amount in millimeters for ROI mask generation.
                Default: 20mm
            log_level: Logging level (default: logging.INFO)

        Note:
            The moving_model and fixed_model are typically extracted from VTU models
            using model.extract_surface(algorithm="dataset_surface") before passing to this class.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.moving_model = moving_model
        self.fixed_model = fixed_model
        self.reference_image = reference_image
        self.roi_dilation_mm = roi_dilation_mm

        # Utilities
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()
        self.labelmap_tools = LabelmapTools(log_level=log_level)

        # Registration instances
        self.registrar_ANTS = RegisterImagesANTS(log_level=log_level)
        self.registrar_ICON = RegisterImagesICON(log_level=log_level)
        self.registrar_ICON.set_modality("ct")
        self.registrar_ICON.set_multi_modality(False)

        # Generated masks (will be created during registration)
        self.fixed_mask_image: Optional[itk.Image] = None
        self.fixed_mask_roi_image: Optional[itk.Image] = None
        self.moving_mask_image: Optional[itk.Image] = None
        self.moving_mask_roi_image: Optional[itk.Image] = None

        # Registration results
        self.forward_transform: Optional[itk.CompositeTransform] = None  # Moving→fixed
        self.inverse_transform: Optional[itk.CompositeTransform] = None  # Fixed→moving
        self.registered_model: Optional[pv.PolyData] = None

    def _create_masks_from_models(self) -> None:
        """Generate binary mask images from moving and fixed models.

        Creates:
            - fixed_mask_image: Binary mask from fixed model
            - fixed_mask_roi_image: Dilated ROI mask from fixed model
            - moving_mask_image: Binary mask from moving model
            - moving_mask_roi_image: Dilated ROI mask from moving model

        Uses self.reference_image for coordinate frame (origin, spacing, direction).
        """
        self.log_info("Generating binary masks from models...")

        # Create fixed mask
        self.fixed_mask_image = self.contour_tools.create_distance_map(
            self.fixed_model,
            self.reference_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=50.0,
        )

        # Create fixed ROI mask with dilation
        self.log_info("Dilating fixed mask by %.1fmm for ROI...", self.roi_dilation_mm)
        mask = self.contour_tools.create_mask_from_mesh(
            self.fixed_model, self.reference_image
        )
        self.fixed_mask_roi_image = self.labelmap_tools.convert_labelmap_to_mask(
            mask, dilation_in_mm=self.roi_dilation_mm
        )

        # Create moving mask
        self.moving_mask_image = self.contour_tools.create_distance_map(
            self.moving_model,
            self.reference_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=50.0,
        )

        # Create moving ROI mask with dilation
        self.log_info("Dilating moving mask by %.1fmm for ROI...", self.roi_dilation_mm)
        mask = self.contour_tools.create_mask_from_mesh(
            self.moving_model, self.reference_image
        )
        self.moving_mask_roi_image = self.labelmap_tools.convert_labelmap_to_mask(
            mask, dilation_in_mm=self.roi_dilation_mm
        )

        self.log_info("Mask generation complete")

    def register(
        self,
        transform_type: str = "Deformable",
        use_ICON: bool = False,
        icon_iterations: int = 50,
    ) -> dict:
        """Perform mask-based registration of moving model to fixed model.

        This method executes progressive multi-stage registration:

        **None transform type:**
            1. No ANTs registration

        **Rigid transform type:**
            1. ANTs rigid registration

        **Affine transform type:**
            1. ANTs affine registration (includes rigid stage)

        **Deformable transform type:**
            1. ANTs SyN deformable registration (includes rigid + affine + deformable stages)

        **Optional ICON refinement** (all transform type):
            1. ICON deep learning registration for fine-tuning

        Args:
            transform_type: Registration transform type - 'None', 'Rigid', 'Affine', or 'Deformable'. Default: 'Deformable'
            use_ICON: Whether to apply ICON registration refinement after ANTs. Default: False
            icon_iterations: Number of ICON optimization iterations if use_ICON=True. Default: 50

        Returns:
            Dictionary containing:
                - 'moving_model': Aligned moving model (PyVista PolyData)
                - 'forward_transform': Moving→fixed transform (ITK CompositeTransform)
                - 'inverse_transform': Fixed→moving transform (ITK CompositeTransform)

        Raises:
            ValueError: If transform_type is not 'None', 'Rigid', 'Affine', or 'Deformable'

        Example:
            >>> # Rigid registration
            >>> result = registrar.register(transform_type='Rigid')
            >>>
            >>> # Affine registration
            >>> result = registrar.register(transform_type='Affine')
            >>>
            >>> # Deformable registration with ICON refinement
            >>> result = registrar.register(
            ...     transform_type='Deformable', use_ICON=True, icon_iterations=100
            ... )
        """
        if transform_type not in ["None", "Rigid", "Affine", "Deformable"]:
            raise ValueError(
                f"Invalid transform type '{transform_type}'. Must be 'None', 'Rigid', 'Affine', or 'Deformable'."
            )

        self.log_section("%s Mask-based Registration", transform_type.upper())

        # Step 1: Generate masks from models
        self._create_masks_from_models()

        self.log_info(
            "Performing ANTs %s registration...",
            transform_type,
        )

        inverse_transform_ANTS = None
        forward_transform_ANTS = None
        if transform_type != "None":
            self.registrar_ANTS.set_fixed_image(self.fixed_mask_image)
            self.registrar_ANTS.set_fixed_mask(self.fixed_mask_roi_image)

            self.registrar_ANTS.set_transform_type(transform_type)
            self.registrar_ANTS.set_metric("MeanSquares")

            result_ANTS = self.registrar_ANTS.register(
                moving_image=self.moving_mask_image,
                moving_mask=self.moving_mask_roi_image,
            )
            inverse_transform_ANTS = result_ANTS["inverse_transform"]
            forward_transform_ANTS = result_ANTS["forward_transform"]
        else:
            identity_transform = itk.AffineTransform[itk.D, 3].New()
            identity_transform.SetIdentity()
            inverse_transform_ANTS = identity_transform
            forward_transform_ANTS = identity_transform

        # Initialize composite transforms
        self.forward_transform = forward_transform_ANTS
        self.inverse_transform = inverse_transform_ANTS

        # Optional ICON refinement
        if use_ICON:
            self.log_info(
                "Performing ICON refinement registration (%d iterations)...",
                icon_iterations,
            )

            # Transform masks with ANTs result for ICON input
            moving_mask_ANTS_transformed = self.transform_tools.transform_image(
                self.moving_mask_image,
                forward_transform_ANTS,
                self.reference_image,
                interpolation_method="linear",
            )

            # Configure ICON
            self.registrar_ICON.set_number_of_iterations(icon_iterations)
            self.registrar_ICON.set_fixed_image(self.fixed_mask_image)
            self.registrar_ICON.set_fixed_mask(self.fixed_mask_roi_image)

            # ICON registration
            result_ICON = self.registrar_ICON.register(
                moving_image=moving_mask_ANTS_transformed,
                moving_mask=self.moving_mask_roi_image,
            )
            inverse_transform_ICON = result_ICON["inverse_transform"]
            forward_transform_ICON = result_ICON["forward_transform"]

            # Compose ANTs and ICON transforms
            composed_forward = (
                self.transform_tools.combine_displacement_field_transforms(
                    forward_transform_ANTS,
                    forward_transform_ICON,
                    reference_image=self.reference_image,
                    mode="compose",
                )
            )

            composed_inverse = (
                self.transform_tools.combine_displacement_field_transforms(
                    inverse_transform_ICON,
                    inverse_transform_ANTS,
                    reference_image=self.reference_image,
                    mode="compose",
                )
            )

            self.forward_transform = composed_forward
            self.inverse_transform = composed_inverse

        # Apply final transform to moving model
        self.log_info("Transforming moving model...")
        self.registered_model = self.transform_tools.transform_pvcontour(
            self.moving_model,
            self.inverse_transform,
            with_deformation_magnitude=True,
        )

        self.log_info("%s mask-based registration complete!", transform_type.upper())

        # Return results as dictionary
        return {
            "forward_transform": self.forward_transform,
            "inverse_transform": self.inverse_transform,
            "registered_model": self.registered_model,
        }
