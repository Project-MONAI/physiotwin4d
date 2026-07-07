"""Distance-map-based model-to-model registration for anatomical models.

This module provides the RegisterModelsDistanceMaps class for aligning anatomical
models using distance-map-based deformable registration. The workflow includes:
1. Generate distance maps from moving and fixed models
2. Generate binary registration masks with dilation
3. Progressive registration stages:
   - rigid: Greedy rigid registration
   - affine: Greedy affine registration
   - deformable: Greedy affine → ICON deformable registration

The registration is particularly useful for aligning anatomical models where
shape differences require deformable transformations beyond rigid/affine ICP.

Key Features:
    - Automatic mask generation from PyVista models
    - Multi-stage Greedy/ICON registration (rigid/affine/deformable)
    - Automatic transform composition
    - Support for PyVista models

Example:
    >>> import itk
    >>> import pyvista as pv
    >>> from physiotwin4d import RegisterModelsDistanceMaps
    >>>
    >>> # Load models and reference image
    >>> moving_model = pv.read('generic_model.vtu').extract_surface(algorithm="dataset_surface")
    >>> fixed_model = pv.read('patient_surface.stl')
    >>> reference_image = itk.imread('patient_ct.nii.gz')
    >>>
    >>> # Run deformable registration (Greedy affine + ICON deformable)
    >>> registrar = RegisterModelsDistanceMaps(
    ...     moving_model=moving_model,
    ...     fixed_model=fixed_model,
    ...     reference_image=reference_image,
    ...     mask_dilation_mm=20,
    ... )
    >>> result = registrar.register(transform_type='Deformable', icon_iterations=50)
    >>>
    >>> # Access results
    >>> aligned_model = result['registered_model']
    >>> forward_transform = result['forward_transform']  # warps moving image -> fixed grid
"""

import logging
from typing import Optional

import itk
import pyvista as pv

from .contour_tools import ContourTools
from .labelmap_tools import LabelmapTools
from .physiotwin4d_base import PhysioTwin4DBase
from .register_images_greedy import RegisterImagesGreedy
from .register_images_icon import RegisterImagesICON
from .transform_tools import TransformTools


class RegisterModelsDistanceMaps(PhysioTwin4DBase):
    """Register anatomical models using distance-map-based deformable registration.

    This class provides distance-map-based alignment of 3D surface models with support
    for rigid, affine, and deformable transformation modes. The registration pipeline
    generates signed distance maps from models, applies optional binary mask dilation,
    and uses Greedy for rigid/affine stages and ICON for deformable registration.

    **Registration Pipelines:**
        - **None mode**: No registration (identity transform)
        - **Rigid mode**: Greedy rigid registration
        - **Affine mode**: Greedy affine registration
        - **Deformable mode**: Greedy affine → ICON deformable registration

    **Transform Convention:**
        These are the underlying image-registration (Greedy/ICON) transforms, so
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
        mask_dilation_mm (float): Dilation amount in mm for binary registration masks
        transform_tools (TransformTools): Transform utility instance
        contour_tools (ContourTools): Model utility instance
        registrar_Greedy (RegisterImagesGreedy): Greedy registration instance
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
        ...     mask_dilation_mm=20,
        ... )
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(transform_type='Rigid')
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(transform_type='Affine')
        >>>
        >>> # Or run deformable (Greedy affine + ICON)
        >>> result = registrar.register(transform_type='Deformable', icon_iterations=50)
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
        mask_dilation_mm: float = 20,
        log_level: int | str = logging.INFO,
    ):
        """Initialize distance-map-based model registration.

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            fixed_model: PyVista target surface model
            reference_image: ITK image providing coordinate frame (origin, spacing, direction)
                for mask generation. Typically the patient CT/MRI image.
            mask_dilation_mm: Dilation amount in millimeters for binary registration
                mask generation. Default: 20mm
            log_level: Logging level (default: logging.INFO)

        Note:
            The moving_model and fixed_model are typically extracted from VTU models
            using model.extract_surface(algorithm="dataset_surface") before passing to this class.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.moving_model = moving_model
        self.fixed_model = fixed_model
        self.reference_image = reference_image
        self.mask_dilation_mm = mask_dilation_mm

        # Utilities
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()
        self.labelmap_tools = LabelmapTools(log_level=log_level)

        # Registration instances
        self.registrar_Greedy = RegisterImagesGreedy(log_level=log_level)
        self.registrar_ICON = RegisterImagesICON(log_level=log_level)
        self.registrar_ICON.set_modality("ct")
        self.registrar_ICON.set_multi_modality(False)

        # Generated distance maps and binary registration masks (created during registration)
        self.fixed_distance_map_image: Optional[itk.Image] = None
        self.fixed_mask_image: Optional[itk.Image] = None
        self.moving_distance_map_image: Optional[itk.Image] = None
        self.moving_mask_image: Optional[itk.Image] = None

        # Registration results
        self.forward_transform: Optional[itk.CompositeTransform] = None  # Moving→fixed
        self.inverse_transform: Optional[itk.CompositeTransform] = None  # Fixed→moving
        self.registered_model: Optional[pv.PolyData] = None

    def _create_masks_from_models(self) -> None:
        """Generate distance maps and binary registration masks from moving and fixed models.

        Creates:
            - fixed_distance_map_image: Signed distance map from fixed model
            - fixed_mask_image: Dilated binary registration mask from fixed model
            - moving_distance_map_image: Signed distance map from moving model
            - moving_mask_image: Dilated binary registration mask from moving model

        Uses self.reference_image for coordinate frame (origin, spacing, direction).
        """
        self.log_info("Generating distance maps and registration masks from models...")

        # Create fixed distance map
        self.fixed_distance_map_image = self.contour_tools.create_distance_map(
            self.fixed_model,
            self.reference_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=50.0,
        )

        # Create fixed binary registration mask with dilation
        self.log_info(
            "Dilating fixed mask by %.1fmm for registration mask...",
            self.mask_dilation_mm,
        )
        binary_mask = self.contour_tools.create_mask_from_mesh(
            self.fixed_model, self.reference_image
        )
        self.fixed_mask_image = self.labelmap_tools.convert_labelmap_to_mask(
            binary_mask, dilation_in_mm=self.mask_dilation_mm
        )

        # Create moving distance map
        self.moving_distance_map_image = self.contour_tools.create_distance_map(
            self.moving_model,
            self.reference_image,
            squared_distance=True,
            negative_inside=True,
            zero_inside=False,
            norm_to_max_distance=50.0,
        )

        # Emulate CT intensity range by multiplying by 1000
        tmp_arr = itk.GetArrayViewFromImage(self.fixed_distance_map_image)
        tmp_arr *= 1000

        tmp_arr = itk.GetArrayViewFromImage(self.moving_distance_map_image)
        tmp_arr *= 1000

        # Create moving binary registration mask with dilation
        self.log_info(
            "Dilating moving mask by %.1fmm for registration mask...",
            self.mask_dilation_mm,
        )
        binary_mask = self.contour_tools.create_mask_from_mesh(
            self.moving_model, self.reference_image
        )
        self.moving_mask_image = self.labelmap_tools.convert_labelmap_to_mask(
            binary_mask, dilation_in_mm=self.mask_dilation_mm
        )

        self.log_info("Distance map and mask generation complete")

    def register(
        self,
        transform_type: str = "Deformable",
        icon_iterations: int = 50,
    ) -> dict:
        """Perform mask-based registration of moving model to fixed model.

        This method executes progressive multi-stage registration:

        **None transform type:**
            1. No registration (identity transform)

        **Rigid transform type:**
            1. Greedy rigid registration

        **Affine transform type:**
            1. Greedy affine registration

        **Deformable transform type:**
            1. Greedy affine registration
            2. ICON deformable registration on the affine-pre-aligned masks

        Args:
            transform_type: Registration transform type - 'None', 'Rigid', 'Affine', or 'Deformable'. Default: 'Deformable'
            icon_iterations: Number of ICON optimization iterations for 'Deformable' mode. Default: 50

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
            >>> # Deformable registration (Greedy affine + ICON)
            >>> result = registrar.register(transform_type='Deformable', icon_iterations=100)
        """
        if transform_type not in ["None", "Rigid", "Affine", "Deformable"]:
            raise ValueError(
                f"Invalid transform type '{transform_type}'. Must be 'None', 'Rigid', 'Affine', or 'Deformable'."
            )

        self.log_section("%s Distance-Map-based Registration", transform_type.upper())

        # Step 1: Generate distance maps and registration masks from models
        self._create_masks_from_models()

        # Step 2: Greedy rigid or affine stage (skipped for None/Deformable uses Affine)
        greedy_type = "Affine" if transform_type == "Deformable" else transform_type

        forward_transform_Greedy = None
        inverse_transform_Greedy = None
        if greedy_type != "None":
            self.log_info("Performing Greedy %s registration...", greedy_type)
            self.registrar_Greedy.set_fixed_image(self.fixed_distance_map_image)
            self.registrar_Greedy.set_fixed_mask(self.fixed_mask_image)
            self.registrar_Greedy.set_transform_type(greedy_type)
            self.registrar_Greedy.set_metric("CC")

            result_Greedy = self.registrar_Greedy.register(
                moving_image=self.moving_distance_map_image,
                moving_mask=self.moving_mask_image,
            )
            forward_transform_Greedy = result_Greedy["forward_transform"]
            inverse_transform_Greedy = result_Greedy["inverse_transform"]
        else:
            identity_transform = itk.AffineTransform[itk.D, 3].New()
            identity_transform.SetIdentity()
            forward_transform_Greedy = identity_transform
            inverse_transform_Greedy = identity_transform

        self.forward_transform = forward_transform_Greedy
        self.inverse_transform = inverse_transform_Greedy

        # Step 3: ICON deformable stage (only for Deformable mode)
        if transform_type == "Deformable":
            self.log_info(
                "Performing ICON deformable registration (%d iterations)...",
                icon_iterations,
            )

            # Pre-align moving distance map and binary mask into the fixed grid using the Greedy affine result
            moving_distance_map_affine_transformed = (
                self.transform_tools.transform_image(
                    self.moving_distance_map_image,
                    forward_transform_Greedy,
                    self.reference_image,
                    interpolation_method="linear",
                )
            )
            # moving_mask_affine_transformed = self.transform_tools.transform_image(
            # self.moving_mask_image,
            # forward_transform_Greedy,
            # self.reference_image,
            # interpolation_method="nearest",
            # )

            # Configure and run ICON
            self.registrar_ICON.set_number_of_iterations(icon_iterations)
            self.registrar_ICON.set_fixed_image(self.fixed_distance_map_image)
            # self.registrar_ICON.set_fixed_mask(self.fixed_mask_image)

            result_ICON = self.registrar_ICON.register(
                moving_image=moving_distance_map_affine_transformed,
                # moving_mask=moving_mask_affine_transformed,
            )
            forward_transform_ICON = result_ICON["forward_transform"]
            inverse_transform_ICON = result_ICON["inverse_transform"]

            # Compose Greedy affine + ICON deformable.
            # ICON runs on images already resampled to the patient (fixed) grid,
            # so its transforms are deformations within patient space.
            # Forward (fixed→moving for image pull-back): apply ICON first
            # (patient-space δ), then Greedy (patient→ICP-template).
            # Inverse (moving→fixed for point push-forward): apply Greedy first
            # (ICP-template→patient), then ICON (patient-space refinement).
            self.forward_transform = (
                self.transform_tools.combine_displacement_field_transforms(
                    forward_transform_ICON,
                    forward_transform_Greedy,
                    reference_image=self.reference_image,
                    mode="compose",
                )
            )
            self.inverse_transform = (
                self.transform_tools.combine_displacement_field_transforms(
                    inverse_transform_Greedy,
                    inverse_transform_ICON,
                    reference_image=self.reference_image,
                    mode="compose",
                )
            )

        # Apply final transform to moving model
        self.log_info("Transforming moving model...")
        self.registered_model = self.transform_tools.transform_pvcontour(
            self.moving_model,
            self.inverse_transform,
            with_deformation_magnitude=True,
        )

        self.log_info(
            "%s distance-map-based registration complete.", transform_type.upper()
        )

        # Return results as dictionary
        return {
            "forward_transform": self.forward_transform,
            "inverse_transform": self.inverse_transform,
            "registered_model": self.registered_model,
        }
