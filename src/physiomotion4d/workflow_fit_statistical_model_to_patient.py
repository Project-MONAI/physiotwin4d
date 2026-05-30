"""Model-to-image and model-to-model registration for anatomical models.

This module provides the WorkflowFitStatisticalModelToPatient class for registering generic
anatomical models to patient-specific imaging data and surface models.
The workflow includes:
1. Rough alignment using ICP (RegisterModelsICP)
1.5. Optional PCA-based registration (RegisterModelsPCA) if PCA data provided
2. Mask-based deformable registration (RegisterModelsDistanceMaps)
3. Optional final mask-to-image refinement using Icon

The registration is particularly useful for cardiac modeling where a generic heart model
needs to be fitted to patient-specific imaging data.

Key Features:
    - Automatic mask generation if not provided by user
    - Modular design using RegisterModelsICP, RegisterModelsPCA, and
        RegisterModelsDistanceMaps
    - Multi-stage registration pipeline:
        ICP → (optional PCA) → mask-to-mask → mask-to-image
    - Optional PCA-based shape fitting
    - Support for multi-label anatomical structures
    - Optional Icon-based final refinement
"""

import logging
from typing import Any, Optional, cast

import itk
import numpy as np
import pyvista as pv

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.labelmap_tools import LabelmapTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_images_ants import RegisterImagesANTS
from physiomotion4d.register_images_icon import RegisterImagesICON
from physiomotion4d.register_models_distance_maps import RegisterModelsDistanceMaps
from physiomotion4d.register_models_icp import RegisterModelsICP
from physiomotion4d.register_models_pca import RegisterModelsPCA
from physiomotion4d.transform_tools import TransformTools
from physiomotion4d.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK


def _load_tubetk() -> Any:
    """Load TubeTK lazily for operations that require its filters."""
    from itk import TubeTK as ttk

    return ttk


def _image_has_isotropic_spacing(image: itk.Image) -> bool:
    """Return whether all image spacing values match within numeric tolerance."""
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    return bool(np.allclose(spacing, spacing[0]))


class WorkflowFitStatisticalModelToPatient(PhysioMotion4DBase):
    """Register anatomical models using multi-stage ICP, mask-based, and image-based
        registration.

    This class provides a flexible workflow for registering generic anatomical models
    (e.g., cardiac models) to patient-specific surface models and images. The
    registration pipeline combines:
    - Initial model alignment using RegisterModelsICP (centroid + affine ICP)
    - Mask-based deformable registration using RegisterModelsDistanceMaps (ANTs/ICON)
    - Optional final mask-to-image refinement using Icon registration

    **Registration Pipeline:**
        1. **ICP Alignment**: Rough affine alignment using RegisterModelsICP
        2. **PCA Registration**: Performs PCA-based shape fitting using
            RegisterModelsPCA
        3. **Mask-to-Mask**: Deformable registration using RegisterModelsDistanceMaps
        4. **Mask-to-Image**: Final refinement

    **Mask Configuration:**
        Masks are automatically generated from models if not provided by the user
        via set_masks(). Auto-generated masks use mask_dilation_mm parameter.

    Attributes:
        template_model (pv.DataSet): Generic anatomical model to be registered
        template_model_surface (pv.PolyData): Surface extracted from
            template_model_surface
        template_model_mask (itk.Image): Binary/multi-label mask for model model
        template_model_roi (itk.Image): ROI mask for model model
        patient_models (list of pv.DataSet): Patient-specific models
        patient_model_surface (pv.PolyData): Primary patient model surface (first in
            list)
        combined_patient_model (pv.PolyData): Merged patient models before surface
            extraction; used when pca_uses_surface=False.
        patient_image (itk.Image): Reference image providing coordinate frame
        patient_mask (itk.Image): Binary/multi-label mask for patient model
        patient_roi (itk.Image): ROI mask for patient model
        mask_dilation_mm (float): Dilation for mask generation
        roi_dilation_mm (float): Dilation for ROI mask
        transform_tools (TransformTools): Transform utilities
        registrar_ICON (RegisterImagesICON): ICON registration instance
        registrar_ANTS (RegisterImagesANTS): ANTs registration instance
        use_pca_registration (bool): Whether PCA registration is enabled (set via set_use_pca_registration)
        pca_model (dict): PCA model dict when PCA enabled; same structure as WorkflowCreateStatisticalModel output
        pca_number_of_modes (int): Number of PCA modes when PCA enabled
        icp_forward_point_transform : ICP transforms
        icp_inverse_point_transform : ICP inverse transforms
        icp_template_model_surface: template model surface after ICP alignment
        icp_template_model: template model (UnstructuredGrid) after ICP alignment
        pca_coefficients: PCA shape coefficients (if PCA used)
        pca_template_model (pv.DataSet): template model after PCA registration
            (if PCA used)
        pca_template_model_surface: template model surface after PCA registration (if
            PCA used)
        m2m_forward_transform: Mask-to-mask forward transform
        m2m_inverse_transform: Mask-to-mask inverse transform
        m2m_template_model_surface: template model surface after mask-to-mask
            registration
        m2i_forward_transform: Mask-to-image forward transform
        m2i_inverse_transform: Mask-to-image inverse transform
        m2i_template_model_surface: template model surface after mask-to-image
            registration
        m2i_template_labelmap: template labelmap after mask-to-image registration
        registered_template_model: Final registered model
        registered_template_model_surface: Final registered model surface

    Example:
        >>> # Initialize with minimal parameters (no labelmap; no patient image -> reference created from patient models)
        >>> registrar = WorkflowFitStatisticalModelToPatient(
        ...     template_model=heart_model,
        ...     patient_models=[lv_model, mc_model, rv_model],
        ... )
        >>> registrar.set_roi_dilation_mm(20)
        >>> # To enable PCA registration, call before run_workflow():
        >>> # registrar.set_use_pca_registration(True, pca_model=pca_model_dict, pca_number_of_modes=10)
        >>> # To enable mask-to-image refinement:
        >>> # registrar.set_use_mask_to_image_registration(True, template_labelmap, organ_mesh_ids, organ_extra_ids, background_ids)
        >>> result = registrar.run_workflow()
    """

    def __init__(
        self,
        template_model: pv.DataSet,
        patient_models: list[pv.DataSet] | None = None,
        patient_image: Optional[itk.Image] = None,
        segmentation_method: str = "HeartSimplewareTrimmedBranches",
        log_level: int | str = logging.INFO,
    ):
        """Initialize the model-to-image-and-model registration pipeline.

        Args:
            template_model: Generic anatomical model to be registered
            patient_models: List of patient-specific models extracted from imaging
                data. Typically 3 models for cardiac applications: LV, myocardium, RV.
            patient_image: Optional patient image providing the target coordinate frame.
                If None, a reference image is created from the patient model surface
                via create_reference_image (contour_tools).
            segmentation_method: Segmentation backend used by
                WorkflowConvertImageToVTK when patient_models is None and
                patient_image is provided. One of
                ``'HeartSimplewareTrimmedBranches'`` (default — produces
                cardiac extent that matches KCL-Heart-Model templates),
                ``'HeartSimpleware'``, or ``'ChestTotalSegmentator'``.
                Ignored when patient_models is supplied.
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO
        """
        # Initialize base class with logging
        super().__init__(
            class_name="WorkflowFitStatisticalModelToPatient", log_level=log_level
        )

        self.template_model = template_model
        self.template_model_surface = template_model.extract_surface(
            algorithm="dataset_surface"
        )
        self.template_labelmap: Optional[itk.Image] = None
        self.template_labelmap_organ_mesh_ids: Optional[list[int]] = None
        self.template_labelmap_organ_extra_ids: Optional[list[int]] = None
        self.template_labelmap_background_ids: Optional[list[int]] = None

        if patient_models is None and patient_image is not None:
            convert_image_to_vtk = WorkflowConvertImageToVTK(
                segmentation_method=segmentation_method,
                log_level=log_level,
            )
            patient_models_data = convert_image_to_vtk.run_workflow(
                input_image=patient_image,
                contrast_enhanced_study=False,
                anatomy_groups=["heart"],
            )
            patient_models = [patient_models_data["meshes"]["heart"]]
        elif patient_models is None:
            raise ValueError("Either patient_models or patient_image must be provided.")
        self.patient_models = patient_models
        patient_models_surfaces = [
            model.extract_surface(algorithm="dataset_surface")
            for model in patient_models
        ]
        self.combined_patient_model = pv.merge(patient_models_surfaces)
        self.patient_model_surface = self.combined_patient_model.extract_surface(
            algorithm="dataset_surface"
        )

        # Utilities (needed for create_reference_image when patient_image is None)
        self.transform_tools = TransformTools()
        self.contour_tools = ContourTools()
        self.labelmap_tools = LabelmapTools()

        if patient_image is not None:
            self.patient_image = patient_image
            if not _image_has_isotropic_spacing(self.patient_image):
                ttk = _load_tubetk()
                resampler = ttk.ResampleImage.New(Input=self.patient_image)
                resampler.SetMakeHighResIso(True)
                resampler.Update()
                self.patient_image = resampler.GetOutput()
        else:
            self.patient_image = self.contour_tools.create_reference_image(
                mesh=self.patient_model_surface,
                spatial_resolution=1.0,
                buffer_factor=0.25,
                ptype=itk.F,
            )

        self.registrar_ANTS = RegisterImagesANTS()
        self.registrar_ANTS.set_number_of_iterations([5, 2, 5])
        # Icon registration for final mask-to-image step
        self.registrar_ICON = RegisterImagesICON()
        self.registrar_ICON.set_modality("ct")
        self.registrar_ICON.set_mass_preservation(False)
        self.registrar_ICON.set_multi_modality(True)
        self.registrar_ICON.set_number_of_iterations(50)

        # Mask configuration (auto-generated)
        self.template_model_mask = None
        self.patient_mask = None
        self.template_model_roi = None
        self.patient_roi = None

        # Parameters for mask generation and processing
        self.mask_dilation_mm: float = 0.0  # For auto-generated mask dilation
        self.roi_dilation_mm: float = 25.0  # For ROI mask generation

        # Stage 1: ICP alignment results
        self.icp_registrar: Optional[RegisterModelsICP] = None
        self.icp_inverse_point_transform: Optional[itk.Transform] = None
        self.icp_forward_point_transform: Optional[itk.Transform] = None
        self.icp_template_model: Optional[pv.DataSet] = None
        self.icp_template_model_surface: Optional[pv.PolyData] = None
        self.icp_template_labelmap: Optional[itk.Image] = None

        # Stage 1.5: PCA registration results (optional; enable via set_use_pca_registration(True, pca_model, pca_number_of_modes))
        self.use_pca_registration = False
        self.pca_registrar: Optional[RegisterModelsPCA] = None
        self.pca_forward_point_transform: Optional[itk.Transform] = None
        self.pca_inverse_point_transform: Optional[itk.Transform] = None
        self.pca_model: Optional[dict[str, Any]] = None
        self.pca_number_of_modes: int = 0
        self.pca_coefficients: Optional[np.ndarray] = None
        self.pca_template_model: Optional[pv.DataSet] = None
        self.pca_template_model_surface: Optional[pv.PolyData] = None
        self.pca_template_labelmap: Optional[itk.Image] = None
        self.pca_uses_surface: bool = False

        # Stage 2: Mask-to-mask registration results
        self.use_m2m_registration = True
        self.m2m_inverse_transform: Optional[itk.Transform] = None
        self.m2m_forward_transform: Optional[itk.Transform] = None
        self.m2m_template_model_surface: Optional[pv.PolyData] = None
        self.m2m_template_labelmap: Optional[itk.Image] = None

        # Stage 3: Mask-to-image registration results (disabled by default; enable via set_use_mask_to_image_registration(True, template_labelmap, ...))
        self.use_m2i_registration = False
        self.m2i_inverse_transform: Optional[itk.Transform] = None
        self.m2i_forward_transform: Optional[itk.Transform] = None
        self.m2i_template_model_surface: Optional[pv.PolyData] = None
        self.m2i_template_labelmap: Optional[itk.Image] = None

        self.use_ICON_registration_refinement = False

        # Final result
        self.registered_template_model: Optional[pv.DataSet] = None
        self.registered_template_model_surface: Optional[pv.PolyData] = None
        self.registered_template_labelmap: Optional[itk.Image] = None

    def _auto_generate_mask(
        self, models: list[pv.DataSet], dilate_mm: Optional[float] = None
    ) -> itk.Image:
        """Auto-generate binary masks from models.

        Creates binary masks from list of models, with dilation
        according to mask_dilation_mm parameter.
        """
        self.log_info(
            f"Auto-generating masks from models (dilation:{self.mask_dilation_mm}mm)..."
        )

        # Generate patient mask (single model or multi-label)
        if len(models) == 1:
            mask = self.contour_tools.create_mask_from_mesh(
                models[0],
                self.patient_image,
            )
        else:
            # Create multi-label mask
            first_mask = self.contour_tools.create_mask_from_mesh(
                models[0],
                self.patient_image,
            )
            mask_arr = np.zeros_like(itk.GetArrayFromImage(first_mask), dtype=np.uint8)
            for i, model in enumerate(models):
                if i == 0:
                    mask = first_mask
                else:
                    mask = self.contour_tools.create_mask_from_mesh(
                        model,
                        self.patient_image,
                    )
                cur_mask = itk.GetArrayFromImage(mask)
                mask_arr[cur_mask > 0] = i + 1
            mask = itk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask.CopyInformation(self.patient_image)

        # Apply dilation if requested
        if dilate_mm is None:
            dilate_mm = self.mask_dilation_mm
        if dilate_mm > 0:
            mask = self.labelmap_tools.convert_labelmap_to_mask(
                mask, dilation_in_mm=dilate_mm
            )

        self.log_info("Masks auto-generated successfully.")

        return mask

    def _auto_generate_roi_mask(
        self, mask: itk.Image, dilate_mm: Optional[float] = None
    ) -> itk.Image:
        """Auto-generate ROI mask from existing masks with dilation.

        Uses self.roi_dilation_mm for dilation amount.

        Note:
            Requires masks to exist (auto-generated or user-provided).
        """
        self.log_info(
            f"Auto-generating ROI masks (dilation: {self.roi_dilation_mm}mm)..."
        )

        if dilate_mm is None:
            dilate_mm = self.roi_dilation_mm

        # Generate model ROI mask
        roi = None
        if dilate_mm > 0:
            roi = self.labelmap_tools.convert_labelmap_to_mask(
                mask, dilation_in_mm=dilate_mm
            )
        else:
            roi = mask

        self.log_info("ROI masks auto-generated successfully.")
        return roi

    def set_mask_dilation_mm(self, mask_dilation_mm: float) -> None:
        """Set mask dilation amount for auto-generated masks.

        Args:
            mask_dilation_mm: Dilation amount in millimeters for mask generation.
                Default: 5mm
        """
        self.mask_dilation_mm = mask_dilation_mm

    def set_roi_dilation_mm(self, roi_dilation_mm: float) -> None:
        """Set ROI mask dilation amount.

        Args:
            roi_dilation_mm: Dilation amount in millimeters for ROI mask generation.
                Default: 20mm
        """
        self.roi_dilation_mm = roi_dilation_mm

    def set_use_pca_registration(
        self,
        use_pca_registration: bool,
        pca_model: Optional[dict[str, Any]] = None,
        pca_number_of_modes: int = 0,
        pca_uses_surface: bool = True,
    ) -> None:
        """Set whether to use PCA-based registration and provide the PCA model.

        When enabling (True), pca_model and pca_number_of_modes must be provided.

        Args:
            use_pca_registration: Whether to use PCA registration after ICP.
            pca_model: Required when use is True. PCA model dict (e.g. from
                WorkflowCreateStatisticalModel result["pca_model"]) with keys
                "eigenvalues" and "components".
            pca_number_of_modes: Required when use is True. Number of PCA modes to use.
                Default 0 means use all modes.

        Raises:
            ValueError: If use is True and pca_model is None.
        """
        if use_pca_registration:
            if pca_model is None:
                raise ValueError(
                    "When enabling PCA registration, pca_model must be provided."
                )
            self.pca_model = pca_model
            self.pca_number_of_modes = pca_number_of_modes
            self.pca_uses_surface = pca_uses_surface
        else:
            self.pca_model = None
            self.pca_number_of_modes = 0
        self.use_pca_registration = use_pca_registration

    def set_use_mask_to_mask_registration(
        self, use_mask_to_mask_registration: bool
    ) -> None:
        """Set whether to use mask-to-mask registration.

        Args:
            use_mask_to_mask_registration: Whether to use mask-to-mask registration.
                Default: True
        """
        self.use_m2m_registration = use_mask_to_mask_registration

    def set_use_mask_to_image_registration(
        self,
        use_mask_to_image_registration: bool,
        template_labelmap: Optional[itk.Image] = None,
        template_labelmap_organ_mesh_ids: Optional[list[int]] = None,
        template_labelmap_organ_extra_ids: Optional[list[int]] = None,
        template_labelmap_background_ids: Optional[list[int]] = None,
    ) -> None:
        """Set whether to use mask-to-image registration.

        When enabling (True), a template labelmap and label IDs must be provided
        so the workflow can propagate and refine the labelmap to the patient image.

        Args:
            use_mask_to_image_registration: Whether to use mask-to-image registration.
            template_labelmap: Required when use is True. Template labelmap in template
                model space (same geometry as template_model).
            template_labelmap_organ_mesh_ids: Required when use is True. Label IDs for
                organ mesh in the template labelmap.
            template_labelmap_organ_extra_ids: Required when use is True. Label IDs for
                organ-extra structures in the template labelmap.
            template_labelmap_background_ids: Required when use is True. Label IDs for
                background in the template labelmap.

        Raises:
            ValueError: If use is True and any of template_labelmap or the id lists
                is None or missing.
        """
        if use_mask_to_image_registration:
            if template_labelmap is None:
                raise ValueError(
                    "When enabling mask-to-image registration, template_labelmap must be provided."
                )
            if template_labelmap_organ_mesh_ids is None:
                raise ValueError(
                    "When enabling mask-to-image registration, "
                    "template_labelmap_organ_mesh_ids must be provided."
                )
            if template_labelmap_organ_extra_ids is None:
                raise ValueError(
                    "When enabling mask-to-image registration, "
                    "template_labelmap_organ_extra_ids must be provided."
                )
            if template_labelmap_background_ids is None:
                raise ValueError(
                    "When enabling mask-to-image registration, "
                    "template_labelmap_background_ids must be provided."
                )
            self.template_labelmap = template_labelmap
            self.template_labelmap_organ_mesh_ids = template_labelmap_organ_mesh_ids
            self.template_labelmap_organ_extra_ids = template_labelmap_organ_extra_ids
            self.template_labelmap_background_ids = template_labelmap_background_ids
        self.use_m2i_registration = use_mask_to_image_registration

    def _transform_model_dataset(
        self,
        model: pv.DataSet,
        tfm: itk.Transform,
        *,
        with_deformation_magnitude: bool = False,
    ) -> pv.DataSet:
        """Transform a model with topology-preserving handling by PyVista type."""
        if isinstance(model, pv.PolyData):
            return self.transform_tools.transform_pvcontour(
                model,
                tfm,
                with_deformation_magnitude=with_deformation_magnitude,
            )
        return self.transform_tools.transform_dataset(
            model,
            tfm,
            with_deformation_magnitude=with_deformation_magnitude,
        )

    def register_model_to_model_icp(self) -> dict:
        """Perform ICP alignment of template model to patient model.

        Uses RegisterModelsICP class for ICP alignment.

        Returns:
            dict: Dictionary containing:
                - 'forward_transform': used to warp an image from model to patient space
                - 'inverse_transform': used to warp an image from patient to model space
                - 'registered_template_model_surface': Transformed model model surface
        """
        self.log_section("Stage 1: ICP Alignment (RegisterModelsICP)", width=70)

        # Create ICP registrar
        self.icp_registrar = RegisterModelsICP(fixed_model=self.patient_model_surface)

        # Run rigid ICP registration
        icp_result = self.icp_registrar.register(
            moving_model=self.template_model_surface,
            transform_type="Affine",
            max_iterations=2000,
        )

        # Store results
        # Note: Point transforms are in opposite direction from image transforms
        self.icp_forward_point_transform = icp_result["forward_point_transform"]
        self.icp_inverse_point_transform = icp_result["inverse_point_transform"]
        self.icp_template_model_surface = icp_result["registered_model"]

        self.icp_template_model = self._transform_model_dataset(
            self.template_model,
            self.icp_forward_point_transform,
        )

        if self.template_labelmap is not None:
            self.icp_template_labelmap = self.transform_tools.transform_image(
                self.template_labelmap,
                self.icp_inverse_point_transform,
                self.patient_image,
                interpolation_method="nearest",
            )
        else:
            self.icp_template_labelmap = None

        self.log_info("Stage 1 complete: ICP alignment finished.")

        self.registered_template_model_surface = self.icp_template_model_surface
        self.registered_template_model = self.icp_template_model
        self.registered_template_labelmap = self.icp_template_labelmap

        return {
            "inverse_point_transform": self.icp_inverse_point_transform,
            "forward_point_transform": self.icp_forward_point_transform,
            "registered_template_model": self.icp_template_model,
            "registered_template_model_surface": self.icp_template_model_surface,
            "registered_template_labelmap": self.icp_template_labelmap,
        }

    def register_model_to_model_pca(self) -> dict:
        """Perform PCA-based registration after ICP alignment.

        Uses RegisterModelsPCA class for intensity-based PCA registration.
        This method requires PCA data to be set via set_pca_data().

        Returns:
            dict: Dictionary containing:
                - 'forward_point_transform': Rigid transform from PCA registration
                - 'pca_coefficients': PCA shape coefficients
                - 'registered_template_model_surface': PCA-registered model surface

        Raises:
            ValueError: If PCA data has not been set
        """
        self.log_section(
            "Stage 2: PCA-Based Registration (RegisterModelsPCA)",
            width=70,
        )

        if not self.use_pca_registration or self.pca_model is None:
            self.pca_template_model = self.icp_template_model
            self.pca_template_model_surface = self.icp_template_model_surface
            self.pca_template_labelmap = self.icp_template_labelmap
            identity_transform = itk.CenteredAffineTransform[itk.D, 3].New()
            identity_transform.SetIdentity()
            self.pca_forward_point_transform = identity_transform
            self.pca_inverse_point_transform = identity_transform
            return {
                "pca_coefficients": None,
                "registered_template_model": self.pca_template_model,
                "registered_template_model_surface": self.pca_template_model_surface,
                "registered_template_labelmap": self.pca_template_labelmap,
                "forward_point_transform": self.pca_forward_point_transform,
                "inverse_point_transform": self.pca_inverse_point_transform,
            }

        pca_template_model: Optional[pv.DataSet]
        if self.pca_uses_surface:
            pca_template_model = self.icp_template_model_surface
            fixed_model = self.patient_model_surface
        else:
            pca_template_model = self.icp_template_model
            fixed_model = self.combined_patient_model
        assert pca_template_model is not None, "PCA template model must be set"

        self.pca_registrar = RegisterModelsPCA.from_pca_model(
            pca_template_model=pca_template_model,
            pca_model=self.pca_model,
            pca_number_of_modes=self.pca_number_of_modes,
            fixed_model=fixed_model,
            reference_image=self.patient_image,
        )

        # Run complete PCA registration
        assert self.pca_registrar is not None, "PCA registrar must be initialized"
        result = self.pca_registrar.register()
        self.pca_coefficients = result["pca_coefficients"]
        registered_model = cast(pv.DataSet, result["registered_model"])
        if self.pca_uses_surface:
            self.pca_template_model_surface = cast(pv.PolyData, registered_model)
        else:
            self.pca_template_model_surface = registered_model.extract_surface(
                algorithm="dataset_surface"
            )

        pca_transforms = self.pca_registrar.compute_pca_transforms(
            reference_image=self.patient_image,
        )
        self.pca_forward_point_transform = pca_transforms["forward_point_transform"]
        self.pca_inverse_point_transform = pca_transforms["inverse_point_transform"]

        if self.log_level == logging.DEBUG:
            tfm_arr = itk.GetArrayFromImage(
                self.pca_forward_point_transform.GetDisplacementField()
            )
            tfm_x_arr = tfm_arr[:, :, :, 0]
            tfm_y_arr = tfm_arr[:, :, :, 1]
            tfm_z_arr = tfm_arr[:, :, :, 2]
            tfm_x_img = itk.GetImageFromArray(tfm_x_arr)
            tfm_y_img = itk.GetImageFromArray(tfm_y_arr)
            tfm_z_img = itk.GetImageFromArray(tfm_z_arr)
            tfm_x_img.CopyInformation(self.patient_image)
            tfm_y_img.CopyInformation(self.patient_image)
            tfm_z_img.CopyInformation(self.patient_image)
            itk.imwrite(tfm_x_img, "pca_forward_point_transform_x.nii.gz")
            itk.imwrite(tfm_y_img, "pca_forward_point_transform_y.nii.gz")
            itk.imwrite(tfm_z_img, "pca_forward_point_transform_z.nii.gz")

        if self.pca_uses_surface:
            assert self.icp_template_model is not None, "ICP template model must be set"
            self.pca_template_model = self._transform_model_dataset(
                self.icp_template_model,
                self.pca_forward_point_transform,
            )
        else:
            self.pca_template_model = registered_model

        # Store results

        self.registered_template_model = self.pca_template_model
        self.registered_template_model_surface = self.pca_template_model_surface

        if self.icp_template_labelmap is not None:
            self.pca_template_labelmap = self.transform_tools.transform_image(
                self.icp_template_labelmap,
                self.pca_inverse_point_transform,
                self.patient_image,
                interpolation_method="nearest",
            )
        else:
            self.pca_template_labelmap = None

        self.registered_template_labelmap = self.pca_template_labelmap

        self.log_info("Stage 2 complete: PCA registration finished.")

        return {
            "pca_coefficients": self.pca_coefficients,
            "forward_point_transform": self.pca_forward_point_transform,
            "inverse_point_transform": self.pca_inverse_point_transform,
            "registered_template_model": self.pca_template_model,
            "registered_template_model_surface": self.pca_template_model_surface,
            "registered_template_labelmap": self.pca_template_labelmap,
        }

    def register_mask_to_mask(
        self, use_ICON_refinement: bool = False
    ) -> Optional[dict]:
        """Perform mask-based deformable registration of model to patient model.

        Uses RegisterModelsDistanceMaps class for ANTs deformable registration.

        Returns:
            dict: Dictionary containing:
                - 'forward_transform': model to patient space transform
                - 'inverse_transform': patient to model space transform
                - 'registered_template_model_surface': Transformed model model
                - 'registered_template_labelmap': Transformed model labelmap
        """
        self.log_section(
            "Stage 3: Mask-to-Mask Deformable Registration",
            width=70,
        )

        if not self.use_m2m_registration:
            self.log_info("Mask-to-mask registration is not enabled.")
            return None

        # Create mask-based registrar
        assert self.pca_template_model_surface is not None, (
            "PCA template model surface must be set"
        )
        mask_registrar = RegisterModelsDistanceMaps(
            moving_model=self.pca_template_model_surface,
            fixed_model=self.patient_model_surface,
            reference_image=self.patient_image,
            roi_dilation_mm=self.roi_dilation_mm,
        )

        # Run deformable registration
        mask_result = mask_registrar.register(
            transform_type="Deformable",
            use_ICON=use_ICON_refinement,
        )

        # Store results
        self.m2m_forward_transform = mask_result["forward_transform"]
        self.m2m_inverse_transform = mask_result["inverse_transform"]
        self.m2m_template_model_surface = mask_result["registered_model"]

        self.registered_template_model_surface = self.m2m_template_model_surface

        if self.pca_template_labelmap is not None:
            self.m2m_template_labelmap = self.transform_tools.transform_image(
                self.pca_template_labelmap,
                self.m2m_forward_transform,
                self.patient_image,
                interpolation_method="nearest",
            )
        else:
            self.m2m_template_labelmap = None

        self.registered_template_labelmap = self.m2m_template_labelmap

        self.log_info("Stage 3 complete: Mask-to-mask registration finished.")

        return {
            "forward_transform": self.m2m_forward_transform,
            "inverse_transform": self.m2m_inverse_transform,
            "registered_template_model_surface": self.m2m_template_model_surface,
            "registered_template_labelmap": self.m2m_template_labelmap,
        }

    def register_labelmap_to_image(
        self, use_ICON_refinement: bool = False
    ) -> Optional[dict]:
        """Perform labelmap-to-image refinement.

        Uses registration to align labelmap to actual image intensities.

        Returns:
            dict: Dictionary containing:
                - 'inverse_transform': patient to model space transform
                - 'forward_transform': model to patient space transform
                - 'registered_template_model_surface': Transformed model model
                - 'registered_template_labelmap': Transformed model labelmap
        """
        self.log_section(
            "Stage 4: Labelmap-to-Image Refinement (Icon Registration)", width=70
        )

        if (
            self.template_labelmap is None
            or self.template_labelmap_organ_mesh_ids is None
            or self.template_labelmap_organ_extra_ids is None
            or self.template_labelmap_background_ids is None
        ):
            raise ValueError(
                "Mask-to-image registration requires template labelmap and label IDs. "
                "Call set_use_mask_to_image_registration(True, template_labelmap, "
                "organ_mesh_ids, organ_extra_ids, background_ids) before run_workflow()."
            )
        if self.m2m_template_labelmap is None:
            raise ValueError(
                "Mask-to-image registration requires a labelmap to have been set "
                "(via set_use_mask_to_image_registration(True, ...)) before running "
                "earlier stages so the labelmap is propagated through ICP/PCA/M2M."
            )

        labelmap_arr = itk.GetArrayFromImage(self.m2m_template_labelmap).astype(
            np.uint16
        )
        labelmap_arr = np.where(
            np.isin(labelmap_arr, self.template_labelmap_background_ids),
            0,
            labelmap_arr,
        )
        labelmap_arr = np.where(
            np.isin(labelmap_arr, self.template_labelmap_organ_mesh_ids),
            0,
            labelmap_arr,
        )
        labelmap_arr = np.where(
            np.isin(labelmap_arr, self.template_labelmap_organ_extra_ids),
            1,
            labelmap_arr,
        )
        labelmap = itk.GetImageFromArray(labelmap_arr)
        labelmap.CopyInformation(self.m2m_template_labelmap)

        labelmap_roi = self._auto_generate_roi_mask(labelmap)

        patient_mask = self._auto_generate_mask(
            [self.patient_model_surface], dilate_mm=0
        )
        patient_roi = self._auto_generate_roi_mask(patient_mask)

        self.registrar_ANTS.set_fixed_image(self.patient_image)
        self.registrar_ANTS.set_fixed_mask(patient_roi)

        result = self.registrar_ANTS.register(
            moving_image=labelmap, moving_mask=labelmap_roi
        )
        self.m2i_inverse_transform = result["inverse_transform"]
        self.m2i_forward_transform = result["forward_transform"]

        if use_ICON_refinement:
            # Configure Icon registration
            self.registrar_ICON.set_fixed_image(self.patient_image)
            self.registrar_ICON.set_fixed_mask(patient_roi)

            # Perform Icon registration
            result = self.registrar_ICON.register(
                initial_forward_transform=self.m2i_forward_transform,
                moving_image=labelmap,
                moving_mask=labelmap_roi,
            )
            self.m2i_inverse_transform = result["inverse_transform"]
            self.m2i_forward_transform = result["forward_transform"]

        # Transform model with Icon result
        assert self.m2m_template_model_surface is not None, (
            "M2M template model surface must be set"
        )
        self.m2i_template_model_surface = cast(
            pv.PolyData,
            self._transform_model_dataset(
                self.m2m_template_model_surface,
                self.m2i_inverse_transform,
                with_deformation_magnitude=True,
            ),
        )

        self.m2i_template_labelmap = self.transform_tools.transform_image(
            self.template_labelmap,
            self.m2i_forward_transform,
            self.patient_image,
            interpolation_method="nearest",
        )

        self.log_info("Stage 4 complete: Mask-to-image registration finished.")

        self.registered_template_model_surface = self.m2i_template_model_surface

        self.registered_template_labelmap = self.m2i_template_labelmap

        return {
            "inverse_transform": self.m2i_inverse_transform,
            "forward_transform": self.m2i_forward_transform,
            "registered_template_model_surface": self.m2i_template_model_surface,
            "registered_template_labelmap": self.m2i_template_labelmap,
        }

    def transform_model(
        self, base_model: Optional[pv.DataSet] = None
    ) -> Optional[pv.DataSet]:
        """Apply registration transforms to the model.

        Transforms the model through all registration stages.

        Args:
            base_model: Base model for generating the new model.
                If None, the template model is used.

        Returns:
            pv.DataSet: Registered model
        """
        self.log_info("Applying transforms to model...")

        if base_model is None:
            self.registered_template_model = self.template_model.copy(deep=True)
            assert self.registered_template_model is not None, (
                "Registered template model must be set"
            )
            transformed_model = self.registered_template_model
        else:
            transformed_model = base_model.copy(deep=True)

        transform_steps: list[tuple[str, itk.Transform]] = []
        if self.icp_forward_point_transform is not None:
            transform_steps.append(("ICP", self.icp_forward_point_transform))
        if self.pca_coefficients is not None:
            assert self.pca_registrar is not None, "PCA registrar must be set"
            pca_transform = (
                self.pca_forward_point_transform
                or self.pca_registrar.forward_point_transform
            )
            if pca_transform is not None:
                transform_steps.append(("PCA", pca_transform))
            if self.pca_registrar.post_pca_transform is not None:
                transform_steps.append(
                    ("PCA post-transform", self.pca_registrar.post_pca_transform)
                )
        if self.use_m2m_registration and self.m2m_inverse_transform is not None:
            transform_steps.append(("Mask-to-mask", self.m2m_inverse_transform))
        if self.use_m2i_registration and self.m2i_inverse_transform is not None:
            transform_steps.append(("Mask-to-image", self.m2i_inverse_transform))

        for i, (name, tfm) in enumerate(transform_steps, start=1):
            self.log_progress(i, len(transform_steps), prefix=f"Applying {name}")
            transformed_model = self._transform_model_dataset(
                transformed_model,
                tfm,
            )

        new_points = np.asarray(transformed_model.points, dtype=float)

        self.log_info("Transform application complete.")

        if base_model is None:
            assert self.registered_template_model is not None, (
                "Registered template model must be set"
            )
            self.registered_template_model.points = new_points
            return self.registered_template_model
        transformed_model.points = new_points
        return transformed_model

    def run_workflow(
        self,
        use_ICON_registration_refinement: bool = False,
    ) -> dict:
        """Execute the complete multi-stage registration workflow.

        Runs registration stages in sequence:

        1. ICP alignment (RegisterModelsICP)
        2. PCA registration (PCA data was provided)
        3. Mask-to-mask deformable registration (RegisterModelsDistanceMaps)
        4. Optional mask-to-image refinement (Icon); requires template labelmap and IDs
            set via set_use_mask_to_image_registration(True, ...).

        Args:
            use_ICON_registration_refinement: Whether to include icon registration
                refinement stage. Default: False

        Returns:
            dict with registered_template_model and registered_template_model_surface
        """
        self.log_section(
            "STARTING COMPLETE MODEL-TO-IMAGE-AND-MODEL REGISTRATION WORKFLOW", width=70
        )

        self.use_ICON_registration_refinement = use_ICON_registration_refinement

        # Stage 1: ICP alignment
        self.register_model_to_model_icp()

        # Stage 2: Optional PCA registration (if PCA data was set)
        self.register_model_to_model_pca()

        # Stage 3: Optional Mask-to-mask deformable registration
        if self.use_m2m_registration:
            self.register_mask_to_mask(
                use_ICON_refinement=use_ICON_registration_refinement
            )

        # Stage 4: Optional mask-to-image refinement
        if self.use_m2i_registration:
            self.register_labelmap_to_image(
                use_ICON_refinement=use_ICON_registration_refinement
            )

        _ = self.transform_model()

        self.log_section("REGISTRATION WORKFLOW COMPLETE", width=70)
        assert self.registered_template_model_surface is not None, (
            "Registered template model surface must be set"
        )
        self.log_info(
            "Final registered patient model surface: %d points.",
            self.registered_template_model_surface.n_points,
        )

        return {
            "registered_template_model": self.registered_template_model,
            "registered_template_model_surface": self.registered_template_model_surface,
            "registered_template_labelmap": self.registered_template_labelmap,
        }
