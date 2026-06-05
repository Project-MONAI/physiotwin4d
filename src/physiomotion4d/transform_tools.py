"""
Tools for transforming and manipulating ITK transforms.

This module provides the TransformTools class with utilities for working with
ITK transforms, including transforming images and contours, generating
deformation fields, interpolating between transforms, and correcting spatial
folding artifacts.

The tools support various transform operations needed for medical image
analysis, particularly in the context of 4D cardiac imaging where transforms
are used to track anatomical motion over time.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias, cast

try:
    import cupy as cp  # optional (GPU)
except (ImportError, OSError):
    # ImportError: cupy not installed or CUDA libraries missing/mismatched.
    # OSError: driver/library load failure on some platforms.
    # In all cases fall back to CPU (NumPy) paths.
    cp = None
import itk
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk
from numpy.typing import NDArray
from pxr import Gf, Sdf, Usd, UsdGeom

from physiomotion4d.image_tools import ImageTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.vtk_to_usd import add_framing_camera

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]


class TransformTools(PhysioMotion4DBase):
    """
    Utilities for transforming and manipulating ITK transforms.

    This class provides a comprehensive set of tools for working with ITK
    transforms in medical image analysis. It supports transforming various
    data types (images, contours), generating visualization aids, and
    performing advanced operations like transform interpolation and spatial
    folding correction.

    The class is particularly useful for 4D cardiac imaging workflows where
    transforms are used to track anatomical motion over time, requiring
    operations like transform chaining, interpolation, and quality control.

    Key capabilities:
    - Transform PyVista contours and ITK images
    - Generate deformation fields from transforms
    - Interpolate between transforms temporally
    - Smooth transforms to reduce noise
    - Combine transforms with spatial masks
    - Detect and correct spatial folding
    - Generate visualization grids

    Example:
        >>> transform_tools = TransformTools()
        >>> # Transform a contour mesh
        >>> transformed_contour = transform_tools.transform_pvcontour(
        ...     contour, transform, with_deformation_magnitude=True
        ... )
        >>> # Generate deformation field
        >>> field = transform_tools.generate_field(transform, reference_image)
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the TransformTools class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def combine_displacement_field_transforms(
        self,
        tfm1: itk.Transform,
        tfm2: itk.Transform,
        reference_image: itk.Image,
        tfm1_weight: float = 1.0,
        tfm2_weight: float = 1.0,
        mode: str = "compose",
        tfm1_blur_sigma: float = 0.0,
        tfm2_blur_sigma: float = 0.0,
    ) -> itk.Transform:
        """
        Compose two displacement field transforms.

        In ``add`` mode, returns a single displacement field transform with
        weighted summed vectors. In ``compose`` mode, returns a composite
        transform containing both weighted displacement field transforms.
        """
        assert mode in ["add", "compose"], "Invalid mode"

        dtfm1 = self.convert_transform_to_displacement_field_transform(
            tfm1, reference_image
        )
        dtfm2 = self.convert_transform_to_displacement_field_transform(
            tfm2, reference_image
        )
        dfield1 = dtfm1.GetDisplacementField()
        dfield2 = dtfm2.GetDisplacementField()
        dfield1_arr = itk.array_from_image(dfield1)
        dfield2_arr = itk.array_from_image(dfield2)
        if tfm1_blur_sigma > 0.0:
            for dim in range(dfield1.GetNumberOfComponentsPerPixel()):
                tmp_field = dfield1_arr[:, :, :, dim]
                tmp_image = itk.image_from_array(tmp_field)
                tmp_image.CopyInformation(dfield1)
                tmp_image = itk.smoothing_recursive_gaussian_image_filter(
                    tmp_image, Sigma=tfm1_blur_sigma
                )
                tmp_field = itk.array_from_image(tmp_image)
                dfield1_arr[:, :, :, dim] = tmp_field
        if tfm2_blur_sigma > 0.0:
            for dim in range(dfield2.GetNumberOfComponentsPerPixel()):
                tmp_field = dfield2_arr[:, :, :, dim]
                tmp_image = itk.image_from_array(tmp_field)
                tmp_image.CopyInformation(dfield2)
                tmp_image = itk.smoothing_recursive_gaussian_image_filter(
                    tmp_image, Sigma=tfm2_blur_sigma
                )
                tmp_field = itk.array_from_image(tmp_image)
                dfield2_arr[:, :, :, dim] = tmp_field
        if mode == "add":
            dfield_composed_arr = tfm1_weight * dfield1_arr + tfm2_weight * dfield2_arr
            image_tools = ImageTools()
            dfield_composed = image_tools.convert_array_to_image_of_vectors(
                dfield_composed_arr,
                ptype=itk.D,
                reference_image=dfield1,
            )
            new_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
            new_tfm.SetDisplacementField(dfield_composed)
            return new_tfm
        # compose
        image_tools = ImageTools()

        dfield1_arr = tfm1_weight * dfield1_arr
        dfield2_arr = tfm2_weight * dfield2_arr
        new_dfield1 = image_tools.convert_array_to_image_of_vectors(
            dfield1_arr,
            ptype=itk.D,
            reference_image=dfield1,
        )
        new_dfield2 = image_tools.convert_array_to_image_of_vectors(
            dfield2_arr,
            ptype=itk.D,
            reference_image=dfield2,
        )
        new_tfm1 = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm1.SetDisplacementField(new_dfield1)
        new_tfm2 = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm2.SetDisplacementField(new_dfield2)
        composite_tfm = itk.CompositeTransform[itk.D, 3].New()
        composite_tfm.AddTransform(new_tfm1)
        composite_tfm.AddTransform(new_tfm2)
        return composite_tfm

    def convert_transform_to_displacement_field(
        self,
        tfm: itk.Transform,
        reference_image: itk.image,
        np_component_type: type[np.float32] | type[np.float64] = np.float64,
        use_reference_image_as_mask: bool = False,
    ) -> itk.image:
        """
        Generate a dense deformation field from an ITK transform.

        Converts any ITK transform into a dense displacement field that
        explicitly stores the displacement vector at each voxel. This is
        useful for visualization, analysis, and storage of transforms.

        Args:
            tfm (itk.Transform): Input transform to convert. Can be any ITK
                transform type (Affine, BSpline, DisplacementField, etc.)
            reference_image (itk.image): Defines the spatial grid for the
                output deformation field (spacing, size, origin, direction)
            use_reference_image_as_mask (bool): If True, applies the reference
                image as a mask to zero out displacement vectors outside
                the image domain

        Returns:
            itk.image: Vector image where each voxel contains a displacement
                vector [dx, dy, dz] in physical coordinates

        Example:
            >>> # Generate deformation field for visualization
            >>> field = transform_tools.generate_field(registration_transform,
                reference_ct)
            >>> # Use as mask to limit field to anatomical regions
            >>> masked_field = transform_tools.generate_field(
            ...     transform, reference_ct, use_reference_image_as_mask=True
            ... )
        """
        # Handle case where tfm is a list (e.g., from itk.transformread)
        if isinstance(tfm, (list, tuple)):
            if len(tfm) == 1:
                tfm = tfm[0]
            else:
                raise ValueError(
                    f"Expected single transform, got list with {len(tfm)} transforms"
                )

        TfmPrecision = itk.template(tfm)[1][0]

        # Create and configure filter
        field = None
        if "DisplacementFieldTransform" not in str(type(tfm)):
            field_filter = itk.TransformToDisplacementFieldFilter[
                itk.Image[itk.Vector[itk.F, 3], 3], TfmPrecision
            ].New()
            field_filter.SetTransform(tfm)
            field_filter.SetReferenceImage(reference_image)
            field_filter.SetUseReferenceImage(True)
            field_filter.Update()
            field = field_filter.GetOutput()
        else:
            field = tfm.GetDisplacementField()
            field_arr = itk.array_view_from_image(tfm.GetDisplacementField())
            reference_image_arr = itk.array_view_from_image(reference_image)
            if field_arr.shape[:3] != reference_image_arr.shape:
                field_filter = itk.TransformToDisplacementFieldFilter[
                    itk.Image[itk.Vector[itk.F, 3], 3], TfmPrecision
                ].New()
                field_filter.SetTransform(tfm)
                field_filter.SetReferenceImage(reference_image)
                field_filter.SetUseReferenceImage(True)
                field_filter.Update()
                field = field_filter.GetOutput()

        field_arr = itk.array_from_image(field)
        field_arr = field_arr.astype(np_component_type)

        image_tools = ImageTools()
        field = image_tools.convert_array_to_image_of_vectors(
            field_arr,
            ptype=np_component_type,
            reference_image=reference_image,
        )

        if use_reference_image_as_mask:
            mask = reference_image
            field = itk.MaskImageFilter(field, mask)

        return field

    def convert_transform_to_displacement_field_transform(
        self, tfm: itk.Transform, reference_image: itk.Image
    ) -> itk.DisplacementFieldTransform:
        """
        Convert an ITK transform to a displacement field transform.
        """
        # TransformToDisplacementFieldFilter only supports float precision
        # so we need to cast the transform to float and then convert
        # to double since most other transform filters require double precision
        field = self.convert_transform_to_displacement_field(
            tfm, reference_image, np_component_type=np.float64
        )

        new_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm.SetDisplacementField(field)
        return new_tfm

    def invert_displacement_field_transform(self, tfm: itk.Transform) -> itk.Transform:
        """
        Invert a displacement field transform.
        """
        assert "DisplacementFieldTransform" in str(type(tfm)), (
            "Input transform must be a displacement field transform"
        )
        image_tools = ImageTools()

        field_itk = tfm.GetDisplacementField()

        field_sitk = image_tools.convert_itk_image_to_sitk(field_itk)

        field_sitk_inv = sitk.InvertDisplacementField(field_sitk)

        field_itk_inv = image_tools.convert_sitk_image_to_itk(field_sitk_inv)

        new_tfm = itk.DisplacementFieldTransform[itk.D, 3].New()
        new_tfm.SetDisplacementField(field_itk_inv)

        return new_tfm

    def transform_pvcontour(
        self,
        contour: pv.PolyData,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.PolyData:
        """
        Transform PyVista contour meshes using an ITK transform.

        Applies an ITK transform to all points in a PyVista PolyData mesh,
        useful for deforming anatomical contours according to computed
        registration transforms. Optionally computes deformation magnitude
        at each point.

        Args:
            contour (pv.PolyData): The input contour mesh to transform
            tfm (itk.Transform): ITK transform to apply. Can be a single
                transform or a list/array containing one transform
            with_deformation_magnitude (bool): If True, adds a
                "DeformationMagnitude" point data array containing the
                Euclidean distance each point moved

        Returns:
            pv.PolyData: The transformed contour mesh with updated point
                coordinates and optionally deformation magnitude data

        Example:
            >>> # Transform cardiac contour with deformation tracking
            >>> transformed_heart = transform_tools.transform_pvcontour(
            ...     heart_contour, cardiac_transform, with_deformation_magnitude=True
            ... )
            >>> # Access deformation magnitudes
            >>> deformation = transformed_heart['DeformationMagnitude']
        """

        return cast(
            pv.PolyData,
            self.transform_dataset(
                contour,
                tfm,
                with_deformation_magnitude=with_deformation_magnitude,
            ),
        )

    def transform_dataset(
        self,
        mesh: pv.DataSet,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.DataSet:
        """Transform a PyVista dataset while preserving mesh topology and data arrays.

        Applies an ITK point transform to every point in the input dataset and
        returns a deep copy with the original cells, cell data, and point data
        preserved. This is appropriate for non-contour datasets such as
        UnstructuredGrid inputs where casting to PolyData would lose topology.
        """

        new_mesh = mesh.copy(deep=True)
        pnts = np.array(new_mesh.points, dtype=float)

        # Handle case where tfm is a list (e.g., from itk.transformread)
        if isinstance(tfm, (list, tuple)):
            if len(tfm) == 1:
                tfm = tfm[0]
            else:
                raise ValueError(
                    f"Expected single transform, got list with {len(tfm)} transforms"
                )

        pnts = np.array(pnts)
        new_pnts = [
            np.array(tfm.TransformPoint((float(p[0]), float(p[1]), float(p[2]))))
            for p in pnts
        ]
        new_mesh.points = np.asarray(new_pnts, dtype=float).reshape(-1, 3)

        if with_deformation_magnitude:
            if cp is not None:
                new_pnts_cp = cp.array(new_pnts)
                pnts_cp = cp.array(pnts)
                new_mesh.point_data["DeformationMagnitude"] = cp.linalg.norm(
                    new_pnts_cp - pnts_cp, axis=1
                ).get()
            else:
                new_mesh.point_data["DeformationMagnitude"] = np.linalg.norm(
                    np.asarray(new_pnts) - np.asarray(pnts), axis=1
                )

        return new_mesh

    def transform_image(
        self,
        img: itk.image,
        tfm: itk.Transform,
        reference_image: itk.image,
        interpolation_method: str = "linear",
    ) -> itk.image:
        """
        Transform an ITK image using a specified transform and interpolation.

        Resamples an image according to a geometric transform, using the
        reference image to define the output grid properties. Different
        interpolation methods are available depending on data type and
        quality requirements.

        Args:
            img (itk.image): The input image to transform
            tfm (itk.Transform): The ITK transform to apply
            reference_image (itk.image): Defines output spacing, size, origin,
                and direction for the transformed image
            tfm_type (str): Interpolation method. Options:
                - "linear": Linear interpolation (default, good for CT/MR)
                - "nearest": Nearest neighbor (preserves discrete values)
                - "sinc": Sinc interpolation (highest quality, slower)

        Returns:
            itk.image: The transformed image resampled to reference grid

        Raises:
            ValueError: If tfm_type is not one of the supported options

        Example:
            >>> # Transform CT image with linear interpolation
            >>> warped_ct = transform_tools.transform_image(
            ...     ct_image, deformation_transform, reference_ct
            ... )
            >>> # Transform label map preserving discrete values
            >>> warped_labels = transform_tools.transform_image(
            ...     labelmap, transform, reference, interpolation_method='nearest'
            ... )
        """
        # Handle case where tfm is a list (e.g., from itk.transformread)
        if isinstance(tfm, (list, tuple)):
            if len(tfm) == 1:
                tfm = tfm[0]
            else:
                raise ValueError(
                    "Expected single transform or list with one transform, got list"
                    f"with {len(tfm)} transforms"
                )

        interpolator = None
        if interpolation_method == "linear":
            interpolator = itk.LinearInterpolateImageFunction.New(img)
        elif interpolation_method == "nearest":
            interpolator = itk.NearestNeighborInterpolateImageFunction.New(img)
        elif interpolation_method == "sinc":
            interpolator = itk.WindowedSincInterpolateImageFunction.New(img)
        else:
            raise ValueError(f"Invalid transform type: {interpolation_method}")

        # This shouldn't be needed, but for certain itk.CompositeTransform types,
        # the resample_image_filter will silently fail and apply the identity
        # transform instead of the one passed.
        dftfm = self.convert_transform_to_displacement_field_transform(
            tfm, reference_image
        )

        img_reg = itk.resample_image_filter(
            Input=img,
            Transform=dftfm,
            Interpolator=interpolator,
            ReferenceImage=reference_image,
            UseReferenceImage=True,
        )
        return img_reg

    def convert_vtk_matrix_to_itk_transform(
        self, vtk_mat: vtk.vtkMatrix4x4
    ) -> itk.Transform:
        """
        Convert a VTK matrix to an ITK transform.

        Converts a VTK matrix object into an equivalent ITK transform.
        This is useful for interoperability between VTK-based processing
        (e.g., mesh manipulation) and ITK-based image processing and
        registration.

        Args:
            vtk_mat (itk.vtkMatrix): The input VTK transform to convert
        Returns:
            itk.Transform: The equivalent ITK transform

        Example:
            >>> # Convert VTK transform from mesh processing
            >>> itk_transform = transform_tools.get_itk_transform_from_vtk_transform
                vtk_transform)
        """
        mat = np.eye(3).astype(np.float64)
        vec = itk.Vector[itk.D, 3]()
        for i in range(3):
            vec[i] = vtk_mat.GetElement(i, 3)
            for j in range(3):
                mat[i, j] = vtk_mat.GetElement(i, j)
        itkmat = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(mat))
        itk_tfm = itk.AffineTransform[itk.D, 3].New()
        itk_tfm.SetIdentity()
        itk_tfm.SetMatrix(itkmat)
        itk_tfm.SetOffset(vec)

        return itk_tfm

    def smooth_transform(
        self, tfm: itk.Transform, sigma: float, reference_image: itk.image
    ) -> itk.Transform:
        """
        Smooth a transform using Gaussian filtering to reduce noise.

        Applies Gaussian smoothing to the displacement field representation
        of a transform to reduce noise and create more regularized
        deformations. This is useful for improving transform quality and
        reducing artifacts.

        Args:
            tfm (itk.Transform): Input transform to smooth
            sigma (float): Standard deviation of Gaussian smoothing kernel
                in physical units (millimeters). Larger values create
                more smoothing
            reference_image (itk.image): Defines spatial grid for field
                generation and smoothing

        Returns:
            itk.Transform: DisplacementFieldTransform with smoothed
                deformation field

        Example:
            >>> # Smooth noisy registration transform
            >>> smooth_transform = transform_tools.smooth_transform(
            ...     noisy_transform, sigma=2.0, reference_ct
            ... )
            >>> # Light smoothing for artifact reduction
            >>> refined_transform = transform_tools.smooth_transform(
            ...     transform, sigma=0.5, reference_image
            ... )
        """
        field = self.convert_transform_to_displacement_field(tfm, reference_image)

        field_arr = itk.array_from_image(field)
        for dim in range(field.GetNumberOfComponentsPerPixel()):
            tmp_field_arr = field_arr[:, :, :, dim]
            tmp_image = itk.image_from_array(tmp_field_arr)
            tmp_image.CopyInformation(field)
            tmp_image = itk.smoothing_recursive_gaussian_image_filter(
                tmp_image, Sigma=sigma
            )
            tmp_field_arr = itk.array_from_image(tmp_image)
            field_arr[:, :, :, dim] = tmp_field_arr
        image_tools = ImageTools()
        field = image_tools.convert_array_to_image_of_vectors(
            field_arr,
            ptype=itk.D,
            reference_image=field,
        )

        tfm_smooth = itk.DisplacementFieldTransform[
            itk.D, field.GetImageDimension()
        ].New()
        tfm_smooth.SetDisplacementField(field)

        return tfm_smooth

    def combine_transforms_with_masks(
        self,
        transform1: itk.Transform,
        transform2: itk.Transform,
        mask1: itk.Image,
        mask2: itk.Image,
        reference_image: itk.Image,
        max_iter: int = 10,
        jacobian_threshold: float = 0.1,
    ) -> itk.Transform:
        """
        Combine two transforms using spatial masks with folding correction.

        Merges two transforms by weighting their displacement fields according
        to provided masks, then iteratively corrects any spatial folding
        (negative Jacobian determinant) that may result from the combination.

        This is useful for combining transforms computed for different
        anatomical regions (e.g., separate heart and lung registration)
        into a single coherent transform.

        Args:
            transform1 (itk.Transform): First transform to combine
            transform2 (itk.Transform): Second transform to combine
            mask1 (itk.Image): Float mask defining spatial influence of
                transform1 (0.0 = no influence, 1.0 = full influence)
            mask2 (itk.Image): Float mask defining spatial influence of
                transform2
            reference_image (itk.Image): Defines output grid properties
            max_iter (int): Maximum iterations for folding correction
            jacobian_threshold (float): Jacobian determinant threshold below
                which folding is detected and corrected

        Returns:
            itk.Transform: DisplacementFieldTransform with combined and
                corrected transformation

        Example:
            >>> # Combine heart and lung transforms
            >>> combined_transform = transform_tools.combine_transforms_with_masks(
            ...     heart_transform, lung_transform, heart_mask, lung_mask, reference_ct
            ... )
        """
        # Generate displacement fields
        field1 = self.convert_transform_to_displacement_field(
            transform1, reference_image
        )
        field2 = self.convert_transform_to_displacement_field(
            transform2, reference_image
        )

        # Weight fields by masks
        mask1_arr = itk.array_from_image(mask1)
        mask2_arr = itk.array_from_image(mask2)

        field1_arr = itk.array_from_image(field1)
        field2_arr = itk.array_from_image(field2)

        # Expand mask dimensions to match vector field (add dimension for vector
        #     components)
        mask1_arr = mask1_arr[..., np.newaxis]
        mask2_arr = mask2_arr[..., np.newaxis]

        sum_fields_arr = mask1_arr * field1_arr + mask2_arr * field2_arr

        denom = mask1_arr + mask2_arr
        denom[denom == 0] = 1.0

        combined_field_arr = sum_fields_arr / denom

        # Copy array data to ITK image
        combined_field = ImageTools().convert_array_to_image_of_vectors(
            combined_field_arr, field1, itk.F
        )

        # Correct spatial folding iteratively
        for _ in range(max_iter):
            jacobian_det = self.compute_jacobian_determinant_from_field(combined_field)
            if not self.detect_folding_in_field(
                jacobian_det, threshold=jacobian_threshold
            ):
                break
            combined_field = self.reduce_folding_in_field(combined_field, jacobian_det)

        # Get dimension and create transform with correct types
        Dimension = combined_field.GetImageDimension()
        tfm_combined = itk.DisplacementFieldTransform[itk.F, Dimension].New()
        tfm_combined.SetDisplacementField(combined_field)

        return tfm_combined

    def compute_jacobian_determinant_from_field(self, field: itk.Image) -> itk.Image:
        """Compute Jacobian determinant of a displacement field.

        Calculates the Jacobian determinant at each voxel of a displacement
        field, which indicates local volume change. Values less than 0
        indicate spatial folding, values between 0-1 indicate compression,
        and values greater than 1 indicate expansion.

        Args:
            field (itk.Image): Vector displacement field image

        Returns:
            itk.Image: Scalar image containing Jacobian determinant values

        Example:
            >>> jacobian = transform_tools.compute_jacobian_determinant_from_field(
                    deformation_field
                )
        """
        if "VF" not in str(type(field)):
            field_arr = itk.array_from_image(field)
            field = ImageTools().convert_array_to_image_of_vectors(
                field_arr, field, itk.F
            )
        jac_filter = itk.DisplacementFieldJacobianDeterminantFilter.New(field)
        jac_filter.SetUseImageSpacing(True)
        jac_filter.Update()
        return jac_filter.GetOutput()

    def detect_folding_in_field(
        self, jacobian_det: itk.Image, threshold: float = 0.1
    ) -> bool:
        """Detect spatial folding in a transform.

        Checks for spatial folding by examining the minimum Jacobian
        determinant value. Folding occurs when the Jacobian determinant
        becomes negative or very small, indicating non-invertible regions.

        Args:
            jacobian_det (itk.Image): Jacobian determinant image
            threshold (float): Threshold below which folding is detected

        Returns:
            bool: True if folding is detected, False otherwise

        Example:
            >>> if transform_tools.detect_folding_in_field(jacobian, 0.1):
            ...     print('Spatial folding detected - transform needs correction')
        """
        stats = itk.StatisticsImageFilter.New(jacobian_det)
        stats.Update()
        return float(stats.GetMinimum()) < threshold

    def reduce_folding_in_field(
        self,
        field: itk.Image,
        jacobian_det: itk.Image,
        reduction_factor: float = 0.8,
        threshold: float = 0.1,
    ) -> itk.Image:
        """Reduce folding by scaling displacement field in problematic regions.

        Corrects spatial folding by reducing the magnitude of displacement
        vectors in regions where the Jacobian determinant is below the
        threshold. This is a simple but effective approach to maintaining
        transform invertibility.

        Args:
            field (itk.Image): Input displacement field to correct
            jacobian_det (itk.Image): Jacobian determinant image
            reduction_factor (float): Factor to multiply displacements in
                folding regions (0.8 = 20% reduction)
            threshold (float): Jacobian threshold for identifying folding

        Returns:
            itk.Image: Corrected displacement field with reduced folding

        Example:
            >>> corrected_field = transform_tools.reduce_folding_in_field(
            ...     folded_field, jacobian, reduction_factor=0.7
            ... )
        """
        # Create correction mask
        thresholder = itk.BinaryThresholdImageFilter.New(jacobian_det)
        thresholder.SetLowerThreshold(-1000)
        thresholder.SetUpperThreshold(threshold)
        thresholder.SetInsideValue(reduction_factor)
        thresholder.SetOutsideValue(1.0)
        thresholder.Update()

        thresh_arr = itk.array_from_image(thresholder.GetOutput())
        field_arr = itk.array_from_image(field)
        for i in range(field_arr.shape[3]):
            field_arr[:, :, :, i] *= thresh_arr
        corrected_field = ImageTools().convert_array_to_image_of_vectors(
            field_arr, field, itk.F
        )
        return corrected_field

    def generate_grid_image(
        self, reference_image: itk.image, grid_size: int = 60, line_width: int = 3
    ) -> itk.image:
        """
        Generate a grid image.
        """
        img_arr = itk.array_from_image(reference_image)
        img_arr_max = np.max(img_arr)
        img_shape = list(img_arr.shape)
        grid_spacing = [s / grid_size for s in img_shape]
        if line_width <= 0:
            line_width = 1
        width_min = line_width // 2
        width_max = width_min + line_width
        for i in range(grid_size):
            for j in range(grid_size):
                min_idx0 = max(0, int(i * grid_spacing[0]) - width_min)
                max_idx0 = min(img_arr.shape[0], int(i * grid_spacing[0]) + width_max)
                min_idx1 = max(0, int(j * grid_spacing[1]) - width_min)
                max_idx1 = min(img_arr.shape[1], int(j * grid_spacing[1]) + width_max)
                if min_idx0 < max_idx0 and min_idx1 < max_idx1:
                    img_arr[min_idx0:max_idx0, min_idx1:max_idx1, :] = img_arr_max

                min_idx2 = max(0, int(j * grid_spacing[2]) - width_min)
                max_idx2 = min(img_arr.shape[2], int(j * grid_spacing[2]) + width_max)
                if min_idx0 < max_idx0 and min_idx2 < max_idx2:
                    img_arr[min_idx0:max_idx0, :, min_idx2:max_idx2] = img_arr_max

                min_idx1 = max(0, int(i * grid_spacing[1]) - width_min)
                max_idx1 = min(img_arr.shape[1], int(i * grid_spacing[1]) + width_max)
                if min_idx1 < max_idx1 and min_idx2 < max_idx2:
                    img_arr[:, min_idx1:max_idx1, min_idx2:max_idx2] = img_arr_max

        grid_image = itk.image_from_array(img_arr)
        grid_image.CopyInformation(reference_image)

        return grid_image

    def convert_field_to_grid_visualization(
        self,
        tfm: itk.Transform,
        reference_image: itk.image,
        grid_size: int = 60,
        line_width: int = 3,
    ) -> itk.image:
        """
        Generate a visual deformation grid for transform visualization.

        Creates a regular grid pattern in the reference image space, then
        applies the transform to visualize the deformation. The resulting
        warped grid shows how the transform deforms space and can reveal
        areas of compression, expansion, or folding.

        Args:
            tfm (itk.Transform): Transform to visualize
            reference_image (itk.image): Defines spatial domain and grid properties
            grid_size (int): Number of grid lines in each dimension

        Returns:
            itk.image: Binary image containing the transformed grid pattern

        Example:
            >>> # Create deformation visualization grid
            >>> grid = transform_tools.generate_visual_grid_from_field(
            ...     cardiac_transform, reference_ct, grid_size=20
            ... )
            >>> # Overlay on original image for visualization
        """
        grid_image = self.generate_grid_image(reference_image, grid_size, line_width)

        grid_image_tfm = self.transform_image(grid_image, tfm, reference_image)

        return grid_image_tfm

    def convert_itk_transform_to_usd_visualization(
        self,
        tfm: itk.Transform,
        reference_image: itk.image,
        output_filename: str,
        visualization_type: str = "arrows",
        subsample_factor: int = 4,
        arrow_scale: float = 1.0,
        magnitude_threshold: float = 0.0,
    ) -> str:
        """
        Convert an ITK transform to a USD visualization for NVIDIA Omniverse.

        Creates a USD file containing either arrows or flow lines that visualize
        the displacement field from the transform. Arrows show direction and
        magnitude at sampled points, while flow lines show particle trajectories
        through the deformation field.

        Args:
            tfm (itk.Transform): Input ITK transform to visualize. Can be any ITK
                transform type (Affine, BSpline, DisplacementField, Composite, etc.)
            reference_image (itk.image): Defines the spatial grid for sampling
                the displacement field (spacing, size, origin, direction)
            output_filename (str): Path to output USD file (e.g., "deformation.usda")
            visualization_type (str): Type of visualization - either "arrows" or
                "flowlines". Default is "arrows".
            subsample_factor (int): Subsample the displacement field by this factor
                in each dimension to reduce primitive count. Default is 4 (64x fewer
                points). Higher values = fewer primitives = better performance.
            arrow_scale (float): Scale factor for arrow length. Default is 1.0.
                Increase to make arrows longer, decrease to make them shorter.
            magnitude_threshold (float): Only visualize displacements with magnitude
                greater than this threshold (in mm). Default is 0.0 (show all).
                Use to filter out small/negligible displacements.

        Returns:
            str: Path to the created USD file

        Example:
            >>> # Create arrow visualization of registration transform
            >>> usd_file = transform_tools.convert_transform_to_usd_visualization(
            ...     registration_transform,
            ...     reference_ct,
            ...     'deformation_arrows.usda',
            ...     visualization_type='arrows',
            ...     subsample_factor=8,  # 512x fewer arrows
            ...     arrow_scale=2.0,  # 2x longer arrows
            ...     magnitude_threshold=1.0,  # Only show displacements > 1mm
            ... )
            >>>
            >>> # Create flow line visualization
            >>> usd_file = transform_tools.convert_transform_to_usd_visualization(
            ...     registration_transform,
            ...     reference_ct,
            ...     'deformation_flowlines.usda',
            ...     visualization_type='flowlines',
            ...     subsample_factor=4,
            ... )

        Note:
            - The USD file can be opened in NVIDIA Omniverse for 3D visualization
            - Arrows are colored by displacement magnitude (blue=low, red=high)
            - Flowlines show particle paths through the deformation field
            - Subsampling is critical for performance - dense fields can have millions
                of points, creating too many primitives for interactive visualization
        """
        # Generate ITK displacement field from transform
        displacement_field = self.convert_transform_to_displacement_field(
            tfm, reference_image
        )

        # Get displacement field as numpy array (z, y, x, 3)
        displacement_array = itk.GetArrayFromImage(displacement_field)

        # Get image size
        size = displacement_field.GetLargestPossibleRegion().GetSize()

        # Subsample the field to reduce number of primitives
        displacement_array = displacement_array[
            ::subsample_factor, ::subsample_factor, ::subsample_factor, :
        ]
        subsampled_size = [
            size[0] // subsample_factor,
            size[1] // subsample_factor,
            size[2] // subsample_factor,
        ]

        # Remove any existing file and evict any stale in-memory USD layer.
        # USD caches layers globally by identifier, so a prior call in the
        # same Python session can block CreateNew even after the file is gone.
        output_path = Path(output_filename)
        if output_path.exists():
            output_path.unlink()
        stale_layer = Sdf.Layer.Find(str(output_path))
        if stale_layer is not None:
            stale_layer.Clear()
            del stale_layer

        # Create USD stage
        stage = Usd.Stage.CreateNew(output_filename)

        # Set up stage metadata
        stage.SetMetadata("upAxis", "Y")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        # Create root xform
        # root_xform = UsdGeom.Xform.Define(stage, "/DeformationVisualization")

        if visualization_type == "arrows":
            self._create_arrow_visualization(
                stage,
                displacement_array,
                displacement_field,
                subsampled_size,
                subsample_factor,
                arrow_scale,
                magnitude_threshold,
            )
        elif visualization_type == "flowlines":
            self._create_flowline_visualization(
                stage,
                displacement_array,
                displacement_field,
                subsampled_size,
                subsample_factor,
                magnitude_threshold,
            )
        else:
            raise ValueError(
                f"Invalid visualization_type: {visualization_type}. "
                "Must be 'arrows' or 'flowlines'."
            )

        # Framing camera with tight near-clip for Omniverse Kit viewer ergonomics.
        add_framing_camera(stage)

        # Save the stage
        stage.Save()
        self.log_info("Created USD visualization: %s", output_filename)
        self.log_info("  Type: %s", visualization_type)
        self.log_info("  Points: %d", np.prod(subsampled_size))
        self.log_info("  Subsample factor: %d", subsample_factor)

        return output_filename

    def _create_arrow_visualization(
        self,
        stage: Usd.Stage,
        displacement_array: FloatArray,
        displacement_field: itk.Image,
        size: Sequence[int],
        subsample_factor: int,
        arrow_scale: float,
        magnitude_threshold: float,
    ) -> None:
        """Create arrow-based visualization of displacement field."""
        # Iterate through all points in the subsampled field
        arrow_count = 0
        for k in range(size[2]):
            for j in range(size[1]):
                for i in range(size[0]):
                    # Get displacement vector at this point (z, y, x, 3)
                    displacement = displacement_array[k, j, i, :]

                    # Calculate magnitude
                    magnitude = np.linalg.norm(displacement)

                    # Skip if below threshold
                    if magnitude < magnitude_threshold:
                        continue

                    # Calculate index in original image space
                    index = [
                        i * subsample_factor,
                        j * subsample_factor,
                        k * subsample_factor,
                    ]

                    # Convert index to physical/world position using ITK
                    world_pos = displacement_field.TransformIndexToPhysicalPoint(index)

                    # Create arrow at this position
                    arrow_path = f"/DeformationVisualization/Arrow_{arrow_count}"
                    self._create_arrow_prim(
                        stage,
                        arrow_path,
                        world_pos,
                        displacement,
                        float(magnitude),
                        arrow_scale,
                    )
                    arrow_count += 1

        self.log_info("  Created %d arrows", arrow_count)

    def _create_arrow_prim(
        self,
        stage: Usd.Stage,
        prim_path: str,
        position: Sequence[float],
        displacement: FloatArray,
        magnitude: float,
        arrow_scale: float,
    ) -> None:
        """Create a single arrow primitive representing a displacement vector."""
        # Create xform for the arrow
        arrow_xform = UsdGeom.Xform.Define(stage, prim_path)

        # Create a cone for the arrow (pointing in +Y direction by default)
        cone = UsdGeom.Cone.Define(stage, f"{prim_path}/cone")

        # Set cone size based on displacement magnitude
        arrow_length = float(magnitude * arrow_scale)
        cone.GetHeightAttr().Set(arrow_length)
        cone.GetRadiusAttr().Set(arrow_length * 0.1)  # 10% of length

        # Calculate rotation to align arrow with displacement direction
        if magnitude > 1e-6:  # Avoid division by zero
            # Normalize displacement to get direction
            direction = displacement / magnitude

            # Default cone points in +Y, we want it to point along displacement
            # Create rotation matrix to align +Y with displacement direction
            up = np.array([0, 1, 0])
            rotation_axis = np.cross(up, direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:  # Not parallel
                rotation_axis = rotation_axis / rotation_axis_norm
                cos_angle = np.dot(up, direction)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                # Convert axis-angle to quaternion
                half_angle = angle / 2.0
                sin_half = np.sin(half_angle)
                quat = Gf.Quatd(
                    np.cos(half_angle),  # w
                    rotation_axis[0] * sin_half,  # x
                    rotation_axis[1] * sin_half,  # y
                    rotation_axis[2] * sin_half,  # z
                )

                # Apply rotation (using Orient for quaternion)
                arrow_xform.AddXformOp(UsdGeom.XformOp.TypeOrient).Set(quat)

        # Position the arrow at the base point + half displacement
        # (so arrow starts at point and extends along displacement)
        arrow_position = [
            position[0] + displacement[0] * 0.5,
            position[1] + displacement[1] * 0.5,
            position[2] + displacement[2] * 0.5,
        ]
        arrow_xform.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(
            Gf.Vec3f(
                float(arrow_position[0]),
                float(arrow_position[1]),
                float(arrow_position[2]),
            )
        )

        # Color based on magnitude (blue to red colormap)
        # Normalize magnitude for coloring (assume max ~20mm displacement)
        max_expected_displacement = 20.0
        normalized_mag = min(magnitude / max_expected_displacement, 1.0)

        # Blue (low) to red (high) colormap
        color = Gf.Vec3f(float(normalized_mag), 0.0, float(1.0 - normalized_mag))
        cone.GetDisplayColorAttr().Set([color])

    def _create_flowline_visualization(
        self,
        stage: Usd.Stage,
        displacement_array: FloatArray,
        displacement_field: itk.Image,
        size: Sequence[int],
        subsample_factor: int,
        magnitude_threshold: float,
    ) -> None:
        """Create flow line visualization by tracing streamlines through displacement
        field."""
        # Seed points - use a sparser grid for flowline seeds
        seed_step = 2  # Every other point in the already-subsampled grid
        flowline_count = 0

        for k in range(0, size[2], seed_step):
            for j in range(0, size[1], seed_step):
                for i in range(0, size[0], seed_step):
                    # Get displacement at seed point
                    displacement = displacement_array[k, j, i, :]
                    magnitude = np.linalg.norm(displacement)

                    # Skip if below threshold
                    if magnitude < magnitude_threshold:
                        continue

                    # Calculate index in original image space
                    index = [
                        i * subsample_factor,
                        j * subsample_factor,
                        k * subsample_factor,
                    ]

                    # Convert index to physical/world position using ITK
                    seed_pos = displacement_field.TransformIndexToPhysicalPoint(index)

                    # Trace streamline from this seed point
                    streamline_points = self._trace_streamline(
                        displacement_array,
                        displacement_field,
                        subsample_factor,
                        seed_pos,
                        max_steps=50,
                        step_size=0.5,
                    )

                    # Create USD curve for this streamline
                    if len(streamline_points) > 1:
                        curve_path = (
                            f"/DeformationVisualization/Flowlines/Line_{flowline_count}"
                        )
                        self._create_curve_prim(stage, curve_path, streamline_points)
                        flowline_count += 1

        self.log_info("  Created %d flowlines", flowline_count)

    def _trace_streamline(
        self,
        displacement_array: FloatArray,
        displacement_field: itk.Image,
        subsample_factor: int,
        seed_pos: Sequence[float],
        max_steps: int,
        step_size: float,
    ) -> list[NDArray[np.float64]]:
        """Trace a streamline through the displacement field using forward Euler
        integration."""
        points = [np.array(seed_pos, dtype=float)]
        current_pos = np.array(seed_pos, dtype=float)

        # Get original image size for bounds checking
        original_size = displacement_field.GetLargestPossibleRegion().GetSize()

        for _ in range(max_steps):
            # Convert world position to array indices using ITK
            index = displacement_field.TransformPhysicalPointToIndex(tuple(current_pos))

            # Convert to subsampled array indices
            subsampled_indices = [
                index[0] // subsample_factor,
                index[1] // subsample_factor,
                index[2] // subsample_factor,
            ]

            # Get subsampled array size
            subsampled_size = [
                original_size[0] // subsample_factor,
                original_size[1] // subsample_factor,
                original_size[2] // subsample_factor,
            ]

            # Check bounds in subsampled array (array is [k, j, i, :])
            if (
                subsampled_indices[0] < 0
                or subsampled_indices[0] >= subsampled_size[0]
                or subsampled_indices[1] < 0
                or subsampled_indices[1] >= subsampled_size[1]
                or subsampled_indices[2] < 0
                or subsampled_indices[2] >= subsampled_size[2]
            ):
                break

            # Get displacement at current position (array is [k, j, i, :])
            displacement = displacement_array[
                subsampled_indices[2], subsampled_indices[1], subsampled_indices[0], :
            ]
            magnitude = np.linalg.norm(displacement)

            # Stop if displacement is too small
            if magnitude < 0.1:
                break

            # Normalize displacement for integration
            direction = displacement / (magnitude + 1e-10)

            # Take step along direction
            current_pos = current_pos + direction * step_size
            points.append(current_pos.copy())

        return points

    def _create_curve_prim(
        self, stage: Usd.Stage, prim_path: str, points: Sequence[NDArray[np.float64]]
    ) -> UsdGeom.BasisCurves:
        """Create a USD BasisCurves primitive for a streamline."""
        curve = UsdGeom.BasisCurves.Define(stage, prim_path)

        # Set curve properties
        curve.GetTypeAttr().Set(UsdGeom.Tokens.linear)  # Linear segments between points
        curve.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)  # Open curve

        # Set points
        points_array = [Gf.Vec3f(*p) for p in points]
        curve.GetPointsAttr().Set(points_array)

        # Set curve vertex counts (one curve with N points)
        curve.GetCurveVertexCountsAttr().Set([len(points)])

        # Set width for visibility
        curve.GetWidthsAttr().Set([0.5])

        # Set color (cyan for flowlines)
        curve.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 1.0)])

        return curve
