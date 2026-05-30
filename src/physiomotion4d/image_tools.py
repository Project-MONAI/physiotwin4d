"""
Image Tools for PhysioMotion4D

This module provides utilities for converting between different medical image formats
and performing image processing operations.
"""

import logging
from typing import Any, Optional

import itk
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase


class ImageTools(PhysioMotion4DBase):
    """
    Utilities for medical image format conversions and processing.

    This class provides methods for converting between ITK (Insight Toolkit) and
    SimpleITK image formats while preserving all metadata (origin, spacing, direction,
    pixel type). Supports both scalar and vector (multi-component) images.

    Example:
        >>> tools = ImageTools()
        >>> # Convert ITK to SimpleITK
        >>> sitk_image = tools.convert_itk_image_to_sitk(itk_image)
        >>> # Convert back to ITK
        >>> itk_image_back = tools.convert_sitk_image_to_itk(sitk_image)
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize ImageTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def imreadVD3(self, filename: str) -> Any:
        """Read an ITK vector image with double precision vectors.

        ITK's imread is not wrapped for itk.Image[itk.Vector[itk.D,3],3],
        so this method reads as itk.Image[itk.Vector[itk.F,3],3] and converts
        to double precision.

        Args:
            filename (str): Path to the image file to read

        Returns:
            itk.Image[itk.Vector[itk.D,3],3]: Vector image with double precision

        Example:
            >>> displacement_field = ImageTools().imreadVD3('deformation.mha')
        """
        # Read as float precision vector image
        image = itk.imread(filename)
        if "VD" in str(type(image)):
            return image

        image_arr = itk.array_from_image(image)
        image_double = self.convert_array_to_image_of_vectors(image_arr, image, itk.D)

        return image_double

    def imwriteVD3(self, image: Any, filename: str, compression: bool = True) -> None:
        """Write an ITK vector image with double precision vectors.

        ITK's imwrite is not wrapped for itk.Image[itk.Vector[itk.D,3],3],
        so this method converts to itk.Image[itk.Vector[itk.F,3],3] and writes.

        Args:
            image (itk.Image[itk.Vector[itk.D,3],3]): Vector image to write
            filename (str): Path to the output file
            compression (bool): Whether to use compression (default: True)

        Example:
            >>> ImageTools().imwriteVD3(displacement_field, 'deformation.mha')
        """
        # Convert to float precision for writing
        if "VD" not in str(type(image)):
            raise ValueError("Image must be a vector image with double precision")

        image_arr = itk.array_from_image(image)
        image_float = self.convert_array_to_image_of_vectors(image_arr, image, itk.F)

        # Write the float image
        itk.imwrite(image_float, filename, compression=compression)

    def convert_itk_image_to_sitk(self, itk_image: itk.Image) -> sitk.Image:
        """
        Convert an ITK image to a SimpleITK image.

        This method converts an ITK (Insight Toolkit) image to SimpleITK format while
        preserving all metadata including origin, spacing, direction, and pixel type.
        Works with both scalar and vector (multi-component) images.

        Args:
            itk_image: Input ITK image (can be scalar or vector image)

        Returns:
            SimpleITK image with identical data and metadata

        Note:
            Memory layout is preserved during conversion. Both ITK and SimpleITK
            use (z, y, x) ordering for 3D images in numpy arrays.

        Example:
            >>> tools = ImageTools()
            >>> itk_image = itk.imread('image.nii.gz')
            >>> sitk_image = tools.convert_itk_image_to_sitk(itk_image)
        """
        # Get numpy array from ITK image
        # ITK array is in (z, y, x) or (z, y, x, components) format
        array = itk.array_from_image(itk_image)

        # Get image metadata
        origin = itk.origin(itk_image)
        spacing = itk.spacing(itk_image)
        direction = itk.array_from_matrix(itk_image.GetDirection())

        # Check if this is a vector image
        is_vector = False
        if hasattr(itk_image, "GetNumberOfComponentsPerPixel"):
            n_components = itk_image.GetNumberOfComponentsPerPixel()
            is_vector = n_components > 1

        if is_vector:
            # For vector images, SimpleITK expects (z, y, x, components) which is what we have
            # Create SimpleITK image from numpy array
            sitk_image = sitk.GetImageFromArray(array, isVector=True)
        else:
            # For scalar images, array is (z, y, x)
            sitk_image = sitk.GetImageFromArray(array, isVector=False)

        # Set metadata
        # Convert origin and spacing to tuples (reverse order for SimpleITK: x, y, z)
        sitk_image.SetOrigin(tuple(origin))
        sitk_image.SetSpacing(tuple(spacing))

        # Direction matrix needs to be flattened
        # ITK and SimpleITK use the same direction convention, we just need to flatten it correctly
        direction_flat = direction.flatten()
        sitk_image.SetDirection(direction_flat.tolist())

        return sitk_image

    def convert_sitk_image_to_itk(self, sitk_image: sitk.Image) -> itk.Image:
        """
        Convert a SimpleITK image to an ITK image.

        This method converts a SimpleITK image to ITK (Insight Toolkit) format while
        preserving all metadata including origin, spacing, direction, and pixel type.
        Works with both scalar and vector (multi-component) images.

        Args:
            sitk_image: Input SimpleITK image (can be scalar or vector image)

        Returns:
            ITK image with identical data and metadata

        Note:
            Memory layout is preserved during conversion. Both SimpleITK and ITK
            use (z, y, x) ordering for 3D images in numpy arrays.

        Example:
            >>> tools = ImageTools()
            >>> sitk_image = sitk.ReadImage('image.nii.gz')
            >>> itk_image = tools.convert_sitk_image_to_itk(sitk_image)
        """
        # Get numpy array from SimpleITK image
        # SimpleITK array is in (z, y, x) or (z, y, x, components) format
        array = sitk.GetArrayFromImage(sitk_image)

        # Get image metadata
        origin = sitk_image.GetOrigin()  # Returns (x, y, z)
        spacing = sitk_image.GetSpacing()  # Returns (x, y, z)
        direction = sitk_image.GetDirection()  # Returns flattened direction matrix
        dimension = sitk_image.GetDimension()
        n_components = sitk_image.GetNumberOfComponentsPerPixel()

        # Check if this is a vector image
        is_vector = n_components > 1

        # Create ITK image from numpy array
        if is_vector:
            # Vector image
            itk_image = itk.image_from_array(array, is_vector=True)
        else:
            # Scalar image
            itk_image = itk.image_from_array(array, is_vector=False)

        # Set origin (reverse order: SimpleITK gives x,y,z, ITK expects x,y,z internally
        # but we set it using the same order)
        itk_image.SetOrigin(origin)

        # Set spacing
        itk_image.SetSpacing(spacing)

        # Set direction matrix
        # Reshape direction to matrix form
        if dimension == 2:
            direction_matrix = np.array(direction).reshape(2, 2)
        elif dimension == 3:
            direction_matrix = np.array(direction).reshape(3, 3)
        else:
            raise ValueError(f"Unsupported image dimension: {dimension}")

        # Convert numpy array to ITK matrix and set
        itk_direction = itk.matrix_from_array(direction_matrix)
        itk_image.SetDirection(itk_direction)

        return itk_image

    def convert_array_to_image_of_vectors(
        self,
        arr_data: NDArray[Any],
        reference_image: Any,
        ptype: Any = itk.D,
    ) -> Any:
        """
        Convert a numpy array to an ITK image of vector type.

        This method is needed because itk in python does not support creating
        images of vectors with itk.D precision.   Luckily array_view_from_image
        does support itk.D precision vectors.
        """
        if ptype not in [itk.F, itk.D]:
            if ptype == np.float32:
                ptype = itk.F
            elif ptype == np.float64:
                ptype = itk.D
            else:
                raise ValueError(f"Unsupported component type: {ptype}")

        itk_image = itk.Image[itk.Vector[ptype, 3], 3].New()
        itk_image.SetRegions(reference_image.GetLargestPossibleRegion())
        itk_image.SetSpacing(reference_image.GetSpacing())
        itk_image.SetOrigin(reference_image.GetOrigin())
        itk_image.SetDirection(reference_image.GetDirection())
        itk_image.Allocate()
        itk.array_view_from_image(itk_image)[:] = arr_data

        return itk_image

    def flip_image(
        self,
        in_image: itk.Image,
        in_mask: Optional[itk.Image] = None,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        flip_and_make_identity: bool = False,
    ) -> Any | tuple[Any, Any]:
        """
        Flip the image and mask.

        Only axis-aligned flips are supported. If ``flip_and_make_identity`` is
        True, the image and mask are first flipped along any axes whose
        corresponding diagonal entries in the direction matrix are negative
        (assuming the direction matrix encodes only axis-aligned flips), then
        any additional requested flips are performed, and finally the direction
        matrix is set to the identity matrix. This is useful when combining ITK
        images with VTK objects (that often do not support a direction matrix).

        Args:
            in_image: The input image to flip
            in_mask: The input mask to flip
            flip_x: Flip the image and mask along the x-axis
            flip_y: Flip the image and mask along the y-axis
            flip_z: Flip the image and mask along the z-axis
            flip_and_make_identity: Flip the image and mask and make the direction
                matrix identity.
        """
        flip0 = False
        flip1 = False
        flip2 = False
        if flip_and_make_identity:
            # itk.array_from_matrix avoids itk.Matrix.__array__, whose missing
            # copy keyword triggers a numpy>=2.0 DeprecationWarning.
            direction = itk.array_from_matrix(in_image.GetDirection())
            flip0 = direction[0, 0] < 0
            flip1 = direction[1, 1] < 0
            flip2 = direction[2, 2] < 0
        if flip_x:
            flip0 = True
        if flip_y:
            flip1 = True
        if flip_z:
            flip2 = True
        if flip0 or flip1 or flip2:
            self.log_info(f"Flipping image: {flip0}, {flip1}, {flip2}")
            flip_filter = itk.FlipImageFilter.New(Input=in_image)
            flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
            flip_filter.SetFlipAboutOrigin(True)
            flip_filter.Update()
            out_image = flip_filter.GetOutput()
            if flip_and_make_identity:
                id_mat = itk.Matrix[itk.D, 3, 3]()
                id_mat.SetIdentity()
                out_image.SetDirection(id_mat)
            if in_mask is not None:
                flip_filter = itk.FlipImageFilter.New(Input=in_mask)
                flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
                flip_filter.SetFlipAboutOrigin(True)
                flip_filter.Update()
                out_mask = flip_filter.GetOutput()
                if flip_and_make_identity:
                    id_mat = itk.Matrix[itk.D, 3, 3]()
                    id_mat.SetIdentity()
                    out_mask.SetDirection(id_mat)
                return out_image, out_mask
            else:
                return out_image
        else:
            if in_mask is not None:
                return in_image, in_mask
            else:
                return in_image
