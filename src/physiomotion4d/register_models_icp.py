"""ICP-based model-to-model registration for anatomical models.

This module provides the RegisterModelsICP class for aligning anatomical
models using Iterative Closest Point (ICP) algorithm. The workflow includes:
1. Initial centroid alignment
2. Rigid or affine ICP alignment

The registration is particularly useful for initial rough alignment of generic
models to patient-specific anatomical data.

Key Features:
    - Centroid-based initial alignment
    - VTK ICP algorithm with rigid or affine transformation modes
    - Three-stage affine pipeline: centroid → rigid ICP → affine ICP
    - Support for PyVista models
    - Automatic transform composition

Example:
    >>> import pyvista as pv
    >>> from physiomotion4d import RegisterModelsICP
    >>>
    >>> # Load models
    >>> moving_model = pv.read('generic_model.vtu')
    >>> fixed_model = pv.read('patient_surface.stl')
    >>>
    >>> # Run affine registration
    >>> registrar = RegisterModelsICP(fixed_model=fixed_model)
    >>> result = registrar.register(
    ...     transform_type='Affine',
    ...     moving_model=moving_model,
    ...     max_iterations=200,
    ... )
    >>>
    >>> # Access results
    >>> aligned_model = result['registered_model']
    >>> forward_point_transform = result['forward_point_transform']  # Moving to fixed
        # transform
"""

import logging
from typing import Optional

import itk
import numpy as np
import pyvista as pv
import vtk

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.transform_tools import TransformTools


class RegisterModelsICP(PhysioMotion4DBase):
    """Register anatomical models using Iterative Closest Point (ICP) algorithm.

    This class provides ICP-based alignment of 3D surface models with support for
    both rigid and affine transformation modes. The registration pipeline uses
    centroid alignment for initialization followed by VTK's ICP algorithm.

    **Registration Pipelines:**
        - **Rigid transform type**: Centroid alignment → Rigid ICP
        - **Affine transform type**: Centroid alignment → Rigid ICP → Affine ICP

    **Transform Convention:**
        These are POINT transforms (applied with TransformPoint, e.g. via
        TransformTools.transform_pvcontour), so their orientation is opposite to
        the image-registration transforms (see
        docs/developer/transform_conventions):

        - forward_point_transform: maps moving points -> fixed points; use it to
          warp the moving model/landmarks onto the fixed model. This is the
          inverse of the transform that would warp the moving IMAGE onto the
          fixed grid.
        - inverse_point_transform: maps fixed points -> moving points.

    Attributes:
        moving_model (pv.PolyData): Surface model to be aligned
        fixed_model (pv.PolyData): Target surface model
        transform_tools (TransformTools): Transform utility instance
        forward_point_transform (itk.AffineTransform): Optimized moving→fixed transform
        inverse_point_transform (itk.AffineTransform): Optimized fixed→moving transform
        registered_model (pv.PolyData): Aligned moving model

    Example:
        >>> # Initialize with model
        >>> registrar = RegisterModelsICP(fixed_model=patient_surface)
        >>>
        >>> # Run rigid registration
        >>> result = registrar.register(
        ...     transform_type='Rigid',
        ...     max_iterations=200,
        ...     moving_model=model_surface,
        ... )
        >>>
        >>> # Or run affine registration
        >>> result = registrar.register(
        ...     transform_type='Affine',
        ...     max_iterations=200,
        ...     moving_model=model_surface,
        ... )
        >>>
        >>> # Get aligned model and transforms
        >>> aligned_model = result['registered_model']
        >>> forward_point_transform = result['forward_point_transform']
    """

    def __init__(
        self,
        fixed_model: pv.PolyData,
        log_level: int | str = logging.INFO,
    ):
        """Initialize ICP-based model registration.

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            fixed_model: PyVista target surface model
            log_level: Logging level (default: logging.INFO)

        Note:
            The moving_model is typically extracted from a VTU model using
            model.extract_surface(algorithm="dataset_surface") before passing to this class.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.moving_model: Optional[pv.PolyData] = None
        self.fixed_model = fixed_model
        self.transform_type = "Affine"

        # Transform utilities
        self.transform_tools = TransformTools()

        # Registration results
        self.forward_point_transform: Optional[itk.AffineTransform] = None
        self.inverse_point_transform: Optional[itk.AffineTransform] = None
        self.registered_model: Optional[pv.PolyData] = None

    def register(
        self,
        moving_model: pv.PolyData,
        transform_type: str = "Affine",
        max_iterations: int = 2000,
    ) -> dict:
        """Perform ICP alignment of moving model to fixed model.

        This method executes alignment with either rigid or affine transformations:

        **Rigid transform type:**
            1. Centroid alignment: Translate moving model to align mass centers
            2. Rigid ICP: Refine with rigid-body transformation (rotation + translation)

        **Affine transform type:**
            1. Centroid alignment: Translate moving model to align mass centers
            2. Rigid ICP: Refine with rigid-body transformation
            3. Affine ICP: Further refine with affine transformation (includes
                scaling/shearing)

        Args:
            moving_model: PyVista surface model to be aligned to fixed model
            transform_type: Registration transform type, either 'Rigid' or 'Affine'.
                Default: 'Affine'
            max_iterations: Maximum number of ICP iterations per stage. Default: 2000

        Returns:
            Dictionary containing:
                - 'registered_model': Aligned moving model (PyVista PolyData)
                - 'forward_point_transform': Moving→fixed transform
                    (ITK AffineTransform)
                - 'inverse_point_transform': Fixed→moving transform
                    (ITK AffineTransform)

        Raises:
            ValueError: If transform_type is not 'Rigid' or 'Affine'

        Example:
            >>> # Rigid registration
            >>> result = registrar.register(
            ...     transform_type='Rigid',
            ...     max_iterations=5000,
            ...     moving_model=moving_model,
            ... )
            >>>
            >>> # Affine registration
            >>> result = registrar.register(
            ...     transform_type='Affine',
            ...     max_iterations=2000,
            ...     moving_model=moving_model,
            ... )
        """
        if transform_type not in ["Rigid", "Affine"]:
            raise ValueError(
                f"Invalid transform '{transform_type}'. Must be 'Rigid' or 'Affine'."
            )

        self.log_section("%s ICP Alignment", transform_type.upper())

        self.moving_model = moving_model
        self.transform_type = transform_type

        # Step 1: Centroid alignment (common to both modes)
        self.registered_model = self.moving_model.copy(deep=True)

        moving_centroid = np.array(self.registered_model.center)
        self.log_debug("Moving model centroid: %s", moving_centroid)
        fixed_centroid = np.array(self.fixed_model.center)
        self.log_debug("Fixed model centroid: %s", fixed_centroid)
        translation = fixed_centroid - moving_centroid
        self.log_info("Step 1: Translating by %s to align centroids...", translation)

        # Create ITK affine transform with translation
        forward_point_transform = itk.AffineTransform[itk.D, 3].New()
        forward_point_transform.SetIdentity()
        forward_point_transform.SetOffset(translation)

        # Apply centroid alignment to model
        self.registered_model = self.transform_tools.transform_pvcontour(
            self.registered_model,
            forward_point_transform,
            with_deformation_magnitude=False,
        )

        self.log_debug("Center after Step 1: %s", self.registered_model.center)

        # Step 2: Rigid ICP (common to both modes)
        self.log_info(
            "Step 2: Performing rigid ICP (max iterations: %d)...", max_iterations
        )
        icp_rigid = vtk.vtkIterativeClosestPointTransform()
        icp_rigid.SetSource(self.registered_model)
        icp_rigid.SetTarget(self.fixed_model)
        icp_rigid.GetLandmarkTransform().SetModeToRigidBody()  # Rigid mode
        icp_rigid.SetMaximumNumberOfIterations(max_iterations)
        icp_rigid.Update()

        # Convert VTK transform to ITK and compose with centroid transform
        rigid_transform = self.transform_tools.convert_vtk_matrix_to_itk_transform(
            icp_rigid.GetMatrix()
        )
        forward_point_transform.Compose(rigid_transform)

        # Apply rigid ICP transform to model
        self.registered_model = self.transform_tools.transform_pvcontour(
            self.registered_model,
            rigid_transform,
            with_deformation_magnitude=False,
        )

        self.log_debug("Center after Step 2: %s", self.registered_model.center)

        # Step 3: Affine ICP (only if affine mode)
        if transform_type == "Affine":
            self.log_info(
                "Step 3: Performing affine ICP (max iterations: %d)...", max_iterations
            )
            icp_affine = vtk.vtkIterativeClosestPointTransform()
            icp_affine.SetSource(self.registered_model)
            icp_affine.SetTarget(self.fixed_model)
            icp_affine.GetLandmarkTransform().SetModeToAffine()  # Affine mode
            icp_affine.SetMaximumNumberOfIterations(max_iterations)
            icp_affine.Update()

            # Convert VTK transform to ITK and compose
            affine_transform = self.transform_tools.convert_vtk_matrix_to_itk_transform(
                icp_affine.GetMatrix()
            )
            forward_point_transform.Compose(affine_transform)

            # Apply affine ICP transform to model
            self.registered_model = self.transform_tools.transform_pvcontour(
                self.registered_model,
                affine_transform,
                with_deformation_magnitude=False,
            )

            self.log_debug("Center after Step 3: %s", self.registered_model.center)

        # Compute inverse transform
        # Ths forward transform for ICP is consistent with the transform convention
        # used with images-to-images registration.
        self.forward_point_transform = forward_point_transform
        self.inverse_point_transform = forward_point_transform.GetInverseTransform()

        self.log_info("%s ICP registration complete!", transform_type.upper())

        # Return results as dictionary
        return {
            "registered_model": self.registered_model,
            "forward_point_transform": self.forward_point_transform,
            "inverse_point_transform": self.inverse_point_transform,
        }
