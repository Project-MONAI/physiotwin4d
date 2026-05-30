from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import itk
import numpy as np
import pyvista as pv
from scipy.optimize import minimize
from typing_extensions import Self

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.transform_tools import TransformTools


class RegisterModelsPCA(PhysioMotion4DBase):
    """Register PCA-based shape models to medical images using mean distance optimization.

    This class implements a registration pipeline for fitting statistical
    shape models to patient-specific medical images:

    **PCA Deformable Registration**
        - Optimizes PCA coefficients
        - Model equation: P = mean + Σ(b_i * std_i * pca_eigenvector_i)
        - Maximizes mean distance at deformed model points P

    **Optimization Objective:**
        Maximize the mean distance of the image sampled at model points using
        ITK's LinearInterpolateImageFunction. This aligns the model with bright
        regions in contrast-enhanced images (e.g., blood pool in cardiac CT).

    Attributes:
        pca_template_model (pv.DataSet): Mean shape model
        pca_eigenvectors (np.ndarray): PCA eigenvectors/components (modes × n_points*3)
        pca_std_deviations (np.ndarray): Standard deviations per mode (modes,)
        fixed_distance_map (itk.Image): Patient image providing distance data
        n_points (int): Number of points in the model
        pca_number_of_modes (int): Number of PCA modes available
        pca_coefficients (np.ndarray): Optimized PCA coefficients
        registered_model (pv.DataSet): Final registered and deformed model
        post_pca_transform (itk.Transform): Transform to apply after PCA registration
        forward_point_transform (itk.DisplacementFieldTransform): POINT transform
            mapping template points -> registered/target points; use it to warp
            the template model/landmarks onto the target. Its orientation is
            opposite to an image-registration forward_transform (see
            docs/developer/transform_conventions). Does not include the post-PCA
            transform.
        inverse_point_transform (itk.DisplacementFieldTransform): POINT transform
            mapping target points -> template points. Does not include the
            post-PCA transform.

    Example:
        >>> # Load PCA model data
        >>> pca_template_model = pv.read('pca_All_mean.vtk')
        >>> with open('pca.json', 'r') as f:
        ...     pca_data = json.load(f)
        >>> pca_group_data = pca_data['All']
        >>> pca_std_deviations = np.sqrt(np.array(pca_group_data['eigenvalues']))
        >>> pca_eigenvectors = np.array(pca_group_data['components'])
        >>>
        >>> # Initialize registrar with loaded data
        >>> registrar = RegisterModelsPCA(
        ...     pca_template_model=pca_template_model,
        ...     pca_eigenvectors=pca_eigenvectors,
        ...     pca_std_deviations=pca_std_deviations,
        ... )
        >>>
        >>> # Run full registration pipeline
        >>> result = registrar.register(pca_number_of_modes=10)
        >>>
        >>> # Save registered model
        >>> result['registered_model'].save('registered_heart.vtk')
        >>>
        >>> # Print optimization results
        >>> print(f'Final mean distance: {result["mean_distance"]:.2f}')
        >>> print(f'PCA coefficients: {result["pca_coefficients"]}')
    """

    def __init__(
        self,
        pca_template_model: pv.DataSet,
        pca_eigenvectors: np.ndarray,
        pca_std_deviations: np.ndarray,
        pca_number_of_modes: int = 0,
        pca_template_model_point_subsample: int = 4,
        post_pca_transform: Optional[itk.Transform] = None,
        fixed_distance_map: Optional[itk.Image] = None,
        fixed_model: Optional[pv.DataSet] = None,
        reference_image: Optional[itk.Image] = None,
        log_level: int | str = logging.INFO,
    ):
        """Initialize the PCA-based model-to-image registration.

        Args:
            pca_template_model: PyVista model containing the mean 3D shape model
                (unstructured grid or polydata)
            pca_eigenvectors: Numpy array of PCA eigenvectors/components. Shape: (modes, n_points*3)
                Each row is a flattened eigenmode with 3D displacements: [x1,y1,z1, x2,y2,z2, ...]
            pca_std_deviations: Numpy array of standard deviations per PCA mode. Shape: (modes,)
                These are the square roots of pca_eigenvalues
            pca_number_of_modes: Number of PCA modes to use. Default: -1 (use all)
            pca_template_model_point_subsample: Step size for subsampling model points. Default: 4
            post_pca_transform: Optional ITK transform to apply after PCA registration.
                Default: None
            fixed_distance_map: ITK image providing the distance map.
                Default: None
            fixed_model: PyVista model used to compute the distance map, if one isn't provided.
            reference_image: ITK image providing coordinate frame for computing the distance map.
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Raises:
            ValueError: If pca_eigenvector dimensions don't match model points
        """
        # Initialize base class with logging
        super().__init__(class_name="RegisterModelsPCA", log_level=log_level)

        # Store model data
        self.pca_template_model: pv.DataSet = pca_template_model
        self.pca_eigenvectors: np.ndarray = pca_eigenvectors
        self.pca_std_deviations: np.ndarray = pca_std_deviations

        self.post_pca_transform = post_pca_transform

        self._contour_tools = ContourTools()

        self.fixed_distance_map = fixed_distance_map
        if (
            self.fixed_distance_map is None
            and fixed_model is not None
            and reference_image is not None
        ):
            self.fixed_model = fixed_model
            self.fixed_distance_map = self._contour_tools.create_distance_map(
                fixed_model,
                reference_image,
                squared_distance=True,
                negative_inside=False,
                zero_inside=True,
                norm_to_max_distance=200.0,
            )
        elif self.fixed_distance_map is not None and (
            fixed_model is not None or reference_image is not None
        ):
            self.log_warning(
                "Fixed model and reference image will be ignored because a distance map is provided."
            )
        elif self.fixed_distance_map is None and (
            fixed_model is None or reference_image is None
        ):
            self.log_error(
                "Fixed model and reference image must be provided if no distance map is provided."
            )
            raise ValueError(
                "Fixed model and reference image must be provided if no distance map is provided."
            )

        self.pca_number_of_modes: int = pca_number_of_modes
        if self.pca_number_of_modes <= 0:
            self.pca_number_of_modes = len(pca_std_deviations)

        self.pca_template_model_point_subsample = pca_template_model_point_subsample

        # outputs
        self.registered_model_pca_coefficients: Optional[np.ndarray] = None
        self.registered_model: Optional[pv.DataSet] = None
        self.registered_model_mean_distance: float = 0.0
        self.registered_model_pca_deformation: Optional[np.ndarray] = None
        self.forward_point_transform: Optional[itk.DisplacementFieldTransform] = None
        self.inverse_point_transform: Optional[itk.DisplacementFieldTransform] = None

        # Image interpolator (created when needed)
        self._fixed_distance_map_interpolator: Optional[
            itk.LinearInterpolateImageFunction
        ] = None
        self._fixed_distance_map_max_distance: float = 0.0

        self._metric_call_count: int = 0

        # Pre-convert mean shape points to ITK format
        self._pca_template_model_points_itk: Optional[list[itk.Point]] = None
        self._create_itk_points()

    @classmethod
    def from_json(
        cls,
        pca_template_model: pv.DataSet,
        pca_json_filename: str,
        pca_number_of_modes: int = 0,
        pca_template_model_point_subsample: int = 4,
        post_pca_transform: Optional[itk.Transform] = None,
        fixed_distance_map: Optional[itk.Image] = None,
        fixed_model: Optional[pv.DataSet] = None,
        reference_image: Optional[itk.Image] = None,
        log_level: int | str = logging.INFO,
    ) -> Self:
        """Create RegisterModelsPCA from PCA model JSON file.

        This method reads PCA statistical shape model data from a JSON file
        containing eigenvalues and principal component vectors.

        The JSON file must contain:
        - 'eigenvalues': Array of eigenvalues (variance) for each component
        - 'components': Array of principal component vectors (flattened shape deformations)

        Args:
            pca_template_model: Mean surface mesh to use as template
            pca_json_filename: Path to the PCA model JSON file
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all)
            pca_template_model_point_subsample: Step size for subsampling model points. Default: 4
            post_pca_transform: Optional ITK transform to apply after PCA registration.
                Default: None
            fixed_distance_map: ITK image providing the distance values
                for registration. If None, must be set later before registration.
            fixed_model: Target surface mesh to register to. Default: None
            reference_image: Reference image defining coordinate space. Default: None
            log_level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING).
                Default: logging.INFO

        Returns:
            RegisterModelsPCA instance

        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If data format is invalid or required fields are missing

        Example:
            >>> registrar = RegisterModelsPCA.from_json(
            ...     pca_template_model=pca_template_model,
            ...     pca_json_filename='path/to/pca_model.json',
            ...     fixed_model=fixed_model,
            ...     reference_image=reference_image,
            ... )
        """
        # Create a logger for the classmethod since superclass hasn't
        # been initialized yet.
        logger = logging.getLogger("PhysioMotion4D")

        json_path = Path(pca_json_filename)

        # Check if JSON file exists
        if not json_path.exists():
            logger.error(f"PCA JSON file not found: {pca_json_filename}")
            raise FileNotFoundError(f"PCA JSON file not found: {pca_json_filename}")

        logger.info("Loading PCA model data...")
        logger.info(f"  JSON file: {json_path}")

        # Load PCA data from JSON
        logger.info("Reading JSON file...")
        with open(json_path, encoding="utf-8") as f:
            pca_data = json.load(f)

        # Extract eigenvalues and convert to standard deviations
        if "eigenvalues" not in pca_data:
            raise ValueError("'eigenvalues' field not found in JSON data")
        pca_std_deviations = np.sqrt(np.array(pca_data["eigenvalues"]))
        logger.info("  Loaded %d eigenvalues", len(pca_std_deviations))

        # Extract principal component vectors
        if "components" not in pca_data:
            raise ValueError("'components' field not found in JSON data")
        pca_eigenvectors = np.array(pca_data["components"], dtype=np.float64)
        logger.info(f"  Loaded components with shape {pca_eigenvectors.shape}")

        # Validate dimensions
        expected_pca_eigenvector_size = pca_template_model.n_points * 3
        actual_pca_eigenvector_size = pca_eigenvectors.shape[1]
        if actual_pca_eigenvector_size != expected_pca_eigenvector_size:
            raise ValueError(
                f"Component dimension mismatch: "
                f"Expected {expected_pca_eigenvector_size} (3 × {pca_template_model.n_points} model points), "
                f"got {actual_pca_eigenvector_size}"
            )

        logger.info("  ✓ Data validation successful!")
        logger.info("PCA model data loaded successfully!")

        return cls.from_pca_model(
            pca_template_model=pca_template_model,
            pca_model=pca_data,
            pca_number_of_modes=pca_number_of_modes,
            pca_template_model_point_subsample=pca_template_model_point_subsample,
            post_pca_transform=post_pca_transform,
            fixed_distance_map=fixed_distance_map,
            fixed_model=fixed_model,
            reference_image=reference_image,
            log_level=log_level,
        )

    @classmethod
    def from_pca_model(
        cls,
        pca_template_model: pv.DataSet,
        pca_model: dict,
        pca_number_of_modes: int = 0,
        pca_template_model_point_subsample: int = 4,
        post_pca_transform: Optional[itk.Transform] = None,
        fixed_distance_map: Optional[itk.Image] = None,
        fixed_model: Optional[pv.DataSet] = None,
        reference_image: Optional[itk.Image] = None,
        log_level: int | str = logging.INFO,
    ) -> Self:
        """Create RegisterModelsPCA from a PCA model dictionary.

        The dict must match the structure produced by
        :class:`WorkflowCreateStatisticalModel` (key ``pca_model``):
        ``explained_variance_ratio``, ``eigenvalues``, ``components``.

        Args:
            pca_template_model: Mean surface mesh to use as template
            pca_model: PCA model dict with 'eigenvalues' and 'components' (and optionally
                'explained_variance_ratio')
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all)
            pca_template_model_point_subsample: Step size for subsampling model points. Default: 4
            post_pca_transform: Optional ITK transform to apply after PCA registration.
            fixed_distance_map: ITK image providing the distance values for registration.
            fixed_model: Target surface mesh to register to.
            reference_image: Reference image defining coordinate space.
            log_level: Logging level.

        Returns:
            RegisterModelsPCA instance

        Raises:
            ValueError: If required keys are missing or dimensions invalid
        """
        if "eigenvalues" not in pca_model:
            raise ValueError("'eigenvalues' field not found in pca_model")
        pca_std_deviations = np.sqrt(np.array(pca_model["eigenvalues"]))
        if "components" not in pca_model:
            raise ValueError("'components' field not found in pca_model")
        pca_eigenvectors = np.array(pca_model["components"], dtype=np.float64)
        expected_size = pca_template_model.n_points * 3
        if pca_eigenvectors.shape[1] != expected_size:
            raise ValueError(
                f"Component dimension mismatch: expected {expected_size} "
                f"(3 × {pca_template_model.n_points} points), got {pca_eigenvectors.shape[1]}"
            )
        return cls(
            pca_template_model=pca_template_model,
            pca_eigenvectors=pca_eigenvectors,
            pca_std_deviations=pca_std_deviations,
            pca_number_of_modes=pca_number_of_modes,
            pca_template_model_point_subsample=pca_template_model_point_subsample,
            post_pca_transform=post_pca_transform,
            fixed_distance_map=fixed_distance_map,
            fixed_model=fixed_model,
            reference_image=reference_image,
            log_level=log_level,
        )

    def _create_itk_points(self) -> None:
        """Pre-convert mean shape points to ITK Point format for efficiency.

        This method creates ITK Point objects once at initialization, avoiding
        repeated conversions during optimization iterations.
        """
        self.log_info("Converting mean shape points to ITK format...")

        self._pca_template_model_points_itk = []
        for point in self.pca_template_model.points:
            itk_point = itk.Point[itk.D, 3]()
            itk_point[0] = float(point[0])
            itk_point[1] = float(point[1])
            itk_point[2] = float(point[2])
            self._pca_template_model_points_itk.append(itk_point)

        self.log_info(
            f"  Converted {len(self._pca_template_model_points_itk)} points to ITK format"
        )

    def set_fixed_model(
        self, fixed_model: pv.UnstructuredGrid, reference_image: Optional[itk.Image]
    ) -> None:
        """Set the fixed model for registration.

        If this is set, the fixed distance map will be set to None.

        Args:
            fixed_model: PyVista model used to compute the distance map, if one isn't provided.
            reference_image: ITK image providing coordinate frame for computing the distance map.
        """
        if reference_image is None:
            raise ValueError(
                "reference_image must not be None when setting a fixed model"
            )

        self.fixed_distance_map = self._contour_tools.create_distance_map(
            fixed_model,
            reference_image,
            squared_distance=True,
            negative_inside=False,
            zero_inside=True,
            norm_to_max_distance=200.0,
        )
        self._fixed_distance_map_interpolator = None

    def set_fixed_distance_map(self, fixed_distance_map: Optional[itk.Image]) -> None:
        """Set the reference image for registration.

        If this is set, the fixed model will be set to None.

        Args:
            fixed_distance_map: ITK image providing distance data
        """
        self.fixed_distance_map = fixed_distance_map
        self._fixed_distance_map_interpolator = None

    def set_pca_template_model(self, pca_template_model: pv.UnstructuredGrid) -> None:
        """Set the average model for registration.

        Args:
            pca_template_model: PyVista model containing the mean 3D shape model
                (unstructured grid or polydata)
        """
        self.pca_template_model = pca_template_model

        self._pca_template_model_points_itk = None

        self._create_itk_points()
        self.log_info("  ✓ Average model set successfully!")

    def _mean_distance_metric(
        self,
        params: np.ndarray,
    ) -> float:
        """Evaluate the optimization metric (mean intensity) at model points.

        This is the objective function to be MAXIMIZED during optimization.
        Higher values indicate better alignment with bright regions.

        Args:
            pca_deformation: Nx3 numpy array of PCA deformation vectors to add to points.
                If None, no deformation is applied.

        Returns:
            Mean distance value across all points
        """
        pca_deformation = self._compute_pca_deformation(params)

        # Create interpolator if not already cached (inline creation)
        if self._fixed_distance_map_interpolator is None:
            if self.fixed_distance_map is None:
                self.log_error("Distance map is not set.")
                raise ValueError("Distance map must be set before registering.")
            ImageType = type(self.fixed_distance_map)
            self._fixed_distance_map_interpolator = itk.LinearInterpolateImageFunction[
                ImageType, itk.D
            ].New()
            self._fixed_distance_map_interpolator.SetInputImage(self.fixed_distance_map)
            fixed_distance_map_array = itk.GetArrayFromImage(self.fixed_distance_map)
            self._fixed_distance_map_max_distance = fixed_distance_map_array.max()
            self.log_debug("Interpolator created")
            self.log_debug(
                "   Max distance = %s", self._fixed_distance_map_max_distance
            )

        self.log_debug("Evaluating params = %s", params)
        self.log_debug("   Max displacement = %s", pca_deformation.max(axis=0))

        # Sample distance at each point
        n_valid_points = 0
        total_distance = 0.0
        center = np.zeros(3)
        point = itk.Point[itk.D, 3]()
        assert self.fixed_distance_map is not None, "fixed_distance_map must be set"
        assert self._pca_template_model_points_itk is not None, (
            "ITK points must be initialized"
        )
        image_size = self.fixed_distance_map.GetBufferedRegion().GetSize()
        for i, base_point in enumerate(self._pca_template_model_points_itk):
            if i % self.pca_template_model_point_subsample != 0:
                continue

            # Start with base point
            point[0] = base_point[0]
            point[1] = base_point[1]
            point[2] = base_point[2]

            # Add PCA deformation if provided
            point[0] += pca_deformation[i, 0]
            point[1] += pca_deformation[i, 1]
            point[2] += pca_deformation[i, 2]

            if self.post_pca_transform is not None:
                point = self.post_pca_transform.TransformPoint(point)

            # Check if point is inside image bounds

            coord_index = (
                self.fixed_distance_map.TransformPhysicalPointToContinuousIndex(point)
            )
            if (
                0 <= coord_index[0] < image_size[0]
                and 0 <= coord_index[1] < image_size[1]
                and 0 <= coord_index[2] < image_size[2]
            ):
                center[0] += point[0]
                center[1] += point[1]
                center[2] += point[2]
                distance = (
                    self._fixed_distance_map_interpolator.EvaluateAtContinuousIndex(
                        coord_index
                    )
                )
                total_distance += distance
                n_valid_points += 1
            else:
                self.log_warning("   Point %d is outside image bounds (%s)", i, point)
                return self._fixed_distance_map_max_distance

        # Compute mean distance
        mean_distance = total_distance / n_valid_points
        center /= n_valid_points

        log_level_int = (
            self.log_level
            if isinstance(self.log_level, int)
            else logging.getLevelName(self.log_level)
        )
        if log_level_int <= logging.DEBUG or self._metric_call_count % 100 == 0:
            self.log_info(
                "   Metric %d: %s -> %f",
                (self._metric_call_count + 1),
                center,
                mean_distance,
            )
            self.log_info(
                "       Params %s",
                params,
            )
        self._metric_call_count += 1

        return mean_distance

    def _compute_pca_deformation(self, pca_coefficients: np.ndarray) -> np.ndarray:
        """Compute PCA deformation vectors for all points.

        Deformation is computed as:
            displacement = Σ(b_i * std_i * pca_eigenvector_i)

        Args:
            pca_coefficients: Array of PCA coefficients b_i (one per mode)
            pca_number_of_modes: Number of PCA modes to use. Default: use all available modes

        Returns:
            Nx3 array of deformation vectors (displacement from mean shape)
        """
        # Initialize deformation to zero
        deformation = np.zeros((self.pca_template_model.n_points, 3), dtype=np.float64)

        # Add contribution from each PCA mode
        for i in range(self.pca_number_of_modes):
            pca_eigenvector_flat = self.pca_eigenvectors[i, :]

            # Reshape to (N, 3)
            pca_eigenvector_3d = pca_eigenvector_flat.reshape(-1, 3)

            # Add weighted deformation: b_i * std_i * pca_eigenvector_i
            deformation += (
                pca_coefficients[i] * self.pca_std_deviations[i] * pca_eigenvector_3d
            )

        return deformation

    def _optimize_pca_coefficients(
        self,
        pca_number_of_modes: int = 0,
        pca_coefficient_bounds: float = 3.0,
        method: str = "L-BFGS-B",
        max_iterations: int = 50,
    ) -> tuple[np.ndarray, float]:
        """Optimize PCA coefficients

        This method optimizes PCA mode coefficients to deform the model to better match
            low values in the distance map.

        Args:
            pca_number_of_modes: Number of PCA modes to use in optimization. Using fewer
                modes provides smoother deformations. Default: 10
            pca_coefficient_bounds: Bound on PCA coefficients in units of std deviations.
                Default: 3.0 (±3 std deviations per mode)
            method: Optimization method for scipy.optimize.minimize.
                Default: 'L-BFGS-B' (supports bounds)
            max_iterations: Maximum number of optimization iterations.
                Default: 50

        Returns:
            Tuple of (pca_coefficients, mean_distance):
                - pca_coefficients: Optimized PCA coefficients
                - mean_distance: Final mean distance metric value

        Raises:
            ValueError: If number of PCA modes to use exceeds available modes
        """
        if pca_number_of_modes <= 0:
            pca_number_of_modes = len(self.pca_eigenvectors)
        if pca_number_of_modes > len(self.pca_eigenvectors):
            raise ValueError(
                f"Number of PCA modes to use ({pca_number_of_modes}) exceeds available modes ({len(self.pca_std_deviations)})"
            )
        self.pca_number_of_modes = pca_number_of_modes

        self.log_info(f"Number of PCA modes: {pca_number_of_modes}")
        self.log_info(
            f"PCA coefficient bounds: ±{pca_coefficient_bounds} std deviations"
        )
        self.log_info(f"Optimization method: {method}")
        self.log_info(f"Max iterations: {max_iterations}")

        bounds = []
        for _ in range(pca_number_of_modes):
            bounds.append((-pca_coefficient_bounds, pca_coefficient_bounds))

        log_level_int = (
            self.log_level
            if isinstance(self.log_level, int)
            else logging.getLevelName(self.log_level)
        )
        disp = log_level_int <= logging.INFO

        self.log_info("Running optimization...")
        result_pca = minimize(  # type: ignore[call-overload]
            lambda params: self._mean_distance_metric(params),
            np.zeros(self.pca_number_of_modes),
            method=method,
            bounds=bounds,
            options={"maxiter": max_iterations, "disp": disp},
        )

        optimized_pca_coefficients = result_pca.x
        optimized_mean_distance = result_pca.fun

        self.log_info("Optimization completed!")
        self.log_info(f"Optimized PCA coefficients: {optimized_pca_coefficients}")
        self.log_info(f"Final mean intensity: {optimized_mean_distance:.2f}")

        return optimized_pca_coefficients, optimized_mean_distance

    def transform_template_model(self) -> pv.DataSet:
        """Create the final registered model by applying PCA deformation.

        Returns:
            Final registered and deformed model as a PyVista dataset.

        Raises:
            ValueError: If registration has not been performed
        """
        if self.registered_model_pca_coefficients is None:
            self.log_error("PCA coefficients are not set.")
            raise ValueError(
                "PCA coefficients must be set before creating registered model"
            )

        self.log_info("Creating final registered model...")

        # Compute PCA deformation
        if self.registered_model_pca_deformation is None:
            self.registered_model_pca_deformation = self._compute_pca_deformation(
                self.registered_model_pca_coefficients,
            )

        # Apply deformation and affine transform to each point
        final_points = np.zeros((self.pca_template_model.n_points, 3), dtype=np.float64)

        n_points = self.pca_template_model.n_points
        progress_interval = max(1, n_points // 10)  # Report progress every 10%

        point = itk.Point[itk.D, 3]()
        for i in range(n_points):
            # Report progress
            if i % progress_interval == 0 or i == n_points - 1:
                self.log_progress(i + 1, n_points, prefix="Transforming points")

            # Start with mean shape point
            point[0] = float(self.pca_template_model.points[i][0])
            point[1] = float(self.pca_template_model.points[i][1])
            point[2] = float(self.pca_template_model.points[i][2])

            # Add PCA deformation
            point[0] += self.registered_model_pca_deformation[i, 0]
            point[1] += self.registered_model_pca_deformation[i, 1]
            point[2] += self.registered_model_pca_deformation[i, 2]

            if self.post_pca_transform is not None:
                point = self.post_pca_transform.TransformPoint(point)

            # Store result
            final_points[i, 0] = point[0]
            final_points[i, 1] = point[1]
            final_points[i, 2] = point[2]

        # Create new model with transformed points
        self.registered_model = self.pca_template_model.copy(deep=True)
        self.registered_model.points = final_points.copy()

        self.log_info(
            f"Registered model created with {self.registered_model.n_points} points"
        )

        return self.registered_model

    def transform_point(
        self,
        point: itk.Point,
        include_post_pca_transform: bool = True,
    ) -> itk.Point:
        """Transform an arbitrary point using nearest neighbor interpolation.

        Args:
            point: ITK point to transform (itk.Point[itk.D, 3])

        Returns:
            Transformed ITK point

        Notes:
            1) if the point is outside the image bounds, the point is not transformed.
            2) if the forward point transform is set, it is applied.
            3) if the post_pca_transform is set and enabled, it is applied.
            4) if the forward point transform is not set, no errors are raised.

        Example:
            >>> p = itk.Point[itk.D, 3]()
            >>> p[0], p[1], p[2] = 10.0, 20.0, 30.0
            >>> transformed_p = registrar.transform_point(p)
        """

        if self.forward_point_transform is not None:
            transformed_point = self.forward_point_transform.TransformPoint(point)
        else:
            transformed_point = point

        if include_post_pca_transform and self.post_pca_transform is not None:
            transformed_point = self.post_pca_transform.TransformPoint(
                transformed_point
            )

        return transformed_point

    def compute_pca_transforms(self, reference_image: itk.Image) -> dict:
        """Compute PCA transforms.

        Returns:
            Dictionary containing:
                - 'forward_point_transform': POINT transform mapping template
                  points -> target points (warps the template onto the target)
                - 'inverse_point_transform': POINT transform mapping target
                  points -> template points

        Note:
            These are point transforms, oriented opposite to image-registration
            transforms; see docs/developer/transform_conventions.
        """
        assert self.registered_model_pca_deformation is not None, (
            "PCA deformation must be computed"
        )
        template_model_pca_deformation_field_image = (
            self._contour_tools.create_deformation_field(
                np.array(self.pca_template_model.points),
                self.registered_model_pca_deformation,
                reference_image=reference_image,
                blur_sigma=2.5,
                ptype=itk.D,
            )
        )

        self.forward_point_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        self.forward_point_transform.SetDisplacementField(
            template_model_pca_deformation_field_image
        )

        transform_tools = TransformTools()
        self.inverse_point_transform = (
            transform_tools.invert_displacement_field_transform(
                self.forward_point_transform
            )
        )
        return {
            "forward_point_transform": self.forward_point_transform,
            "inverse_point_transform": self.inverse_point_transform,
        }

    def register(
        self,
        pca_number_of_modes: int = 0,
        pca_coefficient_bounds: float = 3.5,
        method: str = "L-BFGS-B",
        max_iterations: int = 100,
    ) -> dict:
        """Optimize PCA coefficients to deform the model to better match
        low values in the distance map.

        Args:
            pca_number_of_modes: Number of PCA modes to use. Default: 0 (use all available modes)
            pca_coefficient_bounds: PCA coefficient bounds (±std devs). Default: 3.0
            method: Optimization method for scipy.optimize.minimize.
                Default: 'L-BFGS-B' (supports bounds)
            max_iterations: Maximum number of optimization iterations.
                Default: 50

        Returns:
            Dictionary containing:
                - 'registered_model': Final registered PyVista model
                - 'pca_coefficients': Optimized PCA coefficients
                - 'mean_distance': Final mean distance metric value

        Raises:
            ValueError: If reference image is not set

        Example:
            >>> result = registrar.register(pca_number_of_modes=10)
            >>> result['registered_model'].save('registered_heart.vtk')
        """
        if self.fixed_distance_map is None:
            raise ValueError("Reference image must be set before registration")

        if pca_number_of_modes <= 0:
            pca_number_of_modes = self.pca_number_of_modes

        self.log_section("PCA-BASED MODEL-TO-IMAGE REGISTRATION", width=70)
        self.log_info(f"Number of points: {self.pca_template_model.n_points}")
        self.log_info(f"Modes to use: {pca_number_of_modes}")

        self.registered_model_pca_coefficients, self.registered_model_mean_distance = (
            self._optimize_pca_coefficients(
                pca_number_of_modes=pca_number_of_modes,
                pca_coefficient_bounds=pca_coefficient_bounds,
                method=method,
                max_iterations=max_iterations,
            )
        )

        # Create final registered model
        self.registered_model_pca_deformation = None
        self.registered_model = self.transform_template_model()

        # Return results as dictionary
        return {
            "registered_model": self.registered_model,
            "pca_coefficients": self.registered_model_pca_coefficients,
            "mean_distance": self.registered_model_mean_distance,
        }
