"""
Tools for creating and manipulating contours.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import itk
import numpy as np
import pyvista as pv
import trimesh

from .image_tools import ImageTools
from .physiotwin4d_base import PhysioTwin4DBase
from .transform_tools import TransformTools


class ContourTools(PhysioTwin4DBase):
    """
    Tools for creating and manipulating contours.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize ContourTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    @staticmethod
    def extract_contours(
        labelmap_image: itk.image,
        smoothing_iterations: int = 10,
        smoothing_scale: float = 1.0,
    ) -> pv.PolyData:
        """
        Make contours from a labelmap image.

        Args:
            labelmap_image (itk.image): The labelmap image to create contours from

        Returns:
            pv.PolyData: The contours as a PyVista PolyData object
        """
        labels = pv.wrap(itk.vtk_image_from_image(labelmap_image))
        contours = cast(
            pv.PolyData,
            labels.contour_labels(
                boundary_style="all",
                pad_background=False,
                smoothing=True,
                smoothing_iterations=smoothing_iterations,
                smoothing_scale=smoothing_scale,
                output_mesh_type="triangles",
            ),
        )

        return contours

    @staticmethod
    def smooth_and_decimate_surface(
        surface: pv.PolyData,
        decimation_reduction: float,
        smoothing_iterations: int,
    ) -> pv.PolyData:
        """Optionally decimate then smooth a surface (no-op when disabled).

        Decimation uses ``decimate_pro`` on a triangulated copy; because
        ``decimate_pro`` discards cell data, per-cell ``boundary_labels`` (needed
        for anatomy splitting downstream) are transferred back onto the decimated
        cells from their nearest original cell so anatomy materials still apply.
        Smoothing uses non-shrinking Taubin smoothing, which only moves points and
        therefore preserves cells and their labels.

        Args:
            surface: Input surface.
            decimation_reduction: Fraction of triangles to remove (0.0 disables).
            smoothing_iterations: Taubin smoothing iterations (0 disables).

        Returns:
            The decimated and/or smoothed surface.
        """
        conditioned = surface
        if decimation_reduction > 0.0:
            original = conditioned
            conditioned = conditioned.triangulate().decimate_pro(decimation_reduction)
            if "boundary_labels" in original.cell_data:
                nearest = original.find_closest_cell(conditioned.cell_centers().points)
                conditioned.cell_data["boundary_labels"] = np.asarray(
                    original.cell_data["boundary_labels"]
                )[nearest]
        if smoothing_iterations > 0:
            conditioned = conditioned.smooth_taubin(n_iter=smoothing_iterations)
        return conditioned

    @staticmethod
    def extract_surface(mesh: pv.DataSet) -> pv.PolyData:
        """Extract the surface of a mesh.

        Args:
            mesh: Input mesh (PolyData is returned unchanged; any other DataSet
                is passed through ``extract_surface``).

        Returns:
            pv.PolyData: The surface of the mesh.
        """
        if isinstance(mesh, pv.PolyData):
            return mesh
        return mesh.extract_surface(algorithm="dataset_surface")

    @staticmethod
    def transform_contours(
        contours: pv.PolyData,
        tfm: itk.Transform,
        with_deformation_magnitude: bool = False,
    ) -> pv.PolyData:
        """
        Transform contours using a given transform.

        Args:
            tfm (itk.Transform): The transform to use

        Returns:
            pv.PolyData: The transformed contours with deformation magnitude
        """
        new_contours = TransformTools().transform_pvcontour(
            contours, tfm, with_deformation_magnitude=with_deformation_magnitude
        )

        return new_contours

    def merge_meshes(
        self, meshes: list[pv.PolyData]
    ) -> tuple[pv.PolyData, list[pv.PolyData]]:
        """
        Merge multiple fixed meshes into a single mesh.

        Returns
        -------
        pv.PolyData
            Merged mesh
        """
        self.log_info("Merging meshes...")
        trimesh_meshes: list[trimesh.Trimesh] = []
        if hasattr(meshes[0], "n_faces_strict"):
            trimesh_meshes = [
                trimesh.Trimesh(
                    vertices=mesh.points,
                    faces=mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:],
                )
                for mesh in meshes
            ]
        else:
            trimesh_meshes = [
                trimesh.Trimesh(
                    vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:4]
                )
                for mesh in meshes
            ]

        # Merge meshes
        merged_trimesh = trimesh.util.concatenate(trimesh_meshes)
        flip_matrix = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        merged_trimesh.apply_transform(flip_matrix)  # Apply flip transformation
        for mesh in trimesh_meshes:
            mesh.apply_transform(flip_matrix)

        merged_mesh = pv.wrap(merged_trimesh)
        pv_meshes = [pv.wrap(mesh) for mesh in trimesh_meshes]

        return merged_mesh, pv_meshes

    @staticmethod
    def create_reference_image(
        mesh: pv.DataSet,
        spatial_resolution: float = 0.5,
        buffer_factor: float = 0.25,
        ptype: type = itk.F,
    ) -> itk.Image:
        """
        Create a reference image from a mesh.
        """
        points = np.array(mesh.points)
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        min_bounds = min_bounds - buffer_factor * (max_bounds - min_bounds)
        max_bounds = max_bounds + buffer_factor * (max_bounds - min_bounds)
        region = (
            ((max_bounds - min_bounds) / spatial_resolution + 1)
            .astype(np.int32)
            .tolist()
        )
        itk_region = itk.ImageRegion[3]()
        itk_region.SetSize(region)
        reference_image = itk.Image[ptype, 3].New()
        reference_image.SetRegions(itk_region)
        reference_image.SetSpacing([spatial_resolution] * 3)
        reference_image.SetOrigin(min_bounds.tolist())
        reference_image.Allocate()
        return reference_image

    @staticmethod
    def create_mask_from_mesh(
        mesh: pv.DataSet | pv.UnstructuredGrid,
        reference_image: itk.Image,
    ) -> itk.Image:
        ref_spacing = np.array(reference_image.GetSpacing())

        # Create trimesh object with LPS coordinates
        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.extract_surface(algorithm="dataset_surface")

        if hasattr(mesh, "n_faces_strict"):
            # PyVista PolyData
            num_points_per_face = len(mesh.faces) // mesh.n_faces_strict
            faces = mesh.faces.reshape((mesh.n_faces_strict, num_points_per_face))[
                :, 1:
            ]
        else:
            # Handle other mesh types
            faces = mesh.faces.reshape((-1, 4))[:, 1:]

        trimesh_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)

        # Determine voxel spacing (use minimum spacing from reference)
        voxel_pitch = float(np.min(ref_spacing))

        # Voxelize the mesh
        # trimesh.voxelized() creates a grid aligned with the mesh's bounding box
        # The voxel grid origin is at the minimum corner of the bounding box
        vox = trimesh_mesh.voxelized(pitch=voxel_pitch)
        binary_array = vox.matrix.astype(np.uint8)

        # Get the physical origin of the voxel grid in LPS space
        # trimesh voxel grids use a transformation matrix, and the voxel grid starts
        # at the mesh's minimum bounds. The physical origin is where voxel [0,0,0]
        # center is located.
        # Get mesh bounds in LPS coordinates
        mesh_bounds_lps = (
            trimesh_mesh.bounds
        )  # shape (2, 3): [[x_min, y_min, z_min], [x_max, y_max, z_max]]

        # The voxel grid origin is at the minimum corner, but ITK origin is the CENTER
        # of voxel (0,0,0)
        # So we need to add half a voxel pitch to each dimension
        voxel_grid_origin_lps = mesh_bounds_lps[0] + voxel_pitch / 2.0
        voxel_grid_origin_lps[2] = (
            voxel_grid_origin_lps[2] + voxel_pitch * binary_array.shape[2]
        )

        # transpose to match trimesh XYZ convention
        binary_array_zyx = np.transpose(binary_array, (2, 1, 0))
        binary_array_flip = np.flip(binary_array_zyx, axis=0)
        binary_image = itk.GetImageFromArray(binary_array_flip)

        # Set ITK image metadata in LPS coordinates
        # Origin: where the center of voxel (0,0,0) is located in physical space
        binary_image.SetOrigin(voxel_grid_origin_lps)

        # Spacing: uniform voxel pitch in all directions
        binary_image.SetSpacing([voxel_pitch] * 3)

        # Direction: use identity for now (axis-aligned), will be handled by resampling
        # Flip Z axis to match ITK convention
        ref_dir = itk.array_from_matrix(binary_image.GetDirection())
        ref_dir[2, 2] = -ref_dir[2, 2]
        binary_image.SetDirection(ref_dir)

        # Fill holes to create solid mask
        ImageType = type(binary_image)
        fill_filter = itk.BinaryFillholeImageFilter[ImageType].New()
        fill_filter.SetInput(binary_image)
        fill_filter.SetForegroundValue(1)
        fill_filter.Update()
        mask_image = fill_filter.GetOutput()

        resampler = itk.ResampleImageFilter.New(Input=mask_image)
        resampler.SetReferenceImage(reference_image)
        resampler.SetUseReferenceImage(True)
        resampler.SetInterpolator(
            itk.NearestNeighborInterpolateImageFunction.New(mask_image)
        )
        resampler.SetDefaultPixelValue(0)
        resampler.Update()
        mask_image = resampler.GetOutput()

        return mask_image

    def create_labelmap_from_meshes(
        self,
        meshes: list[pv.DataSet | pv.UnstructuredGrid],
        reference_image: itk.Image,
    ) -> itk.Image:
        """
        Create a labelmap from a list of meshes.
        """
        labelmap_arr = np.zeros(
            (
                reference_image.GetLargestPossibleRegion().GetSize()[2],
                reference_image.GetLargestPossibleRegion().GetSize()[1],
                reference_image.GetLargestPossibleRegion().GetSize()[0],
            ),
            dtype=np.uint16,
        )
        for i, mesh in enumerate(meshes):
            mask_image = self.create_mask_from_mesh(mesh, reference_image)
            mask_arr = itk.GetArrayFromImage(mask_image)
            labelmap_arr[mask_arr > 0] = i + 1

        labelmap_image = itk.GetImageFromArray(labelmap_arr)
        labelmap_image.CopyInformation(reference_image)

        return labelmap_image

    @staticmethod
    def sample_mesh_faces(mesh: pv.DataSet, max_spacing: float) -> np.ndarray:
        """Return mesh points supplemented by samples across the triangle faces.

        Rasterizing vertices alone leaves gaps between them on meshes that are
        coarse relative to the voxel size, which makes a distance map built from
        them ripple. Adding barycentric samples dense enough that consecutive
        samples are closer than ``max_spacing`` closes those gaps.

        Args:
            mesh: Source mesh; its surface is triangulated if needed.
            max_spacing: Target spacing between samples, in mm.

        Returns:
            (n, 3) array of sample points, starting with the mesh's own points.
        """
        points = np.asarray(mesh.points, dtype=np.float64)
        surface = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
        surface = surface.triangulate()
        if surface.faces.size == 0:
            return points
        faces = surface.faces.reshape(-1, 4)[:, 1:]

        vertices = np.asarray(surface.points, dtype=np.float64)
        corners = vertices[faces]  # (n_faces, 3, 3)
        edge_lengths = np.linalg.norm(
            corners - np.roll(corners, 1, axis=1), axis=2
        ).max(axis=1)

        # Group faces by how finely they need to be subdivided so each division
        # level is generated as one vectorized batch.
        divisions = np.maximum(1, np.ceil(edge_lengths / max(max_spacing, 1e-6)))
        divisions = np.minimum(divisions, 64).astype(np.int64)

        samples = [points]
        for level in np.unique(divisions):
            if level < 2:
                continue
            selected = corners[divisions == level]
            # Barycentric lattice with `level` divisions per edge.
            steps = np.arange(level + 1, dtype=np.float64) / level
            u, v = np.meshgrid(steps, steps, indexing="ij")
            mask = (u + v) <= 1.0
            weights = np.column_stack([1.0 - u[mask] - v[mask], u[mask], v[mask]])
            samples.append(np.einsum("fca,kc->fka", selected, weights).reshape(-1, 3))

        return np.concatenate(samples, axis=0)

    def create_distance_map(
        self,
        mesh: pv.DataSet | pv.UnstructuredGrid,
        reference_image: itk.Image,
        squared_distance: bool = False,
        negative_inside: bool = True,
        zero_inside: bool = False,
        norm_to_max_distance: float = 0.0,
        sample_faces: bool = True,
    ) -> itk.Image:
        """Compute a distance map of a mesh on the reference image's grid.

        Args:
            mesh: Mesh whose surface the distances are measured to.
            reference_image: Image defining the output grid.
            squared_distance: Sign-preserving square of the result. Default: False
            negative_inside: Keep the signed output. Default: True
            zero_inside: Clip negative values to zero before anything else.
                Default: False
            norm_to_max_distance: If non-zero, divide by this value and clip to
                [-1, 1]. Default: 0.0 (distances stay in mm)
            sample_faces: Rasterize samples across the triangle faces as well as
                the vertices, so that coarse meshes do not leave gaps in the
                rasterized surface. Default: True

        Returns:
            ITK image of distances on the reference grid.
        """
        self.log_info("Computing signed distance map...")

        size = reference_image.GetLargestPossibleRegion().GetSize()

        if sample_faces:
            points = self.sample_mesh_faces(
                mesh, 0.5 * float(min(reference_image.GetSpacing()))
            )
            self.log_debug(
                "Distance map: %d face samples from %d mesh points",
                len(points),
                mesh.n_points,
            )
        else:
            points = np.asarray(mesh.points, dtype=np.float64)

        # NumPy convention is (z, y, x); ITK GetSize() returns (x, y, z)
        tmp_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

        # Bulk equivalent of TransformPhysicalPointToIndex, which rounds half up.
        index_to_world = itk.array_from_matrix(
            reference_image.GetDirection()
        ) @ np.diag(np.asarray(reference_image.GetSpacing()))
        origin = np.asarray(reference_image.GetOrigin(), dtype=np.float64)
        indices = np.floor(
            (points - origin) @ np.linalg.inv(index_to_world).T + 0.5
        ).astype(np.int64)
        size_arr = np.array([size[0], size[1], size[2]], dtype=np.int64)
        inside = np.all((indices >= 0) & (indices < size_arr), axis=1)
        indices = indices[inside]
        point_count = len(indices)
        if point_count:
            tmp_arr[indices[:, 2], indices[:, 1], indices[:, 0]] = 1

        self.log_info(
            "Distance map: %d/%d surface samples within reference image",
            point_count,
            len(points),
        )
        if point_count == 0:
            self.log_warning(
                "No surface points fall within the reference image! "
                "Distance map will be constant. "
                "Mesh bounds: %s  Image origin: %s  Image size: %s  Image spacing: %s",
                str(mesh.bounds),
                str(reference_image.GetOrigin()),
                str(size),
                str(reference_image.GetSpacing()),
            )
        elif not inside.all():
            # Distances near the dropped region are measured to whatever samples
            # remain in the grid, so they are larger than the true distance.
            self.log_warning(
                "%d of %d surface samples fall outside the reference image; "
                "distances near that boundary are overestimated.",
                len(points) - point_count,
                len(points),
            )

        tmp_binary_image = itk.GetImageFromArray(tmp_arr)
        tmp_binary_image.CopyInformation(reference_image)
        assert (
            tmp_binary_image.GetLargestPossibleRegion().GetSize()
            == reference_image.GetLargestPossibleRegion().GetSize()
        )

        distance_filter = itk.SignedMaurerDistanceMapImageFilter.New(
            Input=tmp_binary_image
        )
        distance_filter.SetSquaredDistance(False)
        distance_filter.SetUseImageSpacing(True)
        distance_filter.Update()
        distance_image = distance_filter.GetOutput()

        distance_arr = itk.GetArrayFromImage(distance_image).astype(np.float32)
        if zero_inside:
            distance_arr = np.clip(distance_arr, 0.0, None)
        if not negative_inside:
            distance_arr = np.abs(distance_arr)
        if squared_distance:
            distance_arr = np.sign(distance_arr) * distance_arr**2
        if norm_to_max_distance != 0.0:
            distance_arr = distance_arr / norm_to_max_distance
            distance_arr = np.clip(distance_arr, -1.0, 1.0)
        distance_image = itk.GetImageFromArray(distance_arr)
        distance_image.CopyInformation(reference_image)

        return distance_image

    @staticmethod
    def create_deformation_field(
        points: np.ndarray,
        point_displacements: np.ndarray,
        reference_image: itk.Image,
        blur_sigma: float = 2.5,
        ptype: type = itk.D,
    ) -> itk.Image:
        """
        Create a displacement map from model points and displacements.
        """
        size = reference_image.GetLargestPossibleRegion().GetSize()
        norm_map = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        displacement_map_x = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        displacement_map_y = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        displacement_map_z = np.zeros((size[2], size[1], size[0])).astype(np.float32)
        itk_point = itk.Point[itk.D, 3]()
        for i, point in enumerate(points):
            itk_point[0] = float(point[0])
            itk_point[1] = float(point[1])
            itk_point[2] = float(point[2])
            indx = reference_image.TransformPhysicalPointToIndex(itk_point)
            if (
                indx[0] < 0
                or indx[1] < 0
                or indx[2] < 0
                or indx[0] >= size[0]
                or indx[1] >= size[1]
                or indx[2] >= size[2]
            ):
                continue
            displacement_map_x[int(indx[2]), int(indx[1]), int(indx[0])] = (
                point_displacements[i, 0]
            )
            displacement_map_y[int(indx[2]), int(indx[1]), int(indx[0])] = (
                point_displacements[i, 1]
            )
            displacement_map_z[int(indx[2]), int(indx[1]), int(indx[0])] = (
                point_displacements[i, 2]
            )
            norm_map[int(indx[2]), int(indx[1]), int(indx[0])] = 1

        norm_img = itk.GetImageFromArray(norm_map)
        norm_img.CopyInformation(reference_image)
        assert (
            norm_img.GetLargestPossibleRegion().GetSize()
            == reference_image.GetLargestPossibleRegion().GetSize()
        )

        blurred_norm = itk.SmoothingRecursiveGaussianImageFilter(
            Input=norm_img, Sigma=blur_sigma
        )
        blurred_norm_arr = itk.GetArrayFromImage(blurred_norm)
        blurred_norm_arr = np.where(blurred_norm_arr < 1.0e-4, 1.0e-4, blurred_norm_arr)

        deformation_field_x_img = itk.GetImageFromArray(displacement_map_x)
        deformation_field_x_img.CopyInformation(reference_image)
        deformation_field_x_img = itk.SmoothingRecursiveGaussianImageFilter(
            Input=deformation_field_x_img, Sigma=blur_sigma
        )

        deformation_field_y_img = itk.GetImageFromArray(displacement_map_y)
        deformation_field_y_img.CopyInformation(reference_image)
        deformation_field_y_img = itk.SmoothingRecursiveGaussianImageFilter(
            Input=deformation_field_y_img, Sigma=blur_sigma
        )

        deformation_field_z_img = itk.GetImageFromArray(displacement_map_z)
        deformation_field_z_img.CopyInformation(reference_image)
        deformation_field_z_img = itk.SmoothingRecursiveGaussianImageFilter(
            Input=deformation_field_z_img, Sigma=blur_sigma
        )

        deformation_field_x = (
            itk.GetArrayFromImage(deformation_field_x_img) / blurred_norm_arr
        )
        deformation_field_y = (
            itk.GetArrayFromImage(deformation_field_y_img) / blurred_norm_arr
        )
        deformation_field_z = (
            itk.GetArrayFromImage(deformation_field_z_img) / blurred_norm_arr
        )

        deformation_field_x = np.where(
            blurred_norm_arr > 1.0e-3, deformation_field_x, 0.0
        )
        deformation_field_y = np.where(
            blurred_norm_arr > 1.0e-3, deformation_field_y, 0.0
        )
        deformation_field_z = np.where(
            blurred_norm_arr > 1.0e-3, deformation_field_z, 0.0
        )

        deformation_field = np.stack(
            [deformation_field_x, deformation_field_y, deformation_field_z], axis=-1
        )

        image_tools = ImageTools()
        deformation_field_img = image_tools.convert_array_to_image_of_vectors(
            deformation_field, reference_image, ptype=ptype
        )

        return deformation_field_img

    # ─────────────────────────── I/O helpers ───────────────────────────────

    @staticmethod
    def save_surfaces(
        surfaces: dict[str, pv.PolyData],
        output_dir: str,
        prefix: str = "",
    ) -> dict[str, str]:
        """Save each named surface to its own VTP file.

        Args:
            surfaces: Mapping of name → surface (e.g. the ``'surfaces'``
                value from :meth:`WorkflowConvertImageToVTK.process`).
            output_dir: Directory to write files into (created if absent).
            prefix: Optional filename prefix.  Each file is named
                ``{prefix}_{name}.vtp`` (or ``{name}.vtp`` when *prefix* is empty).

        Returns:
            Mapping of name → absolute path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: dict[str, str] = {}
        for name, surface in surfaces.items():
            stem = f"{prefix}_{name}" if prefix else name
            path = os.path.join(output_dir, f"{stem}.vtp")
            surface.save(path)
            saved[name] = path
        return saved

    @staticmethod
    def save_combined_surfaces(
        surfaces: dict[str, pv.PolyData],
        output_filename: str,
    ) -> str:
        """Merge all named surfaces into a single VTP file.

        The merged mesh retains per-cell ``Color`` (RGBA uint8) from each
        surface's annotation, enabling colour-by-anatomy rendering in
        Paraview, PyVista, etc.

        It also gains a per-cell ``SegmentationLabelIds`` (int32) array, which
        carries each cell's originating label ID so structure identity survives
        the merge.  Downstream, :class:`ConvertVTKToUSD` splits on this array
        when given ``mask_ids``, giving one prim (and one anatomy material) per
        structure.  A surface whose ``field_data['SegmentationLabelIds']`` does
        not hold exactly one ID has no per-cell attribution — that is the case
        for the per-group surfaces of :class:`WorkflowConvertImageToVTK`, which
        are contoured from a merged binary mask — so its cells are tagged ``0``.
        Pass the per-label surfaces (``'label_surfaces'``) to get real IDs.

        Per-object ``field_data`` is *not* preserved: it is per-object, so a
        single merged mesh cannot carry one value per input surface.  The
        remaining keys set by :meth:`WorkflowConvertImageToVTK._annotate` are
        therefore lost:

        - ``AnatomyGroup`` — group name, e.g. ``'heart'``.
        - ``SegmentationLabelNames`` — structure names within the group.
        - ``AnatomyColor`` — RGB float color (survives indirectly as the
          per-cell ``Color`` array).

        Use :meth:`save_surfaces` instead when structure *names* must be
        recoverable from the saved files.

        Args:
            surfaces: Mapping of name → surface.
            output_filename: Path of the VTP file to write, including its
                directory.  Any missing parent directories are created.

        Returns:
            Path to the saved VTP file.

        Raises:
            ValueError: If *surfaces* is empty.
        """
        if not surfaces:
            raise ValueError("No surfaces to save.")
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Shallow copies so tagging does not add an array to the caller's
        # surfaces; the point/cell arrays themselves stay shared.
        tagged: list[pv.PolyData] = []
        for surface in surfaces.values():
            label_ids = surface.field_data.get("SegmentationLabelIds")
            if label_ids is not None and len(label_ids) == 1:
                label_id = int(label_ids[0])
            else:
                label_id = 0
            tagged_surface = surface.copy(deep=False)
            tagged_surface.cell_data["SegmentationLabelIds"] = np.full(
                tagged_surface.n_cells, label_id, dtype=np.int32
            )
            tagged.append(tagged_surface)
        merged = cast(pv.PolyData, pv.merge(tagged, merge_points=False))
        merged.save(output_filename)
        return output_filename
