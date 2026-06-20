"""
Tools for creating and manipulating contours.
"""

from __future__ import annotations

import logging
from typing import cast

import itk
import numpy as np
import pyvista as pv
import trimesh

from .image_tools import ImageTools
from .physiomotion4d_base import PhysioMotion4DBase
from .transform_tools import TransformTools


class ContourTools(PhysioMotion4DBase):
    """
    Tools for creating and manipulating contours.
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize ContourTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def extract_contours(
        self,
        labelmap_image: itk.image,
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
                smoothing_iterations=10,
                output_mesh_type="triangles",
            ),
        )

        contours.smooth_taubin(
            inplace=True,
            n_iter=50,
            pass_band=0.05,
        )

        # self.contours.decimate_pro(
        # inplace=True,
        # reduction=0.7,
        # feature_angle=45,
        # preserve_topology=True,
        # )

        return contours

    def transform_contours(
        self,
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

    def create_reference_image(
        self,
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

    def create_mask_from_mesh(
        self,
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
            dtype=np.uint8,
        )
        for i, mesh in enumerate(meshes):
            mask_image = self.create_mask_from_mesh(mesh, reference_image)
            mask_arr = itk.GetArrayFromImage(mask_image)
            labelmap_arr[mask_arr > 0] = i + 1

        labelmap_image = itk.GetImageFromArray(labelmap_arr)
        labelmap_image.CopyInformation(reference_image)

        return labelmap_image

    def create_distance_map(
        self,
        mesh: pv.DataSet | pv.UnstructuredGrid,
        reference_image: itk.Image,
        squared_distance: bool = False,
        negative_inside: bool = True,
        zero_inside: bool = False,
        norm_to_max_distance: float = 0.0,
    ) -> itk.Image:
        self.log_info("Computing signed distance map...")

        # Convert mask to binary
        points = mesh.points

        size = reference_image.GetLargestPossibleRegion().GetSize()

        # NumPy convention is (z, y, x); ITK GetSize() returns (x, y, z)
        tmp_arr = np.zeros((size[2], size[1], size[0]), dtype=np.int32)
        itk_point = itk.Point[itk.D, 3]()
        point_count = 0
        for point in points:
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
            tmp_arr[indx[2], indx[1], indx[0]] = 1
            point_count += 1

        self.log_info(
            "Distance map: %d/%d surface points within reference image",
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

        tmp_binary_image = itk.GetImageFromArray(tmp_arr.astype(np.uint8))
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

    def create_deformation_field(
        self,
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
