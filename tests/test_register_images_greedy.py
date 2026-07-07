#!/usr/bin/env python
"""
Tests for Greedy-based image registration.

Uses the same fixtures as test_register_images_ANTS (converted 3D CT images).
Requires the picsl-greedy package and test data.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from physiotwin4d.register_images_greedy import RegisterImagesGreedy
from physiotwin4d.transform_tools import TransformTools


@pytest.mark.slow
class TestRegisterImagesGreedy:
    """Test suite for Greedy-based image registration."""

    def test_registrar_initialization(
        self, registrar_greedy: RegisterImagesGreedy
    ) -> None:
        """Test that RegisterImagesGreedy initializes correctly."""
        assert registrar_greedy is not None, "Registrar not initialized"
        assert hasattr(registrar_greedy, "fixed_image"), "Missing fixed_image attribute"
        assert hasattr(registrar_greedy, "fixed_mask"), "Missing fixed_mask attribute"

        print("\nGreedy registrar initialized successfully")

    def test_set_modality(self, registrar_greedy: RegisterImagesGreedy) -> None:
        """Test setting imaging modality."""
        registrar_greedy.set_modality("ct")
        assert registrar_greedy.modality == "ct", "Modality not set correctly"

        registrar_greedy.set_modality("mr")
        assert registrar_greedy.modality == "mr", "Modality change failed"

        print("\nModality setting works correctly")

    def test_set_transform_type_and_metric(
        self, registrar_greedy: RegisterImagesGreedy
    ) -> None:
        """Test setting transform type and metric."""
        registrar_greedy.set_transform_type("Rigid")
        assert registrar_greedy.transform_type == "Rigid"

        registrar_greedy.set_transform_type("Affine")
        assert registrar_greedy.transform_type == "Affine"

        registrar_greedy.set_transform_type("Deformable")
        assert registrar_greedy.transform_type == "Deformable"

        registrar_greedy.set_metric("CC")
        assert registrar_greedy.metric == "CC"
        registrar_greedy.set_metric("Mattes")
        assert registrar_greedy.metric == "Mattes"
        registrar_greedy.set_metric("MeanSquares")
        assert registrar_greedy.metric == "MeanSquares"

        with pytest.raises(ValueError, match="Invalid transform type"):
            registrar_greedy.set_transform_type("Invalid")
        with pytest.raises(ValueError, match="Invalid metric"):
            registrar_greedy.set_metric("Invalid")

        print("\nTransform type and metric setting work correctly")

    def test_set_fixed_image(
        self, registrar_greedy: RegisterImagesGreedy, test_images: list[Any]
    ) -> None:
        """Test setting fixed image."""
        fixed_image = test_images[0]
        registrar_greedy.set_fixed_image(fixed_image)
        assert registrar_greedy.fixed_image is not None, "Fixed image not set"

        print("\nFixed image set successfully")
        print(f"  Image size: {itk.size(registrar_greedy.fixed_image)}")

    def test_register_affine_without_mask(
        self,
        registrar_greedy: RegisterImagesGreedy,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test affine registration without masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_greedy"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        print("\nGreedy affine registration without mask...")

        registrar_greedy.set_modality("ct")
        registrar_greedy.set_transform_type("Affine")
        registrar_greedy.set_fixed_image(fixed_image)

        result = registrar_greedy.register(moving_image=moving_image)

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "inverse_transform" in result, "Missing inverse_transform in result"
        assert "forward_transform" in result, "Missing forward_transform in result"

        inverse_transform = result["inverse_transform"]
        forward_transform = result["forward_transform"]

        assert inverse_transform is not None, "inverse_transform is None"
        assert forward_transform is not None, "forward_transform is None"

        print("Greedy affine registration complete without mask")

        itk.transformwrite(
            [inverse_transform],
            str(reg_output_dir / "greedy_affine_inverse_no_mask.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(reg_output_dir / "greedy_affine_forward_no_mask.hdf"),
            compression=True,
        )

    def test_register_affine_with_mask(
        self,
        registrar_greedy: RegisterImagesGreedy,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test affine registration with binary masks."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_greedy"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        fixed_size_itk = itk.size(fixed_image)
        moving_size_itk = itk.size(moving_image)
        fixed_size = (
            int(fixed_size_itk[0]),
            int(fixed_size_itk[1]),
            int(fixed_size_itk[2]),
        )
        moving_size = (
            int(moving_size_itk[0]),
            int(moving_size_itk[1]),
            int(moving_size_itk[2]),
        )

        fixed_mask_arr = np.zeros(fixed_size[::-1], dtype=np.uint8)
        moving_mask_arr = np.zeros(moving_size[::-1], dtype=np.uint8)
        fixed_mask_arr[
            fixed_size[2] // 4 : 3 * fixed_size[2] // 4,
            fixed_size[1] // 4 : 3 * fixed_size[1] // 4,
            fixed_size[0] // 4 : 3 * fixed_size[0] // 4,
        ] = 1
        moving_mask_arr[
            moving_size[2] // 4 : 3 * moving_size[2] // 4,
            moving_size[1] // 4 : 3 * moving_size[1] // 4,
            moving_size[0] // 4 : 3 * moving_size[0] // 4,
        ] = 1

        fixed_mask = itk.image_from_array(fixed_mask_arr)
        fixed_mask.CopyInformation(fixed_image)
        moving_mask = itk.image_from_array(moving_mask_arr)
        moving_mask.CopyInformation(moving_image)

        registrar_greedy.set_modality("ct")
        registrar_greedy.set_transform_type("Affine")
        registrar_greedy.set_fixed_image(fixed_image)
        registrar_greedy.set_fixed_mask(fixed_mask)

        result = registrar_greedy.register(
            moving_image=moving_image, moving_mask=moving_mask
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["inverse_transform"] is not None
        assert result["forward_transform"] is not None

        print("Greedy affine registration complete with masks")

    def test_transform_application(
        self,
        registrar_greedy: RegisterImagesGreedy,
        test_images: list[Any],
        test_directories: dict[str, Path],
    ) -> None:
        """Test applying registration transform to moving image."""
        output_dir = test_directories["output"]
        reg_output_dir = output_dir / "registration_greedy"
        reg_output_dir.mkdir(exist_ok=True)

        fixed_image = test_images[0]
        moving_image = test_images[1]

        registrar_greedy.set_modality("ct")
        registrar_greedy.set_transform_type("Affine")
        registrar_greedy.set_fixed_image(fixed_image)
        result = registrar_greedy.register(moving_image=moving_image)

        forward_transform = result["forward_transform"]
        transform_tools = TransformTools()
        registered_image = transform_tools.transform_image(
            moving_image, forward_transform, fixed_image, interpolation_method="linear"
        )

        assert registered_image is not None, "Registered image is None"
        assert itk.size(registered_image) == itk.size(fixed_image), "Size mismatch"

        moving_arr = itk.array_from_image(moving_image)
        registered_arr = itk.array_from_image(registered_image)
        difference = np.sum(
            np.abs(moving_arr.astype(float) - registered_arr.astype(float))
        )

        print("Greedy transform applied successfully")
        print(f"  Registered image size: {itk.size(registered_image)}")
        print(f"  Total difference: {difference:.2f}")

        itk.imwrite(
            registered_image,
            str(reg_output_dir / "greedy_registered_image.mha"),
            compression=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
