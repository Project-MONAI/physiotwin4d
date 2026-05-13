#! /usr/bin/env python
"""
Test for time series image registration.

This test validates the RegisterTimeSeriesImages class which registers
an ordered sequence of images to a fixed reference image.
"""

from pathlib import Path
from typing import Any

import itk
import numpy as np
import pytest

from physiomotion4d import (
    RegisterTimeSeriesImages,
    TestTools,
    TransformTools,
)


@pytest.mark.requires_data
@pytest.mark.slow
class TestRegisterTimeSeriesImages:
    """Test suite for time series image registration."""

    _class_name = "registration_time_series_images"

    def test_registrar_initialization_ants(self) -> None:
        """Test that RegisterTimeSeriesImages initializes correctly with ANTs."""
        registrar = RegisterTimeSeriesImages(registration_method="ants")
        assert registrar is not None, "Registrar not initialized"
        assert registrar.registration_method_name == "ants", "Method not set correctly"
        assert registrar.registrar_ants is not None, (
            "Internal ANTs registrar not created"
        )
        assert registrar.registrar_icon is not None, (
            "Internal ICON registrar not created"
        )

        print("\nTime series registrar initialized with ANTs")

    def test_registrar_initialization_icon(self) -> None:
        """Test that RegisterTimeSeriesImages initializes correctly with ICON."""
        registrar = RegisterTimeSeriesImages(registration_method="icon")
        assert registrar is not None, "Registrar not initialized"
        assert registrar.registration_method_name == "icon", "Method not set correctly"
        assert registrar.registrar_ants is not None, (
            "Internal ANTs registrar not created"
        )
        assert registrar.registrar_icon is not None, (
            "Internal ICON registrar not created"
        )

        print("\nTime series registrar initialized with ICON")

    def test_registrar_initialization_invalid_method(self) -> None:
        """Test that invalid registration method raises error."""
        with pytest.raises(ValueError, match="registration_method must be"):
            RegisterTimeSeriesImages(registration_method="invalid")

        print("\nInvalid method correctly rejected")

    def test_set_modality(self) -> None:
        """Test setting imaging modality."""
        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        assert registrar.modality == "ct", "Modality not set correctly"

        print("\nModality setting works correctly")

    def test_set_fixed_image(self, test_images: list[Any]) -> None:
        """Test setting fixed image."""
        registrar = RegisterTimeSeriesImages(registration_method="ants")
        fixed_image = test_images[0]

        registrar.set_fixed_image(fixed_image)
        assert registrar.fixed_image is not None, "Fixed image not set"

        print("\nFixed image set successfully")
        print(f"  Image size: {itk.size(registrar.fixed_image)}")

    def test_set_number_of_iterations(self) -> None:
        """Test setting number of iterations."""
        registrar_ants = RegisterTimeSeriesImages(registration_method="ants")
        iterations_ants = [30, 15, 5]

        registrar_ants.set_number_of_iterations_ants(iterations_ants)
        assert registrar_ants.number_of_iterations_ants == iterations_ants, (
            "ANTs iterations not set correctly"
        )

        registrar_icon = RegisterTimeSeriesImages(registration_method="icon")
        iterations_icon = 50

        registrar_icon.set_number_of_iterations_icon(iterations_icon)
        assert registrar_icon.number_of_iterations_icon == iterations_icon, (
            "ICON iterations not set correctly"
        )

        print("\nNumber of iterations set successfully")

    def test_register_time_series_basic(
        self, test_images: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test basic time series registration without prior transform."""
        # Use first 3 images for quick test
        fixed_image = test_images[0]
        moving_images = test_images[1:4]

        print("\nRegistering time series (basic)...")
        print(f"  Fixed image: {itk.size(fixed_image)}")
        print(f"  Number of moving images: {len(moving_images)}")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_ants([20, 10, 2])

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=0,
            register_reference=True,
            prior_weight=0.0,
        )

        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "forward_transforms" in result, "Missing forward_transforms in result"
        assert "inverse_transforms" in result, "Missing inverse_transforms in result"
        assert "losses" in result, "Missing losses in result"

        forward_transforms = result["forward_transforms"]
        inverse_transforms = result["inverse_transforms"]
        losses = result["losses"]

        # Verify list lengths
        assert len(forward_transforms) == len(moving_images), (
            "forward_transforms length mismatch"
        )
        assert len(inverse_transforms) == len(moving_images), (
            "inverse_transforms length mismatch"
        )
        assert len(losses) == len(moving_images), "losses length mismatch"

        # Verify all transforms are valid
        for i, (forward_transform, inverse_transform) in enumerate(
            zip(forward_transforms, inverse_transforms, strict=False)
        ):
            assert forward_transform is not None, f"forward_transform[{i}] is None"
            assert inverse_transform is not None, f"inverse_transform[{i}] is None"

        print("Time series registration complete")
        print(f"  Transforms generated: {len(forward_transforms)}")
        print(f"  Average loss: {np.mean(losses):.6f}")

        transform_tools = TransformTools()
        moving_image = transform_tools.transform_image(
            moving_images[0],
            forward_transforms[0],
            fixed_image,
            interpolation_method="linear",
        )

        test_tools = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"] / self._class_name,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )

        test_tools.write_result_transform(
            forward_transforms[0], "basic_forward_transform_0.hdf"
        )
        assert test_tools.compare_result_to_baseline_transform(
            "basic_forward_transform_0.hdf",
        )

        test_tools.write_result_image(
            moving_image, "basic_time_series_registered_0.mha"
        )
        assert test_tools.compare_result_to_baseline_image(
            "basic_time_series_registered_0.mha",
        )

    def test_register_time_series_with_prior(
        self, test_images: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test time series registration with prior transform usage."""
        fixed_image = test_images[0]
        moving_images = test_images[1:4]

        print("\nRegistering time series (with prior)...")
        print(f"  Number of moving images: {len(moving_images)}")
        print("  Using prior transform weight: 0.5")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_ants([20, 10, 2])

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=1,  # Start from middle
            register_reference=True,
            prior_weight=0.5,
        )

        forward_transforms = result["forward_transforms"]
        losses = result["losses"]

        transform_tools = TransformTools()
        moving_image = transform_tools.transform_image(
            moving_images[0],
            forward_transforms[0],
            fixed_image,
            interpolation_method="linear",
        )

        # Verify all transforms generated
        for i, forward_transform in enumerate(forward_transforms):
            assert forward_transform is not None, f"forward_transform[{i}] is None"

        print("Time series registration with prior complete")
        print(f"  Losses: {[f'{loss:.6f}' for loss in losses]}")

        test_tools = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"] / self._class_name,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )

        test_tools.write_result_transform(
            forward_transforms[0], "prior_forward_transform_0.hdf"
        )
        assert test_tools.compare_result_to_baseline_transform(
            "prior_forward_transform_0.hdf",
        )

        test_tools.write_result_image(
            moving_image, "prior_time_series_registered_0.mha"
        )
        assert test_tools.compare_result_to_baseline_image(
            "prior_time_series_registered_0.mha",
        )

    def test_register_time_series_identity_start(self, test_images: list[Any]) -> None:
        """Test time series registration with identity for starting image."""
        fixed_image = test_images[0]
        moving_images = test_images[1:4]

        print("\nRegistering time series (identity start)...")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_ants([20, 10, 2])

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=0,
            register_reference=False,  # Use identity
            prior_weight=0.0,
        )

        # Starting image should have very low/zero loss
        losses = result["losses"]
        print(f"  Starting image loss: {losses[0]}")
        assert losses[0] == 0.0, "Starting image should have zero loss with identity"

        print("Identity start registration complete")

    def test_register_time_series_different_starting_indices(
        self, test_images: list[Any]
    ) -> None:
        """Test time series registration with different starting indices."""
        fixed_image = test_images[0]
        moving_images = test_images[1:3]  # 2 images

        print("\nTesting different starting indices...")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_ants([10, 5, 1])

        # Test starting from beginning, middle, and end
        for starting_index in [0, 1]:
            print(f"  Starting index: {starting_index}")
            result = registrar.register_time_series(
                moving_images=moving_images,
                reference_frame=starting_index,
                register_reference=True,
                prior_weight=0.0,
            )

            assert len(result["forward_transforms"]) == len(moving_images), (
                f"Wrong number of transforms for reference_frame={starting_index}"
            )

        print("Different starting indices work correctly")

    def test_register_time_series_error_no_fixed_image(self) -> None:
        """Test that error is raised if fixed image not set."""
        registrar = RegisterTimeSeriesImages(registration_method="ants")

        moving_images = [None, None, None]  # Dummy list

        with pytest.raises(ValueError, match="Fixed image must be set"):
            registrar.register_time_series(moving_images=moving_images)

        print("\nError correctly raised when fixed image not set")

    def test_register_time_series_error_invalid_starting_index(
        self, test_images: list[Any]
    ) -> None:
        """Test that error is raised for invalid starting index."""
        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_fixed_image(test_images[0])

        moving_images = test_images[1:4]

        # Test negative index
        with pytest.raises(ValueError, match="reference_frame.*out of range"):
            registrar.register_time_series(
                moving_images=moving_images, reference_frame=-1
            )

        # Test index too large
        with pytest.raises(ValueError, match="reference_frame.*out of range"):
            registrar.register_time_series(
                moving_images=moving_images, reference_frame=10
            )

        print("\nInvalid starting index correctly rejected")

    def test_register_time_series_error_invalid_prior_portion(
        self, test_images: list[Any]
    ) -> None:
        """Test that error is raised for invalid prior portion value."""
        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_fixed_image(test_images[0])

        moving_images = test_images[1:4]

        # Test negative value
        with pytest.raises(ValueError, match="must be in"):
            registrar.register_time_series(
                moving_images=moving_images,
                prior_weight=-0.1,
            )

        # Test value > 1
        with pytest.raises(ValueError, match="must be in"):
            registrar.register_time_series(
                moving_images=moving_images,
                prior_weight=1.5,
            )

        print("\nInvalid prior portion correctly rejected")

    def test_transform_application_time_series(
        self, test_images: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test applying transforms from time series registration."""
        fixed_image = test_images[0]
        moving_images = test_images[1:3]

        print("\nTesting transform application...")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_ants([20, 10, 2])

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=0,
            register_reference=True,
            prior_weight=0.0,
        )

        forward_transforms = result["forward_transforms"]

        # Apply transform to first moving image
        transform_tools = TransformTools()
        registered_image = transform_tools.transform_image(
            moving_images[0],
            forward_transforms[0],
            fixed_image,
            interpolation_method="linear",
        )

        assert registered_image is not None, "Registered image is None"
        assert itk.size(registered_image) == itk.size(fixed_image), "Size mismatch"

        print("Transform application successful")
        print(f"  Registered image size: {itk.size(registered_image)}")

        # Save registered image
        test_tools = TestTools(
            class_name=self._class_name,
            results_dir=test_directories["output"] / self._class_name,
            baselines_dir=test_directories["baselines"] / self._class_name,
        )

        test_tools.write_result_image(
            registered_image, "transform_application_time_series_0.mha"
        )
        assert test_tools.compare_result_to_baseline_image(
            "transform_application_time_series_0.mha",
        )

    def test_register_time_series_icon(self, test_images: list[Any]) -> None:
        """Test time series registration with ICON method."""
        fixed_image = test_images[0]
        moving_images = test_images[1:3]

        print("\nTesting time series registration with ICON...")

        registrar = RegisterTimeSeriesImages(registration_method="icon")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_icon(5)  # ICON uses single int

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=0,
            register_reference=True,
            prior_weight=0.0,
        )

        assert len(result["forward_transforms"]) == len(moving_images)
        assert len(result["inverse_transforms"]) == len(moving_images)
        assert len(result["losses"]) == len(moving_images)

        print("ICON time series registration complete")

    def test_register_time_series_with_mask(
        self, test_images: list[Any], test_directories: dict[str, Path]
    ) -> None:
        """Test time series registration with fixed image mask."""
        fixed_image = test_images[0]
        moving_images = test_images[1:3]

        # Create simple binary mask (central region)
        fixed_size_itk = itk.size(fixed_image)
        fixed_size = (
            int(fixed_size_itk[0]),
            int(fixed_size_itk[1]),
            int(fixed_size_itk[2]),
        )

        fixed_mask_arr = np.zeros(fixed_size[::-1], dtype=np.uint8)
        fixed_mask_arr[
            fixed_size[2] // 4 : 3 * fixed_size[2] // 4,
            fixed_size[1] // 4 : 3 * fixed_size[1] // 4,
            fixed_size[0] // 4 : 3 * fixed_size[0] // 4,
        ] = 1

        fixed_mask = itk.image_from_array(fixed_mask_arr)
        fixed_mask.CopyInformation(fixed_image)

        print("\nTesting time series registration with mask...")
        print(f"  Mask voxels: {np.sum(fixed_mask_arr)}")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_fixed_mask(fixed_mask)
        registrar.set_number_of_iterations_ants([20, 10, 2])

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=0,
            register_reference=True,
            prior_weight=0.0,
        )

        assert len(result["forward_transforms"]) == len(moving_images)

        print("Masked time series registration complete")

    def test_bidirectional_registration(self, test_images: list[Any]) -> None:
        """Test that bidirectional registration works correctly."""
        fixed_image = test_images[0]
        moving_images = test_images[1:6]  # 5 images

        print("\nTesting bidirectional registration...")
        print(f"  Total images: {len(moving_images)}")
        print("  Starting from middle (index 2)")

        registrar = RegisterTimeSeriesImages(registration_method="ants")
        registrar.set_modality("ct")
        registrar.set_fixed_image(fixed_image)
        registrar.set_number_of_iterations_ants([20, 10, 2])

        result = registrar.register_time_series(
            moving_images=moving_images,
            reference_frame=2,  # Middle image
            register_reference=True,
            prior_weight=0.0,
        )

        forward_transforms = result["forward_transforms"]

        # All transforms should be generated
        for i, forward_transform in enumerate(forward_transforms):
            assert forward_transform is not None, f"Transform {i} is None"

        print("Bidirectional registration successful")
        print(f"  All {len(forward_transforms)} transforms generated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
