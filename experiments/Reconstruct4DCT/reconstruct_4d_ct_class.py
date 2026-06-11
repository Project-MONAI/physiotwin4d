#!/usr/bin/env python
# %% [markdown]
# # 4D CT Reconstruction Using RegisterTimeSeriesImages Class
#
# This notebook demonstrates the use of the `RegisterTimeSeriesImages` class to register a time series of CT images to a common reference frame.
#
# This is a refactored version of `reconstruct_4d_ct.ipynb` that uses the new class-based approach, including:
# - Registration of time series images using ANTs, ICON, or ANTs+ICON methods
# - Reconstruction of time series using the `reconstruct_time_series()` method
# - Optional upsampling to fixed image resolution while preserving spatial positioning
#

# %%
import os

import itk
import numpy as np

from physiomotion4d import RegisterTimeSeriesImages, TransformTools
from physiomotion4d.test_tools import TestTools

_HERE = os.path.dirname(os.path.abspath(__file__))

# %% [markdown]
# ## Load Data and Set Parameters
#
# Set `quick_run = True` for a fast test with fewer images, or `quick_run = False` for full processing.
#

# %%
# Load image files
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
files = [
    os.path.join(data_dir, f)
    for f in sorted(os.listdir(data_dir))
    if f.endswith(".mha") and f.startswith("slice_")
]

print(f"Found {len(files)} slice files")

# %%
# Configuration: quick run when executed as test (pytest); full run when manual (set quick_run = True for interactive quick test)
quick_run = TestTools.running_as_test()

# Select files and parameters based on mode
if quick_run:
    print("=== QUICK RUN MODE ===")
    total_num_files = len(files)
    target_num_files = 5
    file_step = total_num_files // target_num_files
    files = files[0:total_num_files:file_step]
    files_indx = list(range(0, total_num_files, file_step))
    num_files = len(files)
    reference_image_num = num_files // 2

    # Registration parameters - only Greedy for quick run
    registration_methods = ["Greedy", "ICON", "Greedy_ICON"]
    number_of_iterations_list = [[8, 4, 1], 5, [[8, 4, 1], 5]]  # For Greedy and ICON
else:
    print("=== FULL RUN MODE ===")
    num_files = len(files)
    files_indx = list(range(num_files))
    reference_image_num = 7

    # Registration parameters - Greedy and ICON for full run
    registration_methods = ["Greedy"]  # , "ICON", "Greedy_ICON"]
    number_of_iterations_list = [
        [30, 15, 7, 3],
    ]  # For Greedy
    # 20,  # For ICON
    # [[30, 15, 7, 3], 20],  # For Greedy_ICON
    # ]

# Common parameters
reference_image_file = os.path.join(
    data_dir, f"slice_{files_indx[reference_image_num]:03d}.mha"
)
register_start_to_reference = False
portion_of_prior_transform_to_init_next_transform = 0.0

print(f"Number of files: {num_files}")
print(f"Reference image: slice_{files_indx[reference_image_num]:03d}.mha")
print(f"Registration methods: {registration_methods}")
print(f"Number of iterations: {number_of_iterations_list}")

# %% [markdown]
# ## Load Images
#

# %%
# Load fixed/reference image
fixed_image = itk.imread(reference_image_file, pixel_type=itk.F)
print(f"Fixed image size: {itk.size(fixed_image)}")
print(f"Fixed image spacing: {itk.spacing(fixed_image)}")

# Save fixed image for reference
_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
out_file = os.path.join(_RESULTS_DIR, "slice_fixed.mha")
itk.imwrite(fixed_image, out_file)
print(f"Saved fixed image to: {out_file}")

images = []
for file in files:
    img = itk.imread(file, pixel_type=itk.F)
    images.append(img)

# %%
# This cell will be run for each registration method in the loop below
print(f"Registration methods to run: {registration_methods}")

# %% [markdown]
# ## Perform Time Series Registration
#
# Loop through each registration method and perform registration.
#
# The registration produces:
# - **Forward transforms**: Transform moving images to fixed space (moving → fixed)
# - **Inverse transforms**: Transform fixed image to moving space (fixed → moving)
# - **Losses**: Registration quality metric for each time point
#

# %%
# Store results for each method
all_results = {}

# Loop through each registration method
for method_idx, registration_method in enumerate(registration_methods):
    number_of_iterations = number_of_iterations_list[method_idx]

    print("\n" + "=" * 70)
    print(f"Starting registration with {registration_method.upper()}")
    print("=" * 70)
    print(f"  Starting index: {reference_image_num}")
    print(f"  Register start to reference: {register_start_to_reference}")
    print(
        f"  Prior transform weight: {portion_of_prior_transform_to_init_next_transform}"
    )
    print(f"  Number of iterations: {number_of_iterations}")

    # Create registrar for this method
    registrar = RegisterTimeSeriesImages(registration_method=registration_method)
    registrar.set_modality("ct")
    registrar.set_fixed_image(fixed_image)

    # Set iterations based on registration method
    if registration_method == "Greedy":
        registrar.set_number_of_iterations_greedy(number_of_iterations)
    elif registration_method == "ICON":
        registrar.set_number_of_iterations_ICON(number_of_iterations)
    elif registration_method == "Greedy_ICON":
        registrar.set_number_of_iterations_greedy(number_of_iterations[0])
        registrar.set_number_of_iterations_ICON(number_of_iterations[1])

    # Perform registration
    result = registrar.register_time_series(
        moving_images=images,
        reference_frame=reference_image_num,
        register_reference=register_start_to_reference,
        prior_weight=portion_of_prior_transform_to_init_next_transform,
    )

    # Store results
    all_results[registration_method] = result

    forward_transforms = result["forward_transforms"]
    inverse_transforms = result["inverse_transforms"]
    losses = result["losses"]

    print(f"\n{registration_method.upper()} registration complete!")
    print(f"  Average loss: {np.mean(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print(f"  Max loss: {np.max(losses):.6f}")

print("\n" + "=" * 70)
print("All registrations complete!")
print("=" * 70)

# %% [markdown]
# ## Save Results and Reconstruct Time Series
#
# Using the `reconstruct_time_series()` method to apply inverse transforms and reconstruct the time series in the fixed image space.
#
# The method simplifies the reconstruction process by handling the transformation of all moving images at once.

# %%
# Save registered images and transforms for each method
tfm_tools = TransformTools()

for registration_method in registration_methods:
    result = all_results[registration_method]
    forward_transforms = result["forward_transforms"]
    inverse_transforms = result["inverse_transforms"]

    # Get the registrar used for this method
    registrar = RegisterTimeSeriesImages(registration_method=registration_method)
    registrar.set_fixed_image(fixed_image)

    print(f"Saving {registration_method.upper()} results...")

    # Reconstruct time series using the new method (moving to fixed space)
    # This applies the inverse transforms to each moving image
    print("  Reconstructing time series in fixed image space...")
    reconstructed_images = registrar.reconstruct_time_series(
        moving_images=images,
        inverse_transforms=inverse_transforms,
        upsample_to_fixed_resolution=False,
    )

    # Save reconstructed images and inverse transforms
    for i, img_indx in enumerate(files_indx):
        print(f"  Saving slice {img_indx:03d}...")

        # Save reconstructed image (moving to fixed using inverse transform)
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_{registration_method}_reconstructed_{img_indx:03d}.mha",
        )
        itk.imwrite(reconstructed_images[i], out_file, compression=True)

        # Also save forward-transformed images (moving to fixed using forward transform)
        # This shows the moving image aligned to fixed space
        reg_image = tfm_tools.transform_image(
            images[i], forward_transforms[i], fixed_image
        )
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_{registration_method}_forward_{img_indx:03d}.mha",
        )
        itk.imwrite(reg_image, out_file, compression=True)

        # Apply inverse transform and save (fixed to moving)
        reg_image_inv = tfm_tools.transform_image(
            fixed_image, inverse_transforms[i], images[i]
        )
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_fixed_{registration_method}_inverse_{img_indx:03d}.mha",
        )
        itk.imwrite(reg_image_inv, out_file, compression=True)

        # Save transforms
        itk.transformwrite(
            forward_transforms[i],
            os.path.join(
                _RESULTS_DIR,
                f"slice_{registration_method}_forward_{img_indx:03d}.hdf",
            ),
            compression=True,
        )
        itk.transformwrite(
            inverse_transforms[i],
            os.path.join(
                _RESULTS_DIR,
                f"slice_{registration_method}_inverse_{img_indx:03d}.hdf",
            ),
            compression=True,
        )

print("✓ Results saved to results/ directory")

# %% [markdown]
# ## Reconstruct Time Series with Upsampling
#
# The `reconstruct_time_series()` method provides an optional upsampling feature. When `upsample_to_fixed_resolution=True`, each reconstructed time point:
# - Maintains its original **origin** and **direction** (coordinate system)
# - Uses **isotropic spacing** calculated as the mean of the fixed image's X and Y spacing
#
# This is useful when you want higher resolution reconstructed images with isotropic voxels while preserving the spatial positioning of each time point.

# %%
# Optional: Reconstruct time series with upsampling to fixed image resolution
# This demonstrates the upsampling feature where each time point maintains
# its original origin and direction but uses the fixed image's spacing/resolution

print("\n" + "=" * 70)
print("Reconstructing time series with upsampling to fixed resolution")
print("=" * 70)

for registration_method in registration_methods:
    result = all_results[registration_method]
    inverse_transforms = result["inverse_transforms"]

    # Get the registrar used for this method
    registrar = RegisterTimeSeriesImages(registration_method=registration_method)
    registrar.set_fixed_image(fixed_image)

    print(f"\n{registration_method.upper()}: Reconstructing with upsampling...")

    # Reconstruct with upsampling enabled
    upsampled_images = registrar.reconstruct_time_series(
        moving_images=images,
        inverse_transforms=inverse_transforms,
        upsample_to_fixed_resolution=True,
    )

    # Save upsampled reconstructed images
    for i, img_indx in enumerate(files_indx):
        out_file = os.path.join(
            _RESULTS_DIR,
            f"slice_{registration_method}_upsampled_{img_indx:03d}.mha",
        )
        itk.imwrite(upsampled_images[i], out_file, compression=True)

        # Print comparison of image properties
        if i == 0:  # Only print for first image
            print(f"\n  Image comparison for slice {img_indx:03d}:")
            print("    Original moving image:")
            print(f"      Size: {itk.size(images[i])}")
            print(f"      Spacing: {itk.spacing(images[i])}")
            print("    Fixed image:")
            print(f"      Size: {itk.size(fixed_image)}")
            print(f"      Spacing: {itk.spacing(fixed_image)}")
            print("    Upsampled reconstructed image:")
            print(f"      Size: {itk.size(upsampled_images[i])}")
            print(f"      Spacing: {itk.spacing(upsampled_images[i])}")

print("\n✓ Upsampled reconstructed images saved to results/ directory")

# %%
# Print registration losses for each method
for registration_method in registration_methods:
    result = all_results[registration_method]
    losses = result["losses"]

    print(f"{registration_method.upper()} Registration Losses:")
    print("=" * 50)
    for i, img_indx in enumerate(files_indx):
        status = "(reference)" if i == reference_image_num else ""
        print(f"  Slice {img_indx:03d}: {losses[i]:.6f} {status}")

    print(f"{registration_method.upper()} Statistics:")
    print(f"  Mean loss: {np.mean(losses):.6f}")
    print(f"  Std loss: {np.std(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print(f"  Max loss: {np.max(losses):.6f}")

# %% [markdown]
# ## Visualize Registration Quality
#

# %%
# Generate grid image for visualization
grid_image = tfm_tools.generate_grid_image(fixed_image, 30, 1)

for registration_method in registration_methods:
    result = all_results[registration_method]
    inverse_transforms = result["inverse_transforms"]

    print(f"Generating {registration_method.upper()} grid visualizations...")
    for i, img_indx in enumerate(files_indx):
        print(f"  Generating grid for slice {img_indx:03d}...")

        # Transform grid with inverse transform (FM)
        inverse_grid_image = tfm_tools.transform_image(
            grid_image,
            inverse_transforms[i],
            fixed_image,
        )
        itk.imwrite(
            inverse_grid_image,
            os.path.join(
                _RESULTS_DIR,
                f"slice_fixed_{registration_method}_inverse_grid_{img_indx:03d}.mha",
            ),
            compression=True,
        )

        # Save displacement field as image
        inverse_transform_image = tfm_tools.convert_transform_to_displacement_field(
            inverse_transforms[i],
            fixed_image,
            np_component_type=np.float32,
        )
        itk.imwrite(
            inverse_transform_image,
            os.path.join(
                _RESULTS_DIR,
                f"slice_{registration_method}_inverse_{img_indx:03d}_field.mha",
            ),
            compression=True,
        )

print("✓ Grid visualizations saved")
