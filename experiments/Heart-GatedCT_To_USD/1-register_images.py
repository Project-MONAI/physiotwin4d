#!/usr/bin/env python
# %%
from pathlib import Path

import itk

from physiomotion4d.register_images_ants import RegisterImagesANTs
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.test_tools import TestTools
from physiomotion4d.transform_tools import TransformTools

# nnUNetv2 (used by TotalSegmentator) spawns a multiprocessing.Pool. On Windows
# the spawn start method re-imports this script in each child; without the
# __name__ == "__main__" guard around the top-level work, that re-import fires
# segment() again and Python's spawn-cascade detector raises RuntimeError.
if __name__ == "__main__":
    test_mode = TestTools.running_as_test()

    _HERE = Path(__file__).resolve().parent

    # %%
    # Number of cardiac frames and step size; downstream scripts must use the same values.
    N_FRAMES = 21
    FRAME_STEP = 4 if test_mode else 1

    data_dir = _HERE.parent.parent / "data" / "Slicer-Heart-CT"

    output_dir = _HERE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_image_filename = output_dir / "slice_fixed.mha"
    fixed_image = itk.imread(str(fixed_image_filename))

    # %%
    seg = SegmentChestTotalSegmentator()
    seg.contrast_threshold = 500
    result = seg.segment(fixed_image, contrast_enhanced_study=True)
    # %%
    labelmap_mask = result["labelmap"]
    lung_mask = result["lung"]
    heart_mask = result["heart"]
    major_vessels_mask = result["major_vessels"]
    bone_mask = result["bone"]
    other_mask = result["other"]
    contrast_mask = result["contrast"]

    fixed_image_labelmap = labelmap_mask
    itk.imwrite(
        fixed_image_labelmap,
        str(output_dir / "slice_fixed_mask.mha"),
        compression=True,
    )

    heart_arr = itk.GetArrayFromImage(heart_mask)
    contrast_arr = itk.GetArrayFromImage(contrast_mask)
    major_vessels_arr = itk.GetArrayFromImage(major_vessels_mask)
    fixed_image_dynamic_anatomy_mask = itk.GetImageFromArray(
        heart_arr + contrast_arr + major_vessels_arr
    )
    fixed_image_dynamic_anatomy_mask.CopyInformation(fixed_image)
    itk.imwrite(
        fixed_image_dynamic_anatomy_mask,
        str(output_dir / "slice_fixed.dynamic_anatomy_mask.mha"),
        compression=True,
    )

    lung_arr = itk.GetArrayFromImage(lung_mask)
    bone_arr = itk.GetArrayFromImage(bone_mask)
    other_arr = itk.GetArrayFromImage(other_mask)
    fixed_image_static_mask = itk.GetImageFromArray(lung_arr + bone_arr + other_arr)
    fixed_image_static_mask.CopyInformation(fixed_image)
    itk.imwrite(
        fixed_image_static_mask,
        str(output_dir / "slice_fixed.static_anatomy_mask.mha"),
        compression=True,
    )

    # %%
    reg = RegisterImagesANTs()
    reg.set_mask_dilation(5)
    reg.set_number_of_iterations([10, 5, 2])

    # %%
    for i in range(0, N_FRAMES, FRAME_STEP):
        print(f"Processing slice {i:03d}")
        moving_image = itk.imread(str(data_dir / f"slice_{i:03d}.mha"))
        result = seg.segment(moving_image, contrast_enhanced_study=True)
        labelmap_mask = result["labelmap"]
        lung_mask = result["lung"]
        heart_mask = result["heart"]
        major_vessels_mask = result["major_vessels"]
        bone_mask = result["bone"]
        other_mask = result["other"]
        contrast_mask = result["contrast"]
        itk.imwrite(
            labelmap_mask,
            str(output_dir / f"slice_{i:03d}_mask.mha"),
            compression=True,
        )

        # Register the whole image
        reg.set_fixed_image(fixed_image)
        reg.set_fixed_mask(None)
        results = reg.register(moving_image)
        inverse_transform = results["inverse_transform"]
        forward_transform = results["forward_transform"]
        moving_image_reg = TransformTools().transform_image(
            moving_image, forward_transform, fixed_image, "sinc"
        )  # Final resampling with sinc
        itk.imwrite(
            moving_image_reg,
            str(output_dir / f"slice_{i:03d}.reg_all.mha"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(output_dir / f"slice_{i:03d}.reg_all.forward.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [inverse_transform],
            str(output_dir / f"slice_{i:03d}.reg_all.inverse.hdf"),
            compression=True,
        )

        # Register the dynamic anatomy mask
        heart_arr = itk.GetArrayFromImage(heart_mask)
        contrast_arr = itk.GetArrayFromImage(contrast_mask)
        major_vessels_arr = itk.GetArrayFromImage(major_vessels_mask)
        dynamic_anatomy_arr = heart_arr + contrast_arr + major_vessels_arr
        moving_image_dynamic_anatomy_mask = itk.GetImageFromArray(dynamic_anatomy_arr)
        moving_image_dynamic_anatomy_mask.CopyInformation(moving_image)
        reg.set_fixed_image(fixed_image)
        reg.set_fixed_mask(fixed_image_dynamic_anatomy_mask)
        results = reg.register(moving_image, moving_image_dynamic_anatomy_mask)
        inverse_transform = results["inverse_transform"]
        forward_transform = results["forward_transform"]
        moving_image_reg_dynamic_anatomy = TransformTools().transform_image(
            moving_image, forward_transform, fixed_image, "sinc"
        )  # Final resampling with sinc
        itk.imwrite(
            moving_image_dynamic_anatomy_mask,
            str(output_dir / f"slice_{i:03d}.dynamic_anatomy_mask.mha"),
            compression=True,
        )
        itk.imwrite(
            moving_image_reg_dynamic_anatomy,
            str(output_dir / f"slice_{i:03d}.reg_dynamic_anatomy.mha"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(output_dir / f"slice_{i:03d}.reg_dynamic_anatomy.forward.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [inverse_transform],
            str(output_dir / f"slice_{i:03d}.reg_dynamic_anatomy.inverse.hdf"),
            compression=True,
        )

        # Register the static anatomy mask
        lung_arr = itk.GetArrayFromImage(lung_mask)
        bone_arr = itk.GetArrayFromImage(bone_mask)
        other_arr = itk.GetArrayFromImage(other_mask)
        moving_image_static_mask = itk.GetImageFromArray(
            lung_arr + bone_arr + other_arr
        )
        moving_image_static_mask.CopyInformation(moving_image)
        reg.set_fixed_image(fixed_image)
        reg.set_fixed_mask(fixed_image_static_mask)
        results = reg.register(moving_image, moving_image_static_mask)
        inverse_transform = results["inverse_transform"]
        forward_transform = results["forward_transform"]
        moving_image_reg_static = TransformTools().transform_image(
            moving_image, forward_transform, fixed_image, "sinc"
        )  # Final resampling with sinc
        itk.imwrite(
            moving_image_static_mask,
            str(output_dir / f"slice_{i:03d}.static_anatomy_mask.mha"),
            compression=True,
        )
        itk.imwrite(
            moving_image_reg_static,
            str(output_dir / f"slice_{i:03d}.reg_static_anatomy.mha"),
            compression=True,
        )
        itk.transformwrite(
            [forward_transform],
            str(output_dir / f"slice_{i:03d}.reg_static_anatomy.forward.hdf"),
            compression=True,
        )
        itk.transformwrite(
            [inverse_transform],
            str(output_dir / f"slice_{i:03d}.reg_static_anatomy.inverse.hdf"),
            compression=True,
        )
