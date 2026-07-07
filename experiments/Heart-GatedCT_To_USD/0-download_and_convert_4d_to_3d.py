#!/usr/bin/env python
# %%
import shutil
from pathlib import Path

from physiotwin4d.convert_image_4d_to_3d import ConvertImage4DTo3D
from physiotwin4d.data_download_tools import DataDownloadTools

_HERE = Path(__file__).resolve().parent

# %%
data_dir = _HERE.parent.parent / "data" / "Slicer-Heart-CT"
output_dir = _HERE / "results"

data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

input_image_filename = DataDownloadTools.DownloadSlicerHeartCTData(data_dir)

# %%
conv = ConvertImage4DTo3D()
conv.load_image_4d(str(input_image_filename))
conv.save_3d_images(data_dir, "slice")

# Save the mid-stroke slice as the fixed/reference image
shutil.copyfile(data_dir / "slice_007.mha", output_dir / "slice_fixed.mha")
