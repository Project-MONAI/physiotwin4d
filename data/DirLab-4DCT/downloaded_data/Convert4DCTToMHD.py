# %%
import os

# %%
sizes = [
    [256, 256, 94],
    [256, 256, 112],
    [256, 256, 104],
    [256, 256, 99],
    [256, 256, 106],
    [512, 512, 128],
    [512, 512, 136],
    [512, 512, 128],
    [512, 512, 128],
    [512, 512, 120],
]
spacings = [
    [0.97, 0.97, 2.5],
    [1.16, 1.16, 2.5],
    [1.15, 1.15, 2.5],
    [1.13, 1.13, 2.5],
    [1.10, 1.10, 2.5],
    [0.97, 0.97, 2.5],
    [0.97, 0.97, 2.5],
    [0.97, 0.97, 2.5],
    [0.97, 0.97, 2.5],
    [0.97, 0.97, 2.5],
]
files = [
    "Case1Pack",
    "Case2Pack",
    "Case3Pack",
    "Case4Pack",
    "Case5Pack",
    "Case6Pack",
    "Case7Pack",
    "Case8Deploy",
    "Case9Pack",
    "Case10Pack",
]
file_suffix = [
    "_s",
    "-ssm",
    "-ssm",
    "-ssm",
    "-ssm",
    "",
    "",
    "",
    "",
    "",
]

# %%
for file_num, file_name in enumerate(files):
    spacing = f"{spacings[file_num][0]} {spacings[file_num][1]} {spacings[file_num][2]}"
    size = f"{sizes[file_num][0]} {sizes[file_num][1]} {sizes[file_num][2]}"
    for phase in range(10):
        img_path = f"{file_name}/Images/case{file_num + 1}_T{phase * 10:02d}{file_suffix[file_num]}.img"
        script_dir = os.getcwd()
        hdr_path = f"{script_dir}/{file_name}_T{phase * 10:02d}.mhd"
        with open(hdr_path, "w") as f:
            f.write("ObjectType = Image\n")
            f.write("NDims = 3\n")
            f.write(f"DimSize = {size}\n")
            f.write("HeaderSize = -1\n")
            f.write("BinaryData = True\n")
            f.write("BinaryDataByteOrderMSB = False\n")
            f.write("Orientation = 1 0 0 0 1 0 0 0 -1\n")
            f.write(f"ElementSpacing = {spacing}\n")
            f.write("ElementType = MET_SHORT\n")
            f.write("ElementNumberOfChannels = 1\n")
            f.write(f"ElementDataFile = {img_path}\n")
