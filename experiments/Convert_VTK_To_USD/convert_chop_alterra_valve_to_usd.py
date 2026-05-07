#!/usr/bin/env python
# %% [markdown]
# # Cardiac Valve 4D Time-Series Conversion to USD
#
# This notebook demonstrates converting time-varying cardiac valve simulation data from VTK format to animated USD.
#
# ## Dataset: CHOP-Valve4D (Alterra)
#
# One cardiac valve model with time-varying geometry:
#
# - **Alterra**: 265 time steps (cardiac cycle simulation)
#
# This dataset represents 4D (3D + time) simulation of a prosthetic heart valve during a cardiac cycle.
#
# ## Goals
#
# 1. Load and inspect time-varying VTK data
# 2. Convert entire time series to animated USD
# 3. Handle large datasets efficiently
# 4. Preserve all simulation data as USD primvars
# 5. Create multiple variations (full resolution, subsampled, etc.)

# %% [markdown]
# ## Configuration
#
# Control which time series conversions to compute.

# %%
import re
from pathlib import Path

# Use as a test
from physiomotion4d.notebook_utils import running_as_test

# Import USDTools for post-processing colormap
from physiomotion4d.usd_tools import USDTools

from physiomotion4d import ConvertVTKToUSD

# %% [markdown]
# ## 1. Discover and Organize Time-Series Files

# %%
# Set to True to use as a test.  Automatically done by
#    running_as_test() helper function.
quick_run = running_as_test()
quick_run_step = 4

# Define data directories (Alterra only)
data_dir = Path.cwd().parent.parent / "data" / "CHOP-Valve4D"
alterra_dir = data_dir / "Alterra"

output_dir = Path.cwd() / "results" / "valve4d-alterra"
if quick_run:
    output_usd = output_dir / "alterra_quick.usd"
else:
    output_usd = output_dir / "alterra_full.usd"

colormap_primvar_substrs = ["stress", "strain"]
colormap_name = "jet"  # matplotlib colormap name
colormap_range_min = 25
colormap_range_max = 200

# Conversion parameters
separate_by = "connectivity"  # Essential for alterra vtk file
times_per_second = 60.0
solid_color = (0.5, 0.5, 0.5)

# %%
output_dir.mkdir(parents=True, exist_ok=True)

vtk_files = list(Path(alterra_dir).glob("*.vtk"))
pattern = r"\.t(\d+)\.vtk$"

# Extract time step numbers and pair with files
alterra_series = []
for vtk_file in vtk_files:
    match = re.search(pattern, vtk_file.name)
    if match:
        time_step = int(match.group(1))
        alterra_series.append((time_step, vtk_file))

# Sort by time step
alterra_series.sort(key=lambda x: x[0])

# %% [markdown]
# ## 2. Inspect First Frame
#
# Examine the first time step to understand the data structure.

# %%
# Debugging
first_file = alterra_series[0][1]
mesh_info = ConvertVTKToUSD.inspect_file(first_file, extract_surface=True)

print(f"\nFile: {first_file.name}")
print("\nGeometry:")
print(f"  Points: {mesh_info['points']:,}")
print(f"  Faces: {mesh_info['faces']:,}")
print(f"  Normals: {'Yes' if mesh_info['has_normals'] else 'No'}")
print(f"  Colors: {'Yes' if mesh_info['has_colors'] else 'No'}")

# Bounding box
bbox_min = mesh_info["bounds_min"]
bbox_max = mesh_info["bounds_max"]
bbox_size = mesh_info["bounds_size"]
print("\nBounding Box:")
print(f"  Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
print(f"  Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
print(f"  Size: [{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}]")

print(f"\nData Arrays ({len(mesh_info['arrays'])}):")
for i, array in enumerate(mesh_info["arrays"], 1):
    print(f"  {i}. {array['name']}:")
    print(f"     - Type: {array['data_type']}")
    print(f"     - Components: {array['num_components']}")
    print(f"     - Interpolation: {array['interpolation']}")
    print(f"     - Elements: {array['num_elements']:,}")
    data_range = array["range"]
    if data_range[0] is not None and data_range[1] is not None:
        print(f"     - Range: [{data_range[0]:.6f}, {data_range[1]:.6f}]")

# Cell types (face vertex count = triangle, quad, etc.)
print("\nCell types (faces by vertex count):")
for cell_type in mesh_info["cell_types"]:
    print(
        f"  {cell_type['name']} ({cell_type['vertex_count']} vertices): "
        f"{cell_type['num_faces']:,} faces"
    )

# %% [markdown]
# ## 3. Convert TPV25

# %%
alterra_files = [file_path for _, file_path in alterra_series]
alterra_times = [float(time_step) for time_step, _ in alterra_series]

if quick_run:
    alterra_files = alterra_files[::quick_run_step]
    alterra_times = alterra_times[::quick_run_step]

print(f"\nConverting to: {output_usd}")
print(f"Number of time steps: {len(alterra_times)}")
print("\nThis may take several minutes...\n")

# topology validation and conversion happen inside from_files()
stage = ConvertVTKToUSD.from_files(
    data_basename="AlterraValve",
    vtk_files=alterra_files,
    extract_surface=True,
    separate_by=separate_by,
    times_per_second=times_per_second,
    solid_color=solid_color,
    time_codes=alterra_times,
).convert(str(output_usd))

# %%
usd_tools = USDTools()
# ConvertVTKToUSD places prims at /World/{basename}/{part_name}.
# Discover the target prim dynamically so the path stays valid regardless
# of how many connected components or cell types the VTK file produces.
if separate_by == "connectivity":
    mesh_paths = usd_tools.list_mesh_paths_under(
        stage, parent_path="/World/AlterraValve"
    )
    candidates = [
        p for p in mesh_paths if p.split("/")[-1].startswith("AlterraValve_object")
    ]
    vessel_path = candidates[-1] if candidates else "/World/AlterraValve/Mesh"
elif separate_by == "cell_type":
    mesh_paths = usd_tools.list_mesh_paths_under(
        stage, parent_path="/World/AlterraValve"
    )
    triangle_paths = [p for p in mesh_paths if p.split("/")[-1].endswith("_Triangle")]
    vessel_path = triangle_paths[0] if triangle_paths else "/World/AlterraValve/Mesh"
else:
    vessel_path = "/World/AlterraValve/Mesh"

# Select primvar for coloring
primvars = usd_tools.list_mesh_primvars(str(output_usd), vessel_path)
color_primvar = usd_tools.pick_color_primvar(
    primvars, keywords=tuple(colormap_primvar_substrs)
)
print(f"Chosen primvar = {color_primvar}")

if color_primvar:
    print(f"\nApplying colormap to '{color_primvar}' using {colormap_name}")
    usd_tools.apply_colormap_from_primvar(
        str(output_usd),
        vessel_path,
        color_primvar,
        intensity_range=(colormap_range_min, colormap_range_max),
        cmap=colormap_name,
        use_sigmoid_scale=True,
        bind_vertex_color_material=True,
    )
