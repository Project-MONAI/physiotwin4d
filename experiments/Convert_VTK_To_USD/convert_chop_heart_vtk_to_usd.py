#!/usr/bin/env python
# %%
from pathlib import Path


from physiotwin4d.workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD

_HERE = Path(__file__).parent

# %%
vtknames = ["a", "la", "lca", "lv", "myo", "pa", "ra", "rv"]
usdnames = [
    "Aorta",
    "LeftAtrium",
    "LeftCoronaryArtery",
    "LeftVentricle",
    "Myocardium",
    "PulmonaryArtery",
    "RightAtrium",
    "RightVentricle",
]

input_dir = _HERE / "../../data/CHOP-Valve4D/CT/Simpleware/parts"
output_dir = _HERE / "results" / "heart"

output_dir.mkdir(parents=True, exist_ok=True)

all_files = []
for vtkname, usdname in zip(vtknames, usdnames):
    out_usd = Path.absolute(output_dir / f"RVOT28-Dias-{usdname}.usd")
    if out_usd.exists():
        out_usd.unlink()

    in_name = input_dir / f"{vtkname}.vtk"
    all_files.append(in_name)

    converter = WorkflowConvertVTKToUSD(
        vtk_files=[in_name],
        output_usd=out_usd,
        separate_by_connectivity=False,
        separate_by_cell_type=False,
        mesh_name=f"RVOT28Dias_{usdname}",
        appearance="anatomy",
        anatomy_type="heart",
    )
    converter.run()

converter = WorkflowConvertVTKToUSD(
    vtk_files=all_files,
    output_usd=Path.absolute(output_dir / Path("RVOT28-Dias-WholeHeart.usd")),
    separate_by_connectivity=False,
    separate_by_cell_type=False,
    mesh_name="RVOT28Dias_WholeHeart",
    appearance="anatomy",
    anatomy_type="heart",
)
converter.run()
