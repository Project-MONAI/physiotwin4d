# Duke-Heart-4DLabelmaps

Gated 4D cardiac labelmaps acquired at Duke University by Dr. Paul Segars.

## Availability

This dataset is **not currently available**. It is being considered for public
release; until that happens it cannot be downloaded, and it is not distributed
with this repository.

## Effect on the tutorials

Tutorials that depend on this dataset are named with a `duke_heart` prefix in
their organ field, for example:

- `tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py`

These `duke_heart` tutorials will not run without the data. Every other
tutorial uses a publicly available dataset and is unaffected — see
[../README.md](../README.md) for download instructions.

Downstream tutorials that consume `duke_heart` outputs (such as the finetuned
distance-map ICON weights used by
`tutorials/tutorial_07_heart_fit_statistical_model_to_patient.py`) fall back to
stock weights and still run, with reduced accuracy.

## Expected layout

When available, the data is expected under `data/Duke-Heart-4DLabelmaps/` as
one directory per case (`pm0002/`, `pm0003/`, ...), each holding one labelmap
per gated frame.
