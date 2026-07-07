# PhysioTwin4D

**A collection of methods, workflows, tutorials, and CLI tools for creating personalized physiological digital twins.**

PhysioTwin4D typically begins with a 3D medical image of a subject, extracts anatomic models from that image, and then uses AI surrogates to estimate the subject's physiological processes — initially focusing on cardiac and respiratory motion, and expanding to electrophysiology, blood flow, and organ perfusion. The package provides methods for forming these physiological AI surrogates and for fine-tuning the segmentation and registration AI methods that power them, with special emphasis on statistical shape models: they capture subject-specific characteristics that help determine subject-specific physiological function, and establish correspondence across subjects to aid AI surrogate generalization and simplify the application of traditional solvers.

> **Not validated for clinical use.** PhysioTwin4D is a research toolkit. It
> is not a medical device and must not be used for diagnosis, treatment
> planning, or clinical decision-making.

## Documentation

**https://project-monai.github.io/physiotwin4d/** is the primary entry point
for users and contributors. Key sections:

- [Installation](https://project-monai.github.io/physiotwin4d/installation.html) and [Quickstart](https://project-monai.github.io/physiotwin4d/quickstart.html)
- [Tutorials](https://project-monai.github.io/physiotwin4d/tutorials.html) — runnable end-to-end workflows and their datasets
- [CLI & Scripts Guide](https://project-monai.github.io/physiotwin4d/cli_scripts/overview.html) — command-line tools for conversion, segmentation, registration, and USD workflows
- [API Reference](https://project-monai.github.io/physiotwin4d/api/index.html) — workflow, registration, segmentation, and USD classes
- [Developer Guides](https://project-monai.github.io/physiotwin4d/developer/architecture.html) — architecture, extension points, and implementation conventions
- [Contributing](https://project-monai.github.io/physiotwin4d/contributing.html) and [Testing](https://project-monai.github.io/physiotwin4d/testing.html)
- [FAQ](https://project-monai.github.io/physiotwin4d/faq.html) and [Troubleshooting](https://project-monai.github.io/physiotwin4d/troubleshooting.html)

## Highlights

- **Personalized digital twins**: build subject-specific anatomic models and physiological AI surrogates from 3D/4D medical images
- **Statistical shape models**: capture subject-specific anatomy and establish cross-subject correspondence, aiding AI surrogate generalization and simplifying traditional solver setup
- **Simplified workflows on industry-leading open-source tools**: ICON and Greedy for registration; MONAI with TotalSegmentator and Simpleware for segmentation; Netgen for meshing (LGPL license); scikit-learn for statistical shape modeling; ITK for image processing; PyVista and OpenUSD/Omniverse for geometry manipulation; CuPy for accelerated computing; and PhysicsNeMo for AI surrogates
- **Extensible class hierarchy**: add new segmentation and registration methods, and extend to new data types, organs, and physiological processes, without reworking the workflow layer
- **Physiological motion**: cardiac and respiratory motion today, expanding to electrophysiology, blood flow, and organ perfusion
- **NVIDIA Omniverse as the simulation hub**: the end goal for simulation — a simulation-information hub and gateway to other engines (e.g., Ansys solvers), interactive simulations for treatment planning (e.g., Isaac Sim, Newton), visualization systems (e.g., AR/VR devices), and physical systems (e.g., robots via ROS)
- **CLI and Python API**: installed command-line tools and workflow classes for repeatable, scriptable pipelines

## Installation

```bash
# CPU-only — works out of the box; a runtime warning points to the GPU extra
pip install physiotwin4d

# CUDA 13 (recommended for production)
uv pip install "physiotwin4d[cuda13]"
```

See the [installation guide](https://project-monai.github.io/physiotwin4d/installation.html) for GPU setup, source installs, and optional extras (PhysicsNeMo).

## Quick Start

```bash
physiotwin4d-convert-image-to-usd cardiac_4d.nrrd --contrast --output-dir ./results
```

```python
from physiotwin4d import RegisterImagesICON, WorkflowConvertImageToUSD

processor = WorkflowConvertImageToUSD(
    input_filenames=["path/to/cardiac_4d_ct.nrrd"],
    output_directory="./results",
    project_name="cardiac_model",
    registration_method=RegisterImagesICON(),  # or RegisterImagesGreedy()
)
final_usd = processor.process()
```

See the [quickstart](https://project-monai.github.io/physiotwin4d/quickstart.html) and [tutorials](https://project-monai.github.io/physiotwin4d/tutorials.html) for full walkthroughs covering segmentation, registration, statistical shape modeling, and USD export.

## Contributing

See the [contributing guide](https://project-monai.github.io/physiotwin4d/contributing.html) for code style, testing, IDE setup, and pull request conventions.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

Note that some optional dependencies carry their own license terms: meshing
via Netgen is LGPL-licensed, and NVIDIA Omniverse is distributed under its
own customer license, which is free for academic and commercial use. To our
knowledge, none of the dependencies carry commercially prohibitive licenses
such as GPL, but no guarantees are provided — review the license of each
dependency for your own use case.
