---
description: Inspect a PhysioTwin4D implementation and its existing tests, propose a real-data-driven test plan with baseline comparisons, then create or update pytest tests. Explains how to run them.
---

Write or update tests for the following in PhysioTwin4D:

$ARGUMENTS

Instructions:
1. Read the implementation file(s) to understand the public interface.
2. Read the existing test file for this module if one exists (e.g. `tests/test_<module>.py`).
3. Propose a test plan: list the behaviors to cover and the inputs each behavior
   needs.
4. **Strongly prefer real (downloaded) test data over synthetic data.** Request
   the session fixtures (`test_directories`, `download_test_data`,
   `test_images`) so the standard test datasets are pulled automatically on
   first use. Real data exercises the production code paths — preprocessing,
   resampling, dtype handling, world-frame metadata — that synthetic toy
   volumes silently bypass. Only fall back to synthetic `itk.Image` or
   `pv.PolyData` inputs when:
     - the behavior under test is a pure unit (e.g. axis arithmetic, dict
       routing) where real data adds no signal, or
     - real data would push the test into a slow / GPU / Simpleware bucket
       that doesn't fit the test's purpose.
   When using synthetic inputs anyway, keep volumes ≤64 voxels per side and
   say so in the docstring.
5. **When a test produces an image or surface as output, compare against a
   baseline** using the `test_tools.py` utilities (e.g. `TestTools`) rather
   than ad-hoc value assertions. Store baselines under `tests/baselines/`
   (Git LFS-tracked). Run with `--create-baselines` to materialize missing
   baselines on first use; afterward, regression compares to the stored
   baseline. This catches drift that hand-written numeric thresholds miss.
6. State image shape and axis order in every test docstring (e.g.
   `"""...image shape: (X, Y, Z, T) = (64, 64, 32, 1), LPS world frame."""`).
7. Mark tests that need a GPU, a slow runtime, or a licensed Simpleware
   install with `@pytest.mark.requires_gpu`, `@pytest.mark.slow`, or
   `@pytest.mark.requires_simpleware` so they fall into the right opt-in
   bucket (`--run-gpu`, `--run-slow`, `--run-simpleware`). Tests that just
   need downloadable data need **no** marker — the fixture chain handles it.
8. Show the exact command to run the new tests, including any opt-in flags
   the markers require. Examples:
   - `py -m pytest tests/test_<module>.py -v`
   - `py -m pytest tests/test_<module>.py -v --run-slow`
   - `py -m pytest tests/test_<module>.py -v --run-gpu --run-slow`
   - `py -m pytest tests/test_<module>.py --create-baselines` (first run, to
     materialize new baselines)
