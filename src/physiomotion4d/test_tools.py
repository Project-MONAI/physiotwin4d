"""
Test utilities for comparing images in pytest.

Provides TestTools for baseline vs results comparison with configurable
tolerances. All image I/O uses ITK with .mha (compressed); 3D images are
passed as itk.Image at the API level.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

import itk
import numpy as np

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase

# Repo root: src/physiomotion4d/test_tools.py -> parent.parent.parent
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Set by tests/conftest.py from pytest --create-baselines; applies to entire run
_create_baseline_if_missing = False


def set_create_baseline_if_missing(value: bool) -> None:
    """Set whether to create baseline files when missing (used by pytest conftest)."""
    global _create_baseline_if_missing
    _create_baseline_if_missing = value


class TestTools(PhysioMotion4DBase):
    """
    Utilities for pytest image comparison: baseline directory, result directory,
    and comparison with configurable tolerances. Inherits from PhysioMotion4DBase
    for logging. All image I/O uses ITK with compression where supported.
    """

    # Prevent pytest from collecting this as a test class
    __test__ = False

    def __init__(
        self,
        results_dir: Path,
        baselines_dir: Path,
        class_name: str,
        *,
        results_output_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize test helpers.

        Args:
            results_dir: Root directory for result artifacts.
            baselines_dir: Root directory for baseline artifacts.
            class_name: Subdirectory name for baselines and, by default, results.
            results_output_dir: Optional exact directory for result artifacts.
                When set, results are read and written there instead of
                ``results_dir / class_name`` while baselines still use
                ``baselines_dir / class_name``.
            log_level: Logging level.
        """
        super().__init__(class_name=class_name, log_level=log_level)

        self._results_dir = (
            results_output_dir
            if results_output_dir is not None
            else results_dir / class_name
        )
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._baselines_dir = baselines_dir / class_name
        self._baselines_dir.mkdir(parents=True, exist_ok=True)

        self._last_image_per_pixel_absolute_error_tol: float | None = None
        self._last_image_number_of_pixels_above_tol: int | None = None
        self._last_image_max_number_of_pixels_above_tol: int | None = None
        self._last_image_total_absolute_error: float | None = None
        self._last_image_total_absolute_error_tol: float | None = None
        self._last_image_difference_image: Any = (
            None  # itk.Image (type depends on template)
        )

        self._last_transform_per_value_absolute_error_tol: float | None = None
        self._last_transform_number_of_values_above_tol: int | None = None
        self._last_transform_max_number_of_values_above_tol: int | None = None
        self._last_transform_total_absolute_error: float | None = None
        self._last_transform_total_absolute_error_tol: float | None = None
        self._last_transform_difference_transform: Any = (
            None  # itk.Transform (type depends on template)
        )

    def image_pass_fail_and_pixels_above_tolerance(self) -> tuple[bool, int]:
        """
        Return (pass, value) for number of pixels above tolerance from the most
        recent compare_result_to_baseline_image call.
        pass is True if value <= max_pixels_above_tol that was used in that call.
        """
        if (
            self._last_image_number_of_pixels_above_tol is None
            or self._last_image_max_number_of_pixels_above_tol is None
        ):
            raise RuntimeError("No previous compare_result_to_baseline_image call")
        val = self._last_image_number_of_pixels_above_tol
        passed = val <= self._last_image_max_number_of_pixels_above_tol
        return (passed, val)

    def image_pass_fail_and_total_absolute_error(self) -> tuple[bool, float]:
        """
        Return (pass, value) for total absolute error from the most recent
        compare_result_to_baseline_image call.
        pass is True if value <= total_absolute_error_tol that was used in that call.
        """
        if (
            self._last_image_total_absolute_error is None
            or self._last_image_total_absolute_error_tol is None
        ):
            raise RuntimeError("No previous compare_result_to_baseline_image call")
        val = self._last_image_total_absolute_error
        passed = val <= self._last_image_total_absolute_error_tol
        return (passed, val)

    def image_difference(self) -> Any:
        """Return the difference image (itk.Image) from the most recent
        compare_result_to_baseline_image call."""
        if self._last_image_difference_image is None:
            raise RuntimeError("No previous compare_result_to_baseline_image call")
        return self._last_image_difference_image

    def transform_pass_fail_and_number_of_values_above_tolerance(
        self,
    ) -> tuple[bool, int]:
        """
        Return (pass, value) for number of values above tolerance from the most recent compare_result_to_baseline_transform call.
        pass is True if value <= max_number_of_values_above_tol that was used in that call.
        """
        if (
            self._last_transform_number_of_values_above_tol is None
            or self._last_transform_max_number_of_values_above_tol is None
        ):
            raise RuntimeError("No previous compare_result_to_baseline_transform call")
        val = self._last_transform_number_of_values_above_tol
        passed = val <= self._last_transform_max_number_of_values_above_tol
        return (passed, val)

    def transform_pass_fail_and_total_absolute_error(self) -> tuple[bool, float]:
        """
        Return (pass, value) for total absolute error from the most recent compare_result_to_baseline_transform call.
        pass is True if value <= total_absolute_error_tol that was used in that call.
        """
        if (
            self._last_transform_total_absolute_error is None
            or self._last_transform_total_absolute_error_tol is None
        ):
            raise RuntimeError("No previous compare_result_to_baseline_transform call")
        val = self._last_transform_total_absolute_error
        passed = val <= self._last_transform_total_absolute_error_tol
        return (passed, val)

    def transform_difference(self) -> Any:
        """Return the difference transform (itk.Transform) from the most recent compare_result_to_baseline_transform call."""
        if self._last_transform_difference_transform is None:
            raise RuntimeError("No previous compare_result_to_baseline_transform call")
        return self._last_transform_difference_transform

    def write_result_image(self, image: Any, filename: str) -> None:
        """Write the image to the configured result artifact directory."""
        itk.imwrite(image, str(self._results_dir / filename), compression=True)

    def write_result_transform(self, transform: Any, filename: str) -> None:
        """Write the transform to the configured result artifact directory."""
        itk.transformwrite(
            transform, str(self._results_dir / filename), compression=True
        )

    def compare_result_to_baseline_transform(
        self,
        filename: str,
        *,
        per_value_absolute_error_tol: float = 0.0,
        max_number_of_values_above_tol: int = 0,
        total_absolute_error_tol: float = 0.0,
    ) -> bool:
        """Compare the transform to the baseline transform."""
        results_path = Path(self._results_dir / filename)
        baseline_path = Path(self._baselines_dir / filename)
        if not results_path.exists():
            raise FileNotFoundError(f"Results transform not found: {results_path}")
        if not baseline_path.exists():
            if not _create_baseline_if_missing:
                self.log_error(
                    "Baseline transform missing: %s (run pytest with --create-baselines to create from current output)",
                    baseline_path,
                )
                return False
            shutil.copy(str(results_path), str(baseline_path))
            self.log_warning(
                "Baseline transform did not exist; copied results transform: %s",
                results_path,
            )
        transform = itk.transformread(str(results_path))
        transform_params = np.array(transform[0].GetParameters())

        baseline_transform = itk.transformread(str(baseline_path))
        baseline_transform_params = np.array(baseline_transform[0].GetParameters())

        diff = transform_params - baseline_transform_params
        absolute_err = np.abs(diff)
        number_of_values_above_tol = int(
            np.sum(absolute_err > per_value_absolute_error_tol)
        )
        total_absolute_error = float(np.sum(absolute_err))

        difference_transform = baseline_transform[0]
        difference_transform_params = baseline_transform[0].GetParameters()
        for i in range(len(difference_transform_params)):
            difference_transform_params[i] = diff[i]
        difference_transform.SetParameters(difference_transform_params)

        self._last_transform_difference_transform = difference_transform
        self._last_transform_per_value_absolute_error_tol = per_value_absolute_error_tol
        self._last_transform_number_of_values_above_tol = number_of_values_above_tol
        self._last_transform_max_number_of_values_above_tol = (
            max_number_of_values_above_tol
        )
        self._last_transform_total_absolute_error = total_absolute_error
        self._last_transform_total_absolute_error_tol = total_absolute_error_tol

        passed = (
            self._last_transform_number_of_values_above_tol
            <= self._last_transform_max_number_of_values_above_tol
            and self._last_transform_total_absolute_error
            <= self._last_transform_total_absolute_error_tol
        )

        if passed:
            self.log_info(
                "PASS: number_of_values_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                self._last_transform_number_of_values_above_tol,
                self._last_transform_max_number_of_values_above_tol,
                self._last_transform_total_absolute_error,
                self._last_transform_total_absolute_error_tol,
            )
        else:
            self.log_error(
                "FAIL: number_of_values_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                self._last_transform_number_of_values_above_tol,
                self._last_transform_max_number_of_values_above_tol,
                self._last_transform_total_absolute_error,
                self._last_transform_total_absolute_error_tol,
            )
        return passed

    def compare_result_to_baseline_image(
        self,
        filename: str,
        *,
        per_pixel_absolute_error_tol: float = 0.0,
        max_number_of_pixels_above_tol: int = 0,
        total_absolute_error_tol: float = 0.0,
    ) -> bool:
        """
        Load a 3D result image and a 3D baseline image (.mha), compare the full
        volumes voxel-by-voxel, save the difference image with "_diff" in the
        name on failure, and log pass/fail.

        If the baseline file does not exist and --create-baselines was given,
        the 3D result is copied as the new baseline.

        Returns True if comparison passed (pixels and total absolute error
        within tolerance).

        Args:
            filename: File name (relative to class results/baselines dirs).
            per_pixel_absolute_error_tol: Per-voxel absolute error threshold.
            max_number_of_pixels_above_tol: Max allowed voxels exceeding threshold.
            total_absolute_error_tol: Max allowed sum of absolute differences.
        """
        results_path = Path(self._results_dir / filename)
        baseline_path = Path(self._baselines_dir / filename)
        if not results_path.exists():
            raise FileNotFoundError(f"Results image not found: {results_path}")

        image_result = itk.imread(str(results_path))

        if not baseline_path.exists():
            if not _create_baseline_if_missing:
                self.log_error(
                    "Baseline image missing: %s (run pytest with --create-baselines to create from current output)",
                    baseline_path,
                )
                return False
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(results_path), str(baseline_path))
            self.log_warning(
                "Baseline file did not exist; copied 3D result: %s",
                baseline_path,
            )
            image_baseline = image_result
        else:
            image_baseline = itk.imread(str(baseline_path))

        arr_result = np.asarray(itk.array_from_image(image_result), dtype=np.float64)
        arr_baseline = np.asarray(
            itk.array_from_image(image_baseline), dtype=np.float64
        )

        if arr_result.shape != arr_baseline.shape:
            raise ValueError(
                f"Shape mismatch: result {arr_result.shape} vs baseline {arr_baseline.shape}"
            )

        diff_magnitude = np.abs(arr_result - arr_baseline)
        total_absolute_error = float(np.sum(diff_magnitude))
        number_of_pixels_above_tol = int(
            np.sum(diff_magnitude > per_pixel_absolute_error_tol)
        )

        self._last_image_per_pixel_absolute_error_tol = per_pixel_absolute_error_tol
        self._last_image_number_of_pixels_above_tol = number_of_pixels_above_tol
        self._last_image_max_number_of_pixels_above_tol = max_number_of_pixels_above_tol
        self._last_image_total_absolute_error = total_absolute_error
        self._last_image_total_absolute_error_tol = total_absolute_error_tol
        self._last_image_difference_image = itk.image_from_array(
            diff_magnitude.astype(np.float64)
        )

        passed = (
            number_of_pixels_above_tol <= max_number_of_pixels_above_tol
            and total_absolute_error <= total_absolute_error_tol
        )

        if not passed:
            stem = Path(filename).stem
            if "." in stem:
                stem = stem.split(".")[0]
            diff_path = self._results_dir / (stem + "_diff.mha")
            itk.imwrite(
                self._last_image_difference_image, str(diff_path), compression=True
            )

        if passed:
            self.log_info(
                "PASS: number_of_pixels_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                number_of_pixels_above_tol,
                max_number_of_pixels_above_tol,
                total_absolute_error,
                total_absolute_error_tol,
            )
        else:
            self.log_error(
                "FAIL: number_of_pixels_above_tol=%d (max=%d), total_absolute_error=%.6g (tol=%.6g)",
                number_of_pixels_above_tol,
                max_number_of_pixels_above_tol,
                total_absolute_error,
                total_absolute_error_tol,
            )

        return passed

    def save_screenshot_mesh(
        self,
        mesh: Any,  # pv.PolyData
        filename: str,
        *,
        camera_position: Literal["xy", "xz", "yz", "yx", "zx", "zy", "iso"] = "iso",
        window_size: tuple[int, int] = (800, 600),
        color: str = "pink",
        opacity: float = 0.9,
    ) -> Path:
        """Render a PyVista mesh off-screen and save a PNG.

        Saves to the configured result artifact directory. On Linux headless
        environments, calls pv.start_xvfb() before rendering (no-op when a
        display is present).

        Args:
            mesh: PyVista PolyData or compatible mesh object.
            filename: Output PNG filename, relative to the result artifact dir.
            camera_position: PyVista camera preset, e.g. ``'iso'``, ``'xy'``, ``'xz'``.
            window_size: Off-screen render size ``(width, height)`` in pixels.
            color: Mesh color string accepted by PyVista.
            opacity: Mesh opacity in [0, 1].

        Returns:
            Absolute Path to the saved PNG.
        """
        import pyvista as pv

        xvfb_started = False
        try:
            pv.start_xvfb()
            xvfb_started = True
        except Exception:
            pass

        output_path = self._results_dir / filename
        plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        try:
            plotter.add_mesh(mesh, color=color, opacity=opacity)
            plotter.camera_position = camera_position
            plotter.screenshot(str(output_path))
        finally:
            plotter.close()
            if xvfb_started and hasattr(pv, "stop_xvfb"):
                pv.stop_xvfb()
        self.log_info("Screenshot saved: %s", output_path)
        return output_path

    def save_screenshot_image_slice(
        self,
        image: Any,  # itk.Image, axes X Y Z in RAS world space
        filename: str,
        *,
        axis: int = 0,
        slice_fraction: float = 0.5,
        colormap: str = "gray",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        overlay_mask: Optional[Any] = None,  # itk.Image same spatial extent
        overlay_alpha: float = 0.4,
    ) -> Path:
        """Extract one slice from an ITK image and save a PNG via matplotlib.

        The numpy array from ``itk.array_view_from_image`` has shape ``(Z, Y, X)``
        (ITK stores X fastest; numpy reverses the axis order). Axis indices:
        - axis=0: axial (constant-Z plane)
        - axis=1: coronal (constant-Y plane)
        - axis=2: sagittal (constant-X plane)

        Saves to the configured result artifact directory.

        Args:
            image: 3-D ``itk.Image`` in RAS world space, axes X Y Z.
            filename: Output PNG filename, relative to the result artifact dir.
            axis: Numpy axis along which to slice (0=axial, 1=coronal, 2=sagittal).
            slice_fraction: Fractional position along ``axis`` in [0, 1].
            colormap: Matplotlib colormap name for the base image.
            vmin: Lower clamp for display; None means data minimum.
            vmax: Upper clamp for display; None means data maximum.
            overlay_mask: Optional binary ITK mask rendered as a semi-transparent
                overlay. Must have the same spatial extent as ``image``.
            overlay_alpha: Opacity of the mask overlay in [0, 1].

        Returns:
            Absolute Path to the saved PNG.
        """
        import matplotlib.pyplot as plt

        arr = np.asarray(itk.array_view_from_image(image), dtype=np.float64)
        idx = int(arr.shape[axis] * slice_fraction)
        idx = max(0, min(idx, arr.shape[axis] - 1))

        slices: list[Any] = [slice(None)] * arr.ndim
        slices[axis] = idx
        slice_data = arr[tuple(slices)]

        output_path = self._results_dir / filename
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            ax.imshow(slice_data, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower")

            if overlay_mask is not None:
                mask_arr = np.asarray(
                    itk.array_view_from_image(overlay_mask), dtype=np.float64
                )
                mask_slice = mask_arr[tuple(slices)]
                ax.imshow(
                    np.ma.masked_where(mask_slice == 0, mask_slice),
                    cmap="autumn",
                    alpha=overlay_alpha,
                    origin="lower",
                )

            ax.axis("off")
            fig.savefig(str(output_path), bbox_inches="tight", dpi=100)
        finally:
            plt.close(fig)
        self.log_info("Screenshot saved: %s", output_path)
        return output_path
