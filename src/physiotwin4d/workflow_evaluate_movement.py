"""Accuracy of an inferred moving anatomy, per anatomical structure.

:class:`WorkflowEvaluateMovement` scores a
:class:`physiotwin4d.WorkflowInferMovement` against geometry extracted from a
gated image sequence. For every gated time point it carries the reference
frame's labelmap into that time point with the network's own deformation and
compares the result, structure by structure, to the labelmap of the frame that
was actually acquired: volume difference, Dice, and surface RMSE per lung lobe
or per heart chamber.

Going through labelmaps rather than through the model's own surface is what lets
one workflow serve both anatomies. The lung shape model carries its five lobes
as per-cell labels, but the heart model is a single structure -- the whole heart
minus its chamber cavities -- so its chambers exist only in the acquired
labelmaps. Warping those labelmaps scores every structure the acquisition
contains, whether or not the shape model represents it separately.

Everything is measured on one isotropic evaluation grid built around the
reference anatomy, so a case whose gated frames carry different slice pitches is
still scored on a single, stated voxel volume.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import itk
import numpy as np
import pyvista as pv

from . import physicsnemo_tools as pnt
from .contour_tools import ContourTools
from .physiotwin4d_base import PhysioTwin4DBase
from .workflow_infer_movement import WorkflowInferMovement


class WorkflowEvaluateMovement(PhysioTwin4DBase):
    """Score inferred motion per anatomical structure against acquired frames.

    Args:
        movement_workflow: The displacement decoder whose predictions are
            scored.
        label_names: Structures to score, ``{label_id: name}``. Ids the
            reference frame does not contain are dropped with a warning; ids a
            single acquired frame does not contain are skipped for that frame
            alone, since a structure can leave the field of view.
        log_level: Logging level. Default: ``logging.INFO``.
    """

    # Volume-plot series colors, assigned in this order and never cycled: eight
    # hues whose neighbors stay apart under the common color-vision deficiencies.
    _SERIES_COLORS = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    )

    def __init__(
        self,
        movement_workflow: WorkflowInferMovement,
        label_names: dict[int, str],
        log_level: int | str = logging.INFO,
    ) -> None:
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.movement_workflow = movement_workflow
        self.label_names = dict(label_names)
        self.contour_tools = ContourTools(log_level=log_level)

    # ─────────────────────────── Public API ────────────────────────────────
    def process(
        self,
        case_id: str,
        shape_parameters: Path,
        reference_mesh: Path,
        reference_labelmap: itk.Image,
        ground_truth_labelmaps: dict[float, itk.Image],
        output_directory: Path,
        smoothing_sigma_mm: float = 10.0,
        evaluation_spacing_mm: float = 1.0,
        include_dice: bool = True,
    ) -> dict[str, Any]:
        """Score every gated time point of one case.

        Args:
            case_id: Name of the case being scored, recorded in every output.
            shape_parameters: JSON file with the case's PCA coefficient vector.
            reference_mesh: The case's fitted reference-frame SSM surface. The
                predicted displacements are added to its points, and its extent
                defines the evaluation grid.
            reference_labelmap: Labelmap of the reference frame, the anatomy
                carried into every other time point.
            ground_truth_labelmaps: Acquired labelmap per stage, keyed by the
                normalized stage in ``[0, 1]``.
            output_directory: Directory the report, the CSV and the per-stage
                geometry are written to.
            smoothing_sigma_mm: Gaussian sigma, in millimeters, that turns the
                network's surface-shell deformation into a continuous field.
            evaluation_spacing_mm: Isotropic pitch every metric is measured on.
                It sets both the voxel volume the Dice and volume figures are
                quantized to and the resolution of the deformation fields, whose
                memory grows with its cube.
            include_dice: Report the Dice overlap. Turn it off for a structure
                whose motion is small against its own size: Dice is an overlap
                fraction, so a lung lobe scores over 0.96 undeformed and the
                column says more about the organ's bulk than about the motion.
                The volume and surface figures still resolve it.

        Returns:
            Dict with ``rows`` (every metric row), ``csv_file``,
            ``report_file``, ``volume_plot_file``, ``predicted_surfaces`` and
            ``warped_labelmaps``.

        Raises:
            ValueError: If ``ground_truth_labelmaps`` is empty, or none of the
                requested labels are in the reference frame.
        """
        if not ground_truth_labelmaps:
            raise ValueError("No ground-truth labelmaps to evaluate against.")

        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        stages = sorted(ground_truth_labelmaps)
        self.log_section("EVALUATE MOVEMENT [%s]: %d stages", case_id, len(stages))

        grid = self.contour_tools.create_reference_image(
            mesh=cast(pv.DataSet, pv.read(str(reference_mesh))),
            spatial_resolution=evaluation_spacing_mm,
            buffer_factor=0.25,
            ptype=itk.template(reference_labelmap)[1][0],
        )
        self.log_info(
            "Evaluation grid: %s voxels at %.2f mm",
            list(itk.size(grid)),
            evaluation_spacing_mm,
        )
        reference_on_grid = self._resample_labelmap(reference_labelmap, grid)
        scored_labels = self._labels_present(reference_on_grid)
        provenance = self._provenance(case_id, shape_parameters)

        # One deformation per stage, from the network's own predictions. Each
        # stage's warped image is the reference labelmap carried into that
        # stage, which is exactly what the metrics below compare.
        series = self.movement_workflow.process_time_series(
            shape_parameters=shape_parameters,
            stages=stages,
            output_directory=out_dir,
            reference_mesh=reference_mesh,
            reference_image=reference_on_grid,
            warp_interpolation="nearest",
            warp_background_value=0.0,
            smoothing_sigma_mm=smoothing_sigma_mm,
        )

        rows: list[dict[str, Any]] = []
        for index, stage in enumerate(stages):
            truth = self._resample_labelmap(ground_truth_labelmaps[stage], grid)
            truth_surfaces = self._label_surfaces(truth)
            predicted = itk.imread(str(series["warped_images"][index]))
            rows.extend(
                self._score(
                    case_id,
                    stage,
                    truth,
                    truth_surfaces,
                    predicted,
                    self._label_surfaces(predicted),
                    scored_labels,
                    provenance,
                    include_dice,
                )
            )

        csv_file = self._write_csv(rows, out_dir)
        plot_file = self._write_volume_plot(rows, out_dir)
        report_file = self._write_report(
            rows,
            provenance,
            stages,
            smoothing_sigma_mm,
            evaluation_spacing_mm,
            plot_file,
            out_dir,
        )
        return {
            "rows": rows,
            "csv_file": csv_file,
            "report_file": report_file,
            "volume_plot_file": plot_file,
            "predicted_surfaces": series["predicted_surfaces"],
            "warped_labelmaps": series["warped_images"],
        }

    # ──────────────────────────── Metrics ──────────────────────────────────
    @staticmethod
    def dice(truth: np.ndarray, predicted: np.ndarray, label: int) -> float:
        """Dice overlap of one label. ``nan`` when neither volume contains it."""
        truth_mask = truth == label
        predicted_mask = predicted == label
        denominator = np.count_nonzero(truth_mask) + np.count_nonzero(predicted_mask)
        if denominator == 0:
            return float("nan")
        return float(2.0 * np.count_nonzero(truth_mask & predicted_mask) / denominator)

    @staticmethod
    def volume_mm3(labels: np.ndarray, label: int, voxel_volume_mm3: float) -> float:
        """Volume of one label, in cubic millimeters."""
        return float(np.count_nonzero(labels == label) * voxel_volume_mm3)

    @staticmethod
    def surface_rmse_mm(truth: pv.PolyData, predicted: pv.PolyData) -> float:
        """Symmetric point-to-surface RMSE, in millimeters.

        Both directions are pooled before the root-mean-square. A one-sided RMSE
        misses a prediction that covers the truth everywhere but also bulges
        somewhere the truth does not reach.
        """
        forward = predicted.copy().compute_implicit_distance(truth)
        reverse = truth.copy().compute_implicit_distance(predicted)
        distances = np.concatenate(
            [
                np.asarray(forward["implicit_distance"], dtype=np.float64),
                np.asarray(reverse["implicit_distance"], dtype=np.float64),
            ]
        )
        return float(np.sqrt(np.mean(distances**2)))

    # ──────────────────────────── Internals ────────────────────────────────
    @staticmethod
    def _resample_labelmap(labelmap: itk.Image, grid: itk.Image) -> itk.Image:
        """Resample a labelmap onto ``grid``, preserving its discrete values."""
        return itk.resample_image_filter(
            labelmap,
            use_reference_image=True,
            reference_image=grid,
            interpolator=itk.NearestNeighborInterpolateImageFunction.New(labelmap),
            default_pixel_value=0,
        )

    def _labels_present(self, reference_labelmap: itk.Image) -> dict[int, str]:
        """Drop the requested labels the reference frame does not contain."""
        present = set(np.unique(itk.GetArrayViewFromImage(reference_labelmap)).tolist())
        scored = {
            label: name for label, name in self.label_names.items() if label in present
        }
        missing = sorted(set(self.label_names) - set(scored))
        if missing:
            self.log_warning(
                "Reference frame has no voxels for label(s) %s; not scored.", missing
            )
        if not scored:
            raise ValueError(
                "None of the requested labels are present in the reference frame."
            )
        self.log_info(
            "Scoring %d structure(s): %s",
            len(scored),
            ", ".join(scored[label] for label in sorted(scored)),
        )
        return scored

    def _label_surfaces(self, labelmap: itk.Image) -> dict[int, pv.PolyData]:
        """Contour every label of one labelmap on the evaluation grid's pitch."""
        return self.contour_tools.extract_label_surfaces(labelmap)

    def _provenance(self, case_id: str, shape_parameters: Path) -> dict[str, Any]:
        """Case name, shape parameters and network weights, with their dates."""
        inference = self.movement_workflow.inference_workflow
        checkpoint = Path(inference.checkpoint_file)
        info = checkpoint.stat()
        coefficients = pnt.load_pca_coefficients(shape_parameters)
        provenance: dict[str, Any] = {
            "case_id": case_id,
            "shape_parameters_file": str(shape_parameters),
            "network_weights_file": str(checkpoint),
            "network_weights_created": self._timestamp(info.st_ctime),
            "network_weights_modified": self._timestamp(info.st_mtime),
            "network_epoch": "final" if inference.epoch is None else inference.epoch,
        }
        for index, coefficient in enumerate(coefficients, start=1):
            provenance[f"pca_c{index:02d}"] = float(coefficient)
        return provenance

    @staticmethod
    def _timestamp(seconds: float) -> str:
        """Format a filesystem timestamp as an ISO-8601 UTC string."""
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(
            timespec="seconds"
        )

    def _score(
        self,
        case_id: str,
        stage: float,
        truth: itk.Image,
        truth_surfaces: dict[int, pv.PolyData],
        predicted: itk.Image,
        predicted_surfaces: dict[int, pv.PolyData],
        scored_labels: dict[int, str],
        provenance: dict[str, Any],
        include_dice: bool = True,
    ) -> list[dict[str, Any]]:
        """One metric row per scored label of one stage."""
        truth_array = itk.GetArrayViewFromImage(truth)
        predicted_array = itk.GetArrayViewFromImage(predicted)
        voxel_volume_mm3 = float(np.prod(np.asarray(truth.GetSpacing())))

        rows: list[dict[str, Any]] = []
        for label in sorted(scored_labels):
            truth_volume = self.volume_mm3(truth_array, label, voxel_volume_mm3)
            if truth_volume == 0.0:
                self.log_info(
                    "stage %.3f: %s absent from the acquired frame; skipped.",
                    stage,
                    scored_labels[label],
                )
                continue
            predicted_volume = self.volume_mm3(predicted_array, label, voxel_volume_mm3)
            rmse = (
                self.surface_rmse_mm(truth_surfaces[label], predicted_surfaces[label])
                if label in truth_surfaces and label in predicted_surfaces
                else float("nan")
            )
            row: dict[str, Any] = {
                "case_id": case_id,
                "stage": stage,
                "label_id": label,
                "label_name": scored_labels[label],
            }
            if include_dice:
                row["dice"] = self.dice(truth_array, predicted_array, label)
            row.update(
                {
                    "volume_truth_mm3": truth_volume,
                    "volume_predicted_mm3": predicted_volume,
                    "volume_difference_mm3": predicted_volume - truth_volume,
                    "volume_difference_percent": (
                        100.0 * (predicted_volume - truth_volume) / truth_volume
                    ),
                    "surface_rmse_mm": rmse,
                }
            )
            row.update(
                {key: value for key, value in provenance.items() if key != "case_id"}
            )
            rows.append(row)
            self.log_info(
                "stage %.3f %-24s %sdV=%+.2f%%  rmse=%.3f mm",
                stage,
                scored_labels[label],
                f"dice={row['dice']:.4f}  " if include_dice else "",
                row["volume_difference_percent"],
                rmse,
            )
        return rows

    @staticmethod
    def _write_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
        """Write every metric row, provenance included, to one CSV."""
        csv_file = out_dir / "evaluation_metrics.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return csv_file

    def _write_volume_plot(self, rows: list[dict[str, Any]], out_dir: Path) -> Path:
        """Plot the acquired and predicted volume of every structure against stage.

        One color per structure, taken in a fixed order from a hue set separable
        under color-vision deficiency; the acquired volume is solid and the
        predicted volume dashed, so the two never rest on color alone.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plot_file = out_dir / "volume_vs_stage.png"
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        try:
            ends: list[tuple[float, float, str]] = []
            for index, label in enumerate(
                sorted({int(row["label_id"]) for row in rows})
            ):
                matching = sorted(
                    (row for row in rows if row["label_id"] == label),
                    key=lambda row: float(row["stage"]),
                )
                stages = [float(row["stage"]) for row in matching]
                truth = [float(row["volume_truth_mm3"]) / 1000.0 for row in matching]
                predicted = [
                    float(row["volume_predicted_mm3"]) / 1000.0 for row in matching
                ]
                color = self._SERIES_COLORS[index]
                ax.plot(stages, truth, color=color, linewidth=2.0, marker="o", ms=5)
                ax.plot(stages, predicted, color=color, linewidth=2.0, linestyle="--")
                ends.append((stages[-1], truth[-1], str(matching[0]["label_name"])))

            # Three of the hues fall below 3:1 against a white page, so each line
            # is named where it ends rather than in a color key alone. Structures
            # of similar size end on top of each other, so the names are pushed
            # apart, largest first, before they are drawn.
            span = float(np.ptp(ax.get_ylim()))
            previous = float("inf")
            for x_end, y_end, name in sorted(ends, key=lambda end: -end[1]):
                text_y = min(y_end, previous - 0.05 * span)
                ax.annotate(
                    name,
                    xy=(x_end, text_y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color="#52514e",
                    fontsize=9,
                    va="center",
                )
                previous = text_y

            ax.set_xlabel("Stage", color="#52514e")
            ax.set_ylabel("Volume (mL)", color="#52514e")
            ax.grid(True, color="#e1e0d9", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#c3c2b7")
            ax.spines["bottom"].set_color("#c3c2b7")
            ax.tick_params(colors="#898781", labelsize=9)
            ax.legend(
                handles=[
                    Line2D([], [], color="#898781", linewidth=2.0, label="acquired"),
                    Line2D(
                        [],
                        [],
                        color="#898781",
                        linewidth=2.0,
                        linestyle="--",
                        label="predicted",
                    ),
                ],
                frameon=False,
                loc="best",
                fontsize=9,
                labelcolor="#52514e",
            )
            fig.savefig(str(plot_file), bbox_inches="tight", dpi=150)
        finally:
            plt.close(fig)
        self.log_info("Volume plot: %s", plot_file)
        return plot_file

    def _write_report(
        self,
        rows: list[dict[str, Any]],
        provenance: dict[str, Any],
        stages: list[float],
        smoothing_sigma_mm: float,
        evaluation_spacing_mm: float,
        plot_file: Path,
        out_dir: Path,
    ) -> Path:
        """Write the markdown report beside the CSV."""
        coefficients = [
            provenance[key] for key in sorted(provenance) if key.startswith("pca_c")
        ]
        # Both tables carry whichever metrics the rows were scored with.
        has_dice = "dice" in rows[0]
        lines = [
            f"# Movement accuracy: {provenance['case_id']}",
            "",
            "## Run",
            "",
            f"- Hold-out case: `{provenance['case_id']}`",
            f"- Stages evaluated: {len(stages)} "
            f"({', '.join(f'{stage:.2f}' for stage in stages)})",
            f"- Shape parameters: `{provenance['shape_parameters_file']}`",
            "- Shape parameters (standard deviations): "
            + json.dumps([round(value, 4) for value in coefficients]),
            f"- Network weights: `{provenance['network_weights_file']}`",
            f"- Network weights created: {provenance['network_weights_created']}",
            f"- Network weights modified: {provenance['network_weights_modified']}",
            f"- Network epoch: {provenance['network_epoch']}",
            f"- Deformation smoothing sigma: {smoothing_sigma_mm:.1f} mm",
            f"- Evaluation grid pitch: {evaluation_spacing_mm:.2f} mm isotropic",
            "",
            "Every score compares the reference frame carried into that stage "
            "with the inferred deformation against the frame acquired there.",
            "",
            "## Volume over the stages",
            "",
            f"![Structure volume against stage]({plot_file.name})",
            "",
            "Solid: the volume acquired at that stage. Dashed: the volume the "
            "prediction carries there.",
            "",
            "## Per structure, averaged over stages",
            "",
        ]
        metrics = (["Dice"] if has_dice else []) + [
            "Volume difference (%)",
            "Surface RMSE (mm)",
        ]
        lines += self._table_header(["Structure"], metrics)
        for label in sorted({int(row["label_id"]) for row in rows}):
            matching = [row for row in rows if row["label_id"] == label]
            cells = [str(matching[0]["label_name"])]
            if has_dice:
                cells.append(f"{self._mean(matching, 'dice'):.4f}")
            cells += [
                f"{self._mean(matching, 'volume_difference_percent'):+.2f}",
                f"{self._mean(matching, 'surface_rmse_mm'):.3f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")

        lines += ["", "## Per stage", ""]
        lines += self._table_header(["Stage", "Structure"], metrics)
        for row in rows:
            cells = [f"{row['stage']:.2f}", str(row["label_name"])]
            if has_dice:
                cells.append(f"{row['dice']:.4f}")
            cells += [
                f"{row['volume_difference_percent']:+.2f}",
                f"{row['surface_rmse_mm']:.3f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")

        report_file = out_dir / "evaluation_report.md"
        report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log_info("Report: %s", report_file)
        return report_file

    @staticmethod
    def _table_header(keys: list[str], metrics: list[str]) -> list[str]:
        """Markdown header and alignment rows: keys left, metrics right."""
        return [
            "| " + " | ".join(keys + metrics) + " |",
            "| " + " | ".join(["---"] * len(keys) + ["---:"] * len(metrics)) + " |",
        ]

    @staticmethod
    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        """Mean of one column, ignoring the rows where it could not be measured."""
        values = [float(row[key]) for row in rows if not np.isnan(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")
