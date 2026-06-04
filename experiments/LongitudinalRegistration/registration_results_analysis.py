"""Summarize registration Dice and landmark RMSE results across experiments.

For every ``results_*`` directory under a base directory this script reads:

- ``registration_dice_init.csv`` with columns
  ``subject_id, method, stem, label, dice`` (one row per subject / time point
  / anatomy label).
- ``registration_landmarks_init.csv`` with columns
  ``subject_id, method, stem, name, rms_err_mm`` (one row per subject / time
  point / landmark).

It pools all subjects and all time points (rows) within each
``(results_dir, method)`` group and reports the mean, standard deviation, and
95th percentile of the Dice score per label (1..10) and across all labels, and
the same statistics of the landmark RMSE per landmark and across all landmarks.

**Duplicate handling.** If the same ``(subject_id, method, stem, label)``
combination appears more than once in a single Dice CSV, the *n*-th occurrence
is treated as if it came from a separate directory named
``{results_dir}_{n}`` (e.g. ``results_ml_2``, ``results_ml_3``).  The same
applies for ``(subject_id, method, stem, name)`` duplicates in the landmark
CSV.

Two summary tables are produced, each indexed by ``(results_dir, method)``
(with any ``_n`` suffixes introduced by duplicate splitting): one for label
Dice scores and one for landmark RMSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DICE_CSV = "registration_dice_init.csv"
LANDMARKS_CSV = "registration_landmarks_init.csv"

ALL_KEY = "all"
STATS = ("mean", "std", "p95")


def _split_by_occurrence(
    frame: pd.DataFrame,
    key_cols: list[str],
    dir_name: str,
) -> dict[str, pd.DataFrame]:
    """Split ``frame`` into per-occurrence sub-frames keyed by a group name.

    Within ``frame``, each unique combination of ``key_cols`` may appear
    multiple times.  The *n*-th occurrence (1-indexed) of a given key is
    assigned to group ``dir_name`` (for n=1) or ``{dir_name}_{n}`` (for n>1).
    Sub-frames for different occurrence numbers are returned in a dict whose
    keys follow that naming convention.

    Args:
        frame: Input data frame, one row per observation.
        key_cols: Columns that together identify a unique observation.
        dir_name: Base name used to construct group keys.

    Returns:
        Ordered dict mapping group name → sub-data-frame.
    """
    occurrence = frame.groupby(key_cols, sort=False).cumcount() + 1  # 1-indexed
    max_occ = int(occurrence.max())
    result: dict[str, pd.DataFrame] = {}
    for n in range(1, max_occ + 1):
        sub = frame[occurrence == n]
        if sub.empty:
            continue
        group_name = dir_name if n == 1 else f"{dir_name}_{n}"
        result[group_name] = sub.reset_index(drop=True)
    return result


def _aggregate(
    frames: dict[str, pd.DataFrame],
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Aggregate one value column grouped by ``group_col`` for each (directory, method).

    Args:
        frames: Mapping from ``results_*`` directory name to its data frame.
        group_col: Column whose distinct values become summary categories
            (``"label"`` for Dice, ``"name"`` for landmarks).
        value_col: Column to summarize (``"dice"`` or ``"rms_err_mm"``).

    Returns:
        A data frame indexed by ``(results_dir, method)`` with a two-level
        column ``(category, stat)`` where ``category`` is each distinct group
        value plus ``"all"`` and ``stat`` is one of ``mean``, ``std``, ``p95``.
        Categories are pooled across all subjects and time points within each
        ``(directory, method)`` group.
    """

    def _stats(series: pd.Series) -> dict[str, float]:
        # Drop missing measurements so a single blank row does not poison the
        # mean / std / percentile for an entire landmark or the pooled "all".
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            return {stat: float("nan") for stat in STATS}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "p95": float(np.percentile(values, 95)),
        }

    rows: dict[tuple[str, str], dict[tuple[object, str], float]] = {}
    for dir_name, frame in frames.items():
        method_groups: list[tuple[str, pd.DataFrame]]
        if "method" in frame.columns:
            method_groups = [
                (str(m), sub) for m, sub in frame.groupby("method", sort=True)
            ]
        else:
            method_groups = [("", frame)]
        for method, mframe in method_groups:
            row: dict[tuple[object, str], float] = {}
            for category, group in mframe.groupby(group_col):
                for stat, value in _stats(group[value_col]).items():
                    row[(category, stat)] = value
            for stat, value in _stats(mframe[value_col]).items():
                row[(ALL_KEY, stat)] = value
            rows[(dir_name, method)] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(
        table.index, names=["results_dir", "method"]
    )
    # from_dict yields a flat Index of tuples; promote to a MultiIndex so the
    # reindex below aligns on (category, stat) rather than producing NaNs.
    table.columns = pd.MultiIndex.from_tuples(table.columns)

    # Order columns: numeric/string categories sorted, "all" last.
    categories = sorted(
        {category for category, _ in table.columns if category != ALL_KEY},
        key=lambda c: (0, int(c)) if str(c).isdigit() else (1, str(c)),
    )
    categories.append(ALL_KEY)
    ordered = [(category, stat) for category in categories for stat in STATS]
    table = table.reindex(columns=pd.MultiIndex.from_tuples(ordered))
    return table


def summarize(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Dice and landmark RMSE summary tables for ``base_dir``.

    Args:
        base_dir: Directory containing one or more ``results_*`` subdirectories.

    Returns:
        ``(dice_table, landmark_table)`` summary frames, each indexed by
        ``(results_dir, method)``.
    """
    result_dirs = sorted(p for p in base_dir.glob("results_*") if p.is_dir())
    if not result_dirs:
        raise FileNotFoundError(f"No results_* directories found in {base_dir}")

    dice_frames: dict[str, pd.DataFrame] = {}
    landmark_frames: dict[str, pd.DataFrame] = {}
    for result_dir in result_dirs:
        dice_path = result_dir / DICE_CSV
        landmark_path = result_dir / LANDMARKS_CSV
        if dice_path.is_file():
            dice_frames.update(
                _split_by_occurrence(
                    pd.read_csv(dice_path),
                    ["subject_id", "method", "stem", "label"],
                    result_dir.name,
                )
            )
        if landmark_path.is_file():
            landmark_frames.update(
                _split_by_occurrence(
                    pd.read_csv(landmark_path),
                    ["subject_id", "method", "stem", "name"],
                    result_dir.name,
                )
            )

    dice_table = _aggregate(dice_frames, "label", "dice")
    landmark_table = _aggregate(landmark_frames, "name", "rms_err_mm")
    return dice_table, landmark_table


def _print_table(title: str, table: pd.DataFrame) -> None:
    """Print a summary table with a heading at full width."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.float_format",
        lambda v: f"{v:.4f}",
    ):
        print(table)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing results_* subdirectories "
        "(default: this script's directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write summary CSV files into.",
    )
    args = parser.parse_args(argv)

    dice_table, landmark_table = summarize(args.base_dir)

    _print_table(
        "Dice score by label (mean / std / 95th percentile), grouped by "
        "results_dir + method, pooled over subjects and time points",
        dice_table,
    )
    _print_table(
        "Landmark RMSE [mm] (mean / std / 95th percentile), grouped by "
        "results_dir + method, pooled over subjects and time points",
        landmark_table,
    )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        dice_out = args.out_dir / "summary_dice.csv"
        landmark_out = args.out_dir / "summary_landmarks.csv"
        dice_table.to_csv(dice_out)
        landmark_table.to_csv(landmark_out)
        print(f"\nWrote {dice_out}\nWrote {landmark_out}")


if __name__ == "__main__":
    main()
