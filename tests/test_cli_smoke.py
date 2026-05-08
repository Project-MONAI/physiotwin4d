"""CLI help smoke tests."""

from __future__ import annotations

import subprocess
import sys

import pytest


CLI_MODULES = [
    "physiomotion4d.cli.convert_ct_to_vtk",
    "physiomotion4d.cli.convert_heart_gated_ct_to_usd",
    "physiomotion4d.cli.convert_vtk_to_usd",
    "physiomotion4d.cli.create_statistical_model",
    "physiomotion4d.cli.fit_statistical_model_to_patient",
    "physiomotion4d.cli.reconstruct_highres_4d_ct",
    "physiomotion4d.cli.visualize_pca_modes",
]


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_help(module_name: str) -> None:
    """Each CLI module exits successfully for --help."""
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
