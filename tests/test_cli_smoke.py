"""CLI help smoke tests."""

from __future__ import annotations

import importlib
import sys

import pytest


CLI_MODULES = [
    "physiomotion4d.cli.convert_image_to_vtk",
    "physiomotion4d.cli.convert_image_4d_to_3d",
    "physiomotion4d.cli.convert_image_to_usd",
    "physiomotion4d.cli.convert_vtk_to_usd",
    "physiomotion4d.cli.create_statistical_model",
    "physiomotion4d.cli.fit_statistical_model_to_patient",
    "physiomotion4d.cli.reconstruct_highres_4d_ct",
    "physiomotion4d.cli.visualize_pca_modes",
]


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_help(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each CLI module exits successfully for --help."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage:" in captured.out.lower()
