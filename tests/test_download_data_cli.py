"""Tests for the dataset download CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pytest

from physiotwin4d.cli import download_data
from physiotwin4d.data_download_tools import DataDownloadTools


def test_download_data_cli_uses_default_dataset_and_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default CLI arguments route Slicer-Heart-CT to data/Slicer-Heart-CT."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname) / DataDownloadTools.SLICER_HEART_CT_FILENAME

    monkeypatch.setattr(DataDownloadTools, "DownloadSlicerHeartCTData", fake_download)

    result = download_data.main([])

    assert result == 0
    assert calls == [Path("data/Slicer-Heart-CT")]
    assert "Downloaded Slicer-Heart-CT" in capsys.readouterr().out


def test_download_data_cli_uses_requested_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The --directory option controls where Slicer-Heart-CT is stored."""
    calls: list[Path] = []

    def fake_download(dirname: Union[str, Path]) -> Path:
        calls.append(Path(dirname))
        return Path(dirname) / DataDownloadTools.SLICER_HEART_CT_FILENAME

    monkeypatch.setattr(DataDownloadTools, "DownloadSlicerHeartCTData", fake_download)

    result = download_data.main(["Slicer-Heart-CT", "--directory", str(tmp_path)])

    assert result == 0
    assert calls == [tmp_path]
