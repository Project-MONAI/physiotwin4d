"""
Dataset download and verification helpers.

Only Slicer-Heart-CT is downloaded automatically. Other datasets require
manual download, and the verification helpers check the file layouts used by
the repository tutorials, experiments, and tests.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Union

_DOWNLOAD_TIMEOUT_SECONDS = 60.0


class DataDownloadTools:
    """Download and verify optional PhysioMotion4D example datasets."""

    SLICER_HEART_CT_URL = (
        "https://github.com/SlicerHeart/SlicerHeart/releases/download/"
        "TestingData/TruncalValve_4DCT.seq.nrrd"
    )
    SLICER_HEART_CT_FILENAME = "TruncalValve_4DCT.seq.nrrd"

    @staticmethod
    def DownloadSlicerHeartCTData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the Slicer-Heart-CT 4-D CT sample into ``dirname``.

        Args:
            dirname: Directory where ``TruncalValve_4DCT.seq.nrrd`` should live.

        Returns:
            Path to the downloaded or already-cached ``.seq.nrrd`` file.
        """
        data_dir = Path(dirname)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_file = data_dir / DataDownloadTools.SLICER_HEART_CT_FILENAME
        if data_file.exists() and data_file.stat().st_size > 0:
            return data_file

        # Stream to a unique temp file in the same directory with an explicit
        # timeout, then atomically replace the target on success. The temp
        # name is unique so concurrent callers do not clobber each other.
        # Avoids partial files on interrupt and the indefinite hang that
        # urlretrieve has without a timeout.
        tmp_handle = tempfile.NamedTemporaryFile(
            dir=str(data_dir),
            prefix=f".{DataDownloadTools.SLICER_HEART_CT_FILENAME}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_file = Path(tmp_handle.name)
        try:
            with (
                urllib.request.urlopen(  # noqa: S310
                    DataDownloadTools.SLICER_HEART_CT_URL,
                    timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                ) as response,
                tmp_handle as out,
            ):
                shutil.copyfileobj(response, out)
            if tmp_file.stat().st_size == 0:
                raise RuntimeError(
                    f"Downloaded file is empty: {DataDownloadTools.SLICER_HEART_CT_URL}"
                )
            tmp_file.replace(data_file)
        except BaseException:
            tmp_handle.close()
            if tmp_file.exists():
                tmp_file.unlink()
            raise
        return data_file

    @staticmethod
    def VerifySlicerHeartCTData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when Slicer-Heart-CT has the expected 4-D CT file."""
        return (Path(dirname) / DataDownloadTools.SLICER_HEART_CT_FILENAME).is_file()

    @staticmethod
    def VerifyCHOPValve4DData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when CHOP-Valve4D files referenced by the repo exist.

        Accepted layouts are the CT volume used by Simpleware/model-to-patient
        experiments and the valve time-series folders used by VTK-to-USD
        experiments.
        """
        data_dir = Path(dirname)
        has_ct_volume = any(
            (data_dir / "CT" / filename).is_file()
            for filename in ("RVOT28-Dias.nii.gz", "RVOT28-Dias.mha")
        )
        has_simpleware_parts = (data_dir / "CT" / "Simpleware" / "parts").is_dir()
        has_alterra = (data_dir / "Alterra").is_dir() and any(
            (data_dir / "Alterra").glob("*.vtk")
        )
        has_tpv25 = (data_dir / "TPV25").is_dir() and any(
            (data_dir / "TPV25").glob("*.vtk")
        )
        return has_ct_volume or has_simpleware_parts or (has_alterra and has_tpv25)

    @staticmethod
    def VerifyDirLab4DCTData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when a supported DirLab-4DCT case layout exists."""
        data_dir = Path(dirname)
        case1_dir = data_dir / "Case1"
        has_case_dir_layout = case1_dir.is_dir() and any(case1_dir.glob("*.mha"))
        has_case_dir_layout = has_case_dir_layout or (
            case1_dir.is_dir() and any(case1_dir.glob("*.mhd"))
        )

        has_pack_layout = any(data_dir.glob("Case1Pack_T*.mhd")) or any(
            data_dir.glob("Case1Pack_T*.mha")
        )
        return has_case_dir_layout or has_pack_layout

    @staticmethod
    def VerifyKCLHeartModelData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when KCL-Heart-Model has its expected mesh inputs."""
        data_dir = Path(dirname)
        input_meshes_dir = data_dir / "input_meshes"
        return (data_dir / "average_mesh.vtk").is_file() and any(
            input_meshes_dir.glob("*.vtk")
        )
