"""
Dataset download and verification helpers.

Slicer-Heart-CT, KCL-Heart-Model, and CHOP-Valve4D are downloaded
automatically. Other datasets require manual download, and the verification
helpers check the file layouts used by the repository tutorials,
experiments, and tests.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Union

_DOWNLOAD_TIMEOUT_SECONDS = 60.0


class DataDownloadTools:
    """Download and verify optional PhysioTwin4D example datasets."""

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

    KCL_HEART_MODEL_MESH_COUNT = 20
    KCL_HEART_MODEL_INDIVIDUAL_URL_TEMPLATE = (
        "https://zenodo.org/records/4590294/files/{index:02d}.tar.gz?download=1"
    )
    KCL_HEART_MODEL_AVERAGE_URL = (
        "https://zenodo.org/records/4593739/files/average.tar.gz?download=1"
    )

    @staticmethod
    def DownloadKCLHeartModelData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the KCL-Heart-Model dataset into ``dirname``.

        Downloads and extracts the 20 individual four-chamber heart meshes
        from https://zenodo.org/records/4590294 into
        ``dirname/input_meshes/01.vtk`` through ``20.vtk``, and the average
        heart mesh from https://zenodo.org/records/4593739 into
        ``dirname/average_mesh.vtk``. Already-extracted files are reused.

        Args:
            dirname: Directory where the KCL-Heart-Model dataset should live.

        Returns:
            Path to ``dirname``.
        """
        data_dir = Path(dirname)
        input_meshes_dir = data_dir / "input_meshes"
        input_meshes_dir.mkdir(parents=True, exist_ok=True)

        for index in range(1, DataDownloadTools.KCL_HEART_MODEL_MESH_COUNT + 1):
            target_file = input_meshes_dir / f"{index:02d}.vtk"
            if target_file.exists() and target_file.stat().st_size > 0:
                continue
            url = DataDownloadTools.KCL_HEART_MODEL_INDIVIDUAL_URL_TEMPLATE.format(
                index=index
            )
            DataDownloadTools._DownloadAndExtractTarMember(
                url, member_name=f"{index:02d}.vtk", target_file=target_file
            )

        average_file = data_dir / "average_mesh.vtk"
        if not (average_file.exists() and average_file.stat().st_size > 0):
            DataDownloadTools._DownloadAndExtractTarMember(
                DataDownloadTools.KCL_HEART_MODEL_AVERAGE_URL,
                member_name="average.vtk",
                target_file=average_file,
            )

        return data_dir

    @staticmethod
    def _DownloadAndExtractTarMember(  # noqa: N802
        url: str, member_name: str, target_file: Path
    ) -> None:
        """Download a ``.tar.gz`` archive and extract one member to ``target_file``."""
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_file.parent)) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            archive_file = tmp_dir / "archive.tar.gz"
            with (
                urllib.request.urlopen(  # noqa: S310
                    url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                open(archive_file, "wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if archive_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded archive is empty: {url}")

            with tarfile.open(archive_file) as tar:
                try:
                    tar.extract(member_name, path=tmp_dir, filter="data")
                except TypeError:
                    # Python < 3.12 (without the PEP 706 backport) does not
                    # accept the ``filter`` keyword argument.
                    tar.extract(member_name, path=tmp_dir)
            extracted_file = tmp_dir / member_name
            if not extracted_file.is_file():
                raise RuntimeError(
                    f"Expected member {member_name!r} not found in archive: {url}"
                )
            extracted_file.replace(target_file)

    CHOP_VALVE4D_RELEASE_URL = (
        "https://github.com/Project-MONAI/physiotwin4d/releases/download/2026.07.1/"
    )
    # subdirectory name -> release asset filename. Alterra and TPV25 are
    # each >1 GB (per-frame valve mesh time series); CT is the smaller
    # image/segmentation bundle. See data/CHOP-Valve4D/README.md.
    CHOP_VALVE4D_ASSETS = {
        "Alterra": "CHOP-Valve4D-Alterra.zip",
        "CT": "CHOP-Valve4D-CT.zip",
        "TPV25": "CHOP-Valve4D-TPV25.zip",
    }

    @staticmethod
    def DownloadCHOPValve4DData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the CHOP-Valve4D convenience release into ``dirname``.

        Downloads the three zip archives attached to the PhysioTwin4D
        2026.07.1 GitHub release
        (https://github.com/Project-MONAI/physiotwin4d/releases/tag/2026.07.1)
        and extracts each into its matching subdirectory: ``Alterra/`` and
        ``TPV25/`` (valve mesh time series, >1 GB each) and ``CT/`` (source
        CT volume and Simpleware segmentation). A subdirectory that already
        has any files is left alone, so re-running resumes an interrupted
        download. See ``data/CHOP-Valve4D/README.md`` for what this
        converted data contains and how it relates to the original FEBio
        source model.

        Args:
            dirname: Directory where the CHOP-Valve4D dataset should live.

        Returns:
            Path to ``dirname``.
        """
        data_dir = Path(dirname)
        for subdir_name, asset_name in DataDownloadTools.CHOP_VALVE4D_ASSETS.items():
            target_dir = data_dir / subdir_name
            if target_dir.is_dir() and any(target_dir.iterdir()):
                continue
            url = DataDownloadTools.CHOP_VALVE4D_RELEASE_URL + asset_name
            DataDownloadTools._DownloadAndExtractZip(url, target_dir)
        return data_dir

    @staticmethod
    def _DownloadAndExtractZip(url: str, target_dir: Path) -> None:  # noqa: N802
        """Stream-download a ``.zip`` archive and extract it into ``target_dir``."""
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_dir.parent)) as tmp_dir_name:
            archive_file = Path(tmp_dir_name) / "archive.zip"
            with (
                urllib.request.urlopen(  # noqa: S310
                    url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                open(archive_file, "wb") as out,
            ):
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if archive_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded archive is empty: {url}")

            with zipfile.ZipFile(archive_file) as archive:
                archive.extractall(target_dir)

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
