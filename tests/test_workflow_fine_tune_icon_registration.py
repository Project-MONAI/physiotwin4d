"""Unit tests for WorkflowFineTuneICONRegistration.

Exercises constructor validation, ``prepare_dataset`` / ``prepare_config``
file generation, mask derivation, and the ``run_fine_tuning`` subprocess
launch.  GPU-heavy paths (real uniGradICON training, ``apply_registration``)
are not exercised here — only their input-validation guards.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
import pytest
import yaml

from physiomotion4d.workflow_fine_tune_icon_registration import (
    WorkflowFineTuneICONRegistration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_image(path: Path, value: int = 1) -> None:
    """Write a 3x3x3 ``uint8`` ITK image with a single foreground voxel at center."""
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = value
    img = itk.image_from_array(arr)
    itk.imwrite(img, str(path), compression=True)


@pytest.fixture
def two_subject_dataset(tmp_path: Path) -> dict[str, Any]:
    """Two patients, two frames each, with matching labelmaps on disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "ft_out"

    subject_image_files: list[list[str]] = []
    subject_labelmap_files: list[list[Optional[str]]] = []
    for patient_id in ("pm0001", "pm0002"):
        pdir = data_dir / patient_id
        pdir.mkdir()
        images: list[str] = []
        segs: list[Optional[str]] = []
        for frame in ("g000", "g050"):
            image_path = pdir / f"{patient_id}_{frame}.nii.gz"
            label_path = pdir / f"{patient_id}_{frame}_labelmap.nii.gz"
            _make_image(image_path)
            _make_image(label_path)
            images.append(str(image_path))
            segs.append(str(label_path))
        subject_image_files.append(images)
        subject_labelmap_files.append(segs)

    return {
        "output_dir": output_dir,
        "fine_tune_name": "test_exp",
        "subject_ids": ["pm0001", "pm0002"],
        "subject_image_files": subject_image_files,
        "subject_labelmap_files": subject_labelmap_files,
    }


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_init_requires_output_dir_and_name(tmp_path: Path) -> None:
    """output_dir and fine_tune_name are required positional args."""
    with pytest.raises(TypeError):
        WorkflowFineTuneICONRegistration(
            subject_image_files=[["a.nii.gz"]],
        )


def test_init_rejects_empty_image_files(tmp_path: Path) -> None:
    """Empty subject list raises immediately."""
    with pytest.raises(ValueError, match="must not be empty"):
        WorkflowFineTuneICONRegistration(
            subject_image_files=[],
            output_dir=tmp_path,
            fine_tune_name="x",
        )


def test_init_rejects_mismatched_companion_lengths(tmp_path: Path) -> None:
    """Mask/seg/landmark lists must match subject_image_files shape exactly."""
    with pytest.raises(ValueError, match="subject_mask_files\\[0\\] length"):
        WorkflowFineTuneICONRegistration(
            subject_image_files=[["a.nii.gz", "b.nii.gz"]],
            output_dir=tmp_path,
            fine_tune_name="x",
            subject_mask_files=[["m.nii.gz"]],
        )


def test_init_rejects_duplicate_subject_ids(tmp_path: Path) -> None:
    """Duplicate subject IDs collapse paired groups, so reject them up front."""
    with pytest.raises(ValueError, match="unique"):
        WorkflowFineTuneICONRegistration(
            subject_image_files=[["a"], ["b"]],
            output_dir=tmp_path,
            fine_tune_name="x",
            subject_ids=["same", "same"],
        )


def test_init_rejects_mismatched_subject_ids_length(tmp_path: Path) -> None:
    """subject_ids must have one entry per subject."""
    with pytest.raises(ValueError, match="subject_ids length"):
        WorkflowFineTuneICONRegistration(
            subject_image_files=[["a"]],
            output_dir=tmp_path,
            fine_tune_name="x",
            subject_ids=["a", "b"],
        )


def test_use_segmentations_and_use_masks_flags(tmp_path: Path) -> None:
    """The two helper flags reflect supplied companions independently."""
    base: dict[str, Any] = {
        "subject_image_files": [["a"]],
        "output_dir": tmp_path,
        "fine_tune_name": "x",
    }
    none_wf = WorkflowFineTuneICONRegistration(**base)
    assert not none_wf.use_segmentations
    assert not none_wf.use_masks

    seg_only = WorkflowFineTuneICONRegistration(
        **base, subject_labelmap_files=[["seg.nii.gz"]]
    )
    assert seg_only.use_segmentations
    assert seg_only.use_masks  # derived from segs

    mask_only = WorkflowFineTuneICONRegistration(
        **base, subject_mask_files=[["mask.nii.gz"]]
    )
    assert not mask_only.use_segmentations
    assert mask_only.use_masks


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------


def test_prepare_dataset_uses_real_subject_ids(
    two_subject_dataset: dict[str, Any],
) -> None:
    """Subject IDs round-trip from the caller into every dataset entry."""
    workflow = WorkflowFineTuneICONRegistration(
        log_level=logging.CRITICAL, **two_subject_dataset
    )
    dataset_json_path = workflow.prepare_dataset()

    payload = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    entries = payload["data"]
    assert len(entries) == 4
    ids = {entry["subject_id"] for entry in entries}
    assert ids == {"pm0001", "pm0002"}
    for entry in entries:
        assert set(entry).issuperset({"image", "segmentation", "mask", "subject_id"})
        # Paths are forward-slashed for uniGradICON.
        assert "\\" not in entry["image"]
        assert "\\" not in entry["segmentation"]
        assert "\\" not in entry["mask"]


def test_prepare_dataset_skips_frames_with_missing_segmentation(
    tmp_path: Path,
) -> None:
    """A frame with no seg available is dropped when use_label is required."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    img_a = data_dir / "img_a.nii.gz"
    img_b = data_dir / "img_b.nii.gz"
    seg_a = data_dir / "seg_a.nii.gz"
    _make_image(img_a)
    _make_image(img_b)
    _make_image(seg_a)

    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[[str(img_a), str(img_b)]],
        output_dir=tmp_path / "out",
        fine_tune_name="exp",
        subject_labelmap_files=[[str(seg_a), None]],
        log_level=logging.CRITICAL,
    )
    dataset_json_path = workflow.prepare_dataset()

    entries = json.loads(dataset_json_path.read_text(encoding="utf-8"))["data"]
    assert len(entries) == 1
    assert entries[0]["image"].endswith("img_a.nii.gz")


def test_prepare_dataset_uses_explicit_mask_over_derived(tmp_path: Path) -> None:
    """When subject_mask_files supplies a mask, it overrides the derived one."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = data_dir / "image.nii.gz"
    seg = data_dir / "seg.nii.gz"
    explicit_mask = data_dir / "explicit_mask.nii.gz"
    _make_image(image)
    _make_image(seg)
    _make_image(explicit_mask)

    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[[str(image)]],
        output_dir=tmp_path / "out",
        fine_tune_name="exp",
        subject_labelmap_files=[[str(seg)]],
        subject_mask_files=[[str(explicit_mask)]],
        log_level=logging.CRITICAL,
    )
    dataset_json_path = workflow.prepare_dataset()
    entry = json.loads(dataset_json_path.read_text(encoding="utf-8"))["data"][0]

    assert entry["mask"].endswith("explicit_mask.nii.gz")
    assert entry["segmentation"].endswith("seg.nii.gz")
    # No derived mask file was created because the explicit one was used.
    derived_mask = data_dir / "seg_mask.nii.gz"
    assert not derived_mask.exists()


def test_prepare_dataset_mask_only_no_segmentations(tmp_path: Path) -> None:
    """Mask-only input: entries have ``mask`` but no ``segmentation`` field."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = data_dir / "image.nii.gz"
    mask = data_dir / "mask.nii.gz"
    _make_image(image)
    _make_image(mask)

    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[[str(image)]],
        output_dir=tmp_path / "out",
        fine_tune_name="exp",
        subject_mask_files=[[str(mask)]],
        log_level=logging.CRITICAL,
    )
    entry = json.loads(workflow.prepare_dataset().read_text(encoding="utf-8"))["data"][
        0
    ]
    assert "mask" in entry
    assert "segmentation" not in entry


def test_prepare_dataset_derives_mask_next_to_labelmap_by_default(
    two_subject_dataset: dict[str, Any],
) -> None:
    """Derived masks land next to each labelmap when ``mask_dir`` is not set."""
    workflow = WorkflowFineTuneICONRegistration(
        log_level=logging.CRITICAL, **two_subject_dataset
    )
    assert workflow.mask_dir is None
    workflow.prepare_dataset()

    seg_files = [
        Path(s)
        for inner in workflow.subject_labelmap_files or []
        for s in inner
        if s is not None
    ]
    derived = [s.parent / f"{s.name[: -len('.nii.gz')]}_mask.nii.gz" for s in seg_files]
    for mask_path in derived:
        assert mask_path.exists(), f"missing derived mask: {mask_path}"
    assert len(derived) == 4
    # Sanity: derived masks are binary with at least one foreground voxel.
    arr = itk.array_from_image(itk.imread(str(derived[0])))
    assert set(np.unique(arr).tolist()).issubset({0, 1})
    assert int(arr.sum()) >= 1


def test_prepare_dataset_derives_mask_under_explicit_mask_dir(
    two_subject_dataset: dict[str, Any], tmp_path: Path
) -> None:
    """Explicit ``mask_dir`` collects every derived mask in that single folder."""
    explicit_mask_dir = tmp_path / "explicit_masks"
    workflow = WorkflowFineTuneICONRegistration(
        log_level=logging.CRITICAL,
        mask_dir=explicit_mask_dir,
        **two_subject_dataset,
    )
    workflow.prepare_dataset()
    derived = list(explicit_mask_dir.glob("*_mask.nii.gz"))
    assert len(derived) == 4
    # None of the labelmap-adjacent locations should have been written to.
    seg_files = [
        Path(s)
        for inner in workflow.subject_labelmap_files or []
        for s in inner
        if s is not None
    ]
    for s in seg_files:
        assert not (s.parent / f"{s.name[: -len('.nii.gz')]}_mask.nii.gz").exists()


def test_prepare_dataset_raises_on_missing_image_file(tmp_path: Path) -> None:
    """Image existence is a hard requirement; missing image aborts the build."""
    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[[str(tmp_path / "does_not_exist.nii.gz")]],
        output_dir=tmp_path / "out",
        fine_tune_name="exp",
        log_level=logging.CRITICAL,
    )
    with pytest.raises(FileNotFoundError, match="Image not found"):
        workflow.prepare_dataset()


# ---------------------------------------------------------------------------
# prepare_config
# ---------------------------------------------------------------------------


def test_prepare_config_emits_uniGradICON_yaml(
    two_subject_dataset: dict[str, Any],
) -> None:
    """YAML config matches uniGradICON's expected structure when seg is present."""
    workflow = WorkflowFineTuneICONRegistration(
        log_level=logging.CRITICAL,
        epochs=10,
        batch_size=2,
        learning_rate=1e-4,
        input_shape=(64, 64, 64),
        gpus=[1],
        **two_subject_dataset,
    )
    dataset_json = workflow.prepare_dataset()
    config_yaml = workflow.prepare_config(dataset_json)

    config = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    assert config["experiment"]["model_weights"] == "unigradicon"
    assert config["experiment"]["name"].endswith("test_exp_model")
    training = config["training"]
    assert training["epochs"] == 10
    assert training["batch_size"] == 2
    assert training["learning_rate"] == 1e-4
    assert training["input_shape"] == [64, 64, 64]
    assert training["gpus"] == [1]
    # Driven by data availability.
    assert training["use_label"] is True
    assert training["loss_function_masking"] is True
    assert training["roi_masking"] is False

    dataset_cfg = config["datasets"][0]
    assert dataset_cfg["type"] == "paired"
    assert dataset_cfg["is_ct"] is True
    assert dataset_cfg["json_file"].endswith("test_exp_dataset.json")
    assert "\\" not in dataset_cfg["json_file"]


def test_prepare_config_flags_off_when_no_companions(tmp_path: Path) -> None:
    """Without seg or mask, ``use_label`` and ``loss_function_masking`` are False."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image = data_dir / "image.nii.gz"
    _make_image(image)

    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[[str(image)]],
        output_dir=tmp_path / "out",
        fine_tune_name="exp",
        log_level=logging.CRITICAL,
    )
    dataset_json = workflow.prepare_dataset()
    config = yaml.safe_load(
        workflow.prepare_config(dataset_json).read_text(encoding="utf-8")
    )
    assert config["training"]["use_label"] is False
    assert config["training"]["loss_function_masking"] is False


def test_prepare_config_requires_dataset_json(tmp_path: Path) -> None:
    """Calling prepare_config without first preparing the dataset is an error."""
    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[["a"]],
        output_dir=tmp_path,
        fine_tune_name="x",
        log_level=logging.CRITICAL,
    )
    with pytest.raises(ValueError, match="prepare_dataset"):
        workflow.prepare_config()


# ---------------------------------------------------------------------------
# expected_weights_path
# ---------------------------------------------------------------------------


def test_expected_weights_path_layout(tmp_path: Path) -> None:
    """Weights land at ``output_dir/<name>/<name>_model/checkpoints/...``."""
    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[["a"]],
        output_dir=tmp_path,
        fine_tune_name="exp",
        log_level=logging.CRITICAL,
    )
    expected = workflow.expected_weights_path()
    assert expected == (
        tmp_path / "exp" / "exp_model" / "checkpoints" / "Finetune_multi_final.trch"
    )


# ---------------------------------------------------------------------------
# run_fine_tuning (subprocess is monkey-patched)
# ---------------------------------------------------------------------------


def test_run_fine_tuning_invokes_unigradicon_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    two_subject_dataset: dict[str, Any],
) -> None:
    """run_fine_tuning launches the uniGradICON finetune module with the YAML path."""
    captured: dict[str, Any] = {}

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    unigradicon_src = two_subject_dataset["output_dir"].parent / "fake_unigradicon_src"
    workflow = WorkflowFineTuneICONRegistration(
        log_level=logging.CRITICAL,
        unigradicon_src_path=unigradicon_src,
        **two_subject_dataset,
    )
    weights = workflow.run_fine_tuning()

    assert captured["check"] is True
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:4] == ["-m", "unigradicon.finetuning.finetune", "--config"]
    yaml_arg = Path(captured["cmd"][4])
    assert yaml_arg.exists()
    assert yaml_arg.name == "test_exp_config.yaml"

    # Environment overrides.
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert str(unigradicon_src) in captured["env"]["PYTHONPATH"]

    assert weights == workflow.expected_weights_path()


def test_run_fine_tuning_without_unigradicon_src(
    monkeypatch: pytest.MonkeyPatch,
    two_subject_dataset: dict[str, Any],
) -> None:
    """When unigradicon_src_path is None, PYTHONPATH is not prefixed."""

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        # No leading entry referencing a "fake" src tree.
        assert "fake_unigradicon_src" not in env.get("PYTHONPATH", "")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    workflow = WorkflowFineTuneICONRegistration(
        log_level=logging.CRITICAL,
        **two_subject_dataset,
    )
    workflow.run_fine_tuning()


# ---------------------------------------------------------------------------
# apply_registration — validation guards only (skips real registration)
# ---------------------------------------------------------------------------


def test_apply_registration_rejects_empty_moving(tmp_path: Path) -> None:
    """apply_registration validates inputs before touching the registrar."""
    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[["a"]],
        output_dir=tmp_path,
        fine_tune_name="x",
        log_level=logging.CRITICAL,
    )
    arr = np.zeros((3, 3, 3), dtype=np.float32)
    ref = itk.image_from_array(arr)
    with pytest.raises(ValueError, match="moving_images must not be empty"):
        workflow.apply_registration(reference_image=ref, moving_images=[])


def test_apply_registration_rejects_mismatched_companions(tmp_path: Path) -> None:
    """moving_segmentations / moving_landmarks length must match moving_images."""
    workflow = WorkflowFineTuneICONRegistration(
        subject_image_files=[["a"]],
        output_dir=tmp_path,
        fine_tune_name="x",
        log_level=logging.CRITICAL,
    )
    ref = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    mov = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="moving_segmentations length"):
        workflow.apply_registration(
            reference_image=ref,
            moving_images=[mov],
            moving_segmentations=[],
        )
    with pytest.raises(ValueError, match="moving_landmarks length"):
        workflow.apply_registration(
            reference_image=ref,
            moving_images=[mov],
            moving_landmarks=[],
        )
