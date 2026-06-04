"""Fine-tune uniGradICON registration and apply the fine-tuned weights.

This module provides :class:`WorkflowFineTuneICONRegistration`, which packages
the two halves of the longitudinal-registration ICON fine-tuning experiment
from ``experiments/LongitudinalRegistration``:

1. **Fine-tuning**: build a paired dataset JSON and YAML config from per-subject
   lists of image files (with optional segmentation labelmaps and landmark CSVs)
   and launch ``unigradicon.finetuning.finetune`` as a subprocess.  Mirrors
   ``experiments/LongitudinalRegistration/1-finetune_icon.py``.
2. **Apply**: load a fine-tuned uniGradICON checkpoint and register a list of
   moving images to a single reference image using
   :class:`RegisterTimeSeriesImages` (ICON backend).  Mirrors the per-subject
   registration loop in
   ``experiments/LongitudinalRegistration/recon_4d_icon_eval.py``.

Conventions:
    - Fine-tuning is file-based: it reads images/labelmaps/landmarks from disk
      because ``unigradicon.finetuning.finetune`` is launched as a subprocess
      that consumes JSON paths.
    - Apply is in-memory: takes ``itk.Image`` inputs in LPS space and
      ``dict[name, (x, y, z)]`` landmark dictionaries.  Segmentations are
      resampled with nearest-neighbor interpolation; images use linear
      interpolation.
    - The ``inverse_transform`` returned by ICON is a resampler-convention
      transform that maps moving-grid points back to reference-grid points;
      ``forward_transform`` is the inverse direction (reference grid →
      moving grid).  Landmarks are warped using ``TransformPoint`` and
      images/segmentations are resampled via
      :meth:`TransformTools.transform_image`.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union

import itk
import numpy as np
import yaml

from physiomotion4d.labelmap_tools import LabelmapTools
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.register_time_series_images import RegisterTimeSeriesImages
from physiomotion4d.transform_tools import TransformTools

Landmarks = dict[str, tuple[float, float, float]]


class WorkflowFineTuneICONRegistration(PhysioMotion4DBase):
    """Fine-tune uniGradICON on paired 3D images and apply the fine-tuned weights.

    The workflow has two stages that can be used together or independently:

    **Stage 1: Fine-tuning** (file-based)
        Build a paired dataset JSON and YAML config from per-subject lists of
        image, segmentation, and landmark files, then launch
        ``unigradicon.finetuning.finetune`` as a subprocess.  Each subject's
        time-point images form one paired group (they share a ``subject_id``).

    **Stage 2: Apply** (in-memory)
        Register a list of moving images to a single reference image using the
        fine-tuned ICON weights and return both directions of the warp:

        - moving images / segmentations / landmarks warped into reference space
        - the reference image / segmentation / landmarks warped into each
          moving-image space

    Attributes:
        subject_image_files (list[list[str]]): Per-subject lists of image
            paths.  Images within one inner list share a subject_id during
            fine-tuning.
        output_dir (Path): Directory where dataset JSON, YAML config, derived
            masks, and the uniGradICON ``checkpoints/`` tree are written.
        fine_tune_name (str): Sub-directory name for the experiment outputs.
        subject_ids (Optional[list[str]]): One ID per subject (e.g. patient
            identifiers).  Written into the dataset JSON's ``subject_id``
            field; falls back to synthetic ``subject_NNNN`` when ``None``.
        subject_segmentation_files (Optional[list[list[Optional[str]]]]):
            Per-subject multi-label segmentation/labelmap paths aligned with
            ``subject_image_files``.  ``None`` (or per-image ``None``) means no
            segmentation for that image.  If supplied for at least one image,
            paired-with-seg training is enabled.
        subject_mask_files (Optional[list[list[Optional[str]]]]):
            Per-subject binary mask paths aligned with ``subject_image_files``.
            When supplied for a frame these masks are used directly for
            loss-function masking; otherwise masks are derived from
            ``subject_segmentation_files``.
        subject_landmark_files (Optional[list[list[Optional[str]]]]):
            Per-subject landmark CSV paths (``Name,X,Y,Z`` format) aligned with
            ``subject_image_files``.  Recorded in the dataset JSON for
            traceability; not consumed by uniGradICON fine-tuning itself.
        mask_dilation_mm (float): Millimeters of physical-radius binary
            dilation applied to the >0 labelmap when deriving the loss-masking
            binary mask via :meth:`LabelmapTools.convert_labelmap_to_mask`.
        mask_exclude_labels (Optional[list[int]]): Labels to exclude from the mask.
            Default is None.
        mask_dir (Optional[Path]): Directory where derived binary masks are
            written and looked up.  ``None`` (default) writes each derived
            mask next to its source labelmap on disk.
        registrar (RegisterTimeSeriesImages): ICON-backend registrar used in
            :meth:`apply_registration`.
        transform_tools (TransformTools): Utility for resampling images and
            segmentations.

    Example:
        >>> # Stage 1: fine-tune
        >>> workflow = WorkflowFineTuneICONRegistration(
        ...     subject_image_files=[
        ...         ['pm0001/g000.nii.gz', 'pm0001/g050.nii.gz'],
        ...         ['pm0002/g000.nii.gz', 'pm0002/g050.nii.gz'],
        ...     ],
        ...     output_dir=Path('d:/PhysioMotion4D/icon_finetuned'),
        ...     fine_tune_name='duke_4d_gated_icon_ft',
        ...     subject_segmentation_files=[
        ...         ['pm0001/g000_labelmap.nii.gz', 'pm0001/g050_labelmap.nii.gz'],
        ...         ['pm0002/g000_labelmap.nii.gz', 'pm0002/g050_labelmap.nii.gz'],
        ...     ],
        ... )
        >>> weights_path = workflow.run_fine_tuning()
        >>>
        >>> # Stage 2: apply
        >>> result = workflow.apply_registration(
        ...     reference_image=ref_image,
        ...     moving_images=moving_images,
        ...     weights_path=weights_path,
        ...     reference_segmentation=ref_seg,
        ...     moving_segmentations=moving_segs,
        ... )
        >>> warped_to_ref = result['moving_to_reference_images']
        >>> warped_to_moving = result['reference_to_moving_images']
    """

    def __init__(
        self,
        subject_image_files: list[list[str]],
        output_dir: Path,
        fine_tune_name: str,
        subject_ids: Optional[list[str]] = None,
        subject_segmentation_files: Optional[list[list[Optional[str]]]] = None,
        subject_mask_files: Optional[list[list[Optional[str]]]] = None,
        subject_landmark_files: Optional[list[list[Optional[str]]]] = None,
        epochs: int = 2000,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        input_shape: tuple[int, int, int] = (175, 175, 175),
        similarity: str = "lncc",
        lambda_value: float = 1.5,
        dice_loss_weight: float = 0.5,
        lncc_sigma: int = 1,
        ct_window: tuple[float, float] = (-1000.0, 1000.0),
        is_ct: bool = True,
        gpus: Optional[list[int]] = None,
        eval_period: int = 10,
        save_period: int = 50,
        mask_dilation_mm: float = 5.0,
        mask_exclude_labels: Optional[list[int]] = None,
        mask_dir: Optional[Path] = None,
        unigradicon_src_path: Optional[Path] = None,
        log_level: Union[int, str] = logging.INFO,
    ) -> None:
        """Initialize the ICON fine-tuning workflow.

        Args:
            subject_image_files: Per-subject lists of image file paths.  Each
                inner list groups frames belonging to one subject; all of those
                frames share a ``subject_id`` for paired training.
            output_dir: Directory for dataset JSON, YAML config, derived masks,
                and the uniGradICON checkpoint tree.
            fine_tune_name: Sub-directory name for the experiment outputs
                (used as the uniGradICON ``experiment.name`` stem).
            subject_ids: One ID per subject, in the same order as
                ``subject_image_files``.  Written verbatim into the dataset
                JSON's ``subject_id`` field so paired training groups frames
                that share an ID.  ``None`` falls back to synthetic IDs of the
                form ``subject_0000``, ``subject_0001``, ...  Must be unique.
            subject_segmentation_files: Per-subject multi-label segmentation
                (labelmap) paths matching ``subject_image_files``.  ``None``
                disables paired-with-seg training.
                Individual ``None`` entries inside the inner lists skip just
                those frames when paired-with-seg training is enabled.
            subject_mask_files: Per-subject binary mask paths matching
                ``subject_image_files``.  When supplied these are used directly
                for ICON loss-function masking; otherwise masks are derived
                from ``subject_segmentation_files`` via a >0 threshold and
                dilation by ``mask_dilation_mm``.  Per-image ``None``
                entries fall back to the derived mask for that frame (or skip
                it if no segmentation is available either).
            subject_landmark_files: Per-subject landmark CSV paths matching
                ``subject_image_files``.  Stored in the dataset JSON for
                traceability; not consumed by uniGradICON fine-tuning.
            epochs: uniGradICON ``training.epochs``.
            batch_size: uniGradICON ``training.batch_size``.
            learning_rate: uniGradICON ``training.learning_rate``.
            input_shape: uniGradICON ``training.input_shape`` (voxels, X/Y/Z).
            similarity: uniGradICON ``training.similarity`` metric (e.g. ``lncc``).
            lambda_value: uniGradICON ``training.lambda`` regularization weight.
            dice_loss_weight: uniGradICON ``training.dice_loss_weight``.
            lncc_sigma: uniGradICON ``training.lncc_sigma``.
            ct_window: uniGradICON dataset ``ct_window`` ``[low, high]`` in HU.
            is_ct: Whether the dataset is CT (passes through to dataset config).
            gpus: GPU device indices for training.  Defaults to ``[0]``.
            eval_period: uniGradICON ``training.eval_period``.
            save_period: uniGradICON ``training.save_period``.
            mask_dilation_mm: Physical radius (millimeters) of binary
                dilation applied to the >0 labelmap when deriving the
                loss-masking binary mask via
                :meth:`LabelmapTools.convert_labelmap_to_mask`.  Ignored when
                no segmentations are supplied.  Default 5.0 mm.
            mask_dir: Directory where derived binary masks are written and
                looked up.  ``None`` (default) writes each derived mask next
                to its source labelmap on disk
                (``<labelmap_dir>/<labelmap_stem>_mask.nii.gz``).  An explicit
                path puts all derived masks in that single directory.
            unigradicon_src_path: Optional path to a local uniGradICON source
                tree to prepend to ``PYTHONPATH`` when running fine-tuning.
                Useful for using a checked-out copy instead of the installed
                package.
            log_level: Logging level (``logging.DEBUG``, ``logging.INFO``, ...).

        Raises:
            ValueError: If ``subject_image_files`` is empty.
            ValueError: If ``subject_segmentation_files``,
                ``subject_mask_files``, or ``subject_landmark_files`` is
                provided with a shape that does not match
                ``subject_image_files``.
        """
        super().__init__(
            class_name="WorkflowFineTuneICONRegistration", log_level=log_level
        )

        if not subject_image_files:
            raise ValueError("subject_image_files must not be empty")

        if subject_ids is not None:
            if len(subject_ids) != len(subject_image_files):
                raise ValueError(
                    f"subject_ids length ({len(subject_ids)}) must match "
                    f"subject_image_files length ({len(subject_image_files)})"
                )
            if len(set(subject_ids)) != len(subject_ids):
                raise ValueError(f"subject_ids must be unique, got {subject_ids}")

        self._validate_companion_shape(
            subject_image_files,
            subject_segmentation_files,
            "subject_segmentation_files",
        )
        self._validate_companion_shape(
            subject_image_files, subject_mask_files, "subject_mask_files"
        )
        self._validate_companion_shape(
            subject_image_files, subject_landmark_files, "subject_landmark_files"
        )

        self.subject_image_files = subject_image_files
        self.subject_ids = subject_ids
        self.subject_segmentation_files = subject_segmentation_files
        self.subject_mask_files = subject_mask_files
        self.subject_landmark_files = subject_landmark_files

        self.use_segmentations: bool = subject_segmentation_files is not None
        self.use_masks: bool = (
            subject_mask_files is not None or subject_segmentation_files is not None
        )

        self.output_dir = Path(output_dir).resolve()
        self.fine_tune_name = fine_tune_name
        self.experiment_dir = self.output_dir / fine_tune_name
        self.mask_dir: Optional[Path] = Path(mask_dir) if mask_dir is not None else None

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_shape = tuple(input_shape)
        self.similarity = similarity
        self.lambda_value = lambda_value
        self.dice_loss_weight = dice_loss_weight
        self.lncc_sigma = lncc_sigma
        self.ct_window = tuple(ct_window)
        self.is_ct = is_ct
        self.gpus = list(gpus) if gpus is not None else [0]
        self.eval_period = eval_period
        self.save_period = save_period
        self.mask_exclude_labels = mask_exclude_labels
        self.mask_dilation_mm = float(mask_dilation_mm)
        self.unigradicon_src_path = (
            Path(unigradicon_src_path) if unigradicon_src_path is not None else None
        )

        self.transform_tools = TransformTools()
        self.labelmap_tools = LabelmapTools(log_level=log_level)
        self.registrar: Optional[RegisterTimeSeriesImages] = None

        self._use_segmentations: bool = self.use_segmentations
        self._use_masks: bool = self.use_masks

        self._dataset_json_path: Optional[Path] = None
        self._config_yaml_path: Optional[Path] = None

    @staticmethod
    def _validate_companion_shape(
        image_files: list[list[str]],
        companion: Optional[list[list[Optional[str]]]],
        name: str,
    ) -> None:
        """Confirm a companion list has the same nested shape as ``image_files``."""
        if companion is None:
            return
        if len(companion) != len(image_files):
            raise ValueError(
                f"{name} length ({len(companion)}) must match "
                f"subject_image_files length ({len(image_files)})"
            )
        for i, (images, items) in enumerate(zip(image_files, companion, strict=True)):
            if len(items) != len(images):
                raise ValueError(
                    f"{name}[{i}] length ({len(items)}) must match "
                    f"subject_image_files[{i}] length ({len(images)})"
                )

    @staticmethod
    def _posix(path: Union[str, Path]) -> str:
        """Return a forward-slashed string path (uniGradICON expects POSIX paths)."""
        return str(path).replace("\\", "/")

    def _derive_mask(
        self,
        labelmap_path: Union[str, Path],
    ) -> Path:
        """Create (or reuse) a dilated binary mask from a multi-label labelmap.

        Threshold the labelmap at ``>0`` and dilate by ``mask_dilation_mm`` mm
        of physical radius via
        :meth:`LabelmapTools.convert_labelmap_to_mask` to widen the ROI for
        loss-function masking.

        When :attr:`mask_dir` is ``None`` (the default) the mask is written
        next to the source labelmap as
        ``<labelmap_dir>/<labelmap_stem>_mask.nii.gz``.  Otherwise it goes
        under :attr:`mask_dir`.  Existing masks on disk are reused unmodified.

        Args:
            labelmap_path: Path to a multi-label ``itk.Image`` on disk.

        Returns:
            Path to the binary mask file on disk.
        """
        labelmap_path = Path(labelmap_path)
        stem = labelmap_path.name
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        else:
            stem = labelmap_path.stem
        target_dir = (
            self.mask_dir if self.mask_dir is not None else labelmap_path.parent
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        mask_path = target_dir / f"{stem}_mask.nii.gz"
        if mask_path.exists():
            return mask_path

        labelmap = itk.imread(str(labelmap_path))
        mask = self.labelmap_tools.convert_labelmap_to_mask(
            labelmap,
            dilation_in_mm=self.mask_dilation_mm,
            exclude_labels=self.mask_exclude_labels,
        )
        itk.imwrite(mask, str(mask_path), compression=True)
        return mask_path

    def prepare_dataset(
        self,
        use_segmentations: Optional[bool] = None,
        use_masks: Optional[bool] = None,
    ) -> Path:
        """Write the uniGradICON dataset JSON from the configured file lists.

        Builds one entry per image with ``image``, optional ``segmentation``,
        optional ``mask``, optional ``landmarks`` (path only), and a
        ``subject_id`` derived from the inner-list index.

        Masks are taken from ``subject_mask_files`` when supplied for a frame;
        otherwise they are derived from ``subject_segmentation_files`` via a
        >0 threshold and ``mask_dilation_mm`` mm dilation.  Frames are
        skipped (with a log warning) when a required companion (segmentation
        for paired-with-seg training, or mask for loss-function masking) is
        missing.

        Returns:
            Path to the dataset JSON written under :attr:`experiment_dir`.

        Raises:
            FileNotFoundError: If an image listed in ``subject_image_files``
                does not exist on disk.
        """
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if use_segmentations is None:
            use_segmentations = self.use_segmentations
        if use_masks is None:
            use_masks = self.use_masks

        self._use_segmentations = use_segmentations
        self._use_masks = use_masks

        dataset_entries: list[dict[str, str]] = []
        for subject_index, image_files in enumerate(self.subject_image_files):
            subject_id = (
                self.subject_ids[subject_index]
                if self.subject_ids is not None
                else f"subject_{subject_index:04d}"
            )
            seg_list: list[Optional[str]]
            if not use_segmentations:
                seg_list = [None] * len(image_files)
            else:
                seg_list = (
                    self.subject_segmentation_files[subject_index]
                    if self.subject_segmentation_files is not None
                    else [None] * len(image_files)
                )
            mask_list: list[Optional[str]]
            if not use_masks:
                mask_list = [None] * len(image_files)
            else:
                mask_list = (
                    self.subject_mask_files[subject_index]
                    if self.subject_mask_files is not None
                    else [None] * len(image_files)
                )
            landmark_list = (
                self.subject_landmark_files[subject_index]
                if self.subject_landmark_files is not None
                else [None] * len(image_files)
            )

            for image_file, seg_file, mask_file, landmark_file in zip(
                image_files, seg_list, mask_list, landmark_list, strict=True
            ):
                image_path = Path(image_file)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                entry: dict[str, str] = {
                    "image": self._posix(image_path),
                    "subject_id": subject_id,
                }

                if use_segmentations:
                    if seg_file is None or not Path(seg_file).exists():
                        self.log_warning(
                            "Skipping %s: segmentation missing for paired-with-seg "
                            "training (seg=%s)",
                            image_path,
                            seg_file,
                        )
                        continue
                    entry["segmentation"] = self._posix(seg_file)

                if use_masks:
                    if mask_file is not None and Path(mask_file).exists():
                        resolved_mask: Path = Path(mask_file)
                    elif seg_file is not None and Path(seg_file).exists():
                        resolved_mask = self._derive_mask(seg_file)
                    else:
                        self.log_warning(
                            "Skipping %s: neither explicit mask nor segmentation "
                            "available to derive a loss-function mask "
                            "(mask=%s, seg=%s)",
                            image_path,
                            mask_file,
                            seg_file,
                        )
                        continue
                    entry["mask"] = self._posix(resolved_mask)

                if landmark_file is not None:
                    entry["landmarks"] = self._posix(landmark_file)

                dataset_entries.append(entry)

        dataset_json_path = self.experiment_dir / f"{self.fine_tune_name}_dataset.json"
        with dataset_json_path.open("w") as fh:
            json.dump({"data": dataset_entries}, fh, indent=2)

        self.log_info(
            "Wrote dataset JSON %s with %d entries",
            dataset_json_path,
            len(dataset_entries),
        )
        self._dataset_json_path = dataset_json_path
        return dataset_json_path

    def prepare_config(self, dataset_json_path: Optional[Path] = None) -> Path:
        """Write the uniGradICON fine-tuning YAML config.

        Args:
            dataset_json_path: Path to the dataset JSON to reference.  Defaults
                to the JSON last produced by :meth:`prepare_dataset`.

        Returns:
            Path to the YAML config written under :attr:`experiment_dir`.

        Raises:
            ValueError: If no dataset JSON path is available.
        """
        if dataset_json_path is None:
            dataset_json_path = self._dataset_json_path
        if dataset_json_path is None:
            raise ValueError(
                "dataset_json_path not provided and prepare_dataset() has not "
                "been called yet"
            )

        experiment_name = self.experiment_dir / f"{self.fine_tune_name}_model"

        config: dict[str, Any] = {
            "experiment": {
                "name": self._posix(experiment_name),
                "model_weights": "unigradicon",
            },
            "training": {
                "batch_size": self.batch_size,
                "gpus": self.gpus,
                "epochs": self.epochs,
                "eval_period": self.eval_period,
                "save_period": self.save_period,
                "learning_rate": self.learning_rate,
                "input_shape": list(self.input_shape),
                "similarity": self.similarity,
                "lambda": self.lambda_value,
                "dice_loss_weight": self.dice_loss_weight,
                "lncc_sigma": self.lncc_sigma,
                "loss_function_masking": self._use_masks,
                "use_label": self._use_segmentations,
                "roi_masking": False,
            },
            "datasets": [
                {
                    "name": self.fine_tune_name,
                    "weight": 1.0,
                    "type": "paired",
                    "json_file": self._posix(dataset_json_path),
                    "is_ct": self.is_ct,
                    "ct_window": list(self.ct_window),
                    "shuffle": True,
                    "use_cache": True,
                }
            ],
        }

        config_yaml_path = self.experiment_dir / f"{self.fine_tune_name}_config.yaml"
        with config_yaml_path.open("w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
        self.log_info("Wrote config YAML %s", config_yaml_path)
        self._config_yaml_path = config_yaml_path
        return config_yaml_path

    def expected_weights_path(self) -> Path:
        """Return the path uniGradICON writes its final checkpoint to.

        ``unigradicon.finetuning.finetune`` writes
        ``<experiment.name>/checkpoints/Finetune_multi_final.trch`` at the end of
        training.  Used both as the return value of :meth:`run_fine_tuning` and
        as a default in :meth:`apply_registration`.
        """
        return (
            self.experiment_dir
            / f"{self.fine_tune_name}_model"
            / "checkpoints"
            / "Finetune_multi_final.trch"
        )

    def run_fine_tuning(self) -> Path:
        """Build configs and launch ``unigradicon.finetuning.finetune``.

        Equivalent to running
        ``prepare_dataset()`` → ``prepare_config()`` → subprocess launch.  Any
        existing dataset JSON or YAML in :attr:`experiment_dir` is overwritten.

        Returns:
            Path to the expected final checkpoint
            (``Finetune_multi_final.trch``).  The file is written by the
            subprocess and exists only after a successful run.

        Raises:
            subprocess.CalledProcessError: If the fine-tuning subprocess exits
                with a non-zero status.
        """
        self.log_section("FINE-TUNING UNIGRADICON", width=70)

        dataset_json_path = self.prepare_dataset()
        config_yaml_path = self.prepare_config(dataset_json_path)

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        if self.unigradicon_src_path is not None:
            env["PYTHONPATH"] = (
                str(self.unigradicon_src_path) + os.pathsep + env.get("PYTHONPATH", "")
            )

        cmd = [
            sys.executable,
            "-m",
            "unigradicon.finetuning.finetune",
            "--config",
            str(config_yaml_path),
        ]
        self.log_info("Launching fine-tuning subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        weights_path = self.expected_weights_path()
        self.log_info("Fine-tuning complete. Expected weights at %s", weights_path)
        return weights_path

    @staticmethod
    def _transform_landmarks(
        landmarks: Landmarks, transform: itk.Transform
    ) -> Landmarks:
        """Apply ``transform.TransformPoint`` to every landmark in physical LPS space."""
        transformed: Landmarks = {}
        for name, point in landmarks.items():
            new_point = transform.TransformPoint(point)
            transformed[name] = (
                float(new_point[0]),
                float(new_point[1]),
                float(new_point[2]),
            )
        return transformed

    def apply_registration(
        self,
        reference_image: itk.Image,
        moving_images: list[itk.Image],
        weights_path: Optional[Union[str, Path]] = None,
        reference_segmentation: Optional[itk.Image] = None,
        reference_landmarks: Optional[Landmarks] = None,
        moving_segmentations: Optional[list[Optional[itk.Image]]] = None,
        moving_landmarks: Optional[list[Optional[Landmarks]]] = None,
        number_of_iterations: int = 20,
        modality: str = "ct",
    ) -> dict[str, Any]:
        """Register each moving image to the reference using fine-tuned ICON weights.

        For every moving image this method:

        - Runs ICON registration ``moving → reference``.  When a moving
          segmentation is provided, a binary heart-ROI mask is derived from it
          and passed as the registration mask so the ICON loss only sees the
          ROI; the same is done for the reference segmentation (used as the
          fixed mask).
        - Warps the moving image, segmentation, and landmarks into reference
          space using ``forward_transform``.  Segmentations use nearest-neighbor
          interpolation.  Landmarks use ``inverse_transform.TransformPoint``
          (resampler-convention transform: maps moving-grid points back to
          reference-grid points).
        - Warps the reference image, segmentation, and landmarks into each
          moving-image space using ``inverse_transform`` for image/segmentation
          resampling and ``forward_transform.TransformPoint`` for landmarks.

        Args:
            reference_image: Fixed (reference) ``itk.Image`` in LPS.
            moving_images: List of moving ``itk.Image`` instances to register
                to ``reference_image``.
            weights_path: Path to a uniGradICON checkpoint (e.g.
                ``Finetune_multi_final.trch``).  ``None`` uses the default
                pretrained uniGradICON weights.
            reference_segmentation: Optional multi-label labelmap aligned with
                ``reference_image``.  Used to derive the fixed-image mask and
                returned warped into each moving-image space.
            reference_landmarks: Optional ``{name: (x, y, z)}`` landmark dict in
                LPS that will be warped into each moving-image space.
            moving_segmentations: Optional per-moving multi-label labelmaps
                aligned with ``moving_images``.  Used to derive per-moving
                masks and returned warped into reference space.  Per-image
                ``None`` entries are allowed.
            moving_landmarks: Optional per-moving landmark dicts in LPS.  Each
                set is warped into reference space.  Per-image ``None`` entries
                are allowed.
            number_of_iterations: ICON fine-tuning iterations per registration.
            modality: Imaging modality passed through to the underlying ICON
                registrar (``'ct'`` or ``'mri'``).

        Returns:
            dict with:

                - ``forward_transforms`` (``list[itk.Transform]``): per-moving
                  transforms mapping reference grid → moving grid (used to
                  resample moving → reference).
                - ``inverse_transforms`` (``list[itk.Transform]``): per-moving
                  transforms mapping moving grid → reference grid (used to
                  resample reference → moving).
                - ``losses`` (``list[float]``): per-moving registration loss.
                - ``moving_to_reference_images`` (``list[itk.Image]``): each
                  moving image resampled onto the reference grid.
                - ``moving_to_reference_segmentations`` (``list[Optional[itk.Image]]``):
                  each moving segmentation resampled onto the reference grid
                  with nearest-neighbor interpolation.  ``None`` when the input
                  was ``None``.
                - ``moving_to_reference_landmarks`` (``list[Optional[Landmarks]]``):
                  each moving landmark set warped into reference space.
                  ``None`` when the input was ``None``.
                - ``reference_to_moving_images`` (``list[itk.Image]``): the
                  reference image resampled onto each moving grid.
                - ``reference_to_moving_segmentations`` (``list[Optional[itk.Image]]``):
                  the reference segmentation resampled onto each moving grid
                  with nearest-neighbor interpolation.  ``None`` for every
                  entry when ``reference_segmentation`` was ``None``.
                - ``reference_to_moving_landmarks`` (``list[Optional[Landmarks]]``):
                  reference landmarks warped into each moving space.  ``None``
                  for every entry when ``reference_landmarks`` was ``None``.

        Raises:
            ValueError: If ``moving_images`` is empty.
            ValueError: If ``moving_segmentations`` or ``moving_landmarks`` is
                supplied with a length that does not match ``moving_images``.
        """
        if not moving_images:
            raise ValueError("moving_images must not be empty")
        num_moving = len(moving_images)
        if moving_segmentations is not None and len(moving_segmentations) != num_moving:
            raise ValueError(
                f"moving_segmentations length ({len(moving_segmentations)}) must "
                f"match moving_images length ({num_moving})"
            )
        if moving_landmarks is not None and len(moving_landmarks) != num_moving:
            raise ValueError(
                f"moving_landmarks length ({len(moving_landmarks)}) must match "
                f"moving_images length ({num_moving})"
            )

        self.log_section("APPLYING FINE-TUNED ICON REGISTRATION", width=70)
        self.log_info("Number of moving images: %d", num_moving)
        if weights_path is None:
            self.log_info("ICON weights: <default uniGradICON>")
        else:
            self.log_info("ICON weights: %s", weights_path)

        fixed_mask = (
            self.labelmap_tools.convert_labelmap_to_mask(
                reference_segmentation, dilation_in_mm=self.mask_dilation_mm
            )
            if reference_segmentation is not None
            else None
        )
        moving_masks: Optional[list[Optional[itk.Image]]] = None
        if moving_segmentations is not None:
            moving_masks = [
                (
                    self.labelmap_tools.convert_labelmap_to_mask(
                        seg, dilation_in_mm=self.mask_dilation_mm
                    )
                    if seg is not None
                    else None
                )
                for seg in moving_segmentations
            ]

        self.registrar = RegisterTimeSeriesImages(
            registration_method="ICON", log_level=self.log_level
        )
        self.registrar.set_modality(modality)
        self.registrar.set_fixed_image(reference_image)
        self.registrar.set_fixed_mask(fixed_mask)
        self.registrar.set_number_of_iterations_ICON(number_of_iterations)
        if weights_path is not None:
            self.registrar.registrar_ICON.set_weights_path(str(weights_path))

        result = self.registrar.register_time_series(
            moving_images=moving_images,
            moving_masks=moving_masks,
            moving_labelmaps=None,
            reference_frame=0,
            register_reference=True,
            prior_weight=0.0,
        )
        forward_transforms = result["forward_transforms"]
        inverse_transforms = result["inverse_transforms"]
        losses = result["losses"]

        moving_to_reference_images: list[itk.Image] = []
        moving_to_reference_segmentations: list[Optional[itk.Image]] = []
        moving_to_reference_landmarks: list[Optional[Landmarks]] = []
        reference_to_moving_images: list[itk.Image] = []
        reference_to_moving_segmentations: list[Optional[itk.Image]] = []
        reference_to_moving_landmarks: list[Optional[Landmarks]] = []

        for index in range(num_moving):
            forward_tfm = forward_transforms[index]
            inverse_tfm = inverse_transforms[index]
            moving_image = moving_images[index]

            moving_to_reference_images.append(
                self.transform_tools.transform_image(
                    moving_image, forward_tfm, reference_image
                )
            )
            reference_to_moving_images.append(
                self.transform_tools.transform_image(
                    reference_image, inverse_tfm, moving_image
                )
            )

            moving_seg = (
                moving_segmentations[index]
                if moving_segmentations is not None
                else None
            )
            if moving_seg is not None:
                moving_to_reference_segmentations.append(
                    self.transform_tools.transform_image(
                        moving_seg,
                        forward_tfm,
                        reference_image,
                        interpolation_method="nearest",
                    )
                )
            else:
                moving_to_reference_segmentations.append(None)

            if reference_segmentation is not None:
                reference_to_moving_segmentations.append(
                    self.transform_tools.transform_image(
                        reference_segmentation,
                        inverse_tfm,
                        moving_image,
                        interpolation_method="nearest",
                    )
                )
            else:
                reference_to_moving_segmentations.append(None)

            moving_lms = (
                moving_landmarks[index] if moving_landmarks is not None else None
            )
            if moving_lms is not None:
                moving_to_reference_landmarks.append(
                    self._transform_landmarks(moving_lms, inverse_tfm)
                )
            else:
                moving_to_reference_landmarks.append(None)

            if reference_landmarks is not None:
                reference_to_moving_landmarks.append(
                    self._transform_landmarks(reference_landmarks, forward_tfm)
                )
            else:
                reference_to_moving_landmarks.append(None)

        self.log_info(
            "Average ICON loss: %.6f (min %.6f, max %.6f)",
            float(np.mean(losses)),
            float(np.min(losses)),
            float(np.max(losses)),
        )

        return {
            "forward_transforms": forward_transforms,
            "inverse_transforms": inverse_transforms,
            "losses": losses,
            "moving_to_reference_images": moving_to_reference_images,
            "moving_to_reference_segmentations": moving_to_reference_segmentations,
            "moving_to_reference_landmarks": moving_to_reference_landmarks,
            "reference_to_moving_images": reference_to_moving_images,
            "reference_to_moving_segmentations": reference_to_moving_segmentations,
            "reference_to_moving_landmarks": reference_to_moving_landmarks,
        }
