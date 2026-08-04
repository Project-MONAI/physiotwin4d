"""Inference methods for PhysicsNeMo mesh-stage models.

An inference method owns the network: rebuilding it from a checkpoint's
metadata, loading any architecture-specific artifacts, and running the forward
pass. Everything around it — checkpoint loading, normalization statistics,
manifests and output writing — lives in the workflow that drives the method
(:mod:`physiotwin4d.workflow_infer_physicsnemo`).

:class:`InferPhysicsNeMoBase` mirrors
:class:`physiotwin4d.TrainPhysicsNeMoBase`: the concrete
:class:`physiotwin4d.InferPhysicsNeMoMGN` and
:class:`physiotwin4d.InferPhysicsNeMoMLP` subclasses supply only the
network-specific seams.

PhysicsNeMo (and, for the MGN, PyTorch Geometric) are optional dependencies;
they are imported lazily so ``import physiotwin4d`` works without them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .physiotwin4d_base import PhysioTwin4DBase

if TYPE_CHECKING:  # typed for mypy; imported lazily at runtime
    import torch


class InferPhysicsNeMoBase(PhysioTwin4DBase):
    """Base class for a PhysicsNeMo mesh-stage inference method.

    Not instantiated directly — use :class:`physiotwin4d.InferPhysicsNeMoMGN` or
    :class:`physiotwin4d.InferPhysicsNeMoMLP`. Subclasses implement
    :meth:`build_model`, :meth:`load_artifacts` and :meth:`predict`, and set the
    class attribute ``model_tag``.
    """

    model_tag: str = "base"

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the inference method.

        Args:
            log_level: Logging level. Default: ``logging.INFO``.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self._model: "torch.nn.Module"
        self._device: "torch.device"

    def build_model(self, meta: dict) -> "torch.nn.Module":
        """Rebuild the (uncompiled) network from checkpoint metadata."""
        raise NotImplementedError

    def load_artifacts(
        self, model_directory: Path, n_points: int, device: "torch.device"
    ) -> None:
        """Load any architecture-specific artifacts (MGN graph tensors)."""
        raise NotImplementedError

    def predict(self, node_feats: np.ndarray) -> np.ndarray:
        """Run the network over all nodes; return the ``(n, n_target)`` output."""
        raise NotImplementedError

    def set_model(self, model: "torch.nn.Module", device: "torch.device") -> None:
        """Attach the loaded model and its device before predicting."""
        self._model = model
        self._device = device
