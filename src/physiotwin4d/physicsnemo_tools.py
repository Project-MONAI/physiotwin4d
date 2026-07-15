"""Shared helpers for the PhysicsNeMo train / infer workflows.

This module holds the pieces common to the MeshGraphNet (MGN) and fully
connected (MLP) PhysicsNeMo workflows so the workflow classes stay focused on
orchestration.  It provides:

- :class:`SubjectManifest` / :func:`parse_manifest` — the per-subject JSON
  manifest that lists a reference surface, a PCA shape-parameter file, and the
  gated-phase surfaces with their stages.
- :func:`build_node_features` — the shared per-vertex feature layout
  ``[mean_coords_norm, pca_norm (tiled), stage]`` used by both networks.
- :func:`mesh_to_edge_index` / :func:`compute_edge_features` — MGN mesh-graph
  construction from the shared mean-shape surface.
- :func:`reconstruct_reference_points` — rebuild a subject reference surface
  from PCA shape parameters (``P = mean + Σ b_i·std_i·eigenvector_i``), used for
  manifest-free single-subject inference.
- :func:`uncompiled_state_dict` / :func:`strip_compile_prefix` — checkpoint I/O
  that is robust to ``torch.compile`` wrapping.
- :class:`PhaseSampleDataset` — a lazy ``(subject, phase)`` sample provider with
  a bounded in-RAM cache so the training set need not fit in memory.

``torch`` and ``torch_geometric`` are optional dependencies; every function that
needs them imports them locally so ``import physiotwin4d`` works without the
``[physicsnemo]`` extra installed.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyvista as pv

if TYPE_CHECKING:  # imported lazily at runtime; typed here for mypy only
    import torch


# --------------------------------------------------------------------------- #
# Per-subject manifest                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class PhaseEntry:
    """One gated-phase surface and its normalized cardiac stage."""

    surface: Path
    stage: float


@dataclass
class SubjectManifest:
    """A single subject's training/inference inputs.

    Attributes:
        subject_id: Identifier used for output naming.
        reference_surface: The subject's SSM reference surface (``.vtp``); the
            displacement origin for the Option B convention
            (``target = phase.points - reference.points``).
        pca_coefficients: JSON file holding the subject's PCA shape-parameter
            vector (a flat list of floats).
        phases: One :class:`PhaseEntry` per gated phase (at least one).
    """

    subject_id: str
    reference_surface: Path
    pca_coefficients: Path
    phases: list[PhaseEntry]


def parse_manifest(manifest_path: Path) -> SubjectManifest:
    """Parse a per-subject JSON manifest.

    Paths inside the manifest are resolved relative to the manifest's own
    directory unless already absolute.  Every phase must declare a ``stage``.

    Args:
        manifest_path: Path to the subject manifest JSON file.

    Returns:
        The parsed :class:`SubjectManifest`.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If required fields are missing, a phase lacks ``stage``, or
            no phases are listed.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent

    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p)

    for key in ("subject_id", "reference_surface", "pca_coefficients", "phases"):
        if key not in data:
            raise ValueError(f"Manifest {manifest_path} is missing '{key}'.")

    raw_phases = data["phases"]
    if not raw_phases:
        raise ValueError(f"Manifest {manifest_path} lists no phases.")

    phases: list[PhaseEntry] = []
    for entry in raw_phases:
        if "surface" not in entry or "stage" not in entry:
            raise ValueError(
                f"Manifest {manifest_path} has a phase missing 'surface' or "
                "'stage' (stage must be supplied by the caller)."
            )
        phases.append(
            PhaseEntry(surface=_resolve(entry["surface"]), stage=float(entry["stage"]))
        )

    return SubjectManifest(
        subject_id=str(data["subject_id"]),
        reference_surface=_resolve(data["reference_surface"]),
        pca_coefficients=_resolve(data["pca_coefficients"]),
        phases=phases,
    )


def load_pca_coefficients(path: Path) -> np.ndarray:
    """Load a PCA shape-parameter vector saved as a JSON list of floats."""
    return np.asarray(
        json.loads(Path(path).read_text(encoding="utf-8")), dtype=np.float32
    )


# --------------------------------------------------------------------------- #
# Feature construction (shared by MGN and MLP)                                 #
# --------------------------------------------------------------------------- #
def build_node_features(
    mean_coords_norm: np.ndarray, pca_norm: np.ndarray, stage: float
) -> np.ndarray:
    """Assemble per-vertex node features ``[coords_norm, pca_norm, stage]``.

    Args:
        mean_coords_norm: ``(n_points, 3)`` normalized mean-shape coordinates
            (identical for every subject/phase).
        pca_norm: ``(n_pca,)`` normalized PCA shape parameters for the subject.
        stage: Normalized cardiac stage (RR-interval fraction) for the phase.

    Returns:
        ``(n_points, 3 + n_pca + 1)`` float32 feature array.
    """
    n = len(mean_coords_norm)
    pca_tile = np.tile(pca_norm, (n, 1))
    stage_col = np.full((n, 1), stage, dtype=np.float32)
    return np.hstack([mean_coords_norm, pca_tile, stage_col]).astype(np.float32)


# --------------------------------------------------------------------------- #
# MGN mesh-graph construction                                                  #
# --------------------------------------------------------------------------- #
def mesh_to_edge_index(poly: pv.PolyData) -> "torch.Tensor":
    """Build an undirected ``edge_index`` from triangulated PolyData faces.

    Args:
        poly: Triangulated surface whose ``faces`` array encodes the topology.

    Returns:
        ``(2, n_edges)`` long tensor of undirected edges.
    """
    import torch
    import torch_geometric.utils as pyg_utils

    faces = poly.faces.reshape(-1, 4)[:, 1:]  # (F, 3) - strip leading count
    src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return cast("torch.Tensor", pyg_utils.to_undirected(edge_index))


def compute_edge_features(
    coords: np.ndarray, edge_index: "torch.Tensor"
) -> "torch.Tensor":
    """Build ``(n_edges, 4)`` edge features ``[rel_x, rel_y, rel_z, distance]``."""
    import torch

    ei = edge_index.numpy()
    disp = coords[ei[1]] - coords[ei[0]]
    dist = np.linalg.norm(disp, axis=1, keepdims=True)
    return torch.tensor(np.hstack([disp, dist]), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# PCA reconstruction (manifest-free inference)                                 #
# --------------------------------------------------------------------------- #
def reconstruct_reference_points(
    mean_mesh: pv.DataSet, pca_model: dict, coeffs: np.ndarray
) -> np.ndarray:
    """Reconstruct a subject reference surface from PCA shape parameters.

    Applies the statistical-shape-model equation
    ``P = mean + Σ b_i·std_i·eigenvector_i`` on the PCA template *mesh*
    (whose ``components`` are defined) and returns the extracted surface points.
    Because every subject shares the template topology, the extracted surface
    vertex ordering matches the shared mean-shape surface used for training.

    Args:
        mean_mesh: PCA template mesh (e.g. ``pca_mean.vtu``) whose point count
            matches the model components.
        pca_model: Dict with ``eigenvalues`` and ``components`` (the
            ``pca_model.json`` format).
        coeffs: Subject PCA coefficients ``b_i`` (in units of standard
            deviations); shorter/longer than the mode count is truncated.

    Returns:
        ``(n_surface_points, 3)`` float32 reconstructed surface points.

    Raises:
        ValueError: If the component dimension does not match ``mean_mesh``.
    """
    std = np.sqrt(np.asarray(pca_model["eigenvalues"], dtype=np.float64))
    components = np.asarray(pca_model["components"], dtype=np.float64)
    expected = mean_mesh.n_points * 3
    if components.shape[1] != expected:
        raise ValueError(
            f"PCA component dimension {components.shape[1]} does not match "
            f"mean mesh ({expected} = 3 x {mean_mesh.n_points} points)."
        )

    b = np.asarray(coeffs, dtype=np.float64)
    n_modes = min(len(b), len(std), components.shape[0])
    deform_flat = (b[:n_modes] * std[:n_modes]) @ components[:n_modes]
    deform = deform_flat.reshape(-1, 3)

    deformed = mean_mesh.copy(deep=True)
    deformed.points = np.asarray(mean_mesh.points, dtype=np.float64) + deform
    surface = deformed.extract_surface(algorithm="dataset_surface")
    return np.asarray(surface.points, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Checkpoint I/O                                                               #
# --------------------------------------------------------------------------- #
def uncompiled_state_dict(model: Any) -> dict:
    """Return a model's state dict, unwrapping ``torch.compile`` if applied."""
    return cast(dict, getattr(model, "_orig_mod", model).state_dict())


def strip_compile_prefix(state: dict) -> dict:
    """Strip the ``_orig_mod.`` prefix that ``torch.compile`` adds to keys."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state):
        return {
            k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()
        }
    return state


# --------------------------------------------------------------------------- #
# Lazy dataset with bounded RAM cache                                          #
# --------------------------------------------------------------------------- #
@dataclass
class _Sample:
    """Everything needed to materialize one ``(subject, phase)`` sample."""

    subject_id: str
    pca_norm: np.ndarray  # (n_pca,) normalized shape parameters
    ref_points: np.ndarray  # (n_points, 3) subject reference surface
    phase_surface: Path
    stage: float


class PhaseSampleDataset:
    """Lazy provider of ``(node_features, target_displacement)`` samples.

    One item is one ``(subject, phase)`` pair.  Node features are rebuilt on
    access from the shared normalized mean-shape coordinates plus the subject's
    normalized PCA parameters and the phase stage (cheap).  Only the phase
    surface point arrays are read from disk, and those are held in a bounded
    LRU cache so an arbitrarily large training set streams from disk while a
    small set stays resident.

    Args:
        samples: Flat list of :class:`_Sample` (built by the workflow).
        mean_coords_norm: ``(n_points, 3)`` normalized mean-shape coordinates.
        displacement_scale: Target normalization factor (targets are divided by
            it so displacements land in ``~[-1, 1]``).
        cache_max_samples: Maximum decoded phase arrays to cache. ``0`` means
            unbounded (all-in-RAM, fastest); a small value forces disk streaming.
    """

    def __init__(
        self,
        samples: list[_Sample],
        mean_coords_norm: np.ndarray,
        displacement_scale: float,
        cache_max_samples: int = 0,
    ) -> None:
        self._samples = samples
        self._mean_coords_norm = mean_coords_norm.astype(np.float32)
        self._displacement_scale = float(displacement_scale)
        self._cache_max_samples = int(cache_max_samples)
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()
        self._n_points = int(mean_coords_norm.shape[0])
        self._n_features = int(3 + samples[0].pca_norm.shape[0] + 1) if samples else 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def n_points(self) -> int:
        """Vertices per sample (shared across all subjects)."""
        return self._n_points

    @property
    def n_features(self) -> int:
        """Node feature dimension ``3 + n_pca + 1``."""
        return self._n_features

    def _phase_points(self, path: Path) -> np.ndarray:
        """Read (and cache) a phase surface's point array."""
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        mesh = pv.read(str(path))
        if mesh.n_points != self._n_points:
            raise ValueError(
                f"{path} has {mesh.n_points} points, expected {self._n_points}."
            )
        points = np.asarray(mesh.points, dtype=np.float32)

        if self._cache_max_samples != 0:
            self._cache[path] = points
            while len(self._cache) > self._cache_max_samples:
                self._cache.popitem(last=False)
        return points

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(node_features, normalized_target)`` for one sample."""
        sample = self._samples[index]
        node_feats = build_node_features(
            self._mean_coords_norm, sample.pca_norm, sample.stage
        )
        phase_points = self._phase_points(sample.phase_surface)
        target = (phase_points - sample.ref_points) / self._displacement_scale
        return node_feats, target.astype(np.float32)
