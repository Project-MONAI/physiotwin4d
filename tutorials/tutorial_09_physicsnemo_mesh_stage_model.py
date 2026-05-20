"""
Tutorial 9: Train a PhysicsNeMo model for DirLab mesh time-stage prediction.

This tutorial uses the per-time-point PCA-fitted meshes created by Tutorial 8.
For each case, it trains a small PhysicsNeMo fully connected model that maps
reference mesh point coordinates and a requested normalized respiratory stage to
point displacements. The trained model can then predict a mesh at a new
user-specified stage without running image registration again.

Data Required
-------------
Run Tutorial 8 first so ``output/tutorial_08_dirlab_pca_time_series`` contains
``Case*/meshes/*_pca_fit.vtp`` files.
"""

# %%
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyvista as pv
import torch
from physicsnemo.models.mlp import FullyConnected


# nnUNetv2 (used by TotalSegmentator inside several workflows) spawns a
# multiprocessing.Pool. On Windows the spawn start method re-imports this
# script in each child; without the __name__ == "__main__" guard around
# top-level work, that re-import fires the segmenter again and Python's
# spawn-cascade detector raises RuntimeError. Wrapping consistently across
# tutorials also matches the style of tutorial_01.
if __name__ == "__main__":
    # %%
    TUTORIALS_DIR = Path(__file__).resolve().parent
    TUTORIAL_08_OUTPUT_DIR = (
        TUTORIALS_DIR / "output" / "tutorial_08_dirlab_pca_time_series"
    )
    OUTPUT_DIR = TUTORIALS_DIR / "output" / "tutorial_09_physicsnemo_mesh_stage_model"
    TARGET_STAGE = 0.5
    CASE: Optional[int] = None
    EPOCHS = 500
    POINTS_PER_MESH = 4096
    LEARNING_RATE = 1.0e-3
    LOG_LEVEL = logging.INFO

    DIRLAB_CASE_PREFIXES = [
        "Case1Pack",
        "Case2Pack",
        "Case3Pack",
        "Case4Pack",
        "Case5Pack",
        "Case6Pack",
        "Case7Pack",
        "Case8Deploy",
        "Case9Pack",
        "Case10Pack",
    ]

    def run_tutorial() -> dict[str, Any]:
        """Train PhysicsNeMo stage models and predict meshes at ``target_stage``.

        Returns
        -------
        dict[str, Any]
            Per-case checkpoint, metadata, predicted mesh, and training loss paths.
        """

        tutorial_08_output_dir = TUTORIAL_08_OUTPUT_DIR
        output_dir = OUTPUT_DIR
        target_stage = TARGET_STAGE
        case = CASE
        epochs = EPOCHS
        points_per_mesh = POINTS_PER_MESH
        learning_rate = LEARNING_RATE
        log_level = LOG_LEVEL

        logging.basicConfig(level=log_level)
        if target_stage < 0.0 or target_stage > 1.0:
            raise ValueError("target_stage must be in the normalized range [0.0, 1.0].")

        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_cases = len(DIRLAB_CASE_PREFIXES)
        if case is None:
            case_numbers = list(range(1, num_cases + 1))
        else:
            if not 1 <= case <= num_cases:
                raise ValueError(
                    f"CASE={case} is out of range; must be an integer between 1 "
                    f"and {num_cases} (inclusive)."
                )
            case_numbers = [case]

        tutorial_outputs: dict[str, Any] = {}
        for case_number in case_numbers:
            case_prefix = DIRLAB_CASE_PREFIXES[case_number - 1]
            mesh_dir = tutorial_08_output_dir / case_prefix / "meshes"
            mesh_files = (
                sorted(mesh_dir.glob("*_pca_fit.vtp")) if mesh_dir.exists() else []
            )
            if len(mesh_files) < 2:
                message = (
                    f"Tutorial 8 output for {case_prefix} is missing or incomplete: "
                    f"found {len(mesh_files)} '*_pca_fit.vtp' file(s) in {mesh_dir}, "
                    "expected at least 2. Run Tutorial 8 before Tutorial 9."
                )
                if case is not None:
                    raise FileNotFoundError(message)
                logging.info(f"Skipping {case_prefix}: {message}")
                continue

            case_output_dir = output_dir / case_prefix
            case_output_dir.mkdir(parents=True, exist_ok=True)

            reference_mesh = pv.read(str(mesh_files[0]))
            reference_points = np.asarray(reference_mesh.points, dtype=np.float32)
            if points_per_mesh <= 0 or points_per_mesh >= reference_mesh.n_points:
                point_indices = np.arange(reference_mesh.n_points)
            else:
                point_indices = np.linspace(
                    0,
                    reference_mesh.n_points - 1,
                    points_per_mesh,
                    dtype=np.int64,
                )

            coordinate_mean = reference_points.mean(axis=0)
            coordinate_scale = reference_points.std(axis=0)
            coordinate_scale = np.where(coordinate_scale == 0.0, 1.0, coordinate_scale)
            normalized_reference_points = (
                reference_points[point_indices] - coordinate_mean
            ) / coordinate_scale

            training_inputs: list[np.ndarray] = []
            training_targets: list[np.ndarray] = []
            stage_denominator = max(1, len(mesh_files) - 1)
            for stage_index, mesh_file in enumerate(mesh_files):
                mesh = pv.read(str(mesh_file))
                if mesh.n_points != reference_mesh.n_points:
                    raise ValueError(
                        f"{mesh_file} has {mesh.n_points} points, expected "
                        f"{reference_mesh.n_points}. Tutorial 8 meshes must share topology."
                    )

                stage = stage_index / stage_denominator
                stage_column = np.full((len(point_indices), 1), stage, dtype=np.float32)
                training_inputs.append(
                    np.hstack([normalized_reference_points, stage_column])
                )
                training_targets.append(
                    np.asarray(mesh.points[point_indices], dtype=np.float32)
                    - reference_points[point_indices]
                )

            inputs_array = np.vstack(training_inputs).astype(np.float32)
            targets_array = np.vstack(training_targets).astype(np.float32)
            displacement_scale = float(np.max(np.abs(targets_array)))
            if displacement_scale == 0.0:
                displacement_scale = 1.0
            targets_array = targets_array / displacement_scale

            inputs_tensor = torch.from_numpy(inputs_array).to(device)
            targets_tensor = torch.from_numpy(targets_array).to(device)

            model = FullyConnected(
                in_features=4,
                layer_size=128,
                out_features=3,
                num_layers=4,
                activation_fn="silu",
                skip_connections=True,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_function = torch.nn.MSELoss()

            losses: list[float] = []
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                prediction = model(inputs_tensor)
                loss = loss_function(prediction, targets_tensor)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
                    logging.info(
                        f"{case_prefix} epoch {epoch + 1:04d}/{epochs}: "
                        f"loss={losses[-1]:.6f}"
                    )

            model.eval()
            all_normalized_points = (
                reference_points - coordinate_mean
            ) / coordinate_scale
            all_stage_column = np.full(
                (reference_mesh.n_points, 1),
                target_stage,
                dtype=np.float32,
            )
            prediction_inputs = np.hstack([all_normalized_points, all_stage_column])
            predicted_displacements: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, reference_mesh.n_points, 65536):
                    stop = min(start + 65536, reference_mesh.n_points)
                    prediction_tensor = torch.from_numpy(
                        prediction_inputs[start:stop].astype(np.float32)
                    ).to(device)
                    predicted_displacements.append(
                        model(prediction_tensor).cpu().numpy() * displacement_scale
                    )

            predicted_mesh = reference_mesh.copy(deep=True)
            predicted_mesh.points = reference_points + np.vstack(
                predicted_displacements
            )

            stage_tag = f"{target_stage:.3f}".replace(".", "p")
            checkpoint_file = case_output_dir / "physicsnemo_stage_model.pt"
            metadata_file = case_output_dir / "physicsnemo_stage_model_metadata.json"
            losses_file = case_output_dir / "training_losses.json"
            predicted_mesh_file = (
                case_output_dir / f"{case_prefix}_stage_{stage_tag}.vtp"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "coordinate_mean": coordinate_mean.tolist(),
                    "coordinate_scale": coordinate_scale.tolist(),
                    "displacement_scale": displacement_scale,
                    "target_stage": target_stage,
                    "mesh_files": [str(mesh_file) for mesh_file in mesh_files],
                },
                checkpoint_file,
            )
            metadata_file.write_text(
                json.dumps(
                    {
                        "architecture": "physicsnemo.models.mlp.FullyConnected",
                        "input_features": [
                            "reference_x_normalized",
                            "reference_y_normalized",
                            "reference_z_normalized",
                            "normalized_stage",
                        ],
                        "output_features": ["dx", "dy", "dz"],
                        "target_stage": target_stage,
                        "epochs": epochs,
                        "points_per_mesh": len(point_indices),
                        "learning_rate": learning_rate,
                        "coordinate_mean": coordinate_mean.tolist(),
                        "coordinate_scale": coordinate_scale.tolist(),
                        "displacement_scale": displacement_scale,
                        "training_meshes": [str(mesh_file) for mesh_file in mesh_files],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            losses_file.write_text(json.dumps(losses, indent=2), encoding="utf-8")
            predicted_mesh.save(predicted_mesh_file)

            tutorial_outputs[case_prefix] = {
                "checkpoint_file": checkpoint_file,
                "metadata_file": metadata_file,
                "losses_file": losses_file,
                "predicted_mesh_file": predicted_mesh_file,
                "final_loss": losses[-1],
            }

        return tutorial_outputs

    # %%
    # Run this cell in VS Code or Cursor:
    tutorial_results = run_tutorial()
