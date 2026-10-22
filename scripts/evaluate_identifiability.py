"""
Rotational identifiability analysis.

Author: Gwenolé Quellec
Year: 2026

This module illustrates latent reparameterization ambiguities using a trained
model. It does not retrain the network.

Analysis principles
-------------------
- a full-dimensional orthogonal rotation is applied to the learned latent codes;
- the decoder and transition model are reparameterized consistently;
- reconstruction, rollout prediction, and Euclidean transition errors are
  checked for numerical invariance;
- coordinate-dependent minimality and structural penalties are re-evaluated;
- the paper's visualization projection is reused for both latent scatter plots.

Two distinct PCA projections are used intentionally. An unstandardized PCA of
original latent codes supplies an orthonormal rotation basis. For panels A and
B, the publication projection is fitted by StandardScaler + PCA on the saved
original and counterfactual codes, exactly as in scripts.visualization.

Examples
--------

Assessing identifiability for a trained configuration (e.g., P29, seed 0):

    python -m scripts.evaluate_identifiability \\
        --checkpoint runs/P29-seed-0/best.pt \\
        --data-dir data \\
        --split test \\
        --output-dir identifiability/P29-seed-0 \\
        --device cuda

The default sweep covers 0-360 degrees in 5-degree increments. The example
in panel B uses a 60-degree rotation in the raw four-dimensional latent
space. The displayed projection is not itself rotated by 60 degrees. This
experiment demonstrates coordinate ambiguity, not identifiability of the
underlying physical state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from torch import Tensor

from clsm.datasets import CLSMDataset
from clsm.losses import minimality_loss, structural_constraint_loss
from clsm.models import CLSMModel, CLSMModelConfig
from scripts.visualization import (
    LatentProjection,
    fit_latent_projection,
    load_encoded,
    save_figure,
    transform_latent,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SPLIT = "test"
DEFAULT_BATCH_SIZE = 128
DEFAULT_HORIZON = 5

DEFAULT_MAX_ANGLE = 360.0
DEFAULT_N_ANGLES = 73
DEFAULT_EXAMPLE_ANGLE = 60.0
DEFAULT_ROTATION_AXES = (0, 1)
DEFAULT_PCA_SIGNS = (1, 1)

DEFAULT_MAX_LATENT_POINTS = 5_000
DEFAULT_VISUALIZATION_SEED = 42

DEFAULT_ABSOLUTE_TOLERANCE = 1e-4
DEFAULT_RELATIVE_MSE_TOLERANCE = 1e-4

STRUCTURAL_VARIANCE_EPSILON = 1e-4
NORMALIZATION_EPSILON = 1e-12

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# =============================================================================
# Result data structures
# =============================================================================

@dataclass(frozen=True)
class RotationEvaluation:
    """Measured losses, original latents, and the raw orthogonal PCA basis."""

    rows: list[dict[str, float]]
    latent: Tensor
    raw_pca: PCA
    raw_basis: Tensor


@dataclass(frozen=True)
class DisplayData:
    """Saved evaluation arrays and the projection used in the manuscript."""

    latent: np.ndarray
    true_state: np.ndarray
    projection: LatentProjection
    encoded_path: Path
    max_latent_discrepancy: float


# =============================================================================
# Model and artifact loading
# =============================================================================

def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CLSMModel, dict[str, object]]:
    """Restore a trusted deterministic checkpoint without retraining."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_config = dict(checkpoint["model_config"])

    for key in (
        "encoder_hidden_dims",
        "decoder_hidden_dims",
        "transition_hidden_dims",
    ):
        if key in model_config:
            model_config[key] = tuple(model_config[key])

    model = CLSMModel(
        CLSMModelConfig(**model_config)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    if model.config.variational:
        raise ValueError(
            "This analysis requires the deterministic model used in the paper."
        )

    return model, checkpoint


def load_test_observations(
    data_dir: Path,
    split: str,
    model: CLSMModel,
    device: torch.device,
    max_episodes: int | None,
) -> tuple[Tensor, CLSMDataset, Path]:
    """Load observations from the specified benchmark split."""
    dataset_path = data_dir / f"{split}.npz"
    dataset = CLSMDataset.load(dataset_path)

    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max-episodes must be strictly positive.")

        dataset = dataset.subset(
            slice(0, min(max_episodes, dataset.n_episodes))
        )

    if dataset.observation_dim != model.config.observation_dim:
        raise ValueError(
            "The observation dimensions in the checkpoint and dataset differ."
        )

    observations = torch.as_tensor(
        dataset.observation,
        device=device,
        dtype=torch.float32,
    )

    return observations, dataset, dataset_path


def load_paper_display_data(
    encoded_path: Path,
    original_latent: Tensor,
) -> DisplayData:
    """Reproduce the standardized PCA used by visualization.py."""
    encoded = load_encoded(encoded_path)
    required = (
        "latent",
        "counterfactual_latent",
        "true_state",
    )

    missing = [
        key
        for key in required
        if key not in encoded
    ]

    if missing:
        raise KeyError(
            f"Missing encoded data fields: {missing}."
        )

    latent = np.asarray(
        encoded["latent"]
    )
    counterfactual_latent = np.asarray(
        encoded["counterfactual_latent"]
    )

    if latent.shape != tuple(original_latent.shape):
        raise ValueError(
            f"Saved latents have shape {latent.shape}, but the model produces "
            f"{tuple(original_latent.shape)}. Check the split and checkpoint."
        )

    if counterfactual_latent.shape != latent.shape:
        raise ValueError(
            "Original and counterfactual latents have different shapes."
        )

    discrepancy = float(
        np.max(
            np.abs(
                latent - original_latent.detach().cpu().numpy()
            )
        )
    )

    if discrepancy > 1e-3:
        raise ValueError(
            "The saved encoded artifact differs from the current checkpoint. "
            "Check that both belong to the same run."
        )

    projection = fit_latent_projection(
        np.concatenate(
            [latent, counterfactual_latent],
            axis=0,
        ),
        n_components=2,
    )

    return DisplayData(
        latent=latent,
        true_state=np.asarray(encoded["true_state"]),
        projection=projection,
        encoded_path=encoded_path,
        max_latent_discrepancy=discrepancy,
    )


# =============================================================================
# Latent-space reparameterization
# =============================================================================

def fit_raw_pca_basis(
    latent: Tensor,
    rotation_axes: tuple[int, int],
    pca_signs: tuple[int, int],
) -> tuple[PCA, Tensor]:
    """Fit a raw-space orthonormal PCA basis for defining rotations."""
    dimension = latent.shape[-1]
    first, second = rotation_axes

    if (
        first == second
        or min(first, second) < 0
        or max(first, second) >= dimension
    ):
        raise ValueError(
            f"Invalid rotation axes {rotation_axes} for latent dimension "
            f"{dimension}."
        )

    flat = latent.detach().cpu().numpy().reshape(
        -1,
        dimension,
    )

    pca = PCA(
        n_components=dimension,
        svd_solver="full",
    )
    pca.fit(flat)

    # PCA directions are columns in the latent coordinate basis.
    # Flipping a direction only changes its arbitrary orientation.
    components = pca.components_.T.copy()

    for axis, sign in zip(
        rotation_axes,
        pca_signs,
        strict=True,
    ):
        components[:, axis] *= sign

    basis = torch.as_tensor(
        components,
        dtype=latent.dtype,
        device=latent.device,
    )

    identity = torch.eye(
        dimension,
        dtype=latent.dtype,
        device=latent.device,
    )

    if not torch.allclose(
        basis.T @ basis,
        identity,
        atol=2e-5,
    ):
        raise AssertionError(
            "The PCA rotation basis is not sufficiently orthonormal."
        )

    return pca, basis


def pca_plane_rotation(
    basis: Tensor,
    rotation_axes: tuple[int, int],
    angle_degrees: float,
) -> Tensor:
    """Build Q = U R(theta) U^T in the original latent coordinates."""
    dimension = basis.shape[0]
    first, second = rotation_axes

    if angle_degrees == 0.0:
        return torch.eye(
            dimension,
            dtype=basis.dtype,
            device=basis.device,
        )

    radians = math.radians(
        angle_degrees
    )
    cosine = math.cos(radians)
    sine = math.sin(radians)

    rotation = torch.eye(
        dimension,
        dtype=basis.dtype,
        device=basis.device,
    )

    rotation[first, first] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    rotation[second, second] = cosine

    return basis @ rotation @ basis.T


def structural_penalty_components(
    latent: Tensor,
    variance_target: float,
    variance_weight: float,
    covariance_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the components of clsm.losses.structural_constraint_loss."""
    flat = latent.reshape(
        -1,
        latent.shape[-1],
    )
    centered = flat - flat.mean(
        dim=0,
        keepdim=True,
    )

    standard_deviation = torch.sqrt(
        centered.var(dim=0, unbiased=False)
        + STRUCTURAL_VARIANCE_EPSILON
    )

    variance_penalty = torch.relu(
        variance_target - standard_deviation
    ).square().mean()

    covariance = centered.T @ centered / max(
        centered.shape[0] - 1,
        1,
    )
    off_diagonal = covariance - torch.diag_embed(
        covariance.diagonal()
    )
    covariance_penalty = off_diagonal.square().mean()

    total = (
        variance_weight * variance_penalty
        + covariance_weight * covariance_penalty
    )

    return variance_penalty, covariance_penalty, total


# =============================================================================
# Model evaluation
# =============================================================================

@torch.inference_mode()
def encode_observations(
    model: CLSMModel,
    observations: Tensor,
    batch_size: int,
) -> Tensor:
    """Encode a dataset once, without sampling or updating the network."""
    latent_batches = []

    for start in range(
        0,
        observations.shape[0],
        batch_size,
    ):
        latent_batches.append(
            model.encode(
                observations[start:start + batch_size],
                sample=False,
            )["latent"]
        )

    return torch.cat(
        latent_batches,
        dim=0,
    )


@torch.inference_mode()
def predict_with_rotated_coordinates(
    model: CLSMModel,
    rotated_latent: Tensor,
    rotation: Tensor,
    horizon: int,
) -> Tensor:
    """Roll out the conjugate transition model and decode the result."""
    current = rotated_latent[:, :-horizon, :]

    for _ in range(horizon):
        # Row-vector convention: inverse rotation is multiplication by Q.
        current = model.predict_next_latent(
            current @ rotation
        ) @ rotation.T

    return model.decode(
        current @ rotation
    )


@torch.inference_mode()
def evaluate_rotations(
    model: CLSMModel,
    observations: Tensor,
    angles: np.ndarray,
    *,
    batch_size: int,
    horizon: int,
    rotation_axes: tuple[int, int],
    pca_signs: tuple[int, int],
    loss_config: Mapping[str, object],
    absolute_tolerance: float,
    relative_mse_tolerance: float,
) -> RotationEvaluation:
    """Evaluate conjugate rotations and independently verify their invariants."""
    if horizon < 1 or horizon >= observations.shape[1]:
        raise ValueError(
            "The rollout horizon must be between 1 and T-1."
        )

    latent = encode_observations(
        model,
        observations,
        batch_size,
    )

    raw_pca, basis = fit_raw_pca_basis(
        latent,
        rotation_axes,
        pca_signs,
    )

    print(
        "Raw-space PCA for orthogonal rotations: "
        + ", ".join(
            f"PC{index + 1}={100.0 * ratio:.2f}%"
            for index, ratio in enumerate(raw_pca.explained_variance_ratio_)
        )
    )

    minimality_mode = str(
        loss_config.get("minimality_mode", "l1")
    )

    if minimality_mode == "kl":
        raise ValueError(
            "KL-based minimality is unsupported for the deterministic model."
        )

    variance_target = float(
        loss_config.get("structural_variance_target", 1.0)
    )
    variance_weight = float(
        loss_config.get("structural_variance_weight", 1.0)
    )
    covariance_weight = float(
        loss_config.get("structural_covariance_weight", 1.0)
    )

    reference_reconstructions: list[Tensor] | None = None
    reference_predictions: list[Tensor] | None = None
    reference_residuals: list[Tensor] | None = None
    reference_mse: dict[str, float] | None = None

    rows: list[dict[str, float]] = []

    for angle in angles:
        rotation = pca_plane_rotation(
            basis,
            rotation_axes,
            float(angle),
        )
        rotated_latent = latent @ rotation.T

        reconstruction_error_sum = 0.0
        prediction_error_sum = 0.0
        temporal_error_sum = 0.0

        reconstruction_count = 0
        prediction_count = 0
        temporal_count = 0

        max_reconstruction_difference = 0.0
        max_prediction_difference = 0.0
        max_transition_difference = 0.0

        reconstructions: list[Tensor] = []
        predictions: list[Tensor] = []
        residuals: list[Tensor] = []

        for start in range(
            0,
            observations.shape[0],
            batch_size,
        ):
            batch_index = start // batch_size
            stop = start + batch_size

            observation_batch = observations[start:stop]
            rotated_batch = rotated_latent[start:stop]

            # The conjugate decoder expects the inverse-rotated coordinates.
            reconstruction = model.decode(
                rotated_batch @ rotation
            )
            prediction = predict_with_rotated_coordinates(
                model,
                rotated_batch,
                rotation,
                horizon,
            )

            next_latent = model.predict_next_latent(
                rotated_batch[:, :-1, :] @ rotation
            ) @ rotation.T
            temporal_residual = (
                next_latent - rotated_batch[:, 1:, :]
            )

            reconstruction_error_sum += float(
                (reconstruction - observation_batch).square().sum()
            )
            prediction_error_sum += float(
                (
                    prediction - observation_batch[:, horizon:, :]
                ).square().sum()
            )
            temporal_error_sum += float(
                temporal_residual.square().sum()
            )

            reconstruction_count += reconstruction.numel()
            prediction_count += prediction.numel()
            temporal_count += temporal_residual.numel()

            if reference_predictions is None:
                reconstructions.append(reconstruction.clone())
                predictions.append(prediction.clone())
                residuals.append(temporal_residual.clone())
                continue

            # The observation-space outputs should be pointwise invariant.
            max_reconstruction_difference = max(
                max_reconstruction_difference,
                float(
                    (
                        reconstruction
                        - reference_reconstructions[batch_index]
                    ).abs().max()
                ),
            )
            max_prediction_difference = max(
                max_prediction_difference,
                float(
                    (
                        prediction
                        - reference_predictions[batch_index]
                    ).abs().max()
                ),
            )

            # Transition residuals are equivariant, not individually equal.
            max_transition_difference = max(
                max_transition_difference,
                float(
                    (
                        temporal_residual
                        - reference_residuals[batch_index] @ rotation.T
                    ).abs().max()
                ),
            )

        if reference_predictions is None:
            reference_reconstructions = reconstructions
            reference_predictions = predictions
            reference_residuals = residuals

        variance, covariance, structure = structural_penalty_components(
            rotated_latent,
            variance_target,
            variance_weight,
            covariance_weight,
        )

        official_structure = structural_constraint_loss(
            rotated_latent,
            variance_target=variance_target,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
        )

        if not torch.allclose(
            structure,
            official_structure,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise AssertionError(
                "Structural penalty differs from clsm.losses."
            )

        minimality = minimality_loss(
            rotated_latent,
            mode=minimality_mode,
        )

        row = {
            "angle_degrees": float(angle),
            "reconstruction_mse": reconstruction_error_sum / reconstruction_count,
            f"prediction_mse_h{horizon}": prediction_error_sum / prediction_count,
            "temporal_mse": temporal_error_sum / temporal_count,
            "minimality_raw": float(minimality),
            "structure_raw": float(structure),
            "structure_variance_raw": float(variance),
            "structure_covariance_raw": float(covariance),
            "max_abs_reconstruction_difference": max_reconstruction_difference,
            "max_abs_prediction_difference": max_prediction_difference,
            "max_abs_transition_residual_difference": max_transition_difference,
        }

        # An invariant MSE can exhibit small float32 roundoff differences.
        invariant_keys = (
            "reconstruction_mse",
            f"prediction_mse_h{horizon}",
            "temporal_mse",
        )

        if reference_mse is None:
            reference_mse = {
                key: row[key]
                for key in invariant_keys
            }

        relative_differences = {
            key: abs(row[key] - reference_mse[key])
            / max(abs(reference_mse[key]), 1e-8)
            for key in invariant_keys
        }

        row["max_relative_mse_difference"] = max(
            relative_differences.values()
        )

        maximum_pointwise_difference = max(
            max_reconstruction_difference,
            max_prediction_difference,
            max_transition_difference,
        )

        if maximum_pointwise_difference > absolute_tolerance:
            raise AssertionError(
                f"Numerical equivalence failed at {angle:.1f} degrees: "
                f"maximum absolute difference "
                f"{maximum_pointwise_difference:.3g} exceeds "
                f"{absolute_tolerance:.3g}."
            )

        if row["max_relative_mse_difference"] > relative_mse_tolerance:
            raise AssertionError(
                f"Invariant MSEs changed at {angle:.1f} degrees: "
                f"{relative_differences}."
            )

        rows.append(row)

        print(
            f"theta={angle:6.1f}  "
            f"pred={row[f'prediction_mse_h{horizon}']:.7f}  "
            f"min={row['minimality_raw']:.5g}  "
            f"structure={row['structure_raw']:.5g}  "
            f"max_diff={maximum_pointwise_difference:.2g}  "
            f"max_MSE_rel={row['max_relative_mse_difference']:.2g}"
        )

    return RotationEvaluation(
        rows=rows,
        latent=latent,
        raw_pca=raw_pca,
        raw_basis=basis,
    )


# =============================================================================
# Publication figure
# =============================================================================

def plot_latent_reparameterizations(
    axis_original: Axes,
    axis_rotated: Axes,
    display: DisplayData,
    raw_basis: Tensor,
    *,
    rotation_axes: tuple[int, int],
    pca_signs: tuple[int, int],
    example_angle: float,
) -> None:
    """Render original and rotated latents using the same publication PCA."""
    original = display.latent
    rotation = pca_plane_rotation(
        raw_basis,
        rotation_axes,
        example_angle,
    ).detach().cpu().numpy()

    rotated = original @ rotation.T

    projected_original = transform_latent(
        original,
        display.projection,
    ).reshape(-1, 2)
    projected_rotated = transform_latent(
        rotated,
        display.projection,
    ).reshape(-1, 2)

    # Recenter the rotated projection for DISPLAY ONLY. The actual model
    # reparameterization is a linear rotation about the latent origin.
    projected_rotated -= projected_rotated.mean(
        axis=0,
        keepdims=True,
    )
    projected_rotated += projected_original.mean(
        axis=0,
        keepdims=True,
    )

    signs = np.asarray(
        pca_signs,
        dtype=float,
    )
    projected_original *= signs
    projected_rotated *= signs

    flat_state = display.true_state.reshape(
        -1,
        display.true_state.shape[-1],
    )

    generator = np.random.default_rng(
        DEFAULT_VISUALIZATION_SEED
    )
    n_points = projected_original.shape[0]

    if n_points <= DEFAULT_MAX_LATENT_POINTS:
        indices = np.arange(
            n_points
        )
    else:
        indices = generator.choice(
            n_points,
            size=DEFAULT_MAX_LATENT_POINTS,
            replace=False,
        )

    original_points = projected_original[indices]
    rotated_points = projected_rotated[indices]
    state_colors = flat_state[indices, 0]

    scatter = axis_original.scatter(
        original_points[:, 0],
        original_points[:, 1],
        c=state_colors,
        cmap="viridis",
        s=5,
        alpha=0.65,
        rasterized=True,
    )
    axis_rotated.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        c=state_colors,
        cmap="viridis",
        s=5,
        alpha=0.65,
        rasterized=True,
    )

    axis_original.set_title(
        "A. Original PCA projection"
    )
    axis_rotated.set_title(
        f"B. Rotated representation ({example_angle:g}° in latent space)"
    )

    explained = display.projection.projector.explained_variance_ratio_
    all_points = np.vstack(
        [original_points, rotated_points]
    )
    radius = float(np.max(np.abs(all_points))) * 1.06
    midpoint = np.zeros(2)

    for axis in (axis_original, axis_rotated):
        axis.set_xlabel(
            f"PC1 ({100.0 * explained[0]:.1f}%)"
        )
        axis.set_ylabel(
            f"PC2 ({100.0 * explained[1]:.1f}%)"
        )
        axis.set_aspect(
            "equal",
            adjustable="box",
        )
        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)

    figure = axis_original.figure
    figure.colorbar(
        scatter,
        ax=[axis_original, axis_rotated],
        label="x",
        pad=0.01,
    )


def plot_invariant_losses(
    axis: Axes,
    rows: Sequence[Mapping[str, float]],
    *,
    horizon: int,
) -> None:
    """Display the losses invariant under conjugate reparameterization."""
    angles = np.asarray(
        [row["angle_degrees"] for row in rows]
    )

    for key, label in (
        ("reconstruction_mse", "Reconstruction"),
        (f"prediction_mse_h{horizon}", f"Prediction (h={horizon})"),
        ("temporal_mse", "Temporal coherence"),
    ):
        axis.plot(
            angles,
            [row[key] for row in rows],
            linewidth=1.6,
            label=label,
        )

    axis.set_title(
        "C. Rotation-invariant errors"
    )
    axis.set_xlabel(
        "Rotation angle (degrees)"
    )
    axis.set_ylabel(
        "MSE"
    )
    axis.set_xlim(
        0.0,
        angles[-1],
    )
    axis.set_xticks(
        np.arange(0, angles[-1] + 1, 60)
    )
    axis.legend()


def plot_coordinate_penalties(
    axis: Axes,
    rows: Sequence[Mapping[str, float]],
) -> None:
    """Display relative changes in coordinate-dependent penalty terms."""
    angles = np.asarray(
        [row["angle_degrees"] for row in rows]
    )

    for key, label in (
        ("minimality_raw", "Mean |z|"),
        ("structure_raw", "VICReg-like structure"),
        ("structure_variance_raw", "Variance-floor component"),
        ("structure_covariance_raw", "Covariance component"),
    ):
        values = np.asarray(
            [row[key] for row in rows],
            dtype=float,
        )
        initial_value = float(
            values[0]
        )

        if abs(initial_value) <= NORMALIZATION_EPSILON:
            raise ValueError(
                f"Cannot normalize '{key}' by a near-zero initial value."
            )

        axis.plot(
            angles,
            values / initial_value,
            linewidth=1.5,
            label=label,
        )

    axis.axhline(
        1.0,
        color="0.45",
        linestyle=":",
        linewidth=0.9,
    )
    axis.set_title(
        "D. Coordinate-dependent regularization"
    )
    axis.set_xlabel(
        "Rotation angle (degrees)"
    )
    axis.set_ylabel(
        "Penalty / penalty at 0°"
    )
    axis.set_xlim(
        0.0,
        angles[-1],
    )
    axis.set_xticks(
        np.arange(0, angles[-1] + 1, 60)
    )
    axis.legend()


def create_identifiability_figure(
    evaluation: RotationEvaluation,
    display: DisplayData,
    *,
    rotation_axes: tuple[int, int],
    pca_signs: tuple[int, int],
    example_angle: float,
    horizon: int,
) -> Figure:
    """Create the final four-panel identifiability illustration (A--D)."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 8.6),
        layout="constrained",
    )

    plot_latent_reparameterizations(
        axes[0, 0],
        axes[0, 1],
        display,
        evaluation.raw_basis,
        rotation_axes=rotation_axes,
        pca_signs=pca_signs,
        example_angle=example_angle,
    )
    plot_invariant_losses(
        axes[1, 0],
        evaluation.rows,
        horizon=horizon,
    )
    plot_coordinate_penalties(
        axes[1, 1],
        evaluation.rows,
    )

    # Compute the final constrained-layout positions.
    figure.canvas.draw()

    # Freeze the layout before manually repositioning axes.
    figure.set_layout_engine(None)

    # Horizontally align the centers of the bottom and top panels.
    for col in range(2):
        top = axes[0, col].get_position()
        bottom = axes[1, col].get_position()

        top_center = (top.x0 + top.x1) / 2
        bottom_center = (bottom.x0 + bottom.x1) / 2

        dx = top_center - bottom_center

        axes[1, col].set_position([
            bottom.x0 + dx,
            bottom.y0,
            bottom.width,
            bottom.height,
        ])

    return figure


# =============================================================================
# Serialization and command-line interface
# =============================================================================

def save_experiment(
    evaluation: RotationEvaluation,
    display: DisplayData,
    *,
    output_dir: Path,
    checkpoint_path: Path,
    data_path: Path,
    checkpoint: Mapping[str, object],
    dataset: CLSMDataset,
    args: argparse.Namespace,
) -> None:
    """Write publication figures, measurements, and reproducibility metadata."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "identifiability_rotations.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(evaluation.rows[0]),
        )
        writer.writeheader()
        writer.writerows(
            evaluation.rows
        )

    figure = create_identifiability_figure(
        evaluation,
        display,
        rotation_axes=tuple(args.rotation_axes),
        pca_signs=tuple(args.pca_signs),
        example_angle=args.example_angle,
        horizon=args.horizon,
    )

    save_figure(
        figure,
        output_dir / "fig_identifiability.pdf",
    )
    figure.savefig(
        output_dir / "fig_identifiability.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    max_absolute_difference = max(
        max(
            row["max_abs_reconstruction_difference"],
            row["max_abs_prediction_difference"],
            row["max_abs_transition_residual_difference"],
        )
        for row in evaluation.rows
    )
    max_relative_mse_difference = max(
        row["max_relative_mse_difference"]
        for row in evaluation.rows
    )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "data": str(data_path),
        "episodes": dataset.n_episodes,
        "model_config": checkpoint["model_config"],
        "constraint_weights": checkpoint.get("config", {}).get("loss", {}).get("weights", {}),
        "rotation_axes": list(args.rotation_axes),
        "rotation_basis": "Raw-space orthonormal PCA of original latent representations",
        "rotation_center": "Latent origin",
        "max_angle_degrees": float(args.max_angle),
        "n_angles": int(args.n_angles),
        "example_angle_degrees": float(args.example_angle),
        "raw_pca_explained_variance_ratios": [
            float(value)
            for value in evaluation.raw_pca.explained_variance_ratio_
        ],
        "visualization_artifact": str(display.encoded_path),
        "visualization_projection": "StandardScaler + PCA on original and counterfactual latents",
        "visualization_variance_ratios": [
            float(value)
            for value in display.projection.projector.explained_variance_ratio_
        ],
        "max_saved_latent_discrepancy": display.max_latent_discrepancy,
        "max_absolute_model_difference": max_absolute_difference,
        "max_relative_invariant_mse_difference": max_relative_mse_difference,
        "absolute_tolerance": args.tolerance,
        "relative_mse_tolerance": args.mse_relative_tolerance,
        "note": (
            "No model retraining. The latent transformation is a full-dimensional "
            "orthogonal reparameterization, with inverse-transform decoder and "
            "conjugate transition dynamics. Structural penalties are evaluated "
            "on the full test split rather than training mini-batches. "
            "The PCA shown in the figure is the original paper projection."
        ),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the rotation experiment."""
    parser = argparse.ArgumentParser(
        description="Evaluate rotational ambiguities in a trained CLSM-Toy model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted training checkpoint (best.pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--encoded",
        type=Path,
        default=None,
        help="Saved encoded artifact; defaults to <run>/evaluation/<split>_encoded.npz.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "ood"),
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("identifiability/P29-seed-0"),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional subset for debugging; use the full split for paper figures.",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=DEFAULT_MAX_ANGLE,
        help="Upper end of the rotation sweep, in degrees.",
    )
    parser.add_argument(
        "--n-angles",
        type=int,
        default=DEFAULT_N_ANGLES,
        help="Number of equally spaced rotation angles, including both endpoints.",
    )
    parser.add_argument(
        "--rotation-axes",
        type=int,
        nargs=2,
        metavar=("PC_I", "PC_J"),
        default=DEFAULT_ROTATION_AXES,
        help="Zero-based indices of the raw PCA axes used for the rotation.",
    )
    parser.add_argument(
        "--pca-signs",
        type=int,
        nargs=2,
        choices=(-1, 1),
        metavar=("SIGN_I", "SIGN_J"),
        default=DEFAULT_PCA_SIGNS,
        help="Sign conventions for the displayed PCA axes and rotation basis.",
    )
    parser.add_argument(
        "--example-angle",
        type=float,
        default=DEFAULT_EXAMPLE_ANGLE,
        help="Rotation used in panel B.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Raw-observation prediction horizon for panel C.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        help="Absolute tolerance for pointwise numerical equivalence.",
    )
    parser.add_argument(
        "--mse-relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_MSE_TOLERANCE,
        help="Relative tolerance for invariant MSEs.",
    )

    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """Run the full rotation sweep and serialize its four-panel figure."""
    args = build_arg_parser().parse_args()

    if args.n_angles < 2:
        raise ValueError("n-angles must be at least 2.")

    if args.max_angle <= 0 or args.batch_size < 1:
        raise ValueError("max-angle and batch-size must be positive.")

    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max-episodes must be positive.")

    device_name = args.device

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(
        device_name
    )

    model, checkpoint = load_trained_model(
        args.checkpoint,
        device,
    )
    observations, dataset, dataset_path = load_test_observations(
        args.data_dir,
        args.split,
        model,
        device,
        args.max_episodes,
    )

    loss_config = checkpoint.get(
        "config", {}
    ).get("loss", {})

    angles = np.linspace(
        0.0,
        args.max_angle,
        args.n_angles,
    )

    print(
        f"Checkpoint: {args.checkpoint} "
        f"(epoch {checkpoint.get('epoch', 'unknown')})"
    )
    print(
        f"Dataset: {dataset_path}; "
        f"episodes={dataset.n_episodes}; T={dataset.episode_length}"
    )
    print(
        f"Model: latent_dim={model.config.latent_dim}; device={device}"
    )
    print(
        f"Checkpoint constraint weights: {loss_config.get('weights', {})}"
    )

    evaluation = evaluate_rotations(
        model,
        observations,
        angles,
        batch_size=args.batch_size,
        horizon=args.horizon,
        rotation_axes=tuple(args.rotation_axes),
        pca_signs=tuple(args.pca_signs),
        loss_config=loss_config,
        absolute_tolerance=args.tolerance,
        relative_mse_tolerance=args.mse_relative_tolerance,
    )

    encoded_path = (
        args.encoded
        if args.encoded is not None
        else args.checkpoint.parent / "evaluation" / f"{args.split}_encoded.npz"
    )

    if not encoded_path.is_file():
        raise FileNotFoundError(
            f"Missing visualization artifact: {encoded_path}. "
            "Use --encoded to specify the artifact used by visualization.py."
        )

    display = load_paper_display_data(
        encoded_path,
        evaluation.latent,
    )

    print(
        f"Encoded artifact: {encoded_path}; "
        f"max discrepancy vs checkpoint = "
        f"{display.max_latent_discrepancy:.3g}"
    )
    print(
        "Paper-style standardized PCA (original + counterfactual): "
        + ", ".join(
            f"PC{index + 1}={100.0 * ratio:.2f}%"
            for index, ratio in enumerate(
                display.projection.projector.explained_variance_ratio_
            )
        )
    )

    # A complete rotation must return to the original parameterization.
    if np.isclose(
        args.max_angle % 360.0,
        0.0,
    ):
        first = evaluation.rows[0]
        last = evaluation.rows[-1]

        print(
            "360-degree closure: "
            f"minimality={last['minimality_raw'] - first['minimality_raw']:+.3g}, "
            f"structure={last['structure_raw'] - first['structure_raw']:+.3g}"
        )

    save_experiment(
        evaluation,
        display,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        data_path=dataset_path,
        checkpoint=checkpoint,
        dataset=dataset,
        args=args,
    )

    print(
        f"Results saved to {args.output_dir}/"
    )


if __name__ == "__main__":
    main()
