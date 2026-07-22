"""
Evaluation utilities for CLSM.

Author: Gwenolé Quellec
Year: 2026

This module evaluates trained CLSM models along complementary representation
properties while keeping training and evaluation objectives clearly separated.

Evaluation families
-------------------
1. observation compatibility;
2. latent dynamics and multi-step prediction;
3. state accessibility through linear, nonlinear, and temporal probes;
4. canonical correlation between latent and physical state spaces;
5. local and global neighborhood preservation;
6. latent compactness and effective dimensionality;
7. counterfactual consistency;
8. nuisance accessibility and nuisance-subspace removal;
9. optional metrics from the adversarial nuisance head.

Design principles
-----------------
- the training split is used only to fit post-hoc probes and CCA mappings;
- validation, test, and OOD metrics are computed on independent samples;
- counterfactual pairs are used only as an evaluation oracle;
- nuisance classification metrics are omitted for unseen OOD classes;
- all variational encoders are evaluated deterministically;
- metric names remain flat and backward-compatible where practical.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    pairwise_distances,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from tqdm.auto import tqdm

from clsm.datasets import CLSMDataset, DatasetSplits
from clsm.models import CLSMModel, CLSMModelConfig
from clsm.training import make_loader, move_batch_to_device, resolve_device
from clsm.utils import print_separator


# =============================================================================
# Type aliases
# =============================================================================

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MetricValue = float | None
MetricDict = dict[str, MetricValue]


# =============================================================================
# Evaluation data structures
# =============================================================================

@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for deterministic post-hoc evaluation."""

    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "auto"

    rollout_horizons: tuple[int, ...] = (1, 5, 10)

    probe_max_samples: int = 100_000
    nonlinear_probe_max_samples: int = 25_000
    temporal_probe_max_samples: int = 100_000

    cca_max_samples: int = 50_000
    cca_max_iter: int = 2_000
    cca_tolerance: float = 1e-6

    neighborhood_max_samples: int = 5_000
    neighborhood_neighbors: int = 15

    probe_seed: int = 42

    save_latents: bool = False
    output_dir: str | None = None


@dataclass(frozen=True)
class EncodedDataset:
    """Encoded representation of one complete dataset split."""

    latent: FloatArray
    true_state: FloatArray
    observation: FloatArray
    nuisance_id: IntArray
    nuisance: FloatArray | None
    reconstructed_observation: FloatArray
    state_names: tuple[str, ...]

    episode_index: IntArray
    time_index: IntArray

    counterfactual_observation: FloatArray | None = None
    counterfactual_latent: FloatArray | None = None
    predicted_next_latent: FloatArray | None = None
    nuisance_logits: FloatArray | None = None

    @property
    def n_episodes(self) -> int:
        return int(self.latent.shape[0])

    @property
    def episode_length(self) -> int:
        return int(self.latent.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.latent.shape[-1])

    @property
    def state_dim(self) -> int:
        return int(self.true_state.shape[-1])

    @property
    def has_counterfactuals(self) -> bool:
        return (
            self.counterfactual_observation is not None
            and self.counterfactual_latent is not None
        )


@dataclass(frozen=True)
class CCAAnalysis:
    """Complete CCA artifacts fitted on train and evaluated on one split."""

    canonical_correlations: FloatArray
    state_loadings: FloatArray
    latent_loadings: FloatArray

    state_weights: FloatArray
    latent_weights: FloatArray
    state_rotations: FloatArray
    latent_rotations: FloatArray

    state_scaler_mean: FloatArray
    state_scaler_scale: FloatArray
    latent_scaler_mean: FloatArray
    latent_scaler_scale: FloatArray


# =============================================================================
# Model and dataset loading
# =============================================================================

def load_checkpoint_model(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
) -> tuple[CLSMModel, dict[str, object]]:
    """Load a checkpoint and restore the corresponding model."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_config'.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model_config = CLSMModelConfig(
        **checkpoint["model_config"]
    )

    model = CLSMModel(
        model_config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, checkpoint


def load_splits(
    data_dir: Path | str,
) -> DatasetSplits:
    """Load train, validation, test, and optional OOD splits."""
    data_dir = Path(data_dir)

    required = {
        "train": data_dir / "train.npz",
        "validation": data_dir / "validation.npz",
        "test": data_dir / "test.npz",
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required dataset files: "
            + ", ".join(missing)
        )

    ood_path = data_dir / "ood.npz"

    return DatasetSplits(
        train=CLSMDataset.load(
            required["train"]
        ),
        validation=CLSMDataset.load(
            required["validation"]
        ),
        test=CLSMDataset.load(
            required["test"]
        ),
        ood=(
            CLSMDataset.load(ood_path)
            if ood_path.exists()
            else None
        ),
    )


# =============================================================================
# Dataset encoding
# =============================================================================

@torch.no_grad()
def encode_dataset(
    model: CLSMModel,
    dataset: CLSMDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> EncodedDataset:
    """Encode a complete split and collect all available model outputs."""
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            pin_memory
            and device.type == "cuda"
        ),
    )

    latent_batches: list[Tensor] = []
    true_state_batches: list[Tensor] = []
    observation_batches: list[Tensor] = []
    nuisance_id_batches: list[Tensor] = []
    nuisance_batches: list[Tensor] = []
    reconstruction_batches: list[Tensor] = []

    counterfactual_observation_batches: list[Tensor] = []
    counterfactual_latent_batches: list[Tensor] = []
    predicted_next_batches: list[Tensor] = []
    nuisance_logit_batches: list[Tensor] = []

    model.eval()

    for batch in loader:
        batch = move_batch_to_device(
            batch,
            device,
        )

        counterfactual_observation = batch.get(
            "counterfactual_observation"
        )

        output = model(
            batch["observation"],
            counterfactual_observation=counterfactual_observation,
            sample=False,
            adversarial_coefficient=0.0,
        )

        latent_batches.append(
            output["latent"].cpu()
        )
        true_state_batches.append(
            batch["latent_state"].cpu()
        )
        observation_batches.append(
            batch["observation"].cpu()
        )
        nuisance_id_batches.append(
            batch["nuisance_id"].cpu()
        )
        reconstruction_batches.append(
            output["reconstructed_observation"].cpu()
        )

        if "nuisance" in batch:
            nuisance_batches.append(
                batch["nuisance"].cpu()
            )

        if counterfactual_observation is not None:
            counterfactual_observation_batches.append(
                counterfactual_observation.cpu()
            )

        if "counterfactual_latent" in output:
            counterfactual_latent_batches.append(
                output["counterfactual_latent"].cpu()
            )

        if "predicted_next_latent" in output:
            predicted_next_batches.append(
                output["predicted_next_latent"].cpu()
            )

        if "nuisance_logits" in output:
            nuisance_logit_batches.append(
                output["nuisance_logits"].cpu()
            )

    latent = torch.cat(
        latent_batches,
        dim=0,
    ).numpy()

    n_episodes = latent.shape[0]
    episode_length = latent.shape[1]

    episode_index = np.repeat(
        np.arange(
            n_episodes,
            dtype=np.int64,
        )[:, None],
        episode_length,
        axis=1,
    )

    time_index = np.repeat(
        np.arange(
            episode_length,
            dtype=np.int64,
        )[None, :],
        n_episodes,
        axis=0,
    )

    return EncodedDataset(
        latent=latent,
        true_state=torch.cat(
            true_state_batches,
            dim=0,
        ).numpy(),
        observation=torch.cat(
            observation_batches,
            dim=0,
        ).numpy(),
        nuisance_id=torch.cat(
            nuisance_id_batches,
            dim=0,
        ).numpy().astype(np.int64),
        nuisance=(
            torch.cat(
                nuisance_batches,
                dim=0,
            ).numpy()
            if nuisance_batches
            else None
        ),
        reconstructed_observation=torch.cat(
            reconstruction_batches,
            dim=0,
        ).numpy(),
        state_names=dataset.state_names,
        episode_index=episode_index,
        time_index=time_index,
        counterfactual_observation=(
            torch.cat(
                counterfactual_observation_batches,
                dim=0,
            ).numpy()
            if counterfactual_observation_batches
            else None
        ),
        counterfactual_latent=(
            torch.cat(
                counterfactual_latent_batches,
                dim=0,
            ).numpy()
            if counterfactual_latent_batches
            else None
        ),
        predicted_next_latent=(
            torch.cat(
                predicted_next_batches,
                dim=0,
            ).numpy()
            if predicted_next_batches
            else None
        ),
        nuisance_logits=(
            torch.cat(
                nuisance_logit_batches,
                dim=0,
            ).numpy()
            if nuisance_logit_batches
            else None
        ),
    )


# =============================================================================
# Array utilities
# =============================================================================

def _flatten_time(
    values: NDArray,
) -> NDArray:
    """Flatten episode and time dimensions."""
    return values.reshape(
        -1,
        values.shape[-1],
    )


def _subsample_rows(
    features: FloatArray,
    targets: NDArray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[FloatArray, NDArray]:
    """Deterministically subsample paired rows."""
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "features and targets must have the same number of rows."
        )

    if features.shape[0] <= max_samples:
        return features, targets

    generator = np.random.default_rng(
        seed
    )

    indices = generator.choice(
        features.shape[0],
        size=max_samples,
        replace=False,
    )

    return (
        features[indices],
        targets[indices],
    )


def _finite_pair(
    features: FloatArray,
    targets: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Remove rows containing non-finite values."""
    valid = (
        np.all(
            np.isfinite(features),
            axis=1,
        )
        & np.all(
            np.isfinite(targets),
            axis=1,
        )
    )

    return (
        features[valid],
        targets[valid],
    )


def _safe_pearson(
    values_a: FloatArray,
    values_b: FloatArray,
) -> float:
    """Compute Pearson correlation with degenerate-vector protection."""
    values_a = np.asarray(
        values_a,
        dtype=np.float64,
    ).reshape(-1)

    values_b = np.asarray(
        values_b,
        dtype=np.float64,
    ).reshape(-1)

    if values_a.shape != values_b.shape:
        raise ValueError(
            "Correlation vectors must have identical shapes."
        )

    if (
        np.std(values_a) < 1e-12
        or np.std(values_b) < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            values_a,
            values_b,
        )[0, 1]
    )


def _flatten_latent_and_nuisance(
    encoded: EncodedDataset,
) -> tuple[FloatArray, IntArray]:
    latent = _flatten_time(
        encoded.latent
    )

    nuisance = np.repeat(
        encoded.nuisance_id,
        encoded.episode_length,
    )

    return (
        latent,
        nuisance,
    )


# =============================================================================
# Direct representation metrics
# =============================================================================
    
def observation_compatibility_metrics(
    encoded: EncodedDataset,
) -> dict[str, float]:
    """Evaluate observation reconstruction quality."""
    target = _flatten_time(
        encoded.observation
    )
    prediction = _flatten_time(
        encoded.reconstructed_observation
    )

    feature_variance = np.var(
        target,
        axis=0,
    )
    valid_features = (
        feature_variance > 1e-8
    )

    metrics = {
        "observation_mse": float(
            mean_squared_error(
                target,
                prediction,
            )
        ),
        "observation_mae": float(
            np.mean(
                np.abs(
                    target - prediction
                )
            )
        ),
        "observation_r2": float(
            r2_score(
                target,
                prediction,
                multioutput="variance_weighted",
            )
        ),
    }

    if np.any(valid_features):
        metrics[
            "observation_r2_mean_valid_features"
        ] = float(
            np.mean(
                [
                    r2_score(
                        target[:, index],
                        prediction[:, index],
                    )
                    for index in np.flatnonzero(
                        valid_features
                    )
                ]
            )
        )

    return metrics


def temporal_coherence_metrics(
    encoded: EncodedDataset,
) -> dict[str, float]:
    """Evaluate local latent smoothness and one-step transition quality."""
    latent = encoded.latent

    first_difference = (
        latent[:, 1:, :]
        - latent[:, :-1, :]
    )

    second_difference = (
        latent[:, 2:, :]
        - 2.0 * latent[:, 1:-1, :]
        + latent[:, :-2, :]
    )

    metrics = {
        "latent_step_mse": float(
            np.mean(
                first_difference**2
            )
        ),
        "latent_acceleration_mse": float(
            np.mean(
                second_difference**2
            )
        ),
    }

    if encoded.predicted_next_latent is not None:
        target_next = latent[:, 1:, :]
        error = (
            encoded.predicted_next_latent
            - target_next
        )

        metrics["latent_transition_mse"] = float(
            np.mean(
                error**2
            )
        )
        metrics["latent_transition_mae"] = float(
            np.mean(
                np.abs(error)
            )
        )

    return metrics


@torch.no_grad()
def rollout_metrics(
    model: CLSMModel,
    dataset: CLSMDataset,
    *,
    horizons: Sequence[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate multi-step latent and observation rollouts."""
    horizons = tuple(
        sorted(
            set(
                int(horizon)
                for horizon in horizons
            )
        )
    )

    if not horizons:
        return {}

    if horizons[0] < 1:
        raise ValueError(
            "All rollout horizons must be at least 1."
        )

    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            pin_memory
            and device.type == "cuda"
        ),
    )

    latent_sums = {
        horizon: 0.0
        for horizon in horizons
    }
    observation_sums = {
        horizon: 0.0
        for horizon in horizons
    }
    counts = {
        horizon: 0
        for horizon in horizons
    }

    model.eval()

    for batch in loader:
        batch = move_batch_to_device(
            batch,
            device,
        )

        observation = batch["observation"]

        encoded = model.encode(
            observation,
            sample=False,
        )["latent"]

        for horizon in horizons:
            if observation.shape[1] <= horizon:
                continue

            initial_latent = encoded[
                :,
                :-horizon,
                :,
            ]

            target_latent = encoded[
                :,
                horizon:,
                :,
            ]

            target_observation = observation[
                :,
                horizon:,
                :,
            ]

            flat_initial = initial_latent.reshape(
                -1,
                initial_latent.shape[-1],
            )

            rollout = model.rollout_latent(
                flat_initial,
                horizon=horizon,
            )

            predicted_latent = rollout[
                :,
                -1,
                :,
            ]

            predicted_observation = model.decode(
                predicted_latent
            )

            flat_target_latent = target_latent.reshape(
                -1,
                target_latent.shape[-1],
            )

            flat_target_observation = target_observation.reshape(
                -1,
                target_observation.shape[-1],
            )

            n_samples = (
                flat_target_latent.shape[0]
            )

            latent_error = torch.mean(
                (
                    predicted_latent
                    - flat_target_latent
                ) ** 2
            )

            observation_error = torch.mean(
                (
                    predicted_observation
                    - flat_target_observation
                ) ** 2
            )

            latent_sums[horizon] += (
                float(
                    latent_error.cpu()
                )
                * n_samples
            )

            observation_sums[horizon] += (
                float(
                    observation_error.cpu()
                )
                * n_samples
            )

            counts[horizon] += n_samples

    metrics: dict[str, float] = {}

    for horizon in horizons:
        if counts[horizon] == 0:
            metrics[
                f"rollout_latent_mse_h{horizon}"
            ] = float("nan")

            metrics[
                f"rollout_observation_mse_h{horizon}"
            ] = float("nan")
        else:
            metrics[
                f"rollout_latent_mse_h{horizon}"
            ] = (
                latent_sums[horizon]
                / counts[horizon]
            )

            metrics[
                f"rollout_observation_mse_h{horizon}"
            ] = (
                observation_sums[horizon]
                / counts[horizon]
            )

    return metrics


def neighborhood_preservation_metrics(
    encoded: EncodedDataset,
    *,
    max_samples: int,
    n_neighbors: int,
    seed: int,
) -> dict[str, float]:
    """Measure local and global geometry preservation."""
    latent = _flatten_time(
        encoded.latent
    )
    state = _flatten_time(
        encoded.true_state
    )

    n_samples = latent.shape[0]

    if n_samples < 3:
        return {}

    sample_size = min(
        max_samples,
        n_samples,
    )

    generator = np.random.default_rng(
        seed
    )

    if sample_size < n_samples:
        indices = generator.choice(
            n_samples,
            size=sample_size,
            replace=False,
        )

        latent = latent[indices]
        state = state[indices]

    effective_neighbors = min(
        max(
            1,
            n_neighbors,
        ),
        max(
            1,
            (sample_size - 1) // 2,
        ),
    )

    state_distances = pairwise_distances(
        state
    )

    latent_distances = pairwise_distances(
        latent
    )

    upper = np.triu_indices(
        sample_size,
        k=1,
    )

    correlation = spearmanr(
        state_distances[upper],
        latent_distances[upper],
    ).statistic

    return {
        "neighborhood_state_distance_spearman": float(
            correlation
        ),
        "neighborhood_trustworthiness": float(
            trustworthiness(
                state,
                latent,
                n_neighbors=effective_neighbors,
            )
        ),
        "neighborhood_continuity": float(
            trustworthiness(
                latent,
                state,
                n_neighbors=effective_neighbors,
            )
        ),
    }


def minimality_metrics(
    encoded: EncodedDataset,
) -> dict[str, float]:
    """Characterize compactness, covariance, and effective dimensionality."""
    flat = _flatten_time(
        encoded.latent
    )

    centered = (
        flat
        - flat.mean(
            axis=0,
            keepdims=True,
        )
    )

    variance = np.var(
        flat,
        axis=0,
    )

    total_variance = float(
        np.sum(
            variance
        )
    )

    participation_ratio = (
        0.0
        if total_variance <= 1e-12
        else (
            total_variance**2
            / max(
                float(
                    np.sum(
                        variance**2
                    )
                ),
                1e-12,
            )
        )
    )

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    covariance = np.atleast_2d(
        covariance
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    positive = eigenvalues[
        eigenvalues > 1e-12
    ]

    covariance_condition = (
        float(
            positive.max()
            / positive.min()
        )
        if positive.size >= 2
        else float("inf")
    )

    off_diagonal = (
        covariance
        - np.diag(
            np.diag(covariance)
        )
    )

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    squared = singular_values**2

    explained = (
        squared
        / max(
            float(
                np.sum(squared)
            ),
            1e-12,
        )
    )

    cumulative = np.cumsum(
        explained
    )

    dimensions_95 = int(
        np.searchsorted(
            cumulative,
            0.95,
        )
        + 1
    )

    metrics = {
        "latent_mean_abs": float(
            np.mean(
                np.abs(flat)
            )
        ),
        "latent_mean_squared": float(
            np.mean(
                flat**2
            )
        ),
        "latent_norm_mean": float(
            np.mean(
                np.linalg.norm(
                    flat,
                    axis=1,
                )
            )
        ),
        "latent_std_global": float(
            np.std(flat)
        ),
        "latent_total_variance": total_variance,
        "latent_active_dimensions": float(
            np.sum(
                variance > 1e-4
            )
        ),
        "latent_participation_ratio": float(
            participation_ratio
        ),
        "latent_dimensions_95pct": float(
            dimensions_95
        ),
        "latent_covariance_rank": float(
            np.linalg.matrix_rank(
                covariance,
                tol=1e-10,
            )
        ),
        "latent_covariance_condition_number": covariance_condition,
        "latent_covariance_offdiag_mean_abs": float(
            np.mean(
                np.abs(
                    off_diagonal
                )
            )
        ),
    }

    for dimension_index, value in enumerate(
        variance
    ):
        metrics[
            f"latent_variance_dim_{dimension_index}"
        ] = float(value)

    return metrics


def invariance_metrics(
    encoded: EncodedDataset,
) -> dict[str, float]:
    """Evaluate consistency across counterfactual nuisance interventions."""
    if encoded.counterfactual_latent is None:
        return {}

    latent_a = encoded.latent
    latent_b = encoded.counterfactual_latent

    difference = (
        latent_a
        - latent_b
    )

    flat_a = _flatten_time(
        latent_a
    )
    flat_b = _flatten_time(
        latent_b
    )

    norm_a = np.linalg.norm(
        flat_a,
        axis=1,
    )

    norm_b = np.linalg.norm(
        flat_b,
        axis=1,
    )

    denominator = np.maximum(
        norm_a * norm_b,
        1e-12,
    )

    cosine = (
        np.sum(
            flat_a * flat_b,
            axis=1,
        )
        / denominator
    )

    latent_scale = np.mean(
        np.sum(
            (
                flat_a
                - flat_a.mean(
                    axis=0,
                    keepdims=True,
                )
            ) ** 2,
            axis=1,
        )
    )

    normalized_mse = (
        np.mean(
            difference**2
        )
        / max(
            latent_scale,
            1e-12,
        )
    )

    return {
        "counterfactual_latent_mse": float(
            np.mean(
                difference**2
            )
        ),
        "counterfactual_latent_mae": float(
            np.mean(
                np.abs(difference)
            )
        ),
        "counterfactual_cosine_similarity": float(
            np.mean(cosine)
        ),
        "counterfactual_normalized_mse": float(
            normalized_mse
        ),
    }


def adversarial_head_metrics(
    encoded: EncodedDataset,
) -> MetricDict:
    """Evaluate the nuisance adversary on known nuisance classes."""
    if encoded.nuisance_logits is None:
        return {}

    logits = encoded.nuisance_logits

    if logits.ndim == 3:
        labels = np.repeat(
            encoded.nuisance_id,
            logits.shape[1],
        )
        flat_logits = logits.reshape(
            -1,
            logits.shape[-1],
        )
    elif logits.ndim == 2:
        labels = encoded.nuisance_id
        flat_logits = logits
    else:
        return {}

    if (
        labels.size != flat_logits.shape[0]
    ):
        return {}

    valid = (
        labels >= 0
    ) & (
        labels < flat_logits.shape[-1]
    )

    if not np.all(valid):
        return {
            "adversarial_head_accuracy": None,
            "adversarial_head_unseen_classes": float(
                np.unique(
                    labels[~valid]
                ).size
            ),
        }

    prediction = np.argmax(
        flat_logits,
        axis=-1,
    )

    return {
        "adversarial_head_accuracy": float(
            accuracy_score(
                labels,
                prediction,
            )
        ),
        "adversarial_head_chance": float(
            1.0
            / flat_logits.shape[-1]
        ),
        "adversarial_head_unseen_classes": 0.0,
    }


# =============================================================================
# State probe analysis
# =============================================================================

def _fit_and_evaluate_state_probe(
    probe: Any,
    x_train: FloatArray,
    y_train: FloatArray,
    x_eval: FloatArray,
    y_eval: FloatArray,
    state_names: tuple[str, ...],
    *,
    metric_prefix: str,
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    """
    Fit a regression probe and evaluate it on another split.

    This helper implements the common evaluation procedure shared by the
    linear, nonlinear, and temporal state probes. It performs optional
    subsampling, fits the supplied estimator, computes predictions on the
    evaluation split, and reports global and per-state R² metrics.
    """
    x_train, y_train = _subsample_rows(
        x_train,
        y_train,
        max_samples=max_samples,
        seed=seed,
    )

    probe.fit(
        x_train,
        y_train,
    )

    prediction = probe.predict(
        x_eval
    )

    metrics = {
        f"{metric_prefix}_mse": float(
            mean_squared_error(
                y_eval,
                prediction,
            )
        ),
        f"{metric_prefix}_r2": float(
            r2_score(
                y_eval,
                prediction,
                multioutput="variance_weighted",
            )
        ),
    }

    for index, name in enumerate(state_names):
        metrics[
            f"{metric_prefix}_r2_{name}"
        ] = float(
            r2_score(
                y_eval[:, index],
                prediction[:, index],
            )
        )

    return metrics


def fit_state_probe(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    """Fit a linear probe from latent representations to the true state."""
    return _fit_and_evaluate_state_probe(
        make_pipeline(
            StandardScaler(),
            LinearRegression(),
        ),
        _flatten_time(
            train_encoded.latent
        ),
        _flatten_time(
            train_encoded.true_state
        ),
        _flatten_time(
            evaluation_encoded.latent
        ),
        _flatten_time(
            evaluation_encoded.true_state
        ),
        evaluation_encoded.state_names,
        metric_prefix="state_probe",
        max_samples=max_samples,
        seed=seed,
    )


def fit_nonlinear_state_probe(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    """Fit a small nonlinear probe and quantify nonlinear state access."""
    return _fit_and_evaluate_state_probe(
        make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=seed,
            ),
        ),
        _flatten_time(
            train_encoded.latent
        ),
        _flatten_time(
            train_encoded.true_state
        ),
        _flatten_time(
            evaluation_encoded.latent
        ),
        _flatten_time(
            evaluation_encoded.true_state
        ),
        evaluation_encoded.state_names,
        metric_prefix="state_nonlinear_probe",
        max_samples=max_samples,
        seed=seed,
    )


def fit_temporal_state_probe(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    """
    Fit a linear probe from adjacent latent states ``[z_t, z_{t+1}]``.

    This tests whether dynamical variables, such as velocity, are distributed
    across the latent trajectory rather than encoded instantaneously.
    """
    if (
        train_encoded.episode_length < 2
        or evaluation_encoded.episode_length < 2
    ):
        return {}

    x_train = np.concatenate(
        (
            train_encoded.latent[:, :-1, :],
            train_encoded.latent[:, 1:, :],
        ),
        axis=-1,
    ).reshape(
        -1,
        2 * train_encoded.latent_dim,
    )

    y_train = train_encoded.true_state[
        :,
        :-1,
        :,
    ].reshape(
        -1,
        train_encoded.state_dim,
    )

    x_eval = np.concatenate(
        (
            evaluation_encoded.latent[:, :-1, :],
            evaluation_encoded.latent[:, 1:, :],
        ),
        axis=-1,
    ).reshape(
        -1,
        2 * evaluation_encoded.latent_dim,
    )

    y_eval = evaluation_encoded.true_state[
        :,
        :-1,
        :,
    ].reshape(
        -1,
        evaluation_encoded.state_dim,
    )

    return _fit_and_evaluate_state_probe(
        make_pipeline(
            StandardScaler(),
            LinearRegression(),
        ),
        x_train,
        y_train,
        x_eval,
        y_eval,
        evaluation_encoded.state_names,
        metric_prefix="temporal_state_probe",
        max_samples=max_samples,
        seed=seed,
    )


# =============================================================================
# Canonical state analysis
# =============================================================================

def canonical_state_analysis(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> tuple[dict[str, float], CCAAnalysis | None]:
    """
    Fit CCA on train and evaluate canonical correlations on another split.

    The CCA mappings and standardization parameters are fitted exclusively on
    the training split. Canonical correlations and correlation loadings are
    then computed on the independent evaluation split.

    Returns
    -------
    metrics:
        Scalar CCA metrics suitable for JSON/CSV reporting.
    analysis:
        Arrays required for publication figures. ``state_loadings`` and
        ``latent_loadings`` are correlations between the standardized original
        variables and their corresponding canonical variates on the evaluation
        split. Raw weights and rotations are saved separately.
    """
    train_latent = _flatten_time(
        train_encoded.latent
    )
    train_state = _flatten_time(
        train_encoded.true_state
    )

    evaluation_latent = _flatten_time(
        evaluation_encoded.latent
    )
    evaluation_state = _flatten_time(
        evaluation_encoded.true_state
    )

    train_latent, train_state = _finite_pair(
        train_latent,
        train_state,
    )

    evaluation_latent, evaluation_state = _finite_pair(
        evaluation_latent,
        evaluation_state,
    )

    train_latent, train_state = _subsample_rows(
        train_latent,
        train_state,
        max_samples=max_samples,
        seed=seed,
    )

    n_components = min(
        train_latent.shape[1],
        train_state.shape[1],
    )

    if n_components < 1:
        return {}, None

    latent_scaler = StandardScaler()
    state_scaler = StandardScaler()

    standardized_train_latent = (
        latent_scaler.fit_transform(
            train_latent
        )
    )

    standardized_train_state = (
        state_scaler.fit_transform(
            train_state
        )
    )

    standardized_evaluation_latent = (
        latent_scaler.transform(
            evaluation_latent
        )
    )

    standardized_evaluation_state = (
        state_scaler.transform(
            evaluation_state
        )
    )

    cca = CCA(
        n_components=n_components,
        scale=False,
        max_iter=max_iter,
        tol=tolerance,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=ConvergenceWarning,
        )

        cca.fit(
            standardized_train_latent,
            standardized_train_state,
        )

    canonical_latent, canonical_state = cca.transform(
        standardized_evaluation_latent,
        standardized_evaluation_state,
    )

    correlations = np.asarray(
        [
            _safe_pearson(
                canonical_latent[:, component_index],
                canonical_state[:, component_index],
            )
            for component_index in range(
                n_components
            )
        ],
        dtype=np.float64,
    )

    def correlation_loadings(
        variables: FloatArray,
        canonical_variates: FloatArray,
    ) -> FloatArray:
        """Return corr(original variable, canonical variate)."""
        n_variables = variables.shape[1]
        n_variates = canonical_variates.shape[1]

        loadings = np.empty(
            (
                n_variables,
                n_variates,
            ),
            dtype=np.float64,
        )

        for variable_index in range(
            n_variables
        ):
            for component_index in range(
                n_variates
            ):
                loadings[
                    variable_index,
                    component_index,
                ] = _safe_pearson(
                    variables[:, variable_index],
                    canonical_variates[:, component_index],
                )

        return loadings

    latent_loadings = correlation_loadings(
        standardized_evaluation_latent,
        canonical_latent,
    )

    state_loadings = correlation_loadings(
        standardized_evaluation_state,
        canonical_state,
    )

    finite = correlations[
        np.isfinite(correlations)
    ]

    metrics: dict[str, float] = {}

    for component_index, correlation in enumerate(
        correlations,
        start=1,
    ):
        metrics[
            f"state_cca_correlation_{component_index}"
        ] = float(correlation)

    metrics["state_cca_mean_correlation"] = (
        float(
            np.mean(finite)
        )
        if finite.size
        else float("nan")
    )

    metrics["state_cca_min_correlation"] = (
        float(
            np.min(finite)
        )
        if finite.size
        else float("nan")
    )

    metrics[
        "state_cca_mean_squared_correlation"
    ] = (
        float(
            np.mean(
                finite**2
            )
        )
        if finite.size
        else float("nan")
    )

    metrics["state_cca_effective_components_090"] = float(
        np.sum(
            finite >= 0.90
        )
    )

    metrics["state_cca_effective_components_050"] = float(
        np.sum(
            finite >= 0.50
        )
    )

    analysis = CCAAnalysis(
        canonical_correlations=correlations,
        state_loadings=state_loadings,
        latent_loadings=latent_loadings,
        state_weights=np.asarray(
            cca.y_weights_,
            dtype=np.float64,
        ),
        latent_weights=np.asarray(
            cca.x_weights_,
            dtype=np.float64,
        ),
        state_rotations=np.asarray(
            cca.y_rotations_,
            dtype=np.float64,
        ),
        latent_rotations=np.asarray(
            cca.x_rotations_,
            dtype=np.float64,
        ),
        state_scaler_mean=np.asarray(
            state_scaler.mean_,
            dtype=np.float64,
        ),
        state_scaler_scale=np.asarray(
            state_scaler.scale_,
            dtype=np.float64,
        ),
        latent_scaler_mean=np.asarray(
            latent_scaler.mean_,
            dtype=np.float64,
        ),
        latent_scaler_scale=np.asarray(
            latent_scaler.scale_,
            dtype=np.float64,
        ),
    )

    return metrics, analysis


# =============================================================================
# Nuisance analysis
# =============================================================================

def fit_nuisance_probe(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
) -> MetricDict:
    """Fit a post-hoc linear nuisance classifier on individual latent states."""
    x_train, y_train = _flatten_latent_and_nuisance(
        train_encoded
    )

    x_eval, y_eval = _flatten_latent_and_nuisance(
        evaluation_encoded
    )

    x_train, y_train = _subsample_rows(
        x_train,
        y_train,
        max_samples=max_samples,
        seed=seed,
    )

    x_eval, y_eval = _subsample_rows(
        x_eval,
        y_eval,
        max_samples=max_samples,
        seed=seed + 1,
    )

    train_classes = np.unique(
        y_train
    )

    evaluation_classes = np.unique(
        y_eval
    )

    if train_classes.size < 2:
        return {
            "nuisance_timepoint_probe_accuracy": None,
            "nuisance_timepoint_probe_chance": None,
            "nuisance_timepoint_probe_unseen_classes": 0.0,
        }

    unseen = np.setdiff1d(
        evaluation_classes,
        train_classes,
    )

    if unseen.size > 0:
        return {
            "nuisance_timepoint_probe_accuracy": None,
            "nuisance_timepoint_probe_chance": None,
            "nuisance_timepoint_probe_unseen_classes": float(
                unseen.size
            ),
        }

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2_000,
            random_state=seed,
        ),
    )

    probe.fit(
        x_train,
        y_train,
    )

    prediction = probe.predict(
        x_eval
    )

    return {
        "nuisance_timepoint_probe_accuracy": float(
            accuracy_score(
                y_eval,
                prediction,
            )
        ),
        "nuisance_timepoint_probe_chance": float(
            1.0
            / train_classes.size
        ),
        "nuisance_timepoint_probe_unseen_classes": 0.0,
    }


def fit_conditional_nuisance_probe(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
) -> MetricDict:
    """Probe nuisance accessibility after linearly controlling for state."""
    x_train, y_train = _flatten_latent_and_nuisance(
        train_encoded
    )

    x_eval, y_eval = _flatten_latent_and_nuisance(
        evaluation_encoded
    )

    state_train = _flatten_time(
        train_encoded.true_state
    )

    state_eval = _flatten_time(
        evaluation_encoded.true_state
    )

    train_classes = np.unique(
        y_train
    )

    evaluation_classes = np.unique(
        y_eval
    )

    if train_classes.size < 2:
        return {
            "conditional_nuisance_probe_accuracy": None,
            "conditional_nuisance_probe_chance": None,
            "conditional_nuisance_probe_unseen_classes": 0.0,
        }

    unseen = np.setdiff1d(
        evaluation_classes,
        train_classes,
    )

    if unseen.size > 0:
        return {
            "conditional_nuisance_probe_accuracy": None,
            "conditional_nuisance_probe_chance": None,
            "conditional_nuisance_probe_unseen_classes": float(
                unseen.size
            ),
        }

    residualizer = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )

    residualizer.fit(
        state_train,
        x_train,
    )

    residual_train = (
        x_train
        - residualizer.predict(
            state_train
        )
    )

    residual_eval = (
        x_eval
        - residualizer.predict(
            state_eval
        )
    )

    residual_train, y_train = _subsample_rows(
        residual_train,
        y_train,
        max_samples=max_samples,
        seed=seed,
    )

    residual_eval, y_eval = _subsample_rows(
        residual_eval,
        y_eval,
        max_samples=max_samples,
        seed=seed + 1,
    )

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2_000,
            random_state=seed,
        ),
    )

    probe.fit(
        residual_train,
        y_train,
    )

    prediction = probe.predict(
        residual_eval
    )

    return {
        "conditional_nuisance_probe_accuracy": float(
            accuracy_score(
                y_eval,
                prediction,
            )
        ),
        "conditional_nuisance_probe_chance": float(
            1.0
            / train_classes.size
        ),
        "conditional_nuisance_probe_unseen_classes": 0.0,
    }


def nuisance_subspace_removal_metrics(
    train_encoded: EncodedDataset,
    evaluation_encoded: EncodedDataset,
    *,
    max_samples: int,
    seed: int,
) -> MetricDict:
    """Remove a linear nuisance subspace and re-evaluate state accessibility."""
    x_train, nuisance_train = _flatten_latent_and_nuisance(
        train_encoded
    )

    x_eval, nuisance_eval = _flatten_latent_and_nuisance(
        evaluation_encoded
    )

    state_train = _flatten_time(
        train_encoded.true_state
    )

    state_eval = _flatten_time(
        evaluation_encoded.true_state
    )

    train_classes = np.unique(
        nuisance_train
    )

    evaluation_classes = np.unique(
        nuisance_eval
    )

    if train_classes.size < 2:
        return {}

    if np.setdiff1d(
        evaluation_classes,
        train_classes,
    ).size > 0:
        return {
            "nuisance_removed_probe_accuracy": None,
            "nuisance_removed_state_probe_r2": None,
            "nuisance_removal_state_r2_delta": None,
            "nuisance_subspace_rank": None,
        }

    scaler = StandardScaler()

    standardized_train = scaler.fit_transform(
        x_train
    )

    standardized_eval = scaler.transform(
        x_eval
    )

    probe_train, nuisance_probe_train = _subsample_rows(
        standardized_train,
        nuisance_train,
        max_samples=max_samples,
        seed=seed,
    )

    nuisance_probe = LogisticRegression(
        max_iter=2_000,
        random_state=seed,
    )

    nuisance_probe.fit(
        probe_train,
        nuisance_probe_train,
    )

    coefficients = np.asarray(
        nuisance_probe.coef_,
        dtype=float,
    )

    _, singular_values, right_vectors = np.linalg.svd(
        coefficients,
        full_matrices=False,
    )

    rank = int(
        np.sum(
            singular_values > 1e-10
        )
    )

    if rank == 0:
        cleaned_train = standardized_train
        cleaned_eval = standardized_eval
    else:
        basis = right_vectors[
            :rank
        ].T

        cleaned_train = (
            standardized_train
            - (
                standardized_train
                @ basis
            )
            @ basis.T
        )

        cleaned_eval = (
            standardized_eval
            - (
                standardized_eval
                @ basis
            )
            @ basis.T
        )

    cleaned_probe_train, cleaned_nuisance_train = _subsample_rows(
        cleaned_train,
        nuisance_train,
        max_samples=max_samples,
        seed=seed,
    )

    cleaned_probe_eval, cleaned_nuisance_eval = _subsample_rows(
        cleaned_eval,
        nuisance_eval,
        max_samples=max_samples,
        seed=seed + 1,
    )

    residual_nuisance_probe = LogisticRegression(
        max_iter=2_000,
        random_state=seed,
    )

    residual_nuisance_probe.fit(
        cleaned_probe_train,
        cleaned_nuisance_train,
    )

    nuisance_prediction = residual_nuisance_probe.predict(
        cleaned_probe_eval
    )

    state_train_features, state_train_targets = _subsample_rows(
        cleaned_train,
        state_train,
        max_samples=max_samples,
        seed=seed,
    )

    state_probe = LinearRegression()

    state_probe.fit(
        state_train_features,
        state_train_targets,
    )

    cleaned_state_prediction = state_probe.predict(
        cleaned_eval
    )

    cleaned_state_r2 = float(
        r2_score(
            state_eval,
            cleaned_state_prediction,
            multioutput="variance_weighted",
        )
    )

    original_state_probe = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )

    original_train_features, original_train_targets = _subsample_rows(
        x_train,
        state_train,
        max_samples=max_samples,
        seed=seed,
    )

    original_state_probe.fit(
        original_train_features,
        original_train_targets,
    )

    original_state_r2 = float(
        r2_score(
            state_eval,
            original_state_probe.predict(
                x_eval
            ),
            multioutput="variance_weighted",
        )
    )

    return {
        "nuisance_removed_probe_accuracy": float(
            accuracy_score(
                cleaned_nuisance_eval,
                nuisance_prediction,
            )
        ),
        "nuisance_removed_state_probe_r2": cleaned_state_r2,
        "nuisance_removal_state_r2_delta": float(
            cleaned_state_r2
            - original_state_r2
        ),
        "nuisance_subspace_rank": float(
            rank
        ),
    }


# =============================================================================
# Dataset evaluation orchestration
# =============================================================================

def evaluate_dataset(
    model: CLSMModel,
    dataset: CLSMDataset,
    *,
    train_encoded: EncodedDataset | None,
    config: EvaluationConfig,
    device: torch.device,
) -> tuple[
    MetricDict,
    EncodedDataset,
    CCAAnalysis | None,
]:
    """
    Evaluate one dataset split.

    Training-dependent probes (state probes, CCA, nuisance analyses) are
    always fitted on `train_encoded` and evaluated on the current split,
    ensuring strict separation between fitting and evaluation.
    """
    encoded = encode_dataset(
        model,
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        device=device,
    )

    metrics: MetricDict = {}
    cca_analysis: CCAAnalysis | None = None

    metrics.update(
        observation_compatibility_metrics(
            encoded
        )
    )

    metrics.update(
        temporal_coherence_metrics(
            encoded
        )
    )

    metrics.update(
        rollout_metrics(
            model,
            dataset,
            horizons=config.rollout_horizons,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            device=device,
        )
    )

    metrics.update(
        minimality_metrics(
            encoded
        )
    )

    metrics.update(
        neighborhood_preservation_metrics(
            encoded,
            max_samples=config.neighborhood_max_samples,
            n_neighbors=config.neighborhood_neighbors,
            seed=config.probe_seed,
        )
    )

    metrics.update(
        invariance_metrics(
            encoded
        )
    )

    metrics.update(
        adversarial_head_metrics(
            encoded
        )
    )

    if train_encoded is not None:
        linear_metrics = fit_state_probe(
            train_encoded,
            encoded,
            max_samples=config.probe_max_samples,
            seed=config.probe_seed,
        )

        nonlinear_metrics = fit_nonlinear_state_probe(
            train_encoded,
            encoded,
            max_samples=config.nonlinear_probe_max_samples,
            seed=config.probe_seed,
        )

        temporal_probe_metrics = fit_temporal_state_probe(
            train_encoded,
            encoded,
            max_samples=config.temporal_probe_max_samples,
            seed=config.probe_seed,
        )

        (
            cca_metrics,
            cca_analysis,
        ) = canonical_state_analysis(
            train_encoded,
            encoded,
            max_samples=config.cca_max_samples,
            seed=config.probe_seed,
            max_iter=config.cca_max_iter,
            tolerance=config.cca_tolerance,
        )

        metrics.update(
            linear_metrics
        )
        metrics.update(
            nonlinear_metrics
        )
        metrics.update(
            temporal_probe_metrics
        )
        metrics.update(
            cca_metrics
        )

        if (
            "state_probe_r2" in linear_metrics
            and "state_nonlinear_probe_r2" in nonlinear_metrics
        ):
            metrics["state_linearity_gap"] = float(
                nonlinear_metrics[
                    "state_nonlinear_probe_r2"
                ]
                - linear_metrics[
                    "state_probe_r2"
                ]
            )

        metrics.update(
            fit_nuisance_probe(
                train_encoded,
                encoded,
                max_samples=config.probe_max_samples,
                seed=config.probe_seed,
            )
        )

        metrics.update(
            fit_conditional_nuisance_probe(
                train_encoded,
                encoded,
                max_samples=config.probe_max_samples,
                seed=config.probe_seed,
            )
        )

        metrics.update(
            nuisance_subspace_removal_metrics(
                train_encoded,
                encoded,
                max_samples=config.probe_max_samples,
                seed=config.probe_seed,
            )
        )

    return (
        metrics,
        encoded,
        cca_analysis,
    )


def evaluate_splits(
    model: CLSMModel,
    splits: DatasetSplits,
    *,
    config: EvaluationConfig,
    selected_splits: Sequence[str] = (
        "validation",
        "test",
        "ood",
    ),
) -> dict[str, MetricDict]:
    """
    Evaluate selected splits using probes fitted on the training split.

    The training split is encoded once and reused for all downstream
    probe fitting.
    """
    device = next(
        model.parameters()
    ).device

    train_encoded = encode_dataset(
        model,
        splits.train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        device=device,
    )

    split_map: dict[
        str,
        CLSMDataset | None,
    ] = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
        "ood": splits.ood,
    }

    results: dict[
        str,
        MetricDict,
    ] = {}

    if (
        config.save_latents
        and config.output_dir is not None
        and "train" not in selected_splits
    ):
        save_encoded_dataset(
            train_encoded,
            Path(config.output_dir)
            / "train_encoded.npz",
        )

    for split_name in selected_splits:
        if split_name not in split_map:
            raise KeyError(
                f"Unknown split: {split_name}"
            )

        dataset = split_map[
            split_name
        ]

        if dataset is None:
            continue

        if split_name == "train":
            encoded = train_encoded

            (
                metrics,
                _,
                cca_analysis,
            ) = evaluate_dataset(
                model,
                dataset,
                train_encoded=train_encoded,
                config=config,
                device=device,
            )
        else:
            (
                metrics,
                encoded,
                cca_analysis,
            ) = evaluate_dataset(
                model,
                dataset,
                train_encoded=train_encoded,
                config=config,
                device=device,
            )

        results[
            split_name
        ] = metrics

        if (
            config.save_latents
            and config.output_dir is not None
        ):
            save_encoded_dataset(
                encoded,
                Path(config.output_dir)
                / f"{split_name}_encoded.npz",
            )

        if (
            cca_analysis is not None
            and config.output_dir is not None
        ):
            output_dir = Path(
                config.output_dir
            )

            save_cca_analysis(
                cca_analysis,
                output_dir
                / f"{split_name}_cca_analysis.npz",
            )

            # Backward-compatible alias
            if split_name == "test":
                save_cca_analysis(
                    cca_analysis,
                    output_dir
                    / "cca_analysis.npz",
                )

    return results


# =============================================================================
# Evaluation artifact serialization
# =============================================================================

def save_encoded_dataset(
    encoded: EncodedDataset,
    path: Path | str,
) -> Path:
    """Save encoded representations and associated metadata."""
    path = Path(path)

    if path.suffix != ".npz":
        path = path.with_suffix(
            ".npz"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays: dict[str, object] = {
        "latent": encoded.latent,
        "true_state": encoded.true_state,
        "observation": encoded.observation,
        "nuisance_id": encoded.nuisance_id,
        "reconstructed_observation": (
            encoded.reconstructed_observation
        ),
        "state_names": np.asarray(encoded.state_names, dtype="U"),
        "episode_index": encoded.episode_index,
        "time_index": encoded.time_index,
    }

    if encoded.nuisance is not None:
        arrays["nuisance"] = (
            encoded.nuisance
        )

    optional_arrays = {
        "counterfactual_observation": (
            encoded.counterfactual_observation
        ),
        "counterfactual_latent": (
            encoded.counterfactual_latent
        ),
        "predicted_next_latent": (
            encoded.predicted_next_latent
        ),
        "nuisance_logits": (
            encoded.nuisance_logits
        ),
    }

    for name, value in optional_arrays.items():
        if value is not None:
            arrays[name] = value

    np.savez_compressed(
        path,
        **arrays,
    )

    return path


def save_cca_analysis(
    analysis: CCAAnalysis,
    path: Path | str,
) -> Path:
    """Save complete CCA artifacts for later visualization."""
    path = Path(path)

    if path.suffix != ".npz":
        path = path.with_suffix(
            ".npz"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        canonical_correlations=(
            analysis.canonical_correlations
        ),
        state_loadings=(
            analysis.state_loadings
        ),
        latent_loadings=(
            analysis.latent_loadings
        ),
        state_weights=(
            analysis.state_weights
        ),
        latent_weights=(
            analysis.latent_weights
        ),
        state_rotations=(
            analysis.state_rotations
        ),
        latent_rotations=(
            analysis.latent_rotations
        ),
        state_scaler_mean=(
            analysis.state_scaler_mean
        ),
        state_scaler_scale=(
            analysis.state_scaler_scale
        ),
        latent_scaler_mean=(
            analysis.latent_scaler_mean
        ),
        latent_scaler_scale=(
            analysis.latent_scaler_scale
        ),
    )

    return path


def _json_safe_value(
    value: MetricValue,
) -> MetricValue:
    """Convert non-finite values to JSON-safe null."""
    if value is None:
        return None

    numeric = float(value)

    if not np.isfinite(
        numeric
    ):
        return None

    return numeric


def _json_safe_results(
    results: Mapping[
        str,
        Mapping[str, MetricValue],
    ],
) -> dict[
    str,
    dict[str, MetricValue],
]:
    """Convert nested metric dictionaries to JSON-safe values."""
    return {
        split_name: {
            metric_name: _json_safe_value(
                value
            )
            for metric_name, value in metrics.items()
        }
        for split_name, metrics in results.items()
    }


def save_results_json(
    results: Mapping[
        str,
        Mapping[str, MetricValue],
    ],
    path: Path | str,
) -> Path:
    """Save nested split metrics to JSON."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            _json_safe_results(
                results
            ),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    return path


def save_results_csv(
    results: Mapping[
        str,
        Mapping[str, MetricValue],
    ],
    path: Path | str,
) -> Path:
    """Save split metrics in long-format CSV."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        {
            "split": split_name,
            "metric": metric_name,
            "value": (
                ""
                if value is None
                or not np.isfinite(
                    float(value)
                )
                else float(value)
            ),
        }
        for split_name, metrics in results.items()
        for metric_name, value in metrics.items()
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "split",
                "metric",
                "value",
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return path


# =============================================================================
# Reporting and aggregation
# =============================================================================

def print_compact_evaluation_summary(
    run_name: str,
    results: Mapping[
        str,
        Mapping[str, MetricValue],
    ],
    state_names: tuple[str, ...],
) -> None:
    """Print principal comparable metrics."""
    selected_metrics = [
        "observation_mse",
        "rollout_observation_mse_h5",
        "rollout_latent_mse_h5",
        "state_linearity_gap",
        "state_cca_mean_correlation",
        "state_cca_min_correlation",
        "neighborhood_trustworthiness",
        "counterfactual_normalized_mse",
        "conditional_nuisance_probe_accuracy",
        "state_probe_r2",
        "temporal_state_probe_r2",
    ]
    for state_name in state_names:
        selected_metrics.append(f"state_probe_r2_{state_name}")
        selected_metrics.append(f"temporal_state_probe_r2_{state_name}")
        

    print()
    print(
        f"Evaluation summary: {run_name}"
    )
    print_separator("-")

    for split_name in (
        "test",
        "ood",
    ):
        if split_name not in results:
            continue

        print(
            split_name.upper()
        )

        for metric_name in selected_metrics:
            value = results[
                split_name
            ].get(
                metric_name
            )

            if value is None:
                print(
                    f"  {metric_name:<42} {'N/A':>12}"
                )
            else:
                print(
                    f"  {metric_name:<42} {value:>12.6f}"
                )

    print_separator("-")


def aggregate_seed_results(
    seed_results: Mapping[
        str,
        Mapping[
            str,
            Mapping[str, MetricValue],
        ],
    ],
) -> dict[
    str,
    dict[
        str,
        dict[str, MetricValue],
    ],
]:
    """Aggregate split metrics across runs."""
    aggregated: dict[
        str,
        dict[
            str,
            dict[str, MetricValue],
        ],
    ] = {}

    split_names = sorted(
        {
            split_name
            for run_results in seed_results.values()
            for split_name in run_results
        }
    )

    for split_name in split_names:
        metric_names = sorted(
            {
                metric_name
                for run_results in seed_results.values()
                if split_name in run_results
                for metric_name in run_results[
                    split_name
                ]
            }
        )

        aggregated[
            split_name
        ] = {}

        for metric_name in metric_names:
            raw_values = [
                run_results[
                    split_name
                ][
                    metric_name
                ]
                for run_results in seed_results.values()
                if (
                    split_name in run_results
                    and metric_name
                    in run_results[
                        split_name
                    ]
                    and run_results[
                        split_name
                    ][
                        metric_name
                    ]
                    is not None
                )
            ]

            values = np.asarray(
                raw_values,
                dtype=float,
            )

            finite = values[
                np.isfinite(values)
            ]

            if finite.size == 0:
                summary: dict[
                    str,
                    MetricValue,
                ] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "n": 0.0,
                }
            else:
                summary = {
                    "mean": float(
                        np.mean(finite)
                    ),
                    "std": float(
                        np.std(
                            finite,
                            ddof=1,
                        )
                        if finite.size > 1
                        else 0.0
                    ),
                    "min": float(
                        np.min(finite)
                    ),
                    "max": float(
                        np.max(finite)
                    ),
                    "n": float(
                        finite.size
                    ),
                }

            aggregated[
                split_name
            ][
                metric_name
            ] = summary

    return aggregated


# =============================================================================
# Command-line configuration
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CLSM model.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Single checkpoint path, "
            "e.g. runs/full-seed-0/best.pt."
        ),
    )

    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help=(
            "Checkpoint template for several seeds, e.g. "
            "'runs/full-seed-{}/best.pt'."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "test",
            "ood",
            "all",
        ),
        default="all",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=[
            1,
            5,
            10,
        ],
    )

    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--nonlinear-probe-max-samples",
        type=int,
        default=25_000,
    )

    parser.add_argument(
        "--temporal-probe-max-samples",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--cca-max-samples",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--cca-max-iter",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--cca-tolerance",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--neighborhood-max-samples",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--neighborhood-neighbors",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--probe-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--save-latents",
        action="store_true",
    )

    parser.add_argument(
        "--aggregate-output",
        default=None,
    )

    return parser


# =============================================================================
# Checkpoint evaluation orchestration
# =============================================================================

def resolve_evaluation_seeds(
    args: argparse.Namespace,
) -> list[int | None]:
    """Resolve single-checkpoint and multi-seed evaluation modes."""
    if args.checkpoint is not None:
        if args.checkpoint_template is not None:
            raise ValueError(
                "Use either --checkpoint or --checkpoint-template, "
                "not both."
            )

        if (
            args.seed is not None
            or args.seeds is not None
        ):
            raise ValueError(
                "--seed/--seeds require --checkpoint-template."
            )

        return [
            None
        ]

    if args.checkpoint_template is None:
        raise ValueError(
            "Provide either --checkpoint or --checkpoint-template."
        )

    if (
        args.seed is not None
        and args.seeds is not None
    ):
        raise ValueError(
            "Use either --seed or --seeds, not both."
        )

    if args.seeds is not None:
        seeds = list(
            args.seeds
        )
    elif args.seed is not None:
        seeds = [
            args.seed
        ]
    else:
        raise ValueError(
            "--checkpoint-template requires --seed or --seeds."
        )

    if len(
        set(seeds)
    ) != len(seeds):
        raise ValueError(
            "Seeds must be unique."
        )

    return seeds


def format_checkpoint_path(
    template: str,
    seed: int,
) -> Path:
    """Format one checkpoint path from a seed template."""
    if "{}" in template:
        return Path(
            template.format(seed)
        )

    if "{seed}" in template:
        return Path(
            template.format(
                seed=seed
            )
        )

    raise ValueError(
        "--checkpoint-template must contain '{}' or '{seed}'."
    )


def evaluate_one_checkpoint(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, MetricDict]:
    """Evaluate one checkpoint and save all artifacts."""
    model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        device=device,
    )

    splits = load_splits(
        data_dir
    )
    state_names = splits.train.state_names

    selected_splits = (
        (
            "train",
            "validation",
            "test",
            "ood",
        )
        if args.split == "all"
        else (
            args.split,
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir is not None
        else (
            checkpoint_path.parent
            / "evaluation"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = EvaluationConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        rollout_horizons=tuple(
            args.rollout_horizons
        ),
        probe_max_samples=args.probe_max_samples,
        nonlinear_probe_max_samples=(
            args.nonlinear_probe_max_samples
        ),
        temporal_probe_max_samples=(
            args.temporal_probe_max_samples
        ),
        cca_max_samples=args.cca_max_samples,
        cca_max_iter=args.cca_max_iter,
        cca_tolerance=args.cca_tolerance,
        neighborhood_max_samples=(
            args.neighborhood_max_samples
        ),
        neighborhood_neighbors=(
            args.neighborhood_neighbors
        ),
        probe_seed=args.probe_seed,
        save_latents=args.save_latents,
        output_dir=str(
            output_dir
        ),
    )

    results = evaluate_splits(
        model,
        splits,
        config=config,
        selected_splits=selected_splits,
    )

    save_results_json(
        results,
        output_dir
        / "evaluation_metrics.json",
    )

    save_results_csv(
        results,
        output_dir
        / "evaluation_metrics.csv",
    )

    print_compact_evaluation_summary(
        checkpoint_path.parent.name,
        results,
        state_names,
    )

    return results


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = build_arg_parser()

    args = parser.parse_args()

    try:
        seeds = resolve_evaluation_seeds(
            args
        )
    except ValueError as error:
        parser.error(
            str(error)
        )

    device = resolve_device(
        args.device
    )

    data_dir = Path(
        args.data_dir
    )

    print()
    print_separator()
    print(
        f"Data     : {data_dir}"
    )
    print(
        f"Device   : {device}"
    )

    if device.type == "cuda":
        print(
            "GPU      : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        "Seeds    : "
        + (
            "single checkpoint"
            if seeds == [None]
            else " ".join(
                str(seed)
                for seed in seeds
            )
        )
    )
    print_separator()

    seed_results: dict[
        str,
        dict[str, MetricDict],
    ] = {}

    progress = tqdm(
        seeds,
        desc="Evaluating",
        unit="run",
        dynamic_ncols=True,
    )

    for seed in progress:
        if seed is None:
            checkpoint_path = Path(
                args.checkpoint
            )
        else:
            checkpoint_path = format_checkpoint_path(
                args.checkpoint_template,
                seed,
            )

        run_label = (
            checkpoint_path.parent.name
        )

        progress.set_postfix(
            run=run_label
        )

        seed_results[
            run_label
        ] = evaluate_one_checkpoint(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            args=args,
            device=device,
        )

    if len(
        seed_results
    ) > 1:
        aggregated = aggregate_seed_results(
            seed_results
        )

        if args.aggregate_output is not None:
            aggregate_path = Path(
                args.aggregate_output
            )
        else:
            first_seed = next(
                seed
                for seed in seeds
                if seed is not None
            )

            first_checkpoint = format_checkpoint_path(
                args.checkpoint_template,
                first_seed,
            )

            aggregate_path = (
                first_checkpoint.parent.parent
                / "aggregate_evaluation_metrics.json"
            )

        aggregate_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "runs": {
                run_name: _json_safe_results(
                    run_results
                )
                for run_name, run_results in seed_results.items()
            },
            "aggregate": aggregated,
        }

        with aggregate_path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

        print()
        print(
            "Aggregated metrics saved to: "
            f"{aggregate_path}"
        )

        selected_metrics = (
            "observation_mse",
            "rollout_observation_mse_h5",
            "rollout_latent_mse_h5",
            "counterfactual_normalized_mse",
            "state_probe_r2",
            "temporal_state_probe_r2",
            "state_cca_mean_correlation",
            "state_linearity_gap",
            "neighborhood_trustworthiness",
            "conditional_nuisance_probe_accuracy",
        )

        for split_name in (
            "test",
            "ood",
        ):
            if split_name not in aggregated:
                continue

            print()
            print(
                f"{split_name.upper()} aggregate"
            )
            print_separator("-")

            for metric_name in selected_metrics:
                if metric_name not in aggregated[
                    split_name
                ]:
                    continue

                summary = aggregated[
                    split_name
                ][
                    metric_name
                ]

                if summary["mean"] is None:
                    print(
                        f"{metric_name:<42} {'N/A':>20}"
                    )
                else:
                    print(
                        f"{metric_name:<42} "
                        f"{summary['mean']:.6f} ± "
                        f"{summary['std']:.6f}"
                    )


if __name__ == "__main__":
    main()
