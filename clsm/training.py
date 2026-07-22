"""
Generic training utilities for CLSM.

Author: Gwenolé Quellec
Year: 2026

This script trains the CLSM model with configurable weights
for the six CLSM constraint families:

1. predictive sufficiency;
2. minimality;
3. temporal coherence;
4. observation compatibility;
5. invariance to nuisance factors;
6. structural constraints.

The implementation uses plain PyTorch.

Outputs are written to ``runs/<run-name>/``:
- ``config.json``
- ``history.csv``
- ``metrics.json``
- ``best.pt``
- ``last.pt``

Training uses standard observation sequences without counterfactual pairing.
When enabled, the invariance objective is implemented through categorical
nuisance prediction and gradient reversal. Counterfactual views are reserved
for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from clsm.utils import print_banner, print_separator

from .datasets import CLSMDataset, DatasetSplits
from .losses import (
    CLSMLoss,
    ConstraintWeights,
    minimality_loss,
    nuisance_adversarial_accuracy,
    nuisance_adversarial_loss,
    observation_compatibility_loss,
    predictive_sufficiency_loss,
    structural_constraint_loss,
    temporal_coherence_loss,
)
from .models import CLSMModel, CLSMModelConfig


# =============================================================================
# Constants
# =============================================================================

UNRESOLVED_OBSERVATION_DIM = 1


# =============================================================================
# Training configuration
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """Dataset loading configuration."""

    data_dir: Path | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimization configuration."""

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float | None = 5.0
    num_workers: int = 0
    pin_memory: bool = True
    early_stopping_patience: int = 20
    seed: int = 42
    device: str = "auto"
    adversarial_warmup_epochs: int = 20


@dataclass(frozen=True)
class LossConfig:
    """
    CLSM loss and surrogate configuration.

    ``predictive_horizon_decay`` is applied to the order of the selected horizons
    rather than to their actual temporal distance.
    """

    weights: ConstraintWeights = field(
        default_factory=ConstraintWeights
    )

    prediction_loss_type: str = "mse"
    reconstruction_loss_type: str = "mse"
    minimality_mode: str = "l1"
    temporal_mode: str = "dynamics"

    predictive_horizons: tuple[int, ...] = (
        1,
        5,
        10,
    )

    predictive_horizon_decay: float = 1.0

    structural_variance_target: float = 1.0
    structural_variance_weight: float = 1.0
    structural_covariance_weight: float = 1.0


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Complete training configuration."""

    model: CLSMModelConfig
    run_name: str = "clsm"
    output_dir: Path = Path("runs")
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    loss: LossConfig = field(default_factory=LossConfig)


PRESET_WEIGHTS: dict[str, ConstraintWeights] = {

    # Pure presets
    "reconstruction": ConstraintWeights(
        predictive=0.0,
        minimality=0.0,
        temporal=0.0,
        observation=1.0,
        invariance=0.0,
        structural=0.0,
    ),
    "predictive": ConstraintWeights(
        predictive=1.0,
        minimality=0.0,
        temporal=0.0,
        observation=0.0,
        invariance=0.0,
        structural=0.0,
    ),
    "temporal": ConstraintWeights(
        predictive=0.0,
        minimality=0.0,
        temporal=1.0,
        observation=0.0,
        invariance=0.0,
        structural=0.0,
    ),
    "structural": ConstraintWeights(
        predictive=0.0,
        minimality=0.0,
        temporal=0.0,
        observation=0.0,
        invariance=0.0,
        structural=1.0,
    ),

    # Useful presets
    "reconstruction_predictive": ConstraintWeights(
        predictive=1.0,
        minimality=0.0,
        temporal=0.0,
        observation=1.0,
        invariance=0.0,
        structural=0.0,
    ),
    "base": ConstraintWeights(
        predictive=1.0,
        minimality=0.0,
        temporal=0.5,
        observation=0.5,
        invariance=0.0,
        structural=0.0,
    ),
    "base_invariance": ConstraintWeights(
        predictive=1.0,
        minimality=0.0,
        temporal=0.5,
        observation=0.5,
        invariance=0.1,
        structural=0.0,
    ),
    "base_structural": ConstraintWeights(
        predictive=1.0,
        minimality=0.0,
        temporal=0.5,
        observation=0.5,
        invariance=0.0,
        structural=0.1,
    ),
    "base_minimality": ConstraintWeights(
        predictive=1.0,
        minimality=0.01,  # small weight to reduce the risk of representational collapse.
        temporal=0.5,
        observation=0.5,
        invariance=0.0,
        structural=0.0,
    ),
    "full": ConstraintWeights(
        predictive=0.15,
        minimality=0.0025,
        temporal=0.25,
        observation=0.5,
        invariance=0.003,
        structural=0.05,
    ),
}


# =============================================================================
# PyTorch data utilities
# =============================================================================

class TorchEpisodeDataset(Dataset):
    """Expose a :class:`CLSMDataset` through the PyTorch Dataset API."""

    def __init__(self, dataset: CLSMDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset.n_episodes

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item: dict[str, Tensor] = {
            "observation": torch.from_numpy(
                self.dataset.observation[index]
            ).float(),
            "latent_state": torch.from_numpy(
                self.dataset.latent_state[index]
            ).float(),
            "nuisance": torch.from_numpy(
                self.dataset.nuisance[index]
            ).float(),
            "nuisance_id": torch.tensor(
                self.dataset.nuisance_id[index],
                dtype=torch.long,
            ),
        }

        if self.dataset.has_counterfactuals:
            item["counterfactual_observation"] = torch.from_numpy(
                self.dataset.counterfactual_observation[index]
            ).float()

        return item


def make_loader(
    dataset: CLSMDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a PyTorch data loader."""
    if dataset.n_episodes < 1:
        raise ValueError("Cannot create a loader for an empty dataset.")

    return DataLoader(
        TorchEpisodeDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def move_batch_to_device(
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    """Move every tensor in a batch to the selected device."""
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in batch.items()
    }


# =============================================================================
# Reproducibility and device handling
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """
    Resolve ``auto``, ``cpu``, ``cuda``, or a concrete device such as
    ``cuda:0``.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    return device


# =============================================================================
# Dataset loading and provenance
# =============================================================================

def load_splits(
    data_dir: str | Path,
) -> DatasetSplits:
    """
    Load required train, validation, and test splits and an optional OOD split.

    All loaded splits must share the same observation dimension and state
    schema.
    """
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

    splits = DatasetSplits(
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

    # Verify all datasets use the same schema
    datasets = (
        splits.train,
        splits.validation,
        splits.test,
    )
    if splits.ood is not None:
        datasets += (splits.ood,)
    for dataset in datasets[1:]:
        if dataset.observation_dim != splits.train.observation_dim:
            raise ValueError(
                "All splits must have the same observation dimension."
            )
        if dataset.state_names != splits.train.state_names:
            raise ValueError(
                "All splits must have the same state names."
            )

    return splits


def save_data_manifest(
    run_dir: Path,
    data_dir: str | Path,
) -> Path:
    """Save the paths of the datasets used for a training run."""
    data_dir = Path(data_dir)

    split_filenames = {
        "train": "train.npz",
        "validation": "validation.npz",
        "test": "test.npz",
        "ood": "ood.npz",
    }

    files = {}

    for split_name, filename in split_filenames.items():
        path = data_dir / filename

        if path.exists():
            files[split_name] = str(path)

    required_splits = {"train", "validation", "test"}
    missing_required = required_splits - files.keys()

    if missing_required:
        raise FileNotFoundError(
            "Cannot create data manifest. Missing required dataset files: "
            + ", ".join(sorted(missing_required))
        )

    manifest = {
        "data_dir": str(data_dir),
        "files": files,
    }

    manifest_path = run_dir / "data_manifest.json"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            manifest,
            stream,
            indent=2,
            sort_keys=True,
        )

    print(f"Data manifest saved to: {manifest_path}")

    return manifest_path


# =============================================================================
# Loss preparation and computation
# =============================================================================

def build_multi_horizon_observation_predictions(
    model: CLSMModel,
    latent: Tensor,
    observation: Tensor,
    *,
    horizons: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """
    Build open-loop future-observation predictions from each valid latent
    starting point.

    Parameters
    ----------
    model:
        CLSM model containing the transition and observation decoder.
    latent:
        Encoded sequence with shape ``(B, T, latent_dim)``.
    observation:
        Observation sequence with shape ``(B, T, observation_dim)``.
    horizons:
        Strictly positive prediction horizons, for example ``(1, 5, 10)``.

    Returns
    -------
    predicted_future:
        Tensor with shape ``(B, T-H, n_horizons, observation_dim)``.
    target_future:
        Ground-truth tensor with the same shape.
    """
    horizons = tuple(
        sorted(
            {
                int(horizon)
                for horizon in horizons
            }
        )
    )

    if not horizons:
        raise ValueError(
            "At least one prediction horizon must be provided."
        )

    if horizons[0] < 1:
        raise ValueError(
            "Prediction horizons must be strictly positive."
        )

    if latent.ndim != 3:
        raise ValueError(
            "latent must have shape (B, T, latent_dim)."
        )

    if observation.ndim != 3:
        raise ValueError(
            "observation must have shape (B, T, observation_dim)."
        )

    if latent.shape[:2] != observation.shape[:2]:
        raise ValueError(
            "latent and observation must share batch and time axes."
        )

    maximum_horizon = horizons[-1]
    sequence_length = latent.shape[1]

    if maximum_horizon >= sequence_length:
        raise ValueError(
            f"Maximum horizon {maximum_horizon} must be smaller than "
            f"sequence length {sequence_length}."
        )

    # All starting points have all requested futures available.
    current_latent = latent[
        :,
        : sequence_length - maximum_horizon,
        :,
    ]

    predicted_by_horizon: list[Tensor] = []
    target_by_horizon: list[Tensor] = []

    requested_horizons = set(
        horizons
    )

    for step in range(
        1,
        maximum_horizon + 1,
    ):
        current_latent = model.predict_next_latent(
            current_latent
        )

        if step not in requested_horizons:
            continue

        predicted_observation = model.decode(
            current_latent
        )

        target_observation = observation[
            :,
            step : sequence_length - maximum_horizon + step,
            :,
        ]

        predicted_by_horizon.append(
            predicted_observation
        )

        target_by_horizon.append(
            target_observation
        )

    # Horizon is the penultimate axis expected by
    # predictive_sufficiency_loss().
    predicted_future = torch.stack(
        predicted_by_horizon,
        dim=-2,
    )

    target_future = torch.stack(
        target_by_horizon,
        dim=-2,
    )

    return (
        predicted_future,
        target_future,
    )


def compute_loss_components(
    model: CLSMModel,
    batch: Mapping[str, Tensor],
    loss_config: LossConfig,
    *,
    sample_latent: bool,
    adversarial_coefficient: float | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Run the model and compute all active CLSM loss components.

    Distinction between predictive and temporal terms
    -------------------------------------------------
    - predictive sufficiency is approximated by decoding the predicted next
      latent state and comparing it with the next observation;
    - temporal coherence compares the predicted next latent with the encoded
      next latent.

    This prevents the two terms from being exact duplicates.
    """
    observation = batch["observation"]
    output = model(
        observation,
        sample=sample_latent,
        adversarial_coefficient=adversarial_coefficient,
    )

    latent = output["latent"]
    weights = loss_config.weights
    components: dict[str, Tensor] = {}
    diagnostics: dict[str, Tensor] = {}

    if weights.observation > 0:
        components["observation"] = observation_compatibility_loss(
            output["reconstructed_observation"],
            observation,
            loss_type=loss_config.reconstruction_loss_type,
        )

    if weights.predictive > 0:
        (
            predicted_future_observation,
            target_future_observation,
        ) = build_multi_horizon_observation_predictions(
            model,
            latent,
            observation,
            horizons=loss_config.predictive_horizons,
        )

        horizon_weights = torch.tensor(
            [
                loss_config.predictive_horizon_decay
                ** horizon_index
                for horizon_index in range(
                    len(
                        loss_config.predictive_horizons
                    )
                )
            ],
            device=latent.device,
            dtype=latent.dtype,
        )

        components["predictive"] = predictive_sufficiency_loss(
            predicted_future_observation,
            target_future_observation,
            loss_type=loss_config.prediction_loss_type,
            horizon_weights=horizon_weights,
        )

    if weights.temporal > 0:
        predicted_next_latent = None

        if loss_config.temporal_mode == "dynamics":
            predicted_next_latent = model.predict_next_latent(
                latent[..., :-1, :]
            )

        components["temporal"] = temporal_coherence_loss(
            latent,
            predicted_next_latent=predicted_next_latent,
            mode=loss_config.temporal_mode,
        )

    if weights.invariance > 0:
        if "nuisance_logits" not in output:
            raise ValueError(
                "The invariance loss requires a configured categorical "
                "nuisance adversary."
            )
        if "nuisance_id" not in batch:
            raise KeyError(
                "The invariance loss requires 'nuisance_id' in the batch."
            )

        nuisance_logits = output["nuisance_logits"]
        nuisance_id = batch["nuisance_id"]

        components["invariance"] = nuisance_adversarial_loss(
            nuisance_logits,
            nuisance_id,
        )
        diagnostics["nuisance_adversarial_accuracy"] = (
            nuisance_adversarial_accuracy(
                nuisance_logits,
                nuisance_id,
            )
        )

    if weights.minimality > 0:
        if loss_config.minimality_mode == "kl":
            if "mean" not in output or "log_variance" not in output:
                raise ValueError(
                    "KL minimality requires a variational encoder."
                )

            components["minimality"] = minimality_loss(
                latent,
                mode="kl",
                mean=output["mean"],
                log_variance=output["log_variance"],
            )

        else:
            components["minimality"] = minimality_loss(
                latent,
                mode=loss_config.minimality_mode,
            )

    if weights.structural > 0:
        components["structural"] = structural_constraint_loss(
            latent,
            variance_target=(
                loss_config.structural_variance_target
            ),
            variance_weight=(
                loss_config.structural_variance_weight
            ),
            covariance_weight=(
                loss_config.structural_covariance_weight
            ),
        )

    flat_latent = latent.reshape(-1, latent.shape[-1])
    latent_mean_per_dim = flat_latent.mean(dim=0)
    latent_std_per_dim = flat_latent.std(dim=0, unbiased=False)
    for dimension_index in range(latent.shape[-1]):
        diagnostics[f"latent_mean_dim_{dimension_index}"] = latent_mean_per_dim[dimension_index]
        diagnostics[f"latent_std_dim_{dimension_index}"] = latent_std_per_dim[dimension_index]

    return components, diagnostics


# =============================================================================
# Metric accumulation
# =============================================================================

@dataclass
class MetricAccumulator:
    """Accumulate batch-weighted scalar metrics."""

    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(
        self,
        metrics: Mapping[str, Tensor],
        batch_size: int,
    ) -> None:
        self.count += int(batch_size)
        for name, value in metrics.items():
            scalar = float(value.detach().cpu())
            self.sums[name] = self.sums.get(name, 0.0) + scalar * batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            name: value / self.count
            for name, value in self.sums.items()
        }


# =============================================================================
# Epoch execution
# =============================================================================

def run_epoch(
    model: CLSMModel,
    loader: DataLoader,
    objective: CLSMLoss,
    loss_config: LossConfig,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    adversarial_coefficient: float | None = None,
) -> dict[str, float]:
    """
    Run one training or evaluation epoch.

    If ``optimizer`` is ``None``, the model is evaluated without gradient
    updates.
    """
    training = optimizer is not None
    model.train(training)

    accumulator = MetricAccumulator()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["observation"].shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            components, diagnostics = compute_loss_components(
                model,
                batch,
                loss_config,
                sample_latent=(
                    training
                    and loss_config.minimality_mode == "kl"
                ),
                adversarial_coefficient=adversarial_coefficient,
            )

            if not components:
                if training:
                    raise RuntimeError(
                        "No active loss component was produced during training."
                    )

                reference = batch["observation"]
                total = reference.new_zeros(())
                weighted = {}
            else:
                total, weighted = objective(
                    components
                )

            if training:
                total.backward()

                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )

                optimizer.step()

        non_adversarial_weighted = {
            name: value
            for name, value in weighted.items()
            if name != "invariance"
        }

        if non_adversarial_weighted:
            selection_total = torch.stack(
                list(
                    non_adversarial_weighted.values()
                )
            ).sum()
        else:
            selection_total = total

        metrics: dict[str, Tensor] = {
            "total": total,
            "selection_total": selection_total,
        }
        metrics.update(
            {
                f"raw_{name}": value
                for name, value in components.items()
            }
        )
        metrics.update(
            {
                f"weighted_{name}": value
                for name, value in weighted.items()
            }
        )
        metrics.update(
            {
                name: value
                for name, value in diagnostics.items()
                if value.ndim == 0
            }
        )

        accumulator.update(metrics, batch_size)

    return accumulator.compute()


# =============================================================================
# Training orchestration
# =============================================================================

def train_model(
    config: TrainConfig,
    *,
    show_run_header: bool = True,
) -> tuple[CLSMModel, dict[str, float]]:
    """
    Train one CLSM model and save artifacts.

    Returns
    -------
    model:
        Model restored to the best validation checkpoint.
    final_metrics:
        Test and optional OOD metrics.
    """
    set_global_seed(config.optimization.seed)
    device = resolve_device(config.optimization.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.data.data_dir is None:
        raise ValueError(
            "A dataset directory must be provided."
        )
    effective_data_dir = Path(
        config.data.data_dir
    )

    splits = load_splits(
        effective_data_dir
    )

    print(
        f"Using datasets from: {effective_data_dir}"
    )

    save_data_manifest(
        run_dir=run_dir,
        data_dir=effective_data_dir,
    )

    n_nuisances: int | None = None

    if config.loss.weights.invariance > 0:
        unique_nuisance_ids = np.unique(
            splits.train.nuisance_id
        )

        if unique_nuisance_ids.size < 2:
            raise ValueError(
                "Adversarial invariance requires at least two nuisance "
                "classes in the training split."
            )

        expected_ids = np.arange(
            unique_nuisance_ids.size,
            dtype=np.int64,
        )
        if not np.array_equal(
            unique_nuisance_ids,
            expected_ids,
        ):
            raise ValueError(
                "Training nuisance identifiers must be contiguous and "
                "zero-based for categorical adversarial training. "
                f"Found {unique_nuisance_ids.tolist()}."
            )

        n_nuisances = int(
            unique_nuisance_ids.size
        )

    effective_model_config = CLSMModelConfig(
        **{
            **asdict(config.model),
            "observation_dim": (
                splits.train.observation_dim
            ),
            "n_nuisances": n_nuisances,
            "variational": (
                config.loss.minimality_mode == "kl"
            ),
        }
    )

    config = replace(
        config,
        model=effective_model_config,
    )

    save_config(
        config,
        run_dir / "config.json",
    )

    model = CLSMModel(config.model).to(device)
    objective = CLSMLoss(config.loss.weights)

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    train_loader = make_loader(
        splits.train,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        num_workers=config.optimization.num_workers,
        pin_memory=(
            config.optimization.pin_memory and device.type == "cuda"
        ),
    )
    validation_loader = make_loader(
        splits.validation,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.optimization.num_workers,
        pin_memory=(
            config.optimization.pin_memory and device.type == "cuda"
        ),
    )
    test_loader = make_loader(
        splits.test,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.optimization.num_workers,
        pin_memory=(
            config.optimization.pin_memory and device.type == "cuda"
        ),
    )
    ood_loader = (
        None
        if splits.ood is None
        else make_loader(
            splits.ood,
            batch_size=config.optimization.batch_size,
            shuffle=False,
            num_workers=config.optimization.num_workers,
            pin_memory=(
                config.optimization.pin_memory and device.type == "cuda"
            ),
        )
    )

    history_path = run_dir / "history.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    best_validation = math.inf
    epochs_without_improvement = 0
    start_time = time.time()

    if show_run_header:
        print()
        print_separator()
        print(f"Run      : {config.run_name}")
        print(f"Data     : {effective_data_dir}")
        print(f"Device   : {device}")
        if device.type == "cuda":
            print(f"GPU      : {torch.cuda.get_device_name(device)}")
        print(
            "Episodes : "
            f"train={splits.train.n_episodes}, "
            f"validation={splits.validation.n_episodes}, "
            f"test={splits.test.n_episodes}, "
            f"ood={0 if splits.ood is None else splits.ood.n_episodes}"
        )
        print(f"Weights  : {config.loss.weights.as_dict()}")
        print_separator()

    history_rows = []

    progress = tqdm(
        range(1, config.optimization.epochs + 1),
        desc=config.run_name,
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in progress:
        if (
            config.loss.weights.invariance > 0
            and config.optimization.adversarial_warmup_epochs > 0
        ):
            warmup_fraction = min(
                1.0,
                epoch
                / config.optimization.adversarial_warmup_epochs,
            )
            adversarial_coefficient = (
                config.model.gradient_reversal_coefficient
                * warmup_fraction
            )
        else:
            adversarial_coefficient = (
                config.model.gradient_reversal_coefficient
            )

        train_metrics = run_epoch(
            model,
            train_loader,
            objective,
            config.loss,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.optimization.gradient_clip_norm,
            adversarial_coefficient=adversarial_coefficient,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            objective,
            config.loss,
            device=device,
            adversarial_coefficient=adversarial_coefficient,
        )

        row: dict[str, float] = {"epoch": float(epoch)}
        row.update(
            {
                f"train_{name}": value
                for name, value in train_metrics.items()
            }
        )
        row.update(
            {
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            }
        )
        history_rows.append(row)
        write_history(history_path, history_rows)

        validation_total = validation_metrics["total"]
        validation_selection = validation_metrics[
            "selection_total"
        ]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_total": validation_total,
            "validation_selection_total": validation_selection,
            "config": config_to_dict(config),
            "model_config": asdict(config.model),
        }
        torch.save(checkpoint, last_path)

        improved = validation_selection < best_validation
        if improved:
            best_validation = validation_selection
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - start_time
        postfix = {
            "train": f"{train_metrics['selection_total']:.5f}",
            "val": f"{validation_selection:.5f}",
            "best": f"{best_validation:.5f}",
            "wait": epochs_without_improvement,
            "grl": f"{adversarial_coefficient:.3f}",
            "elapsed": f"{elapsed:.1f}s",
        }
        for name in (
            "raw_observation",
            "raw_predictive",
            "raw_invariance",
            "raw_structural",
            "nuisance_adversarial_accuracy",
        ):
            if name in train_metrics:
                display_name = name.replace(
                    "raw_",
                    "",
                ).replace(
                    "nuisance_adversarial_accuracy",
                    "adv_acc",
                )
                postfix[display_name] = f"{train_metrics[name]:.4f}"
        progress.set_postfix(postfix)

        if (
            config.optimization.early_stopping_patience > 0
            and epochs_without_improvement
            >= config.optimization.early_stopping_patience
        ):
            tqdm.write(
                f"{config.run_name}: early stopping after "
                f"{epochs_without_improvement} epochs without improvement."
            )
            break

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        test_loader,
        objective,
        config.loss,
        device=device,
    )

    final_metrics: dict[str, float] = {
        f"test_{name}": value
        for name, value in test_metrics.items()
    }

    if ood_loader is not None:
        ood_loss_config = config.loss
        ood_objective = objective

        if config.loss.weights.invariance > 0:
            ood_weights = replace(
                config.loss.weights,
                invariance=0.0,
            )
            ood_loss_config = replace(
                config.loss,
                weights=ood_weights,
            )
            ood_objective = CLSMLoss(
                ood_weights
            )

        ood_metrics = run_epoch(
            model,
            ood_loader,
            ood_objective,
            ood_loss_config,
            device=device,
        )

        final_metrics.update(
            {
                f"ood_{name}": value
                for name, value in ood_metrics.items()
            }
        )

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(final_metrics, stream, indent=2, sort_keys=True)

    print_compact_final_summary(config.run_name, final_metrics, config.loss.weights)

    return model, final_metrics


def print_compact_final_summary(
    run_name: str,
    metrics: Mapping[str, float],
    weights: ConstraintWeights,
) -> None:
    """Print a compact summary of final train-script metrics."""

    rows = [
        ("Test selection total", "test_selection_total", None),
        ("Test optimization total", "test_total", None),

        ("Test observation", "test_raw_observation", weights.observation),
        ("Test prediction", "test_raw_predictive", weights.predictive),
        ("Test temporal", "test_raw_temporal", weights.temporal),
        ("Test minimality", "test_raw_minimality", weights.minimality),
        ("Test adversarial CE", "test_raw_invariance", weights.invariance),
        ("Test adversarial acc.", "test_nuisance_adversarial_accuracy", None),
        ("Test structural", "test_raw_structural", weights.structural),

        ("OOD selection total", "ood_selection_total", None),
        ("OOD observation", "ood_raw_observation", weights.observation),
        ("OOD prediction", "ood_raw_predictive", weights.predictive),
        ("OOD temporal", "ood_raw_temporal", weights.temporal),
        ("OOD minimality", "ood_raw_minimality", weights.minimality),
        ("OOD structural", "ood_raw_structural", weights.structural),
    ]

    print()
    print(f"Summary: {run_name}")
    print_separator("-")

    for label, key, weight in rows:
        if key not in metrics:
            continue

        value = metrics[key]

        if weight is None:
            print(f"{label:<24} {value:>10.6f}")
        else:
            print(
                f"{label:<24} "
                f"{value:>10.6f} "
                f"({value * weight:>10.6f})"
            )

    print_separator("-")


# =============================================================================
# Serialization utilities
# =============================================================================

def config_to_dict(config: TrainConfig) -> dict[str, object]:
    """Convert nested dataclasses to a JSON-safe dictionary."""

    def make_json_safe(value: object) -> object:
        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                make_json_safe(item)
                for item in value
            ]

        return value

    return make_json_safe(asdict(config))


def save_config(config: TrainConfig, path: Path) -> None:
    """Save the complete training configuration."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            config_to_dict(config),
            stream,
            indent=2,
            sort_keys=True,
        )


def write_history(
    path: Path,
    rows: Iterable[Mapping[str, float]],
) -> None:
    """Write the complete training history to CSV."""
    rows = list(rows)
    if not rows:
        return

    fieldnames = sorted(
        {
            field
            for row in rows
            for field in row.keys()
        },
        key=lambda name: (name != "epoch", name),
    )

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Command-line configuration
# =============================================================================

def add_training_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments shared by CLSM training entry points."""

    # Experiment
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_WEIGHTS),
        default="full",
        help="Predefined combination of CLSM constraints.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional run name. When several seeds are used, the seed is "
            "appended automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory in which checkpoints and metrics are saved.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing train.npz, validation.npz, test.npz, "
            "and optionally ood.npz."
        ),
    )

    # Optimization
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    # Runtime and reproducibility
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Execution device: 'auto', 'cpu', 'cuda', or a concrete CUDA "
            "device such as 'cuda:0'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single model seed, retained for backward compatibility.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more model seeds, for example: --seeds 0 1 2 3 4.",
    )

    # Adversarial nuisance removal
    parser.add_argument(
        "--gradient-reversal-coefficient",
        type=float,
        default=1.0,
        help=(
            "Maximum gradient-reversal strength used by the categorical "
            "nuisance adversary."
        ),
    )
    parser.add_argument(
        "--adversarial-warmup-epochs",
        type=int,
        default=20,
        help=(
            "Number of epochs used to linearly increase the gradient-"
            "reversal coefficient from zero to its configured value."
        ),
    )

    # Constraint weights
    parser.add_argument(
        "--weight-predictive",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-minimality",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-temporal",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-observation",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-invariance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-structural",
        type=float,
        default=None,
    )

    # Constraint variants
    parser.add_argument(
        "--minimality-mode",
        choices=(
            "kl",
            "l1",
            "participation_ratio",
        ),
        default="l1",
    )
    parser.add_argument(
        "--temporal-mode",
        choices=(
            "velocity",
            "dynamics",
            "acceleration",
        ),
        default="dynamics",
    )


def config_from_args(
    args: argparse.Namespace,
    *,
    seed: int,
    run_name: str,
    model_config: CLSMModelConfig,
) -> TrainConfig:
    """Build the generic training configuration from CLI arguments."""
    weights = PRESET_WEIGHTS[args.preset]
    weights = override_constraint_weights(
        weights,
        args,
    )

    data_config = DataConfig(
        data_dir=args.data_dir,
    )

    optimization_config = OptimizationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=seed,
        device=args.device,
        adversarial_warmup_epochs=args.adversarial_warmup_epochs,
    )

    loss_config = LossConfig(
        weights=weights,
        minimality_mode=args.minimality_mode,
        temporal_mode=args.temporal_mode,
    )

    if args.adversarial_warmup_epochs < 0:
        raise ValueError(
            "adversarial_warmup_epochs must be non-negative."
        )

    return TrainConfig(
        run_name=run_name,
        output_dir=args.output_dir,
        data=data_config,
        optimization=optimization_config,
        loss=loss_config,
        model=model_config,
    )


def override_constraint_weights(
    weights: ConstraintWeights,
    args: argparse.Namespace,
) -> ConstraintWeights:
    """Override preset loss weights using optional CLI arguments."""
    overrides = {}

    argument_mapping = {
        "predictive": args.weight_predictive,
        "minimality": args.weight_minimality,
        "temporal": args.weight_temporal,
        "observation": args.weight_observation,
        "invariance": args.weight_invariance,
        "structural": args.weight_structural,
    }

    for name, value in argument_mapping.items():
        if value is None:
            continue

        if value < 0.0:
            raise ValueError(
                f"Loss weight '{name}' must be non-negative, "
                f"got {value}."
            )

        overrides[name] = float(value)

    if not overrides:
        return weights

    return replace(
        weights,
        **overrides,
    )


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Resolve --seed and --seeds without breaking old commands."""
    if args.seeds is not None:
        if args.seed is not None:
            raise ValueError("Use either --seed or --seeds, not both.")
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def resolve_run_name(
    template: str | None,
    *,
    preset: str,
    seed: int,
    multiple_runs: bool,
) -> str:
    """Resolve one run name from an optional seed template."""
    if template is None:
        return f"{preset}-seed-{seed}"
    if "{}" in template:
        return template.format(seed)
    if "{seed}" in template:
        return template.format(seed=seed)
    if multiple_runs:
        return f"{template}-seed-{seed}"
    return template


# =============================================================================
# Experiment orchestration
# =============================================================================

def run_experiments(
    *,
    args: argparse.Namespace,
    model_config_factory: Callable[
        [argparse.Namespace],
        CLSMModelConfig,
    ],
) -> dict[str, dict[str, float]]:
    """
    Run one or more CLSM training experiments.

    Parameters
    ----------
    args:
        Parsed command-line arguments.
    model_config_factory:
        Callback constructing the environment-specific model configuration.
        The observation dimension may remain unresolved because
        ``train_model`` replaces it after loading the datasets.
    """
    seeds = resolve_seeds(args)
    device = resolve_device(args.device)

    model_config = model_config_factory(args)

    print()
    print_separator()
    print(f"Preset     : {args.preset}")
    print(f"Data       : {args.data_dir}")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(device)}")
    print(f"Seeds      : {' '.join(str(seed) for seed in seeds)}")
    print(f"Minimality : {args.minimality_mode}")
    print(f"Temporal   : {args.temporal_mode}")
    print_separator()

    completed: dict[str, dict[str, float]] = {}

    for run_index, seed in enumerate(seeds, start=1):
        run_name = resolve_run_name(
            args.run_name,
            preset=args.preset,
            seed=seed,
            multiple_runs=len(seeds) > 1,
        )

        tqdm.write(
            f"[{run_index}/{len(seeds)}] Starting {run_name}"
        )

        config = config_from_args(
            args,
            seed=seed,
            run_name=run_name,
            model_config=model_config,
        )

        _, metrics = train_model(
            config,
            show_run_header=False,
        )

        completed[run_name] = metrics

    if len(completed) > 1:
        print_banner("Completed runs")

        for run_name, metrics in completed.items():
            test_total = metrics.get(
                "test_total",
                float("nan"),
            )
            ood_total = metrics.get(
                "ood_total",
                float("nan"),
            )

            print(
                f"{run_name:<32} "
                f"test={test_total:.6f}  "
                f"ood={ood_total:.6f}"
            )

    return completed
