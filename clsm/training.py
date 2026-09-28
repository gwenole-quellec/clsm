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
Integrated two-phase adversarial training uses a separately optimized strong
nuisance adversary followed by the refresh phase. Counterfactual views are
reserved for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from .datasets import CLSMDataset, DatasetSplits
from .losses import (
    CLSMLoss,
    ConstraintWeights,
    minimality_loss,
    observation_compatibility_loss,
    predictive_sufficiency_loss,
    structural_constraint_loss,
    temporal_coherence_loss,
)
from .models import CLSMModel, CLSMModelConfig
from .strong_adversary import (
    make_adversary,
    make_adversary_generator,
    standardize_batch,
    update_adversary,
)
from .utils import print_banner, print_separator


# =============================================================================
# Constants
# =============================================================================

DEFAULT_EPOCHS = 50
DEFAULT_ADVERSARY_STEPS = 15
DEFAULT_BATCH_SIZE = 128
DEFAULT_GRADIENT_CLIP_NORM = 5.0
DEFAULT_MODEL_SEED = 42
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = True
DEFAULT_EARLY_STOPPING_PATIENCE = 0
DEFAULT_DEVICE = "auto"
DEFAULT_REFRESH_PRETRAIN_EPOCHS = 40
DEFAULT_ADVERSARIAL_WARMUP_EPOCHS = 20
DEFAULT_MODEL_LEARNING_RATE = 1e-3
DEFAULT_MODEL_WEIGHT_DECAY = 1e-5
DEFAULT_CONFIGURATIONS_PATH = Path(
    "toy/configurations.json"
)

ADVERSARY_LEARNING_RATE = 1e-3
ADVERSARY_WEIGHT_DECAY = 1e-4

TRAIN_LOADER_SEED_OFFSET = 100_000

MIN_NUISANCE_CLASSES = 2

INVARIANCE_METHOD = "strong_adversary_refresh"
PROTOCOL_VERSION = "strong_adversary_refresh"

UNRESOLVED_OBSERVATION_DIM = 1

_ADVERSARY = None
_ADVERSARY_OPT = None
_ADVERSARY_GENERATOR = None
_ADVERSARY_STEPS = DEFAULT_ADVERSARY_STEPS


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

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_MODEL_LEARNING_RATE
    weight_decay: float = DEFAULT_MODEL_WEIGHT_DECAY
    gradient_clip_norm: float | None = DEFAULT_GRADIENT_CLIP_NORM
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    seed: int = DEFAULT_MODEL_SEED
    device: str = DEFAULT_DEVICE
    refresh_pretrain_epochs: int = DEFAULT_REFRESH_PRETRAIN_EPOCHS
    adversary_steps: int = DEFAULT_ADVERSARY_STEPS
    adversarial_warmup_epochs: int = DEFAULT_ADVERSARIAL_WARMUP_EPOCHS


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

    invariance_method: str = INVARIANCE_METHOD
    prediction_loss_type: str = "mse"
    reconstruction_loss_type: str = "mse"
    minimality_mode: str = "l1"
    temporal_mode: str = "dynamics"

    predictive_horizons: tuple[int, ...] = (1, 5, 10)

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
    seed: int | None = None,
) -> DataLoader:
    """Create a PyTorch data loader."""
    if dataset.n_episodes < 1:
        raise ValueError("Cannot create a loader for an empty dataset.")

    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("A seed must be provided when shuffle=True.")
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        TorchEpisodeDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
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
        "sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in files.items()},
        "protocol_version": PROTOCOL_VERSION,
    }

    manifest_path = run_dir / "data_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as stream:
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

    requested_horizons = set(horizons)

    for step in range(
        1,
        maximum_horizon + 1,
    ):
        current_latent = model.predict_next_latent(current_latent)

        if step not in requested_horizons:
            continue

        predicted_observation = model.decode(current_latent)

        target_observation = observation[
            :,
            step : sequence_length - maximum_horizon + step,
            :,
        ]

        predicted_by_horizon.append(predicted_observation)
        target_by_horizon.append(target_observation)

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

    return (predicted_future, target_future)


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

    Predictive sufficiency is approximated by decoding predicted future
    latent states and comparing them with future observations, whereas
    temporal coherence compares predicted latent dynamics with the encoded
    latent trajectory.

    When invariance is active, the nuisance adversary is optimized
    separately. Its cross-entropy is negated so that the encoder is
    optimized to make nuisance prediction difficult. During the initial
    training phase, this contribution can be linearly ramped up through
    ``adversarial_coefficient``.
    """

    observation = batch["observation"]

    output = model(
        observation,
        sample=sample_latent,
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
                loss_config.predictive_horizon_decay ** horizon_index
                for horizon_index in range(len(loss_config.predictive_horizons))
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
        flat_latent = latent.reshape(
            -1,
            latent.shape[-1],
        )

        standardized_latent = standardize_batch(
            flat_latent
        )

        labels = batch["nuisance_id"].repeat_interleave(
            latent.shape[1]
        )

        logits = _ADVERSARY(
            standardized_latent
        )

        adversarial_loss = nn.functional.cross_entropy(
            logits,
            labels,
        )

        coefficient = (
            1.0
            if adversarial_coefficient is None
            else adversarial_coefficient
        )

        components["invariance"] = (
            -coefficient
            * adversarial_loss
        )

        diagnostics[
            "nuisance_adversarial_accuracy"
        ] = (
            logits.argmax(dim=-1) == labels
        ).float().mean()

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

    flat_latent = latent.reshape(
        -1,
        latent.shape[-1],
    )

    latent_mean_per_dim = flat_latent.mean(dim=0)
    latent_std_per_dim = flat_latent.std(
        dim=0,
        unbiased=False,
    )

    for dimension_index in range(latent.shape[-1]):
        diagnostics[
            f"latent_mean_dim_{dimension_index}"
        ] = latent_mean_per_dim[dimension_index]
        diagnostics[
            f"latent_std_dim_{dimension_index}"
        ] = latent_std_per_dim[dimension_index]

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
    updates. During training with invariance enabled, the nuisance adversary
    is first updated on detached latent features before the encoder update.
    """

    training = optimizer is not None

    model.train(training)

    accumulator = MetricAccumulator()

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        batch_size = batch["observation"].shape[0]

        if training:
            if loss_config.weights.invariance > 0:
                with torch.no_grad():
                    frozen = model.encode(
                        batch["observation"],
                        sample=False,
                    )["latent"]

                    features = standardize_batch(
                        frozen.reshape(
                            -1,
                            frozen.shape[-1],
                        )
                    )

                    labels = batch["nuisance_id"].repeat_interleave(frozen.shape[1])

                update_adversary(
                    _ADVERSARY,
                    _ADVERSARY_OPT,
                    features,
                    labels,
                    generator=_ADVERSARY_GENERATOR,
                    steps=_ADVERSARY_STEPS,
                )

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
                total, weighted = objective(components)

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
                list(non_adversarial_weighted.values())
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

        accumulator.update(
            metrics,
            batch_size,
        )

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
        Model restored to the prespecified final checkpoint.
    final_metrics:
        Final validation monitoring metrics (test is evaluated separately).
    """
    if config.optimization.adversary_steps < 1:
        raise ValueError("adversary_steps must be positive.")
    if config.optimization.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.optimization.early_stopping_patience != 0:
        raise ValueError("Fixed-duration protocol requires --early-stopping-patience 0")
    if (
        config.loss.weights.invariance > 0
        and not 0 < config.optimization.refresh_pretrain_epochs < config.optimization.epochs
    ):
        raise ValueError("Require 0 < --refresh-pretrain-epochs < --epochs (total duration)")
    if (
        config.loss.weights.invariance > 0
        and (config.loss.minimality_mode == "kl" or config.model.dropout != 0)
    ):
        raise ValueError("Validated refresh protocol requires deterministic encoding and dropout=0")

    global standardize_batch
    from .strong_adversary import standardize_batch as batch_standardizer
    standardize_batch = batch_standardizer

    set_global_seed(config.optimization.seed)
    device = resolve_device(config.optimization.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    if config.data.data_dir is None:
        raise ValueError(
            "A dataset directory must be provided."
        )
    effective_data_dir = Path(config.data.data_dir)

    splits = load_splits(effective_data_dir)

    print(
        f"Using datasets from: {effective_data_dir}"
    )

    save_data_manifest(
        run_dir=run_dir,
        data_dir=effective_data_dir,
    )

    n_nuisances = MIN_NUISANCE_CLASSES

    if config.loss.weights.invariance > 0:
        unique_nuisance_ids = np.unique(splits.train.nuisance_id)

        if unique_nuisance_ids.size < MIN_NUISANCE_CLASSES:
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

        n_nuisances = int(unique_nuisance_ids.size)

    effective_model_config = CLSMModelConfig(
        **{
            **asdict(config.model),
            "observation_dim": splits.train.observation_dim,
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
    global _ADVERSARY
    global _ADVERSARY_OPT
    global _ADVERSARY_GENERATOR
    global _ADVERSARY_STEPS

    _ADVERSARY = make_adversary(
        latent_dim=config.model.latent_dim,
        n_classes=n_nuisances or MIN_NUISANCE_CLASSES,
        seed=config.optimization.seed,
    ).to(device)

    _ADVERSARY.requires_grad_(False)
    _ADVERSARY_OPT = torch.optim.Adam(
        _ADVERSARY.parameters(),
        lr=ADVERSARY_LEARNING_RATE,
        weight_decay=ADVERSARY_WEIGHT_DECAY,
    )
    _ADVERSARY_GENERATOR = make_adversary_generator(config.optimization.seed)
    _ADVERSARY_STEPS = (config.optimization.adversary_steps)

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
        seed=TRAIN_LOADER_SEED_OFFSET + config.optimization.seed,
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
    refresh_phase = None

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
            adversarial_coefficient = min(
                1.0,
                epoch / config.optimization.adversarial_warmup_epochs,
            )
        else:
            adversarial_coefficient = 1.0

        if config.loss.weights.invariance > 0 and epoch > config.optimization.refresh_pretrain_epochs:
            if refresh_phase is None:
                from .refresh import RefreshPhase
                refresh_phase = RefreshPhase(model, splits.train, _ADVERSARY, _ADVERSARY_OPT, config)
                standardize_batch = refresh_phase.standardize
            train_metrics = refresh_phase.epoch(model, optimizer, objective, config.loss)
        else:
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
        validation_selection = validation_metrics["selection_total"]

        checkpoint = {
            "epoch": epoch,
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint_selection": "fixed_final_epoch",
            "adversary_mean": (
                None
                if refresh_phase is None
                else refresh_phase.mean
            ),
            "adversary_std": (
                None
                if refresh_phase is None
                else refresh_phase.std
            ),
            "model_state_dict": model.state_dict(),
            "strong_adversary_state_dict": _ADVERSARY.state_dict(),
            "strong_adversary_optimizer_state_dict": (
                _ADVERSARY_OPT.state_dict()
            ),
            "adversary_rng_state": (
                _ADVERSARY_GENERATOR.get_state()
                if refresh_phase is None
                else refresh_phase.generator.get_state()
            ),
            "adversary_updates_per_encoder_step": (
                config.optimization.adversary_steps
                if config.loss.weights.invariance > 0
                else 0
            ),
            "optimized_backbone_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            "optimized_adversary_parameters": (
                sum(
                    parameter.numel()
                    for parameter in _ADVERSARY.parameters()
                )
                if config.loss.weights.invariance > 0
                else 0
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_total": validation_total,
            "validation_selection_total": validation_selection,
            "config": config_to_dict(config),
            "model_config": asdict(config.model),
        }
        torch.save(checkpoint, last_path)
        if epoch == config.optimization.refresh_pretrain_epochs:
            torch.save(checkpoint, run_dir / "pretrain.pt")
        # best.pt is the pipeline-compatible alias of the prespecified final epoch.
        if epoch == config.optimization.epochs:
            torch.save(checkpoint, best_path)

        improved = validation_selection < best_validation
        if improved:
            best_validation = validation_selection
            epochs_without_improvement = 0
            # Selection is fixed-duration; validation is monitoring only.
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - start_time
        postfix = {
            "train": f"{train_metrics['selection_total']:.5f}",
            "val": f"{validation_selection:.5f}",
            "best": f"{best_validation:.5f}",
            "wait": epochs_without_improvement,
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
    _ADVERSARY.load_state_dict(best_checkpoint["strong_adversary_state_dict"])

    final_metrics = {"validation_selection_total": best_checkpoint["validation_selection_total"],
                     "epoch": float(best_checkpoint["epoch"])}
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    return model, final_metrics


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
        "--configurations-file",
        type=Path,
        default=DEFAULT_CONFIGURATIONS_PATH,
        help="JSON file containing named CLSM constraint configurations.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Named configuration loaded from --configurations-file. "
            "If omitted, all six --weight-* arguments must be provided."
        ),
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
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_MODEL_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_MODEL_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "--adversarial-warmup-epochs",
        type=int,
        default=DEFAULT_ADVERSARIAL_WARMUP_EPOCHS,
            help=(
                "Number of initial epochs over which the adversarial invariance "
                "contribution is linearly increased from zero to full strength."
            ),
    )

    # Runtime and reproducibility
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
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
    parser.add_argument(
        "--refresh-pretrain-epochs",
        type=int,
        default=DEFAULT_REFRESH_PRETRAIN_EPOCHS
    )
    parser.add_argument(
        "--adversary-steps",
        type=int,
        default=DEFAULT_ADVERSARY_STEPS,
        help="Adversary updates per encoder step.",
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
    weights: ConstraintWeights,
) -> TrainConfig:
    """Build the generic training configuration from CLI arguments."""

    data_config = DataConfig(data_dir=args.data_dir)

    optimization_config = OptimizationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        seed=seed,
        device=args.device,
        refresh_pretrain_epochs=args.refresh_pretrain_epochs,
        adversary_steps=args.adversary_steps,
        adversarial_warmup_epochs=args.adversarial_warmup_epochs,
    )

    loss_config = LossConfig(
        weights=weights,
        minimality_mode=args.minimality_mode,
        temporal_mode=args.temporal_mode,
    )

    return TrainConfig(
        run_name=run_name,
        output_dir=args.output_dir,
        data=data_config,
        optimization=optimization_config,
        loss=loss_config,
        model=model_config,
    )


def load_configurations(
    path: str | Path,
) -> dict[str, ConstraintWeights]:
    """Load named CLSM constraint configurations from JSON."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    raw_configurations = payload.get("configurations")

    if not isinstance(raw_configurations, dict):
        raise ValueError(
            "Configuration file must contain a 'configurations' object."
        )

    configurations: dict[str, ConstraintWeights] = {}

    for name, specification in raw_configurations.items():
        if not isinstance(specification, dict):
            raise ValueError(
                f"Configuration {name!r} must be an object."
            )

        raw_weights = specification.get("weights")

        if not isinstance(raw_weights, dict):
            raise ValueError(
                f"Configuration {name!r} must contain a 'weights' object."
            )

        try:
            configurations[name] = ConstraintWeights(**raw_weights)
        except TypeError as exc:
            raise ValueError(
                f"Invalid weights for configuration {name!r}: {exc}"
            ) from exc

    return configurations


def override_constraint_weights(
    weights: ConstraintWeights,
    args: argparse.Namespace,
) -> ConstraintWeights:
    """Override constraint weights using optional CLI arguments."""
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

    return replace(weights, **overrides)


def resolve_constraint_weights(
    args: argparse.Namespace,
) -> tuple[str, ConstraintWeights]:
    """Resolve constraint weights from a preset or six explicit CLI weights."""

    explicit_weights = {
        "predictive": args.weight_predictive,
        "minimality": args.weight_minimality,
        "temporal": args.weight_temporal,
        "observation": args.weight_observation,
        "invariance": args.weight_invariance,
        "structural": args.weight_structural,
    }

    if args.preset is None:
        missing = [
            name
            for name, value in explicit_weights.items()
            if value is None
        ]

        if missing:
            raise SystemExit(
                "Provide either --preset or all six --weight-* arguments. "
                "Missing explicit weights: "
                + ", ".join(missing)
            )

        weights = override_constraint_weights(
            ConstraintWeights(),
            args,
        )

        return ("explicit", weights)

    configurations = load_configurations(args.configurations_file)

    if args.preset not in configurations:
        available = ", ".join(
            sorted(configurations)
        )

        raise SystemExit(
            f"Unknown preset {args.preset!r}. "
            f"Available presets: {available}"
        )

    weights = override_constraint_weights(
        configurations[args.preset],
        args,
    )

    return (args.preset, weights)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Resolve --seed and --seeds without breaking old commands."""
    if args.seeds is not None:
        if args.seed is not None:
            raise ValueError("Use either --seed or --seeds, not both.")
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [DEFAULT_MODEL_SEED]

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def resolve_run_name(
    template: str | None,
    *,
    configuration: str,
    seed: int,
    multiple_runs: bool,
) -> str:
    """Resolve one run name from an optional seed template."""

    if template is None:
        return f"{configuration}-seed-{seed}"

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
    configuration_name, weights = resolve_constraint_weights(args)

    seeds = resolve_seeds(args)
    device = resolve_device(args.device)

    model_config = model_config_factory(args)

    existing_run_dirs = []
    for seed in seeds:
        run_name = resolve_run_name(
            args.run_name,
            configuration=configuration_name,
            seed=seed,
            multiple_runs=len(seeds) > 1,
        )
        run_dir = Path(args.output_dir) / run_name
        if run_dir.exists():
            existing_run_dirs.append(run_dir)

    if existing_run_dirs:
        message = [
            "Refusing to overwrite existing run directories:",
            *(f"  - {path}" for path in existing_run_dirs),
            "",
            "Choose another --output-dir or remove/rename these directories.",
        ]
        raise SystemExit("\n".join(message))

    print()
    print_separator()
    print(f"Configuration : {configuration_name}")
    print(f"Data          : {args.data_dir}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(device)}")
    print(f"Seeds         : {' '.join(str(seed) for seed in seeds)}")
    print(f"Minimality    : {args.minimality_mode}")
    print(f"Temporal      : {args.temporal_mode}")
    print_separator()

    completed: dict[str, dict[str, float]] = {}

    for run_index, seed in enumerate(seeds, start=1):
        run_name = resolve_run_name(
            args.run_name,
            configuration=configuration_name,
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
            weights=weights,
        )

        _, metrics = train_model(
            config,
            show_run_header=False,
        )

        completed[run_name] = metrics

    if len(completed) > 1:
        print_banner("Completed runs")

        for run_name, metrics in completed.items():
            print(f"{run_name:<32} epoch={metrics['epoch']:.0f} "
                  f"validation={metrics['validation_selection_total']:.6f}")

    return completed
