"""
Run the reproducible sweep design and the four selected pairwise Pareto fronts.

Author: Gwenolé Quellec
Year: 2026

The script samples CLSM constraint weights, trains and evaluates each
configuration, aggregates results across model seeds, and generates the four
pairwise Pareto curves retained for the paper induced by the five paper metrics.
Representative Pareto-optimal configurations receive stable global labels
(P1, P2, ...) shared by all figures.

Requires ``adjustText`` for automatic label placement.

Examples
--------
Run a complete sweep:

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --sweep-config toy/constraint_sweep_config.json \
        --num-configurations 200 \
        --device cuda

Rebuild the aggregate analysis from existing per-run evaluations:

    python -m scripts.constraint_sweep \
        --analyze-only

Regenerate the four pairwise Pareto figures from saved aggregate results:

    python -m scripts.constraint_sweep \
        --figures-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from tqdm.auto import tqdm

from clsm.training import INVARIANCE_METHOD
from clsm.utils import module_command, print_banner, print_separator


# =============================================================================
# Sweep data structures
# =============================================================================

@dataclass(frozen=True)
class SweepWeights:
    predictive: float
    minimality: float
    temporal: float
    observation: float
    invariance: float
    structural: float


@dataclass(frozen=True)
class WeightRange:
    minimum: float
    maximum: float


# =============================================================================
# Constants
# =============================================================================

DEFAULT_NUM_CONFIGURATIONS = 200

DEFAULT_SWEEP_SEED = 12_345
DEFAULT_MODEL_SEEDS = (0, 1, 2, 3, 4)

DEFAULT_TRAINING_EPOCHS = 50
DEFAULT_REFRESH_PRETRAIN_EPOCHS = 40
DEFAULT_ADVERSARY_STEPS = 15

DEFAULT_NUISANCE_PROBE_EPOCHS = 500
DEFAULT_PROBE_SEED = 42
DEFAULT_NONLINEAR_PROBE_MAX_SAMPLES = 25_000

DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_EVALUATION_BATCH_SIZE = 256

DEFAULT_ROLLOUT_HORIZONS = (1, 5, 10)
REQUIRED_PARETO_ROLLOUT_HORIZON = 5

DEFAULT_PROBE_WORKERS = 1
DEFAULT_PROBE_CACHE_DIR = ".probe-cache"

DEFAULT_PARETO_ABSOLUTE_TOLERANCE = 0.0
DEFAULT_PARETO_RELATIVE_TOLERANCE = 0.0
DEFAULT_PARETO_LABEL_MINIMUM_DISTANCE = 0.01

DEFAULT_PLOT_OUTLIER_IQR_MULTIPLIER = 1.5

DEFAULT_SWEEP_CONFIG_PATH = Path(
    "toy/constraint_sweep_config.json"
)

DEFAULT_RUNS_DIR = Path(
    "runs_weight_sweep"
)

DEFAULT_OUTPUT_DIR = Path(
    "analysis_weight_sweep"
)

DEFAULT_DATA_DIR = Path(
    "data"
)

ANALYSIS_PROTOCOL_ARGUMENTS = (
    "pareto_absolute_tolerance",
    "pareto_relative_tolerance",
    "pareto_label_minimum_distance",
)

WEIGHT_NAMES = (
    "predictive",
    "minimality",
    "temporal",
    "observation",
    "invariance",
    "structural",
)

# True means that larger values are preferable.
OBJECTIVES = {
    "rollout_observation_mse_h5": False,
    "state_probe_r2": True,
    "neighborhood_trustworthiness": True,
    "counterfactual_relative_energy": False,
    "nuisance_latent_strong_class_balanced_accuracy": False,
}

OBJECTIVE_LABELS = {
    "rollout_observation_mse_h5": "Raw-observation prediction MSE (h=5) ↓",
    "state_probe_r2": r"Linear state $R^2$ ↑",
    "neighborhood_trustworthiness": "Neighborhood trustworthiness ↑",
    "counterfactual_relative_energy": "Counterfactual relative energy ↓",
    "nuisance_latent_strong_class_balanced_accuracy": r"Nuisance $Z$ balanced accuracy ↓",
}

OBJECTIVE_SHORT_NAMES = {
    "rollout_observation_mse_h5": "prediction",
    "state_probe_r2": "state_accessibility",
    "neighborhood_trustworthiness": "neighborhood_preservation",
    "counterfactual_relative_energy": "counterfactual_consistency",
    "nuisance_latent_strong_class_balanced_accuracy": "nuisance_suppression",
}

# Pairwise Pareto fronts retained for the paper.
# The order determines the output numbering and the panel order A--D.
OBJECTIVE_PAIRS = (
    (
        "rollout_observation_mse_h5",
        "counterfactual_relative_energy",
    ),
    (
        "rollout_observation_mse_h5",
        "nuisance_latent_strong_class_balanced_accuracy",
    ),
    (
        "state_probe_r2",
        "counterfactual_relative_energy",
    ),
    (
        "state_probe_r2",
        "neighborhood_trustworthiness",
    ),
)

WEIGHT_RANGES = {
    "predictive": WeightRange(0.01, 5.0),
    "minimality": WeightRange(1e-5, 5e-2),
    "temporal": WeightRange(0.01, 2.0),
    "observation": WeightRange(0.003, 3.0),
    "invariance": WeightRange(0.01, 5.0),
    "structural": WeightRange(1e-3, 1.0),
}

ANCHOR_CONFIGURATIONS = (
    SweepWeights(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    SweepWeights(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    SweepWeights(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    SweepWeights(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    *(SweepWeights(1.0, 0.0, 0.5, 0.5, value, 0.0)
      for value in (0.0, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0)),
)


# =============================================================================
# Weight sampling
# =============================================================================

def generate_configurations(
    *,
    total: int,
    seed: int,
    focus_centers: Sequence[Mapping[str, object]],
    include_anchors: bool = True,
):
    """Generate the reproducible quota-based sweep design."""
    rng = random.Random(seed)
    configurations, plan, seen = [], [], set()

    def append(values, source, **metadata):
        key = tuple(round(values[name], 14) for name in WEIGHT_NAMES)
        if key in seen:
            return False
        seen.add(key)
        index = len(configurations)
        configurations.append(SweepWeights(**values))
        plan.append(dict(configuration_id=index, source=source,
                         zero_weights=[k for k in WEIGHT_NAMES if values[k] == 0], **metadata))
        return True

    if include_anchors:
        for w in ANCHOR_CONFIGURATIONS:
            append(asdict(w), "reference")
        for center in focus_centers:
            append(center["weights"].copy(), "pilot_reference", pilot_id=center["pilot_id"])
    remaining = total - len(configurations)
    if remaining < 0:
        raise ValueError("Total is smaller than the number of references")
    focused = round(remaining * 2 / 3)
    pair_masks = list(itertools.combinations(WEIGHT_NAMES, 2))
    for source, count in (("focused", focused), ("broad", remaining-focused)):
        # Equal single-zero quotas across the six constraints; rounding goes to all-positive.
        singles_each = int(count * 0.4) // 6
        doubles = round(count * 0.2)
        masks = [()] * (count - 6*singles_each - doubles)
        masks += [(name,) for name in WEIGHT_NAMES for _ in range(singles_each)]
        pairs = pair_masks.copy()
        rng.shuffle(pairs)
        masks += [pairs[i % len(pairs)] for i in range(doubles)]
        rng.shuffle(masks)
        centers = [
            focus_centers[
                i % len(focus_centers)
            ]
            for i in range(count)
        ]
        rng.shuffle(centers)
        for mask, center in zip(masks, centers):
            for attempt in range(1000):
                values = {}
                for name in WEIGHT_NAMES:
                    if name in mask:
                        values[name] = 0.0
                        continue
                    bounds = WEIGHT_RANGES[name]
                    lo, hi = math.log10(bounds.minimum), math.log10(bounds.maximum)
                    central = center['weights'][name]
                    if source == "focused" and central > 0:
                        lo = max(lo, math.log10(central) - 0.5)
                        hi = min(hi, math.log10(central) + 0.5)
                    # A zero pilot weight activated by the quota uses the global positive interval.
                    values[name] = 10 ** rng.uniform(lo, hi)
                if append(values, source, pilot_id=center['pilot_id'] if source == "focused" else None):
                    break
            else:
                raise RuntimeError("Could not generate a unique configuration")
    assert len(configurations) == total
    return configurations, plan


def load_sweep_config(
    path: Path,
) -> dict[str, object]:
    """Load and validate the sweep sampling configuration."""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Expected a JSON object in {path}."
        )

    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported sweep configuration schema in {path}."
        )

    focus_centers = payload.get("focus_centers")

    if not isinstance(
        focus_centers,
        list,
    ) or not focus_centers:
        raise ValueError(
            f"{path} must contain a non-empty 'focus_centers' list."
        )

    for index, center in enumerate(focus_centers):
        if not isinstance(
            center,
            dict,
        ):
            raise TypeError(
                f"Focus center {index} must be an object."
            )

        if "pilot_id" not in center:
            raise KeyError(
                f"Focus center {index} has no 'pilot_id'."
            )

        weights = center.get("weights")

        if not isinstance(
            weights,
            dict,
        ):
            raise TypeError(
                f"Focus center {index} has no valid 'weights' object."
            )

        missing = set(WEIGHT_NAMES).difference(weights)
        extra = set(weights).difference(WEIGHT_NAMES)

        if missing or extra:
            raise ValueError(
                f"Invalid weights for focus center {index}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )

        for name in WEIGHT_NAMES:
            value = float(weights[name])

            if value < 0.0:
                raise ValueError(
                    f"Negative weight '{name}' in focus center {index}."
                )

    return payload


# =============================================================================
# External command execution
# =============================================================================

def build_evaluation_command(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    batch_size: int,
    rollout_horizons: Sequence[int],
    device: str,
    probe_workers: int = DEFAULT_PROBE_WORKERS,
    probe_cache_dir: str = DEFAULT_PROBE_CACHE_DIR,
    strong_probe_epochs: int | None = None,
    physical_probe_epochs: int | None = None,
    probe_profile: str = "pareto",
    nuisance_probe_epochs: int = DEFAULT_NUISANCE_PROBE_EPOCHS,
) -> list[str]:
    """Build the command invoking the evaluation module."""
    command = module_command(
        "scripts.evaluation",
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        str(data_dir),
        "--split",
        "validation",
        "--nuisance-probe-epochs",
        str(nuisance_probe_epochs),
        "--batch-size",
        str(batch_size),
        "--rollout-horizons",
        *[str(horizon) for horizon in rollout_horizons],
        "--device",
        device,
    )

    if probe_workers is not None:
        command.extend(["--probe-workers", str(probe_workers)])

    if probe_cache_dir is not None:
        command.extend(["--probe-cache-dir", str(probe_cache_dir)])

    if strong_probe_epochs is not None:
        command.extend(["--strong-probe-epochs", str(strong_probe_epochs)])

    if physical_probe_epochs is not None:
        command.extend(["--physical-probe-epochs", str(physical_probe_epochs)])

    if probe_profile is not None:
        command.extend(["--probe-profile", str(probe_profile)])
    return command


def build_train_command(
    *,
    train_module: str,
    run_name: str,
    runs_dir: Path,
    data_dir: Path,
    model_seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    weights: SweepWeights,
    adversary_steps: int = DEFAULT_ADVERSARY_STEPS,
    refresh_pretrain_epochs: int = DEFAULT_REFRESH_PRETRAIN_EPOCHS,
) -> list[str]:
    """Build the command invoking the training module."""

    command = module_command(
        train_module,
        "--run-name",
        run_name,
        "--output-dir",
        str(runs_dir),
        "--data-dir",
        str(data_dir),
        "--seed",
        str(model_seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    )

    command.extend(
        [
            "--refresh-pretrain-epochs",
            str(refresh_pretrain_epochs),
        ]
    )

    command.extend(
        [
            "--adversary-steps",
            str(adversary_steps),
        ]
    )

    for name in WEIGHT_NAMES:
        command.extend(
            [
                f"--weight-{name}",
                str(
                    getattr(weights, name)
                ),
            ]
        )

    return command


def run_command(command: Sequence[str], *, title: str) -> None:
    print_banner(title)
    print(subprocess.list2cmdline(list(command)))

    result = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{subprocess.list2cmdline(list(command))}\n\n"
            f"{result.stdout}"
        )


# =============================================================================
# Sweep result I/O
# =============================================================================

def load_configuration_results(output_dir: Path) -> list[dict[str, object]]:
    """Load previously aggregated sweep results."""
    path = output_dir / "sweep_configuration_results.json"
    if not path.exists():
        raise FileNotFoundError(
            "Cannot generate figures because aggregated sweep results are "
            f"missing: {path}"
        )

    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)

    if not isinstance(records, list):
        raise TypeError(f"Expected a list of records in {path}.")

    required = {
        "configuration_id",
        *WEIGHT_NAMES,
        *OBJECTIVES,
    }

    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise KeyError(
                f"Record {index} in {path} is missing: {sorted(missing)}"
            )

    return records


def load_search_protocol(output_dir: Path) -> dict[str, object]:
    """Load the immutable protocol of an existing sweep."""
    path = search_protocol_path(output_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"Search protocol not found: {path}"
        )

    with path.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)

    if not isinstance(protocol, dict):
        raise TypeError(
            f"Expected a JSON object in {path}."
        )

    return protocol


def load_sweep_manifest(
    output_dir: Path,
) -> list[dict[str, object]]:
    """Load the immutable manifest of an existing sweep."""

    path = (
        output_dir
        / "sweep_manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Sweep manifest not found: {path}"
        )

    with path.open("r", encoding="utf-8") as stream:
        records = json.load(
            stream
        )

    if not isinstance(records, list):
        raise TypeError(
            f"Expected a list of records in {path}."
        )

    return records


def load_validation_objectives(metrics_path: Path) -> dict[str, float]:
    """Load validation metrics only; reject legacy nuisance definitions."""
    with metrics_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if "validation" not in payload:
        raise KeyError(f"'validation' split missing from {metrics_path}.")

    if payload["validation"].get("nuisance_metric_version") != 2.0:
        raise ValueError("Legacy evaluation: rerun with the independent v2 nuisance probes")
    objectives = {}
    for metric_name in OBJECTIVES:
        value = payload["validation"].get(metric_name)
        if value is None:
            raise ValueError(
                f"Metric '{metric_name}' is missing or invalid in "
                f"{metrics_path}."
            )

        value = float(value)
        if not np.isfinite(value):
            raise ValueError(
                f"Metric '{metric_name}' is non-finite in {metrics_path}."
            )

        objectives[metric_name] = value

    return objectives


def save_records(
    records: Sequence[Mapping[str, object]],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Save a sequence of flat records to JSON and CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            list(records),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    if not records:
        return

    fieldnames = sorted(
        {
            key
            for record in records
            for key in record
        }
    )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_search_protocol_once(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    sampling_plan: Sequence[Mapping[str, object]],
    focus_centers: Sequence[Mapping[str, object]],
) -> None:
    """Create the sweep protocol once; never overwrite it."""
    path = search_protocol_path(output_dir)

    if path.exists():
        return

    train_module_path = Path(
        *args.train_module.split(".")
    ).with_suffix(".py")

    payload = {
        "release": "clsm-weight-sweep-2",
        "arguments": vars(args),
        "sweep_config": {
            "path": str(args.sweep_config),
            "sha256": sha256_file(args.sweep_config),
        },
        "weight_ranges": {
            name: asdict(weight_range)
            for name, weight_range in WEIGHT_RANGES.items()
        },
        "anchors": [
            asdict(weights)
            for weights in ANCHOR_CONFIGURATIONS
        ],
        "selection_split": "validation",
        "sampling_plan": sampling_plan,
        "focus_centers": list(focus_centers),
        "source_sha256": {
            "scripts/constraint_sweep.py": sha256_file(Path(__file__)),
            "clsm/training.py": sha256_file(Path("clsm/training.py")),
            "clsm/models.py": sha256_file(Path("clsm/models.py")),
            "clsm/losses.py": sha256_file(Path("clsm/losses.py")),
            "clsm/refresh.py": sha256_file(Path("clsm/refresh.py")),
            str(train_module_path): sha256_file(train_module_path),
        },
        "local_log10_half_width": 0.5,
    }

    path.write_text(
        json.dumps(
            payload,
            default=str,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def search_protocol_path(output_dir: Path) -> Path:
    """Return the immutable search protocol path for one sweep."""
    return output_dir / "search_protocol.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =============================================================================
# Result aggregation
# =============================================================================

def aggregate_configuration_records(
    run_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Average each metric across model seeds for every configuration."""
    grouped = {}
    for record in run_records:
        configuration_id = int(record["configuration_id"])
        grouped.setdefault(configuration_id, []).append(record)

    aggregated = []

    for configuration_id, records in sorted(grouped.items()):
        first = records[0]
        result = {
            "configuration_id": configuration_id,
            "n_model_seeds": len(records),
        }

        for name in WEIGHT_NAMES:
            result[name] = float(first[name])

        for metric_name in OBJECTIVES:
            values = np.asarray(
                [float(record[metric_name]) for record in records],
                dtype=float,
            )
            result[metric_name] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(
                np.std(values, ddof=1)
                if values.size > 1
                else 0.0
            )

        aggregated.append(result)

    return aggregated


# =============================================================================
# Pareto analysis
# =============================================================================

def load_analysis_protocol_arguments(
    output_dir: Path,
) -> dict[str, object]:
    """Load figure/Pareto settings from the original sweep protocol."""
    protocol = load_search_protocol(output_dir)

    arguments = protocol.get("arguments")
    if not isinstance(arguments, dict):
        raise KeyError(
            "search_protocol.json does not contain a valid "
            "'arguments' object."
        )

    missing = [
        name
        for name in ANALYSIS_PROTOCOL_ARGUMENTS
        if name not in arguments
    ]
    if missing:
        raise KeyError(
            "Missing analysis settings in search_protocol.json: "
            + ", ".join(missing)
        )

    return {
        name: arguments[name]
        for name in ANALYSIS_PROTOCOL_ARGUMENTS
    }


def oriented_pair_matrix(
    records: Sequence[Mapping[str, object]],
    *,
    x_metric: str,
    y_metric: str,
) -> np.ndarray:
    """Return two objectives in minimization form."""
    return np.asarray(
        [
            [
                float(record[x_metric])
                * (-1.0 if OBJECTIVES[x_metric] else 1.0),
                float(record[y_metric])
                * (-1.0 if OBJECTIVES[y_metric] else 1.0),
            ]
            for record in records
        ],
        dtype=float,
    )


def pareto_mask(
    objectives: np.ndarray,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> np.ndarray:
    """Identify non-dominated points for a minimization problem."""
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("objectives must have shape (N, K).")

    non_dominated = np.ones(objectives.shape[0], dtype=bool)

    for candidate_index, candidate in enumerate(objectives):
        tolerance = (
            absolute_tolerance
            + relative_tolerance
            * np.maximum(np.abs(candidate), 1.0)
        )

        no_worse = np.all(
            objectives <= candidate + tolerance,
            axis=1,
        )
        strictly_better = np.any(
            objectives < candidate - tolerance,
            axis=1,
        )

        dominates_candidate = no_worse & strictly_better
        dominates_candidate[candidate_index] = False

        if np.any(dominates_candidate):
            non_dominated[candidate_index] = False

    return non_dominated


def pairwise_pareto_mask(
    records: Sequence[Mapping[str, object]],
    *,
    x_metric: str,
    y_metric: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> np.ndarray:
    """Compute the Pareto mask for one pair of paper metrics."""
    objectives = oriented_pair_matrix(
        records,
        x_metric=x_metric,
        y_metric=y_metric,
    )
    return pareto_mask(
        objectives,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


# =============================================================================
# Pairwise Pareto plots
# =============================================================================

def representative_pareto_mask(
    records: Sequence[Mapping[str, object]],
    pareto: np.ndarray,
    *,
    x_metric: str,
    y_metric: str,
    minimum_distance: float,
) -> np.ndarray:
    """Select representative points from a 2-D Pareto front.

    Distances are measured after min-max normalization of the two displayed
    metrics. The first and last Pareto points are always retained. Intermediate
    points are retained only when they are sufficiently far from the last
    selected point. A non-positive threshold keeps every Pareto point.
    """
    pareto = np.asarray(pareto, dtype=bool)
    representative = np.zeros_like(pareto)

    indices = ordered_front_indices(
        records,
        pareto,
        x_metric=x_metric,
    )

    if len(indices) == 0:
        return representative

    if minimum_distance <= 0.0 or len(indices) <= 2:
        representative[indices] = True
        return representative

    x = np.asarray(
        [float(records[index][x_metric]) for index in indices],
        dtype=float,
    )
    y = np.asarray(
        [float(records[index][y_metric]) for index in indices],
        dtype=float,
    )

    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        if span <= 0.0:
            return np.zeros_like(values)
        return (values - np.min(values)) / span

    points = np.column_stack([normalize(x), normalize(y)])

    selected_positions = [0]
    last_selected = 0

    for position in range(1, len(indices) - 1):
        distance = float(
            np.linalg.norm(points[position] - points[last_selected])
        )
        if distance >= minimum_distance:
            selected_positions.append(position)
            last_selected = position

    if selected_positions[-1] != len(indices) - 1:
        selected_positions.append(len(indices) - 1)

    representative[
        indices[np.asarray(selected_positions, dtype=int)]
    ] = True

    return representative


def build_global_pareto_labels(
    records: Sequence[Mapping[str, object]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    outlier_iqr_multiplier: float,
    label_minimum_distance: float,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[int, str],
]:
    """Compute pairwise Pareto fronts and assign stable global labels.

    A configuration receives one label, such as P1, and keeps that label
    across figures. Labels are assigned by increasing configuration
    identifier over the union of panel-specific representative Pareto points.
    """

    pair_masks = {}
    pair_inlier_masks = {}
    pair_representative_masks = {}
    representative_configuration_ids = set()

    for x_metric, y_metric in OBJECTIVE_PAIRS:
        x_values = np.asarray(
            [float(record[x_metric]) for record in records],
            dtype=float,
        )
        y_values = np.asarray(
            [float(record[y_metric]) for record in records],
            dtype=float,
        )

        # Pareto front: always computed from all configurations.
        mask = pairwise_pareto_mask(
            records,
            x_metric=x_metric,
            y_metric=y_metric,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        pair_masks[(x_metric, y_metric)] = mask

        # Visualization-only outlier filtering for dominated configurations.
        plot_inliers = pairwise_inlier_mask(
            x_values,
            y_values,
            iqr_multiplier=outlier_iqr_multiplier,
        )
        plot_inliers |= mask
        pair_inlier_masks[(x_metric, y_metric)] = plot_inliers

        representative = representative_pareto_mask(
            records,
            mask,
            x_metric=x_metric,
            y_metric=y_metric,
            minimum_distance=label_minimum_distance,
        )
        pair_representative_masks[(x_metric, y_metric)] = representative

        representative_configuration_ids.update(
            int(record["configuration_id"])
            for record, is_representative in zip(
                records,
                representative,
                strict=True,
            )
            if is_representative
        )

    labels = {
        configuration_id: f"P{label_index}"
        for label_index, configuration_id in enumerate(
            sorted(representative_configuration_ids),
            start=1,
        )
    }

    return (
        pair_masks,
        pair_inlier_masks,
        pair_representative_masks,
        labels,
    )


def global_pareto_configuration_records(
    records: Sequence[Mapping[str, object]],
    labels: Mapping[int, str],
) -> list[dict[str, object]]:
    """Return one table row per globally labeled Pareto configuration."""
    rows = []

    for record in records:
        configuration_id = int(record["configuration_id"])
        if configuration_id not in labels:
            continue

        rows.append(
            {
                "label": labels[configuration_id],
                "configuration_id": configuration_id,
                **{
                    weight_name: float(record[weight_name])
                    for weight_name in WEIGHT_NAMES
                },
                **{
                    metric_name: float(record[metric_name])
                    for metric_name in OBJECTIVES
                },
            }
        )

    rows.sort(
        key=lambda row: int(str(row["label"])[1:])
    )
    return rows


def ordered_front_indices(
    records: Sequence[Mapping[str, object]],
    pareto: np.ndarray,
    *,
    x_metric: str,
) -> np.ndarray:
    """Order Pareto points from best to worst along the oriented x objective."""
    indices = np.flatnonzero(pareto)
    oriented_x = np.asarray(
        [
            float(records[index][x_metric])
            * (-1.0 if OBJECTIVES[x_metric] else 1.0)
            for index in indices
        ],
        dtype=float,
    )
    return indices[np.argsort(oriented_x)]


def pairwise_inlier_mask(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    iqr_multiplier: float,
) -> np.ndarray:
    """Return configurations retained for visualization.

    Tukey fences are estimated independently on both displayed metrics.
    Dominated configurations outside either fence may be hidden from the plot.
    Pareto computation itself is unaffected. A negative multiplier disables
    filtering.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if x_values.shape != y_values.shape:
        raise ValueError("x_values and y_values must have identical shapes.")

    if iqr_multiplier < 0.0 or x_values.size < 4:
        return np.ones(x_values.shape, dtype=bool)

    keep = np.ones(x_values.shape, dtype=bool)

    for values in (x_values, y_values):
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1

        if not np.isfinite(iqr) or iqr <= 0.0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        keep &= (values >= lower) & (values <= upper)

    return keep


def plot_pairwise_pareto(
    records: Sequence[Mapping[str, object]],
    *,
    x_metric: str,
    y_metric: str,
    pareto: np.ndarray,
    representative: np.ndarray,
    inliers: np.ndarray,
    labels: Mapping[int, str],
    output_path: Path,
) -> tuple[Path, dict[str, object]]:
    """Plot one 2-D Pareto front as a staircase."""
    x_values = np.asarray(
        [float(record[x_metric]) for record in records],
        dtype=float,
    )
    y_values = np.asarray(
        [float(record[y_metric]) for record in records],
        dtype=float,
    )

    pareto = np.asarray(pareto, dtype=bool)
    if pareto.shape != (len(records),):
        raise ValueError(
            "pareto mask must have shape "
            f"({len(records)},), got {pareto.shape}."
        )

    front_indices = ordered_front_indices(
        records,
        pareto,
        x_metric=x_metric,
    )

    inliers = np.asarray(inliers, dtype=bool)
    if inliers.shape != (len(records),):
        raise ValueError(
            "inlier mask must have shape "
            f"({len(records)},), got {inliers.shape}."
        )

    dominated_inliers = inliers & (~pareto)
    n_excluded_outliers = int(np.sum(~inliers))

    figure, axis = plt.subplots(figsize=(7.2, 5.6))

    axis.scatter(
        x_values[dominated_inliers],
        y_values[dominated_inliers],
        s=28,
        alpha=0.22,
        label="Dominated configurations",
        zorder=1,
    )

    axis.scatter(
        x_values[pareto],
        y_values[pareto],
        s=62,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.55,
        label="2-D Pareto front",
        zorder=3,
    )

    representative = np.asarray(representative, dtype=bool)
    if representative.shape != (len(records),):
        raise ValueError(
            "representative mask must have shape "
            f"({len(records)},), got {representative.shape}."
        )

    representative_indices = ordered_front_indices(
        records,
        representative,
        x_metric=x_metric,
    )

    label_texts = []

    for index in representative_indices:
        configuration_id = int(records[index]["configuration_id"])
        label_texts.append(
            axis.text(
                x_values[index],
                y_values[index],
                labels[configuration_id],
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
                zorder=4,
            )
        )

    if label_texts:
        adjust_text(
            label_texts,
            ax=axis,
            x=x_values[representative_indices],
            y=y_values[representative_indices],
            only_move={
                "text": "xy",
                "static": "xy",
                "explode": "xy",
                "pull": "xy",
            },
            force_text=(0.5, 0.8),
            force_static=(0.2, 0.2),
            expand=(1.2, 1.3),
            min_arrow_len=3,
            arrowprops={
                "arrowstyle": "-",
                "color": "0.4",
                "linewidth": 0.6,
            },
        )

    if len(front_indices) >= 2:
        axis.step(
            x_values[front_indices],
            y_values[front_indices],
            where="post",
            linewidth=1.6,
            alpha=0.9,
            zorder=2,
        )

    axis.set_xlabel(OBJECTIVE_LABELS[x_metric])
    axis.set_ylabel(OBJECTIVE_LABELS[y_metric])
    axis.grid(alpha=0.22)
    axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    representative_pareto_solutions = [
        {
            "label": labels[int(records[index]["configuration_id"])],
            "configuration_id": int(records[index]["configuration_id"]),
            x_metric: float(records[index][x_metric]),
            y_metric: float(records[index][y_metric]),
            **{
                weight_name: float(records[index][weight_name])
                for weight_name in WEIGHT_NAMES
            },
        }
        for index in representative_indices
    ]

    summary = {
        "x_metric": x_metric,
        "y_metric": y_metric,
        "n_configurations": len(records),
        "n_pareto": int(np.sum(pareto)),
        "n_labeled_pareto": int(np.sum(representative)),
        "pareto_fraction": float(np.mean(pareto)),
        "n_excluded_outliers": n_excluded_outliers,
        "representative_pareto_configuration_ids": [
            solution["configuration_id"]
            for solution in representative_pareto_solutions
        ],
        "representative_pareto_solutions": representative_pareto_solutions,
    }

    return output_path, summary


def plot_all_pairwise_pareto_fronts(
    records: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    pair_masks: Mapping[tuple[str, str], np.ndarray],
    pair_inlier_masks: Mapping[tuple[str, str], np.ndarray],
    pair_representative_masks: Mapping[tuple[str, str], np.ndarray],
    labels: Mapping[int, str],
) -> tuple[list[Path], list[dict[str, object]]]:
    """Generate the four selected pairwise Pareto figures with shared P labels."""
    paths = []
    summaries = []

    for pair_index, (x_metric, y_metric) in enumerate(
        OBJECTIVE_PAIRS,
        start=1,
    ):
        filename = (
            f"pareto_2d_{pair_index:02d}_"
            f"{OBJECTIVE_SHORT_NAMES[x_metric]}_vs_"
            f"{OBJECTIVE_SHORT_NAMES[y_metric]}.pdf"
        )

        path, summary = plot_pairwise_pareto(
            records,
            x_metric=x_metric,
            y_metric=y_metric,
            pareto=pair_masks[(x_metric, y_metric)],
            representative=pair_representative_masks[(x_metric, y_metric)],
            inliers=pair_inlier_masks[(x_metric, y_metric)],
            labels=labels,
            output_path=output_dir / filename,
        )

        paths.append(path)
        summaries.append(summary)

    return paths, summaries


# =============================================================================
# Figure orchestration
# =============================================================================

def generate_analysis_figures(
    records: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    outlier_iqr_multiplier: float,
    label_minimum_distance: float,
) -> tuple[
    list[Path],
    list[dict[str, object]],
    dict[int, str],
]:
    """Generate the four pairwise Pareto figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        pair_masks,
        pair_inlier_masks,
        pair_representative_masks,
        labels,
    ) = build_global_pareto_labels(
        records,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        label_minimum_distance=label_minimum_distance,
    )

    paths, summaries = plot_all_pairwise_pareto_fronts(
        records,
        output_dir=output_dir,
        pair_masks=pair_masks,
        pair_inlier_masks=pair_inlier_masks,
        pair_representative_masks=pair_representative_masks,
        labels=labels,
    )

    return paths, summaries, labels


def print_representative_pareto_solutions(
    summaries: Sequence[Mapping[str, object]],
) -> None:
    """Print all representative pairwise Pareto solutions and their six weights."""
    for summary in summaries:
        print()
        print(
            f"{summary['x_metric']} vs {summary['y_metric']} "
            f"({summary['n_pareto']} Pareto solutions; "
            f"{summary['n_labeled_pareto']} labeled representatives; "
            f"{summary['n_excluded_outliers']} dominated outliers hidden in plot)"
        )
        print("-" * 96)

        for solution in summary["representative_pareto_solutions"]:
            weights = ", ".join(
                f"{name}={float(solution[name]):.4g}"
                for name in WEIGHT_NAMES
            )
            print(
                f"{solution['label']} | "
                f"config {int(solution['configuration_id']):04d} | "
                f"{summary['x_metric']}="
                f"{float(solution[summary['x_metric']]):.6g} | "
                f"{summary['y_metric']}="
                f"{float(solution[summary['y_metric']]):.6g} | "
                f"{weights}"
            )


def save_pareto_table(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Export Table 4 with the same P labels and panel memberships as the figures."""

    headers = [
        "Configuration",
        *WEIGHT_NAMES,
        "Panels",
    ]

    markdown_lines = [
        "# Table 4 — Representative pairwise Pareto configurations",
        "",
        (
            "Weights of configurations labeled in the four panels. "
            "Membership is based on validation metrics averaged across "
            "model seeds."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]

    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        (
            r"  \caption{\textbf{Constraint weights of representative configurations "
            r"on the empirical pairwise Pareto fronts.} Panel membership refers "
            r"to the validation fronts.}"
        ),
        r"  \label{tab:pareto-weights}",
        r"  \begin{tabular}{lrrrrrrl}",
        r"    \hline",
        (
            r"    Configuration & "
            r"$\lambda_{\rm pred}$ & "
            r"$\lambda_{\rm min}$ & "
            r"$\lambda_{\rm temp}$ & "
            r"$\lambda_{\rm obs}$ & "
            r"$\lambda_{\rm inv}$ & "
            r"$\lambda_{\rm struct}$ & "
            r"Panels \\"
        ),
        r"    \hline",
    ]

    for row in rows:
        markdown_values = [
            str(
                row["label"]
            ),
            *(
                f"{float(row[name]):.6g}"
                for name in WEIGHT_NAMES
            ),
            str(
                row["panels"]
            ),
        ]

        markdown_lines.append(
            "| "
            + " | ".join(
                markdown_values
            )
            + " |"
        )

        latex_values = [
            str(
                row["label"]
            ),
            *(
                f"{float(row[name]):.3g}"
                for name in WEIGHT_NAMES
            ),
            str(
                row["panels"]
            ),
        ]

        latex_lines.append(
            "    "
            + " & ".join(
                latex_values
            )
            + r" \\"
        )

    latex_lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

    markdown_path = output_dir / "pareto_table.md"

    latex_path = output_dir / "pareto_table.tex"

    markdown_path.write_text(
        "\n".join(
            markdown_lines
        )
        + "\n",
        encoding="utf-8",
    )

    latex_path.write_text(
        "\n".join(
            latex_lines
        )
        + "\n",
        encoding="utf-8",
    )


def save_global_pareto_configurations(
    records: Sequence[Mapping[str, object]],
    labels: Mapping[int, str],
    *,
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    """Save the global P-label-to-configuration correspondence."""
    rows = global_pareto_configuration_records(records, labels)
    for row in rows:
        row["panels"] = ", ".join(
            f"({chr(97 + index)})" for index, summary in enumerate(summaries)
            if row["configuration_id"] in summary["representative_pareto_configuration_ids"]
        )
    save_pareto_table(rows, output_dir)

    save_records(
        rows,
        json_path=output_dir / "pareto_configurations.json",
        csv_path=output_dir / "pareto_configurations.csv",
    )


def save_pareto_summaries(
    summaries: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
) -> None:
    """Save pairwise Pareto counts and ordered configuration identifiers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "pareto_2d_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            list(summaries),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    csv_records = [
        {
            **{
                key: value
                for key, value in summary.items()
                if key != "representative_pareto_configuration_ids"
            },
            "representative_pareto_configuration_ids": ",".join(
                str(configuration_id)
                for configuration_id in summary["representative_pareto_configuration_ids"]
            ),
        }
        for summary in summaries
    ]

    save_records(
        csv_records,
        json_path=output_dir / "pareto_2d_summary_flat.json",
        csv_path=output_dir / "pareto_2d_summary.csv",
    )


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a six-weight CLSM sweep, evaluate five metrics, "
            "and compute the four selected pairwise 2-D Pareto fronts."
        )
    )

    parser.add_argument(
        "--num-configurations",
        type=int,
        default=DEFAULT_NUM_CONFIGURATIONS,
        help="Total number of configurations, including anchors."
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=DEFAULT_SWEEP_CONFIG_PATH,
        help=(
            "JSON file defining the pilot focus centers "
            "used by the sweep sampler."
        ),
    )
    parser.add_argument(
        "--sweep-seed",
        type=int,
        default=DEFAULT_SWEEP_SEED
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_MODEL_SEEDS
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--train-module",
        default=None,
        help="Training module to execute.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS
    )
    parser.add_argument(
        "--refresh-pretrain-epochs",
        type=int,
        default=DEFAULT_REFRESH_PRETRAIN_EPOCHS
    )
    parser.add_argument(
        "--nuisance-probe-epochs",
        type=int,
        default=DEFAULT_NUISANCE_PROBE_EPOCHS
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=DEFAULT_EVALUATION_BATCH_SIZE
    )
    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=DEFAULT_ROLLOUT_HORIZONS,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pareto-absolute-tolerance",
        type=float,
        default=DEFAULT_PARETO_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--pareto-relative-tolerance",
        type=float,
        default=DEFAULT_PARETO_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--plot-outlier-iqr-multiplier",
        type=float,
        default=DEFAULT_PLOT_OUTLIER_IQR_MULTIPLIER,
        help=(
            "Hide dominated configurations outside Tukey fences on either axis "
            "for visualization only. Pareto configurations are always retained "
            "and Pareto fronts are computed from all configurations. "
            "Use a negative value to disable filtering."
        ),
    )
    parser.add_argument(
        "--pareto-label-minimum-distance",
        type=float,
        default=DEFAULT_PARETO_LABEL_MINIMUM_DISTANCE,
        help=(
            "Minimum Euclidean spacing used to select intermediate labeled "
            "Pareto solutions after min-max normalization of the displayed "
            "metrics. Front endpoints are always retained. The full front "
            "remains plotted. Use 0 to label every Pareto solution."
        ),
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="Stop scheduling when this file exists; finish active work and save results."
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Overlap one evaluation with the next training; same evaluation device."
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Archive a partial training directory without best.pt and restart only that run."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help=(
            "Skip training and evaluation, reload existing per-run evaluation "
            "files, and rebuild the aggregate analysis."
        ),
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Generate the four pairwise Pareto figures directly from the "
            "existing sweep_configuration_results.json file."
        ),
    )

    parser.add_argument("--adversary-steps", type=int, default=DEFAULT_ADVERSARY_STEPS)
    parser.add_argument("--probe-workers", type=int, default=DEFAULT_PROBE_WORKERS)
    parser.add_argument("--probe-cache-dir", type=str, default=DEFAULT_PROBE_CACHE_DIR)
    parser.add_argument("--strong-probe-epochs", type=int, default=None)
    parser.add_argument("--physical-probe-epochs", type=int, default=None)
    parser.add_argument("--probe-profile", type=str, default="pareto")
    return parser


# =============================================================================
# Sweep orchestration
# =============================================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.analyze_only and args.figures_only:
        parser.error(
            "--analyze-only and --figures-only are mutually exclusive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.figures_only:
        records = load_configuration_results(
            args.output_dir
        )

        analysis_arguments = load_analysis_protocol_arguments(
            args.output_dir
        )

        figure_paths, summaries, labels = generate_analysis_figures(
            records,
            output_dir=args.output_dir / "figures",
            absolute_tolerance=float(
                analysis_arguments[
                    "pareto_absolute_tolerance"
                ]
            ),
            relative_tolerance=float(
                analysis_arguments[
                    "pareto_relative_tolerance"
                ]
            ),
            outlier_iqr_multiplier=args.plot_outlier_iqr_multiplier,
            label_minimum_distance=float(
                analysis_arguments[
                    "pareto_label_minimum_distance"
                ]
            ),
        )

        save_pareto_summaries(
            summaries,
            output_dir=args.output_dir,
        )

        save_global_pareto_configurations(
            records,
            labels,
            summaries=summaries,
            output_dir=args.output_dir,
        )

        print(
            "Using Pareto settings from the original search_protocol.json; "
            "plot-only outlier filtering uses the current CLI setting."
        )

        print_representative_pareto_solutions(summaries)

        print()
        print_separator()
        print(
            "FIGURE GENERATION COMPLETED"
        )
        print(
            f"Configurations : {len(records)}"
        )
        print(
            f"Pareto fronts  : {len(summaries)}"
        )

        for summary in summaries:
            print(
                f"  {summary['x_metric']} vs "
                f"{summary['y_metric']}: "
                f"{summary['n_pareto']}/"
                f"{summary['n_configurations']} "
                f"({summary['pareto_fraction']:.3f})"
            )

        print(
            f"Figures dir    : "
            f"{args.output_dir / 'figures'}"
        )

        print_separator()
        return

    if (
        not args.analyze_only
        and args.train_module is None
    ):
        parser.error(
            "--train-module is required when running training and evaluation."
        )

    if args.analyze_only:
        protocol = load_search_protocol(
            args.output_dir
        )

        protocol_arguments = protocol.get(
            "arguments"
        )

        if not isinstance(
            protocol_arguments,
            dict,
        ):
            raise KeyError(
                "search_protocol.json does not contain "
                "a valid 'arguments' object."
            )

        historical_runs_dir = protocol_arguments.get(
            "runs_dir"
        )

        if historical_runs_dir is None:
            raise KeyError(
                "search_protocol.json does not contain "
                "the historical runs_dir."
            )

        run_records_dir = Path(
            str(historical_runs_dir)
        )

        analysis_arguments = load_analysis_protocol_arguments(
            args.output_dir
        )

        manifest_records = load_sweep_manifest(
            args.output_dir
        )

    else:
        run_records_dir = args.runs_dir
        analysis_arguments = None

        sweep_config = load_sweep_config(
            args.sweep_config
        )

        focus_centers = sweep_config[
            "focus_centers"
        ]

        minimum_configurations = (
            len(ANCHOR_CONFIGURATIONS)
            + len(focus_centers)
            if args.include_anchors
            else 0
        )

        if (
            args.num_configurations
            < minimum_configurations
        ):
            parser.error(
                "--num-configurations must be at least "
                f"{minimum_configurations} "
                "when anchors are included."
            )

        if not args.model_seeds:
            parser.error(
                "At least one model seed must be provided."
            )

        if (
            len(set(args.model_seeds))
            != len(args.model_seeds)
        ):
            parser.error(
                "--model-seeds must be unique."
            )

        if (
            REQUIRED_PARETO_ROLLOUT_HORIZON
            not in args.rollout_horizons
        ):
            parser.error(
                "The five-metric analysis requires "
                "rollout horizon "
                f"{REQUIRED_PARETO_ROLLOUT_HORIZON}."
            )

        configurations, sampling_plan = generate_configurations(
            total=args.num_configurations,
            seed=args.sweep_seed,
            focus_centers=focus_centers,
            include_anchors=args.include_anchors,
        )

        manifest_records = []

        for configuration_id, weights in enumerate(
            configurations
        ):
            for model_seed in args.model_seeds:
                run_name = (
                    f"constraint-sweep/"
                    f"config-{configuration_id:04d}/"
                    f"seed-{model_seed}"
                )

                manifest_records.append(
                    {
                        "configuration_id": configuration_id,
                        "run_name": run_name,
                        "adversary_steps": (
                            args.adversary_steps
                        ),
                        "model_seed": model_seed,
                        **asdict(weights),
                    }
                )

        manifest_path = (
            args.output_dir
            / "sweep_manifest.json"
        )

        if manifest_path.exists():
            previous_manifest = load_sweep_manifest(
                args.output_dir
            )

            if (
                previous_manifest
                != manifest_records
            ):
                raise ValueError(
                    "Search manifest differs. "
                    "Keep the original seeds, bounds, "
                    "and configuration count, or choose "
                    "a new output directory."
                )

        save_records(
            manifest_records,
            json_path=manifest_path,
            csv_path=(
                args.output_dir
                / "sweep_manifest.csv"
            ),
        )

        save_search_protocol_once(
            output_dir=args.output_dir,
            args=args,
            sampling_plan=sampling_plan,
            focus_centers=focus_centers,
        )

    if args.dry_run:
        return

    run_records = []
    progress = tqdm(
        manifest_records,
        desc="Constraint sweep",
        unit="run",
        dynamic_ncols=True,
    )

    def finish_run(record, metrics_path, command):
        if command is not None:
            run_command(
                command,
                title=f"EVALUATING CONFIGURATION {record['configuration_id']} SEED {record['model_seed']}"
            )
        return {**record, **load_validation_objectives(metrics_path)}

    def record_result(record):
        run_records.append(record)
        save_records(run_records, json_path=args.output_dir / "sweep_run_results.json",
                     csv_path=args.output_dir / "sweep_run_results.csv")

    executor = (
        ThreadPoolExecutor(max_workers=1)
        if args.pipeline
        and not args.analyze_only
        else None
    )

    pending = deque()

    try:
        for manifest_record in progress:
            if (
                args.stop_file is not None
                and args.stop_file.exists()
            ):
                print(
                    "Stop requested: finishing queued evaluations.",
                    flush=True,
                )
                break

            # At most one evaluation running plus one queued.
            # Any failure is propagated when the result is collected.
            while pending and (
                pending[0].done()
                or len(pending) >= 2
            ):
                completed = pending.popleft().result()
                record_result(completed)

            run_name = str(manifest_record["run_name"])
            model_seed = int(manifest_record["model_seed"])
            progress.set_postfix(run=run_name)

            weights = SweepWeights(
                **{
                    name: float(manifest_record[name])
                    for name in WEIGHT_NAMES
                }
            )

            run_dir = run_records_dir / run_name
            checkpoint_path = run_dir / "best.pt"
            metrics_path = run_dir / "evaluation" / "evaluation_metrics.json"

            if (
                not args.analyze_only
                and args.skip_existing
                and checkpoint_path.exists()
            ):
                saved_config = json.loads(
                    (run_dir / "config.json").read_text()
                )

                if (
                    saved_config["loss"].get("invariance_method")
                    != INVARIANCE_METHOD
                ):
                    raise ValueError(
                        "Cannot reuse a training run "
                        f"with a different invariance protocol: {run_dir}"
                    )

                expected_optimization = {
                    "epochs": args.epochs,
                    "batch_size": args.train_batch_size,
                    "refresh_pretrain_epochs": (args.refresh_pretrain_epochs),
                    "seed": model_seed,
                    "adversary_steps": (args.adversary_steps),
                }

                optimization_differs = any(
                    saved_config["optimization"].get(name)
                    != value
                    for name, value
                    in expected_optimization.items()
                )

                if optimization_differs:
                    raise ValueError(
                        f"Cached training settings differ: {run_dir}"
                    )

                expected_weights = {
                    name: getattr(weights, name)
                    for name in WEIGHT_NAMES
                }

                if (saved_config["loss"]["weights"] != expected_weights):
                    raise ValueError(
                        f"Cached weights differ: {run_dir}"
                    )

                manifest = json.loads(
                    (run_dir / "data_manifest.json").read_text()
                )

                for split_name in (
                    "train",
                    "validation",
                ):
                    split_path = args.data_dir / f"{split_name}.npz"

                    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()

                    if manifest.get("sha256", {}).get(split_name) != digest:
                        raise ValueError(
                            "Cached dataset differs or is unverified: "
                            f"{run_dir}"
                        )

                if metrics_path.exists():
                    fit_report = json.loads(
                        (metrics_path.parent / "nuisance_probe_fits.json").read_text()
                    )

                    saved_evaluation = json.loads(
                        (metrics_path.parent / "evaluation_protocol.json").read_text()
                    )

                    desired_evaluation = {
                        "probe_profile": (args.probe_profile),
                        "nuisance_probe_epochs": (args.nuisance_probe_epochs),
                        "strong_probe_epochs": (args.strong_probe_epochs),
                        "physical_probe_epochs": (args.physical_probe_epochs),
                        "probe_seed": DEFAULT_PROBE_SEED,
                        "nonlinear_probe_max_samples": DEFAULT_NONLINEAR_PROBE_MAX_SAMPLES,
                    }

                    if (saved_evaluation != desired_evaluation):
                        raise ValueError(
                            "Cached evaluation settings differ: "
                            f"{run_dir}"
                        )

                    if fit_report["settings"]["max_epochs"] != args.nuisance_probe_epochs:
                        raise ValueError(
                            "Cached probe budget differs: "
                            f"{run_dir}"
                        )

            evaluation_command = None

            if not args.analyze_only:
                incomplete_run = (
                    not checkpoint_path.exists()
                    and run_dir.exists()
                    and any(run_dir.iterdir())
                )

                if incomplete_run:
                    if not args.restart_incomplete:
                        raise ValueError(
                            f"Partial training found: {run_dir}. "
                            "Use --restart-incomplete to archive it "
                            "and restart this run only."
                        )

                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

                    archived_run_dir = run_dir.with_name(
                        run_dir.name
                        + ".interrupted-"
                        + timestamp
                    )

                    shutil.move(
                        str(run_dir),
                        str(archived_run_dir),
                    )

                    print(
                        f"Archived incomplete run: {archived_run_dir}",
                        flush=True,
                    )

                needs_training = (
                    not args.skip_existing
                    or not checkpoint_path.exists()
                )

                if needs_training:
                    training_command = build_train_command(
                        train_module=args.train_module,
                        run_name=run_name,
                        runs_dir=args.runs_dir,
                        data_dir=args.data_dir,
                        model_seed=model_seed,
                        epochs=args.epochs,
                        batch_size=args.train_batch_size,
                        device=args.device,
                        weights=weights,
                        adversary_steps=args.adversary_steps,
                        refresh_pretrain_epochs=(
                            args.refresh_pretrain_epochs
                        ),
                    )

                    run_command(
                        training_command,
                        title=(
                            "TRAINING CONFIGURATION "
                            f"{manifest_record['configuration_id']} "
                            f"SEED {model_seed}"
                        ),
                    )

                needs_evaluation = (
                    not args.skip_existing
                    or not metrics_path.exists()
                )

                if needs_evaluation:
                    evaluation_command = build_evaluation_command(
                        checkpoint_path=checkpoint_path,
                        data_dir=args.data_dir,
                        batch_size=args.evaluation_batch_size,
                        probe_workers=args.probe_workers,
                        probe_cache_dir=args.probe_cache_dir,
                        strong_probe_epochs=(
                            args.strong_probe_epochs
                        ),
                        physical_probe_epochs=(
                            args.physical_probe_epochs
                        ),
                        probe_profile=args.probe_profile,
                        nuisance_probe_epochs=(
                            args.nuisance_probe_epochs
                        ),
                        rollout_horizons=args.rollout_horizons,
                        device=args.device,
                    )

            if executor is not None and not args.analyze_only:
                future = executor.submit(
                    finish_run,
                    manifest_record,
                    metrics_path,
                    evaluation_command,
                )

                pending.append(future)

            else:
                completed = finish_run(
                    manifest_record,
                    metrics_path,
                    evaluation_command,
                )

                record_result(completed)

        while pending:
            completed = pending.popleft().result()
            record_result(completed)

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if len(run_records) != len(manifest_records):
        print(f"Stopped cleanly: {len(run_records)}/{len(manifest_records)} results collected. "
              f"Resume with the same command after removing the stop file.")
        return

    aggregated_records = aggregate_configuration_records(run_records)

    save_records(
        aggregated_records,
        json_path=(
            args.output_dir
            / "sweep_configuration_results.json"
        ),
        csv_path=(
            args.output_dir
            / "sweep_configuration_results.csv"
        ),
    )

    if analysis_arguments is not None:
        absolute_tolerance = float(
            analysis_arguments["pareto_absolute_tolerance"]
        )

        relative_tolerance = float(
            analysis_arguments["pareto_relative_tolerance"]
        )

        label_minimum_distance = float(
            analysis_arguments["pareto_label_minimum_distance"]
        )

    else:
        absolute_tolerance = (
            args.pareto_absolute_tolerance
        )

        relative_tolerance = (
            args.pareto_relative_tolerance
        )

        label_minimum_distance = (
            args.pareto_label_minimum_distance
        )

    figure_paths, summaries, labels = generate_analysis_figures(
        aggregated_records,
        output_dir=args.output_dir / "figures",
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        outlier_iqr_multiplier=(
            args.plot_outlier_iqr_multiplier
        ),
        label_minimum_distance=label_minimum_distance,
    )

    save_pareto_summaries(
        summaries,
        output_dir=args.output_dir,
    )
    save_global_pareto_configurations(
        aggregated_records,
        labels,
        summaries=summaries,
        output_dir=args.output_dir,
    )
    print_representative_pareto_solutions(summaries)

    print()
    print_separator()
    print("SWEEP COMPLETED")
    print(f"Configurations : {len(aggregated_records)}")
    print(f"Model runs     : {len(run_records)}")
    print(f"Pareto fronts  : {len(summaries)}")
    for summary in summaries:
        print(
            f"  {summary['x_metric']} vs {summary['y_metric']}: "
            f"{summary['n_pareto']}/{summary['n_configurations']} "
            f"({summary['pareto_fraction']:.3f})"
        )
    print(f"Analysis dir   : {args.output_dir}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")
    print_separator()


if __name__ == "__main__":
    main()

