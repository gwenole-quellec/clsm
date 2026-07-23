"""
Run a large-scale CLSM weight sweep and analyze all pairwise 2-D Pareto fronts.

Author: Gwenolé Quellec
Year: 2026

The script samples CLSM constraint weights, trains and evaluates each
configuration, aggregates results across model seeds, and generates the four
pairwise Pareto curves retained for the paper induced by the five paper metrics. Pareto-optimal
configurations receive stable global labels (P1, P2, ...) shared by all figures.

Requires ``adjustText`` for automatic label placement.

Examples
--------
Run a complete sweep:

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --num-configurations 100 \
        --device cuda

Rebuild the aggregate analysis from existing per-run evaluations:

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --analyze-only

Regenerate the ten pairwise Pareto figures from saved aggregate results:

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --figures-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from tqdm.auto import tqdm

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
    zero_probability: float = 0.0


# =============================================================================
# Sweep constants
# =============================================================================

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
    "counterfactual_normalized_mse": False,
    "conditional_nuisance_probe_accuracy": False,
}

OBJECTIVE_LABELS = {
    "rollout_observation_mse_h5": "Prediction MSE (h=5) ↓",
    "state_probe_r2": r"State accessibility ($R^2$) ↑",
    "neighborhood_trustworthiness": "Neighborhood preservation ↑",
    "counterfactual_normalized_mse": "Counterfactual NMSE ↓",
    "conditional_nuisance_probe_accuracy": "Conditional nuisance accuracy ↓",
}

OBJECTIVE_SHORT_NAMES = {
    "rollout_observation_mse_h5": "prediction",
    "state_probe_r2": "state_accessibility",
    "neighborhood_trustworthiness": "neighborhood_preservation",
    "counterfactual_normalized_mse": "counterfactual_consistency",
    "conditional_nuisance_probe_accuracy": "nuisance_suppression",
}

# Pairwise Pareto fronts retained for the paper.
# The order determines the output numbering and the panel order A--D.
OBJECTIVE_PAIRS = (
    (
        "rollout_observation_mse_h5",
        "counterfactual_normalized_mse",
    ),
    (
        "rollout_observation_mse_h5",
        "conditional_nuisance_probe_accuracy",
    ),
    (
        "neighborhood_trustworthiness",
        "conditional_nuisance_probe_accuracy",
    ),
    (
        "state_probe_r2",
        "neighborhood_trustworthiness",
    ),
)

WEIGHT_RANGES = {
    "predictive": WeightRange(0.05, 2.0, 0.05),
    "minimality": WeightRange(1e-5, 5e-2, 0.25),
    "temporal": WeightRange(0.01, 1.5, 0.10),
    "observation": WeightRange(0.01, 1.5, 0.10),
    "invariance": WeightRange(1e-3, 2e-1, 0.20),
    "structural": WeightRange(1e-3, 2e-1, 0.20),
}

ANCHOR_CONFIGURATIONS = (
    SweepWeights(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    SweepWeights(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    SweepWeights(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    SweepWeights(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    SweepWeights(1.0, 0.0, 0.5, 0.5, 0.0, 0.0),
    SweepWeights(1.0, 0.01, 0.5, 0.5, 0.1, 0.1),
)


# =============================================================================
# Weight sampling
# =============================================================================

def sample_zero_or_log_uniform(
    rng: np.random.Generator,
    *,
    minimum: float,
    maximum: float,
    zero_probability: float,
) -> float:
    """Sample zero or a positive log-uniform value."""
    if rng.random() < zero_probability:
        return 0.0

    return float(
        10.0
        ** rng.uniform(
            np.log10(minimum),
            np.log10(maximum),
        )
    )


def sample_weights(rng: np.random.Generator) -> SweepWeights:
    """Sample all six CLSM constraint weights."""
    sampled = {
        name: sample_zero_or_log_uniform(
            rng,
            minimum=weight_range.minimum,
            maximum=weight_range.maximum,
            zero_probability=weight_range.zero_probability,
        )
        for name, weight_range in WEIGHT_RANGES.items()
    }

    # Regularization-only objectives are not useful without at least one
    # information-preserving objective.
    if (
        sampled["predictive"]
        + sampled["temporal"]
        + sampled["observation"]
        == 0.0
    ):
        name = str(
            rng.choice(
                (
                    "predictive",
                    "temporal",
                    "observation",
                )
            )
        )
        weight_range = WEIGHT_RANGES[name]
        sampled[name] = sample_zero_or_log_uniform(
            rng,
            minimum=weight_range.minimum,
            maximum=weight_range.maximum,
            zero_probability=0.0,
        )

    return SweepWeights(**sampled)


def generate_configurations(
    *,
    num_random_configurations: int,
    rng: np.random.Generator,
    include_anchors: bool,
) -> list[SweepWeights]:
    """Generate the CLSM hyperparameter sweep configurations."""
    configurations = []

    if include_anchors:
        configurations.extend(ANCHOR_CONFIGURATIONS)

    configurations.extend(
        sample_weights(rng)
        for _ in range(num_random_configurations)
    )

    unique = {}
    for weights in configurations:
        key = tuple(
            round(getattr(weights, name), 12)
            for name in WEIGHT_NAMES
        )
        unique[key] = weights

    return list(unique.values())


# =============================================================================
# External command execution
# =============================================================================

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
) -> list[str]:
    """Build the command invoking the training module."""
    command = module_command(
        train_module,
        "--preset",
        "full",
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

    for name in WEIGHT_NAMES:
        command.extend(
            [
                f"--weight-{name}",
                str(getattr(weights, name)),
            ]
        )

    return command


def build_evaluation_command(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    batch_size: int,
    rollout_horizons: Sequence[int],
    device: str,
) -> list[str]:
    """Build the command invoking the evaluation module."""
    return module_command(
        "scripts.evaluation",
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        str(data_dir),
        "--split",
        "all",
        "--batch-size",
        str(batch_size),
        "--rollout-horizons",
        *[str(horizon) for horizon in rollout_horizons],
        "--device",
        device,
    )


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


def load_test_objectives(metrics_path: Path) -> dict[str, float]:
    """Load the five paper metrics from one evaluation file."""
    with metrics_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if "test" not in payload:
        raise KeyError(f"'test' split missing from {metrics_path}.")

    objectives = {}
    for metric_name in OBJECTIVES:
        value = payload["test"].get(metric_name)
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
    """Compute all pairwise fronts and assign stable global labels.

    A configuration receives one label, such as P1, and keeps that label in
    every figure in which it appears. Labels are assigned by increasing
    configuration identifier over the union of all pairwise Pareto sets.
    """
    pair_masks = {}
    pair_inlier_masks = {}
    pair_representative_masks = {}
    pareto_configuration_ids = set()

    for x_metric, y_metric in OBJECTIVE_PAIRS:
        x_values = np.asarray(
            [float(record[x_metric]) for record in records],
            dtype=float,
        )
        y_values = np.asarray(
            [float(record[y_metric]) for record in records],
            dtype=float,
        )

        inliers = pairwise_inlier_mask(
            x_values,
            y_values,
            iqr_multiplier=outlier_iqr_multiplier,
        )
        pair_inlier_masks[(x_metric, y_metric)] = inliers

        inlier_records = [
            record
            for record, keep in zip(records, inliers, strict=True)
            if keep
        ]

        inlier_pareto = pairwise_pareto_mask(
            inlier_records,
            x_metric=x_metric,
            y_metric=y_metric,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

        mask = np.zeros(len(records), dtype=bool)
        mask[np.flatnonzero(inliers)] = inlier_pareto
        pair_masks[(x_metric, y_metric)] = mask

        representative = representative_pareto_mask(
            records,
            mask,
            x_metric=x_metric,
            y_metric=y_metric,
            minimum_distance=label_minimum_distance,
        )
        pair_representative_masks[(x_metric, y_metric)] = representative

        pareto_configuration_ids.update(
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
            sorted(pareto_configuration_ids),
            start=1,
        )
    }

    return pair_masks, pair_inlier_masks, pair_representative_masks, labels


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
    """Return configurations retained for one pairwise Pareto analysis.

    Tukey fences are estimated independently on both metrics using all
    configurations. Points outside either fence are excluded before the
    Pareto set is computed. A negative multiplier disables filtering.
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

    pareto_solutions = [
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
        "pareto_configuration_ids": [
            solution["configuration_id"]
            for solution in pareto_solutions
        ],
        "pareto_solutions": pareto_solutions,
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
    """Generate the ten pairwise Pareto figures."""
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



def save_global_pareto_configurations(
    records: Sequence[Mapping[str, object]],
    labels: Mapping[int, str],
    *,
    output_dir: Path,
) -> None:
    """Save the global P-label-to-configuration correspondence."""
    rows = global_pareto_configuration_records(records, labels)

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
                if key != "pareto_configuration_ids"
            },
            "pareto_configuration_ids": ",".join(
                str(configuration_id)
                for configuration_id in summary["pareto_configuration_ids"]
            ),
        }
        for summary in summaries
    ]

    save_records(
        csv_records,
        json_path=output_dir / "pareto_2d_summary_flat.json",
        csv_path=output_dir / "pareto_2d_summary.csv",
    )


def print_pareto_solutions(
    summaries: Sequence[Mapping[str, object]],
) -> None:
    """Print all pairwise Pareto solutions and their six weights."""
    for summary in summaries:
        print()
        print(
            f"{summary['x_metric']} vs {summary['y_metric']} "
            f"({summary['n_pareto']} Pareto solutions; "
            f"{summary['n_labeled_pareto']} labeled representatives; "
            f"{summary['n_excluded_outliers']} outliers excluded before analysis)"
        )
        print("-" * 96)

        for solution in summary["pareto_solutions"]:
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

    parser.add_argument("--num-configurations", type=int, default=100)
    parser.add_argument("--sweep-seed", type=int, default=12345)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--train-module",
        required=True,
        help="Training module to execute.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/constraint-sweep-analysis"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10],
    )
    parser.add_argument("--device", default="cuda")
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
        default=0.0,
    )
    parser.add_argument(
        "--pareto-relative-tolerance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--plot-outlier-iqr-multiplier",
        type=float,
        default=1.0,
        help=(
            "Exclude configurations outside Tukey fences on either axis before "
            "computing each 2-D Pareto front. Use a negative value to disable "
            "filtering."
        ),
    )
    parser.add_argument(
        "--pareto-label-minimum-distance",
        type=float,
        default=0.01,
        help=(
            "Minimum Euclidean distance between labeled Pareto solutions "
            "after min-max normalization of the displayed metrics. The full "
            "front remains plotted; only representative solutions are named. "
            "Use 0 to label every Pareto solution."
        ),
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
            "Generate the ten pairwise Pareto figures directly from the "
            "existing sweep_configuration_results.json file."
        ),
    )

    return parser


# =============================================================================
# Sweep orchestration
# =============================================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.analyze_only and args.figures_only:
        parser.error("--analyze-only and --figures-only are mutually exclusive.")

    if args.num_configurations < 0:
        parser.error("--num-configurations must be non-negative.")
    if not args.model_seeds:
        parser.error("At least one model seed must be provided.")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("--model-seeds must be unique.")
    if 5 not in args.rollout_horizons:
        parser.error(
            "The five-metric analysis requires rollout horizon 5."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        records = load_configuration_results(args.output_dir)
        figure_paths, summaries, labels = generate_analysis_figures(
            records,
            output_dir=args.output_dir / "figures",
            absolute_tolerance=args.pareto_absolute_tolerance,
            relative_tolerance=args.pareto_relative_tolerance,
            outlier_iqr_multiplier=args.plot_outlier_iqr_multiplier,
            label_minimum_distance=args.pareto_label_minimum_distance,
        )
        save_pareto_summaries(
            summaries,
            output_dir=args.output_dir,
        )
        save_global_pareto_configurations(
            records,
            labels,
            output_dir=args.output_dir,
        )
        print_pareto_solutions(summaries)

        print()
        print_separator()
        print("FIGURE GENERATION COMPLETED")
        print(f"Configurations : {len(records)}")
        print(f"Pareto fronts  : {len(summaries)}")
        for summary in summaries:
            print(
                f"  {summary['x_metric']} vs {summary['y_metric']}: "
                f"{summary['n_pareto']}/{summary['n_configurations']} "
                f"({summary['pareto_fraction']:.3f})"
            )
        print(f"Figures dir    : {args.output_dir / 'figures'}")
        print_separator()
        return

    rng = np.random.default_rng(args.sweep_seed)
    configurations = generate_configurations(
        num_random_configurations=args.num_configurations,
        rng=rng,
        include_anchors=args.include_anchors,
    )

    manifest_records = []
    for configuration_id, weights in enumerate(configurations):
        for model_seed in args.model_seeds:
            run_name = (
                f"constraint-sweep/config-{configuration_id:04d}/"
                f"seed-{model_seed}"
            )
            manifest_records.append(
                {
                    "configuration_id": configuration_id,
                    "run_name": run_name,
                    "model_seed": model_seed,
                    **asdict(weights),
                }
            )

    save_records(
        manifest_records,
        json_path=args.output_dir / "sweep_manifest.json",
        csv_path=args.output_dir / "sweep_manifest.csv",
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

    for manifest_record in progress:
        run_name = str(manifest_record["run_name"])
        model_seed = int(manifest_record["model_seed"])
        progress.set_postfix(run=run_name)

        weights = SweepWeights(
            **{
                name: float(manifest_record[name])
                for name in WEIGHT_NAMES
            }
        )

        run_dir = args.runs_dir / run_name
        checkpoint_path = run_dir / "best.pt"
        metrics_path = (
            run_dir
            / "evaluation"
            / "evaluation_metrics.json"
        )

        if not args.analyze_only:
            if not args.skip_existing or not checkpoint_path.exists():
                run_command(
                    build_train_command(
                        train_module=args.train_module,
                        run_name=run_name,
                        runs_dir=args.runs_dir,
                        data_dir=args.data_dir,
                        model_seed=model_seed,
                        epochs=args.epochs,
                        batch_size=args.train_batch_size,
                        device=args.device,
                        weights=weights,
                    ),
                    title=(
                        f"TRAINING CONFIGURATION "
                        f"{manifest_record['configuration_id']} "
                        f"SEED {model_seed}"
                    ),
                )

            if not args.skip_existing or not metrics_path.exists():
                run_command(
                    build_evaluation_command(
                        checkpoint_path=checkpoint_path,
                        data_dir=args.data_dir,
                        batch_size=args.evaluation_batch_size,
                        rollout_horizons=args.rollout_horizons,
                        device=args.device,
                    ),
                    title=(
                        f"EVALUATING CONFIGURATION "
                        f"{manifest_record['configuration_id']} "
                        f"SEED {model_seed}"
                    ),
                )

        objectives = load_test_objectives(metrics_path)
        run_records.append(
            {
                **manifest_record,
                **objectives,
            }
        )

        save_records(
            run_records,
            json_path=args.output_dir / "sweep_run_results.json",
            csv_path=args.output_dir / "sweep_run_results.csv",
        )

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

    figure_paths, summaries, labels = generate_analysis_figures(
        aggregated_records,
        output_dir=args.output_dir / "figures",
        absolute_tolerance=args.pareto_absolute_tolerance,
        relative_tolerance=args.pareto_relative_tolerance,
        outlier_iqr_multiplier=args.plot_outlier_iqr_multiplier,
        label_minimum_distance=args.pareto_label_minimum_distance,
    )

    save_pareto_summaries(
        summaries,
        output_dir=args.output_dir,
    )
    save_global_pareto_configurations(
        aggregated_records,
        labels,
        output_dir=args.output_dir,
    )
    print_pareto_solutions(summaries)

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