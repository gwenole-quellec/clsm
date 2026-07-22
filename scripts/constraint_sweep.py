"""
Run a large-scale CLSM weight sweep and analyze the resulting Pareto frontier.

Author: Gwenolé Quellec
Year: 2026

This script samples CLSM constraint weights, trains and evaluates each
configuration, identifies the nine-dimensional Pareto-optimal set, and
generates summary figures illustrating the trade-offs between representation
properties and the influence of individual constraint weights.

Examples
--------
Run a complete sweep:

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --num-configurations 100 \
        --device cuda

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --analyze-only

    python -m scripts.constraint_sweep \
        --train-module toy.train \
        --figures-only

Outputs are written to ``runs/constraint-sweep-analysis/`` and include the sweep
manifest, aggregated results, Pareto-optimal configurations, and analysis
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from clsm.utils import module_command, print_banner, print_separator


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

# True means that larger values are preferable
OBJECTIVES = {
    "observation_mse": False,
    "rollout_observation_mse_h5": False,
    "rollout_latent_mse_h5": False,
    "state_probe_r2": True,
    "state_cca_mean_correlation": True,
    "state_linearity_gap": False,
    "neighborhood_trustworthiness": True,
    "counterfactual_normalized_mse": False,
    "conditional_nuisance_probe_accuracy": False,
}

DEFAULT_PROJECTIONS = (
    (
        "state_probe_r2",
        "counterfactual_normalized_mse",
        "rollout_observation_mse_h5",
    ),
    (
        "state_cca_mean_correlation",
        "conditional_nuisance_probe_accuracy",
        "neighborhood_trustworthiness",
    ),
    (
        "rollout_observation_mse_h5",
        "state_probe_r2",
        "counterfactual_normalized_mse",
    ),
    (
        "state_linearity_gap",
        "neighborhood_trustworthiness",
        "state_cca_mean_correlation",
    ),
)

WEIGHT_RANGES = {
    "predictive":  WeightRange(0.05, 2.0, 0.05),
    "minimality":  WeightRange(1e-5, 5e-2, 0.25),
    "temporal":    WeightRange(0.01, 1.5, 0.10),
    "observation": WeightRange(0.01, 1.5, 0.10),
    "invariance":  WeightRange(1e-3, 2e-1, 0.20),
    "structural":  WeightRange(1e-3, 2e-1, 0.20),
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

    # Minimality, invariance, and structural regularization are not useful
    # without at least one information-preserving objective
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
    """Load previously aggregated and Pareto-annotated sweep results."""
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
    required = {"is_pareto_9d", *WEIGHT_NAMES, *OBJECTIVES}
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

def oriented_objective_matrix(
    records: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Return objectives in minimization form."""
    return np.asarray(
        [
            [
                float(record[metric_name])
                * (-1.0 if higher_is_better else 1.0)
                for metric_name, higher_is_better in OBJECTIVES.items()
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


# =============================================================================
# Plot labeling utilities
# =============================================================================

def axis_label(metric_name: str) -> str:
    labels = {
        "observation_mse": "Observation reconstruction MSE",
        "rollout_observation_mse_h5": "Observation rollout MSE (h=5)",
        "rollout_latent_mse_h5": "Latent rollout MSE (h=5)",
        "state_probe_r2": r"State probe $R^2$",
        "state_cca_mean_correlation": r"State CCA mean $\rho$",
        "state_linearity_gap": r"State linearity gap $\Delta_{\mathrm{lin}}$",
        "neighborhood_trustworthiness": "Neighborhood trustworthiness",
        "counterfactual_normalized_mse": "Counterfactual NMSE",
        "conditional_nuisance_probe_accuracy": "Conditional nuisance accuracy",
    }
    return labels.get(metric_name, metric_name.replace("_", " "))


def weight_label(weight_name: str) -> str:
    abbreviations = {
        "predictive": "pred",
        "minimality": "min",
        "temporal": "temp",
        "observation": "obs",
        "invariance": "inv",
        "structural": "struct",
    }
    suffix = abbreviations.get(weight_name, weight_name)
    return rf"$\lambda_{{\mathrm{{{suffix}}}}}$"


# =============================================================================
# Correlation analysis
# =============================================================================

def weight_metric_correlation_matrix(
    records: Sequence[Mapping[str, object]],
    *,
    pareto_only: bool,
) -> np.ndarray:
    """Compute Spearman associations with all metrics oriented as better-up."""
    selected = [
        record for record in records
        if not pareto_only or bool(record["is_pareto_9d"])
    ]
    if len(selected) < 3:
        raise ValueError("At least three configurations are required.")
    matrix = np.empty((len(WEIGHT_NAMES), len(OBJECTIVES)), dtype=float)
    for wi, weight_name in enumerate(WEIGHT_NAMES):
        weight_values = np.asarray(
            [float(record[weight_name]) for record in selected], dtype=float
        )
        for mi, (metric_name, higher_is_better) in enumerate(OBJECTIVES.items()):
            metric_values = np.asarray(
                [float(record[metric_name]) for record in selected], dtype=float
            )
            if not higher_is_better:
                metric_values = -metric_values
            if (
                np.allclose(weight_values, weight_values[0])
                or np.allclose(metric_values, metric_values[0])
            ):
                correlation = float("nan")
            else:
                correlation = float(
                    spearmanr(weight_values, metric_values).statistic
                )
            matrix[wi, mi] = correlation
    return matrix


# =============================================================================
# Pareto projection plots
# =============================================================================

def plot_pareto_projection(
    records: Sequence[Mapping[str, object]],
    *,
    x_metric: str,
    y_metric: str,
    color_metric: str,
    output_path: Path,
) -> Path:
    x_values = np.asarray(
        [float(record[x_metric]) for record in records]
    )
    y_values = np.asarray(
        [float(record[y_metric]) for record in records]
    )
    color_values = np.asarray(
        [float(record[color_metric]) for record in records]
    )
    pareto = np.asarray(
        [bool(record["is_pareto_9d"]) for record in records]
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.6))

    axis.scatter(
        x_values[~pareto],
        y_values[~pareto],
        s=28,
        alpha=0.18,
        label="Dominated configurations",
    )

    scatter = axis.scatter(
        x_values[pareto],
        y_values[pareto],
        c=color_values[pareto],
        s=64,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.5,
        label="9-D Pareto set",
    )

    axis.set_xlabel(axis_label(x_metric))
    axis.set_ylabel(axis_label(y_metric))
    axis.grid(alpha=0.22)
    axis.legend()

    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label(axis_label(color_metric))

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_default_projections(
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> list[Path]:
    """Generate the default Pareto projection figures."""
    paths = []

    for index, (x_metric, y_metric, color_metric) in enumerate(
        DEFAULT_PROJECTIONS,
        start=1,
    ):
        path = output_dir / (
            f"pareto_projection_{index:02d}_"
            f"{x_metric}_vs_{y_metric}_"
            f"color_{color_metric}.pdf"
        )
        paths.append(
            plot_pareto_projection(
                records,
                x_metric=x_metric,
                y_metric=y_metric,
                color_metric=color_metric,
                output_path=path,
            )
        )

    return paths
    

# =============================================================================
# Constraint–metric graph
# =============================================================================

def plot_constraint_metric_bipartite_graph(
    records: Sequence[Mapping[str, object]],
    *,
    output_path: Path,
    pareto_only: bool = False,
    minimum_absolute_correlation: float = 0.25,
) -> Path:
    """
    Plot a bipartite graph linking constraint weights to representation
    properties.

    Edge width and opacity encode the absolute Spearman correlation.
    Edge color encodes its sign:

    - positive: increasing the weight is associated with better oriented
      performance;
    - negative: increasing the weight is associated with worse oriented
      performance.

    Parameters
    ----------
    records:
        Aggregated sweep records containing the six weights, nine metrics,
        and ``is_pareto_9d``.
    output_path:
        Destination PNG or PDF path.
    pareto_only:
        Restrict the analysis to Pareto-optimal configurations.
    minimum_absolute_correlation:
        Hide associations weaker than this threshold.
    """
    if not 0.0 <= minimum_absolute_correlation <= 1.0:
        raise ValueError(
            "minimum_absolute_correlation must lie in [0, 1]."
        )

    correlation_matrix = weight_metric_correlation_matrix(
        records,
        pareto_only=pareto_only,
    )

    weight_names = list(
        WEIGHT_NAMES
    )

    metric_names = list(
        OBJECTIVES
    )

    edges: list[
        tuple[
            int,
            int,
            float,
        ]
    ] = []

    for weight_index in range(
        correlation_matrix.shape[0]
    ):
        for metric_index in range(
            correlation_matrix.shape[1]
        ):
            correlation = float(
                correlation_matrix[
                    weight_index,
                    metric_index,
                ]
            )

            if not np.isfinite(
                correlation
            ):
                continue

            if abs(
                correlation
            ) < minimum_absolute_correlation:
                continue

            edges.append(
                (
                    weight_index,
                    metric_index,
                    correlation,
                )
            )

    if not edges:
        raise ValueError(
            "No association exceeds the requested correlation threshold."
        )

    # Fixed bipartite geometry
    metric_node_left_x = 0.8
    weight_node_right_x = 0.1
    edge_gap = 0.01

    weight_y = np.linspace(
        0.88,
        0.12,
        len(weight_names),
    )

    metric_y = np.linspace(
        0.94,
        0.06,
        len(metric_names),
    )

    figure, axis = plt.subplots(
        figsize=(
            12.0,
            8.5,
        )
    )

    axis.set_xlim(
        0.0,
        1.0,
    )
    axis.set_ylim(
        0.0,
        1.0,
    )
    axis.axis(
        "off"
    )

    axis.text(
        weight_node_right_x + 0.006,
        0.985,
        "Constraint\nweights",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=13,
        fontweight="bold",
    )

    axis.text(
        metric_node_left_x - 0.006,
        0.985,
        "Representation properties",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=13,
        fontweight="bold",
    )

    # Nodes
    for weight_index, weight_name in enumerate(
        weight_names
    ):
        axis.text(
            weight_node_right_x,
            weight_y[
                weight_index
            ],
            weight_label(
                weight_name
            ),
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=13,
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.1,
            },
            zorder=4,
        )

    for metric_index, metric_name in enumerate(metric_names):
        axis.text(
            metric_node_left_x,
            metric_y[metric_index],
            axis_label(metric_name),
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.0,
            },
            zorder=4,
        )

    normalization = Normalize(
        vmin=-1.0,
        vmax=1.0,
    )

    colormap = plt.get_cmap(
        "RdYlGn"
    )

    # Draw weak edges first so that strong associations remain visible
    for (
        weight_index,
        metric_index,
        correlation,
    ) in sorted(
        edges,
        key=lambda item: abs(
            item[2]
        ),
    ):
        absolute_correlation = abs(
            correlation
        )

        linewidth = (
            0.6
            + 5.0
            * absolute_correlation
        )

        alpha = (
            0.12
            + 0.78
            * absolute_correlation
        )

        start = (
            weight_node_right_x + edge_gap,
            weight_y[weight_index],
        )

        end = (
            metric_node_left_x - edge_gap,
            metric_y[metric_index],
        )

        # Curvature reduces complete overlap between nearby edges
        vertical_difference = (
            metric_y[
                metric_index
            ]
            - weight_y[
                weight_index
            ]
        )

        curvature = float(
            np.clip(
                0.08
                * np.sign(
                    vertical_difference
                ),
                -0.12,
                0.12,
            )
        )

        edge = FancyArrowPatch(
            start,
            end,
            arrowstyle="-",
            connectionstyle=(
                f"arc3,rad={curvature}"
            ),
            linewidth=linewidth,
            color=colormap(
                normalization(
                    correlation
                )
            ),
            alpha=alpha,
            zorder=1,
        )

        axis.add_patch(
            edge
        )

    scalar_mappable = ScalarMappable(
        norm=normalization,
        cmap=colormap,
    )

    scalar_mappable.set_array(
        []
    )

    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axis,
        fraction=0.025,
        pad=0.05,
    )

    colorbar.set_label(
        "Spearman correlation with oriented performance",
        fontsize=12,
    )

    colorbar.ax.tick_params(
        labelsize=11,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.04,
            1.0,
            1.0,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    return output_path


# =============================================================================
# Figure orchestration
# =============================================================================
    
def generate_analysis_figures(
    records: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate all Pareto and weight-analysis figures from saved records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_default_projections(records, output_dir / "pareto_projections")
    paths.append(
        plot_constraint_metric_bipartite_graph(
            records,
            pareto_only=False,
            minimum_absolute_correlation=0.25,
            output_path=(output_dir / "constraint_metric_graph_all.pdf"),
        )
    )
    paths.append(
        plot_constraint_metric_bipartite_graph(
            records,
            pareto_only=True,
            minimum_absolute_correlation=0.25,
            output_path=(output_dir / "constraint_metric_graph_pareto.pdf"),
        )
    )
    return paths


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a six-weight CLSM sweep, evaluate nine objectives, "
            "and compute the discrete nine-dimensional Pareto set."
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help=(
            "Skip training and evaluation, reload existing per-run evaluation "
            "files, and rebuild the aggregate Pareto analysis."
        ),
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Generate all figures directly from the existing "
            "sweep_configuration_results.json file, without training, "
            "evaluation, aggregation, or Pareto recomputation."
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
            "The nine-objective analysis requires rollout horizon 5."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        records = load_configuration_results(args.output_dir)
        figure_paths = generate_analysis_figures(
            records,
            output_dir=args.output_dir / "figures",
        )
        print()
        print_separator()
        print("FIGURE GENERATION COMPLETED")
        print(f"Configurations : {len(records)}")
        print(
            "Pareto points  : "
            f"{sum(bool(record['is_pareto_9d']) for record in records)}"
        )
        print(f"Figures dir    : {args.output_dir / 'figures'}")
        print("Figures:")
        for path in figure_paths:
            print(f"  {path}")
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
    objective_matrix = oriented_objective_matrix(aggregated_records)
    mask = pareto_mask(
        objective_matrix,
        absolute_tolerance=args.pareto_absolute_tolerance,
        relative_tolerance=args.pareto_relative_tolerance,
    )

    annotated_records = [
        {
            **record,
            "is_pareto_9d": bool(is_pareto),
        }
        for record, is_pareto in zip(
            aggregated_records,
            mask,
            strict=True,
        )
    ]

    save_records(
        annotated_records,
        json_path=(
            args.output_dir
            / "sweep_configuration_results.json"
        ),
        csv_path=(
            args.output_dir
            / "sweep_configuration_results.csv"
        ),
    )

    pareto_records = [
        record
        for record in annotated_records
        if record["is_pareto_9d"]
    ]

    save_records(
        pareto_records,
        json_path=args.output_dir / "pareto_9d.json",
        csv_path=args.output_dir / "pareto_9d.csv",
    )

    projection_paths = generate_analysis_figures(
        annotated_records,
        output_dir=args.output_dir / "figures",
    )

    print()
    print_separator()
    print("SWEEP COMPLETED")
    print(f"Configurations : {len(aggregated_records)}")
    print(f"Model runs     : {len(run_records)}")
    print(f"Pareto points  : {len(pareto_records)}")
    print(
        "Pareto fraction: "
        f"{len(pareto_records) / max(len(aggregated_records), 1):.3f}"
    )
    print(f"Analysis dir   : {args.output_dir}")
    print("Figures:")
    for path in projection_paths:
        print(f"  {path}")
    print_separator()


if __name__ == "__main__":
    main()
