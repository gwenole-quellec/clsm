"""
Generate a heatmap comparing CLSM presets across representation properties.

Author: Gwenolé Quellec
Year: 2026

This script loads aggregated evaluation results for several CLSM presets,
normalizes each representation metric across models, and produces a color-coded
heatmap summarizing their relative performance. The displayed values correspond
to the original evaluation metrics, while the colors represent normalized
performance (higher is better).

Example
--------

    python -m scripts.representation_heatmap --output representation_heatmap.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Figure configuration
# =============================================================================

PRESETS = {

    # Pure presets
    "Reconstruction": Path(
        "runs/reconstruction-aggregate-evaluation.json"
    ),
    "Predictive": Path(
        "runs/predictive-aggregate-evaluation.json"
    ),
    "Temporal": Path(
        "runs/temporal-aggregate-evaluation.json"
    ),
    "Structural": Path(
        "runs/structural-aggregate-evaluation.json"
    ),

    # Functional presets
    "Reconstruction\n+ predictive": Path(
        "runs/reconstruction_predictive-aggregate-evaluation.json"
    ),
    "Base": Path(
        "runs/base-aggregate-evaluation.json"
    ),
    "Base\n+ invariance": Path(
        "runs/base_invariance-aggregate-evaluation.json"
    ),
    "Base\n+ structural": Path(
        "runs/base_structural-aggregate-evaluation.json"
    ),
    "Base\n+ minimality": Path(
        "runs/base_minimality-aggregate-evaluation.json"
    ),
    "Full": Path(
        "runs/full-aggregate-evaluation.json"
    ),
}

# The third element indicates whether larger values are preferable.
# The order matches the evaluation-metrics section of the manuscript.
METRICS = {
    "Observation\nrecons-\ntruction MSE ↓": (
        "test",
        "observation_mse",
        False,
    ),
    "Observation\nrollout MSE\n(h=5) ↓": (
        "test",
        "rollout_observation_mse_h5",
        False,
    ),
    "Latent\nrollout MSE\n(h=5) ↓": (
        "test",
        "rollout_latent_mse_h5",
        False,
    ),
    "State\nprobe\n$R^2$ ↑": (
        "test",
        "state_probe_r2",
        True,
    ),
    "State\nCCA ↑": (
        "test",
        "state_cca_mean_correlation",
        True,
    ),
    "State\nlinearity\ngap ↓": (
        "test",
        "state_linearity_gap",
        False,
    ),
    "Neighbor-\nhood trust\nworthiness ↑": (
        "test",
        "neighborhood_trustworthiness",
        True,
    ),
    "Counter-\nfactual\nNMSE ↓": (
        "test",
        "counterfactual_normalized_mse",
        False,
    ),
    "Conditional\nnuisance\naccuracy ↓": (
        "test",
        "conditional_nuisance_probe_accuracy",
        False,
    ),
}


# =============================================================================
# Metric formatting
# =============================================================================

def format_raw_value(
    property_name: str,
    value: float,
) -> str:
    """Format original metric values with meaningful units."""

    if property_name == "Observation\nrecons-\ntruction MSE ↓":
        return f"{value:.3f}\nMSE"

    if property_name == "Observation\nrollout MSE\n(h=5) ↓":
        return f"{value:.3f}\nMSE"

    if property_name == "Latent\nrollout MSE\n(h=5) ↓":
        return f"{value:.3f}\nMSE"

    if property_name == "State\nprobe\n$R^2$ ↑":
        return f"{value:.3f}\n$R^2$"

    if property_name == "State\nCCA ↑":
        return f"{value:.3f}\nmean $\\rho$"

    if property_name == "State\nlinearity\ngap ↓":
        return f"{value:.3f}\n$\\Delta_{{\\mathrm{{lin}}}}$"

    if property_name == "Neighbor-\nhood trust\nworthiness ↑":
        return f"{value:.3f}\nscore"

    if property_name == "Counter-\nfactual\nNMSE ↓":
        return f"{value:.4f}\nNMSE"

    if property_name == "Conditional\nnuisance\naccuracy ↓":
        return f"{100.0 * value:.1f}%\naccuracy"

    raise KeyError(
        f"Unknown metric: {property_name}"
    )


# =============================================================================
# Result loading
# =============================================================================

def read_aggregate_mean(
    data: dict,
    *,
    split: str,
    metric: str,
) -> float:
    """
    Read one aggregated metric mean.

    Expected JSON structure:
        data["aggregate"][split][metric]["mean"]
    """
    try:
        metric_data = data["aggregate"][split][metric]
    except KeyError as error:
        available_splits = sorted(
            data.get("aggregate", {}).keys()
        )

        available_metrics = sorted(
            data.get("aggregate", {})
            .get(split, {})
            .keys()
        )

        raise KeyError(
            f"Metric not found: aggregate/{split}/{metric}\n"
            f"Available splits: {available_splits}\n"
            f"Available metrics for '{split}': "
            f"{available_metrics}"
        ) from error

    if isinstance(metric_data, dict):
        if "mean" not in metric_data:
            raise KeyError(
                f"Metric aggregate/{split}/{metric} "
                "does not contain a 'mean' field."
            )

        metric_data = metric_data["mean"]

    if metric_data is None:
        raise ValueError(
            f"Metric aggregate/{split}/{metric} "
            "has no valid mean."
        )

    return float(metric_data)


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a heatmap comparing CLSM presets across representation properties.",
    )
    parser.add_argument(
        "--output",
        default="representation_heatmap.pdf",
        help="Output PDF file."
    )
    args = parser.parse_args()

    values = np.zeros(
        (len(PRESETS), len(METRICS)),
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Load aggregated evaluation metrics
    # -------------------------------------------------------------------------

    for preset_index, (preset_name, filename) in enumerate(
        PRESETS.items()
    ):
        if not filename.exists():
            raise FileNotFoundError(
                f"Missing file for '{preset_name}': {filename}"
            )

        with filename.open(
            "r",
            encoding="utf-8",
        ) as stream:
            data = json.load(stream)

        for metric_index, (
            _display_name,
            (
                split,
                metric_name,
                _higher_is_better,
            ),
        ) in enumerate(METRICS.items()):
            values[
                preset_index,
                metric_index,
            ] = read_aggregate_mean(
                data,
                split=split,
                metric=metric_name,
            )

    # -------------------------------------------------------------------------
    # Normalize each metric independently
    # -------------------------------------------------------------------------

    normalized = np.zeros_like(values)

    for metric_index, (
        _display_name,
        (
            _split,
            _metric_name,
            higher_is_better,
        ),
    ) in enumerate(METRICS.items()):
        column = values[:, metric_index]

        minimum = float(np.min(column))
        maximum = float(np.max(column))
        value_range = maximum - minimum

        if np.isclose(value_range, 0.0):
            normalized[:, metric_index] = 1.0
            continue

        if higher_is_better:
            normalized[:, metric_index] = (
                column - minimum
            ) / value_range
        else:
            normalized[:, metric_index] = (
                maximum - column
            ) / value_range

    # -------------------------------------------------------------------------
    # Build the heatmap    
    # -------------------------------------------------------------------------

    column_labels = list(METRICS.keys())

    figure, axis = plt.subplots(
        figsize=(12.5, 8),
    )

    image = axis.imshow(
        normalized,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_xticks(
        np.arange(len(column_labels)),
        labels=column_labels,
        rotation=0,
        horizontalalignment="center",
    )

    axis.set_yticks(
        np.arange(len(PRESETS)),
        labels=list(PRESETS.keys()),
    )

    axis.tick_params(
        axis="x",
        labelsize=9,
    )

    axis.tick_params(
        axis="y",
        labelsize=10,
    )

    for row_index, preset_name in enumerate(PRESETS):
        for column_index, property_name in enumerate(METRICS.keys()):
            normalized_value = normalized[
                row_index,
                column_index,
            ]

            raw_value = values[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if normalized_value < 0.28
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                format_raw_value(
                    property_name,
                    raw_value,
                ),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=8.8,
                fontweight=(
                    "bold"
                    if preset_name == "Full"
                    else "normal"
                ),
                linespacing=1.15,
            )

    preset_names = list(PRESETS.keys())

    last_pure_index = preset_names.index("Structural")

    axis.axhline(
        last_pure_index + 0.5,
        linewidth=2.0,
        color="black",
    )

    full_index = preset_names.index("Full")

    axis.axhline(
        full_index - 0.5,
        linewidth=2.5,
        color="black",
    )

    full_rectangle = plt.Rectangle(
        (
            -0.5,
            full_index - 0.5,
        ),
        width=len(column_labels),
        height=1.0,
        fill=False,
        linewidth=2.4,
        edgecolor="black",
    )

    axis.add_patch(full_rectangle)

    axis.get_yticklabels()[
        full_index
    ].set_fontweight("bold")

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.045,
        pad=0.035,
    )

    colorbar.set_label(
        "Normalized performance (higher is better)"
    )

    figure.subplots_adjust(
        left=0.28,
        right=0.90,
        top=0.88,
        bottom=0.27,
    )

    # Finalize and save the figure
    output_pdf = Path(args.output)

    figure.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
