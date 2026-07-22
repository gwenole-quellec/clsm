"""
Visualization utilities for CLSM-Toy.

Author: Gwenolé Quellec
Year: 2026

This module produces publication-oriented and diagnostic visualizations for
CLSM-Toy experiments.

Visualization principles
------------------------
- visualization functions consume artifacts and optional metadata produced by ``evaluation.py``;
- statistical analyses are not silently re-fitted inside plotting functions;
- counterfactual pairs are used only as an evaluation oracle;
- latent-space panels within a figure use a common fitted projection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clsm.datasets import load_metadata


# =============================================================================
# Type aliases
# =============================================================================

MetricValue = float | None
MetricDict = dict[str, MetricValue]


# =============================================================================
# Visualization data structure
# =============================================================================

@dataclass(frozen=True)
class LatentProjection:
    """Standardization and PCA objects fitted on a reference latent space."""

    scaler: StandardScaler
    projector: PCA


# =============================================================================
# Artifact loading
# =============================================================================

def load_encoded(
    path: str | Path,
) -> dict[str, np.ndarray]:
    """Load an encoded dataset saved by evaluation.py."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        return {
            name: archive[name]
            for name in archive.files
        }

def load_metrics(
    path: str | Path,
) -> dict[str, MetricDict]:
    """Load split-wise evaluation metrics."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        return json.load(stream)


def load_history(
    path: str | Path,
) -> pd.DataFrame:
    """Load training history produced by train.py."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def load_cca_analysis(
    path: str | Path,
) -> dict[str, np.ndarray]:
    """
    Load saved CCA artifacts.

    Expected keys are ``canonical_correlations``, ``state_loadings``, and
    ``latent_loadings``.
    """
    arrays = load_encoded(path)

    required = (
        "canonical_correlations",
        "state_loadings",
        "latent_loadings",
    )

    _require_keys(
        arrays,
        required,
        context="CCA analysis",
    )

    return arrays


# =============================================================================
# Figure serialization
# =============================================================================

def save_figure(
    figure: Figure,
    path: str | Path,
) -> Path:
    """Save a figure with a tight bounding box."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        bbox_inches="tight",
    )

    return path


# =============================================================================
# Internal utilities
# =============================================================================

def _require_keys(
    mapping: Mapping[str, object],
    keys: Sequence[str],
    *,
    context: str,
) -> None:
    missing = [
        key
        for key in keys
        if key not in mapping
    ]

    if missing:
        raise KeyError(
            f"Missing {context} fields: {missing}."
        )


def _sample_indices(
    n_points: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if n_points <= max_points:
        return np.arange(n_points)

    generator = np.random.default_rng(
        seed
    )

    return generator.choice(
        n_points,
        size=max_points,
        replace=False,
    )


def _resolve_dimension_names(
    dimension: int,
    names: Sequence[str] | None = None,
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Validate explicit names or generate generic dimension names."""
    if dimension < 1:
        raise ValueError(
            "dimension must be strictly positive."
        )

    if names is None:
        return tuple(
            f"{prefix}_{index}"
            for index in range(dimension)
        )

    resolved = tuple(
        str(name)
        for name in names
    )

    if len(resolved) != dimension:
        raise ValueError(
            f"Expected {dimension} names for '{prefix}', "
            f"but received {len(resolved)}."
        )

    if len(set(resolved)) != dimension:
        raise ValueError(
            f"Names for '{prefix}' must be unique."
        )

    return resolved


def _resolve_dimension_labels(
    names: Sequence[str],
    labels: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate display labels or use dimension names as labels."""
    if labels is None:
        return tuple(names)

    resolved = tuple(
        str(label)
        for label in labels
    )

    if len(resolved) != len(names):
        raise ValueError(
            "Dimension labels and names must have the same length."
        )

    return resolved


def _infer_state_names(
    metrics: Mapping[str, object],
    state_dimension: int,
) -> tuple[str, ...]:
    """Infer state names from probe metric keys when metadata are unavailable."""
    prefix = "state_probe_r2_"
    names = tuple(
        key[len(prefix):]
        for key in metrics
        if (
            key.startswith(prefix)
            and not key.startswith("temporal_")
            and key != "state_probe_r2"
        )
    )

    if len(names) == state_dimension:
        return names

    return _resolve_dimension_names(
        state_dimension,
        prefix="state",
    )


def _metric_value(
    metrics: Mapping[str, object],
    metric_name: str,
) -> float:
    value = metrics.get(
        metric_name
    )

    if value is None:
        return float("nan")

    return float(value)


# =============================================================================
# Latent-space projection
# =============================================================================

def fit_latent_projection(
    reference_latent: np.ndarray,
    *,
    n_components: int = 2,
) -> LatentProjection:
    """Fit a standardized PCA projection on a reference latent dataset."""
    if reference_latent.ndim != 3:
        raise ValueError(
            "reference_latent must have shape (N, T, D)."
        )

    flat = reference_latent.reshape(
        -1,
        reference_latent.shape[-1],
    )

    scaler = StandardScaler()
    standardized = scaler.fit_transform(
        flat
    )

    projector = PCA(
        n_components=n_components
    )
    projector.fit(
        standardized
    )

    return LatentProjection(
        scaler=scaler,
        projector=projector,
    )


def transform_latent(
    latent: np.ndarray,
    projection: LatentProjection,
) -> np.ndarray:
    """Apply a fitted standardized PCA projection to latent trajectories."""
    if latent.ndim != 3:
        raise ValueError(
            "latent must have shape (N, T, D)."
        )

    flat = latent.reshape(
        -1,
        latent.shape[-1],
    )

    transformed = projection.projector.transform(
        projection.scaler.transform(
            flat
        )
    )

    return transformed.reshape(
        latent.shape[0],
        latent.shape[1],
        -1,
    )


# =============================================================================
# Primitive visualization functions
# =============================================================================

def plot_observation_views(
    observation: np.ndarray,
    counterfactual_observation: np.ndarray,
    *,
    episode_index: int = 0,
    feature_indices: tuple[int, int] = (0, 1),
    feature_labels: Sequence[str] | None = None,
    connector_alpha: float = 0.18,
    connector_linewidth: float = 0.6,
    ax: Axes | None = None,
) -> Axes:
    """Plot two nuisance-conditioned views of the same trajectory."""
    if observation.ndim != 3:
        raise ValueError(
            "observation must have shape (N, T, observation_dim)."
        )

    if counterfactual_observation.shape != observation.shape:
        raise ValueError(
            "counterfactual_observation must match observation."
        )

    if not 0 <= episode_index < observation.shape[0]:
        raise IndexError(
            "episode_index is out of range."
        )

    first, second = feature_indices
    observation_dimension = observation.shape[-1]

    if (
        first < 0
        or second < 0
        or first >= observation_dimension
        or second >= observation_dimension
    ):
        raise IndexError(
            "feature_indices exceed observation dimension."
        )

    default_feature_names = _resolve_dimension_names(
        observation_dimension,
        prefix="observation",
    )

    if feature_labels is None:
        first_label = default_feature_names[first]
        second_label = default_feature_names[second]
    else:
        resolved_labels = tuple(
            str(label)
            for label in feature_labels
        )

        if len(resolved_labels) == observation_dimension:
            first_label = resolved_labels[first]
            second_label = resolved_labels[second]
        elif len(resolved_labels) == len(feature_indices):
            first_label, second_label = resolved_labels
        else:
            raise ValueError(
                "feature_labels must contain either one label per "
                "observation dimension or one label per displayed feature."
            )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(5, 5)
        )

    view_a = observation[
        episode_index
    ]
    view_b = counterfactual_observation[
        episode_index
    ]

    for point_a, point_b in zip(
        view_a,
        view_b,
        strict=True,
    ):
        ax.plot(
            [
                point_a[first],
                point_b[first],
            ],
            [
                point_a[second],
                point_b[second],
            ],
            linewidth=connector_linewidth,
            alpha=connector_alpha,
            zorder=0,
        )

    ax.plot(
        view_a[:, first],
        view_a[:, second],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="View A",
        zorder=2,
    )

    ax.plot(
        view_b[:, first],
        view_b[:, second],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Counterfactual view B",
        zorder=2,
    )

    ax.set_xlabel(
        first_label
    )
    ax.set_ylabel(
        second_label
    )
    ax.set_title(
        "Two measurements of the same physical trajectory"
    )
    ax.legend()

    return ax


def plot_latent_space(
    latent: np.ndarray,
    true_state: np.ndarray,
    nuisance_id: np.ndarray,
    *,
    projection: LatentProjection | None = None,
    state_names: Sequence[str] | None = None,
    state_labels: Sequence[str] | None = None,
    color_by: str | int = 0,
    max_points: int = 5_000,
    seed: int = 42,
    ax: Axes | None = None,
) -> Axes:
    """Plot a shared PCA projection colored by a state dimension or nuisance."""
    if latent.shape[:2] != true_state.shape[:2]:
        raise ValueError(
            "latent and true_state must share episode and time axes."
        )

    if nuisance_id.shape != (
        latent.shape[0],
    ):
        raise ValueError(
            "nuisance_id must have shape (N,)."
        )

    resolved_state_names = _resolve_dimension_names(
        true_state.shape[-1],
        state_names,
        prefix="state",
    )
    resolved_state_labels = _resolve_dimension_labels(
        resolved_state_names,
        state_labels,
    )

    if projection is None:
        projection = fit_latent_projection(
            latent
        )

    projected = transform_latent(
        latent,
        projection,
    )

    flat_projection = projected.reshape(
        -1,
        projected.shape[-1],
    )
    flat_state = true_state.reshape(
        -1,
        true_state.shape[-1],
    )

    if color_by == "nuisance":
        colors = np.repeat(
            nuisance_id,
            latent.shape[1],
        )
        color_label = "Nuisance ID"
        color_title = "nuisance"
    else:
        if isinstance(color_by, int):
            state_index = color_by
        else:
            try:
                state_index = resolved_state_names.index(
                    color_by
                )
            except ValueError as error:
                raise ValueError(
                    f"Unknown state dimension '{color_by}'. "
                    f"Expected one of {resolved_state_names} "
                    "or 'nuisance'."
                ) from error

        if not 0 <= state_index < true_state.shape[-1]:
            raise IndexError(
                "The selected state dimension is out of range."
            )

        colors = flat_state[
            :,
            state_index,
        ]
        color_label = resolved_state_labels[
            state_index
        ]
        color_title = resolved_state_names[
            state_index
        ]

    indices = _sample_indices(
        flat_projection.shape[0],
        max_points,
        seed,
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6, 5)
        )

    scatter = ax.scatter(
        flat_projection[
            indices,
            0,
        ],
        flat_projection[
            indices,
            1,
        ],
        c=colors[indices],
        s=10,
        alpha=0.7,
    )

    explained = (
        projection.projector.explained_variance_ratio_
    )

    ax.set_xlabel(
        f"PC1 ({100.0 * explained[0]:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({100.0 * explained[1]:.1f}%)"
    )
    ax.set_title(
        f"Learned latent space colored by {color_title}"
    )

    colorbar_axis = inset_axes(
        ax,
        width="4%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )

    colorbar = ax.figure.colorbar(
        scatter,
        cax=colorbar_axis,
    )
    colorbar.set_label(
        color_label,
    )

    return ax


def plot_latent_trajectory(
    latent: np.ndarray,
    *,
    episode_index: int = 0,
    projection: LatentProjection | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot one temporally ordered latent trajectory in PCA space."""
    if not 0 <= episode_index < latent.shape[0]:
        raise IndexError(
            "episode_index is out of range."
        )

    if projection is None:
        projection = fit_latent_projection(
            latent
        )

    projected = transform_latent(
        latent,
        projection,
    )

    trajectory = projected[
        episode_index
    ]

    if ax is None:
        _, ax = plt.subplots(
            figsize=(5, 5)
        )

    time = np.arange(
        trajectory.shape[0]
    )

    scatter = ax.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=time,
        s=28,
        zorder=2,
    )

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        linewidth=1.2,
        alpha=0.8,
        zorder=1,
    )

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        marker="s",
        s=65,
        label="Start",
    )

    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        marker="X",
        s=75,
        label="End",
    )

    explained = (
        projection.projector.explained_variance_ratio_
    )

    ax.set_xlabel(
        f"PC1 ({100.0 * explained[0]:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({100.0 * explained[1]:.1f}%)"
    )
    ax.set_title(
        "Temporally ordered latent trajectory"
    )
    ax.legend()

    colorbar_axis = inset_axes(
        ax,
        width="4%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )

    colorbar = ax.figure.colorbar(
        scatter,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Time index",
    )

    return ax


def plot_counterfactual_alignment(
    latent: np.ndarray,
    counterfactual_latent: np.ndarray,
    *,
    projection: LatentProjection | None = None,
    episode_index: int | None = None,
    max_pairs: int = 1_000,
    seed: int = 42,
    connector_alpha: float = 0.07,
    connector_linewidth: float = 0.35,
    ax: Axes | None = None,
) -> Axes:
    """Plot paired counterfactual representations in a shared PCA space."""
    if latent.shape != counterfactual_latent.shape:
        raise ValueError(
            "latent and counterfactual_latent must have identical shapes."
        )

    if projection is None:
        projection = fit_latent_projection(
            np.concatenate(
                [
                    latent,
                    counterfactual_latent,
                ],
                axis=0,
            )
        )

    projected_a = transform_latent(
        latent,
        projection,
    )

    projected_b = transform_latent(
        counterfactual_latent,
        projection,
    )

    if episode_index is not None:
        if not 0 <= episode_index < latent.shape[0]:
            raise IndexError(
                "episode_index is out of range."
            )

        points_a = projected_a[
            episode_index
        ]
        points_b = projected_b[
            episode_index
        ]
    else:
        flat_a = projected_a.reshape(
            -1,
            2,
        )

        flat_b = projected_b.reshape(
            -1,
            2,
        )

        indices = _sample_indices(
            flat_a.shape[0],
            max_pairs,
            seed,
        )

        points_a = flat_a[
            indices
        ]
        points_b = flat_b[
            indices
        ]

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6, 5)
        )

    for point_a, point_b in zip(
        points_a,
        points_b,
        strict=True,
    ):
        ax.plot(
            [
                point_a[0],
                point_b[0],
            ],
            [
                point_a[1],
                point_b[1],
            ],
            linewidth=connector_linewidth,
            alpha=connector_alpha,
            zorder=0,
        )

    ax.scatter(
        points_a[:, 0],
        points_a[:, 1],
        s=7,
        alpha=0.62,
        label="View A",
        zorder=2,
    )

    ax.scatter(
        points_b[:, 0],
        points_b[:, 1],
        s=7,
        alpha=0.62,
        label="Counterfactual view B",
        zorder=2,
    )

    explained = (
        projection.projector.explained_variance_ratio_
    )

    ax.set_xlabel(
        f"PC1 ({100.0 * explained[0]:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({100.0 * explained[1]:.1f}%)"
    )
    ax.set_title(
        "Counterfactual views align in latent space"
    )
    ax.legend()

    return ax


def plot_state_probe_comparison(
    metrics: Mapping[str, MetricValue],
    *,
    state_names: Sequence[str],
    state_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Compare instantaneous and temporal probes for arbitrary state dimensions."""
    resolved_state_names = tuple(
        state_names
    )
    resolved_state_labels = _resolve_dimension_labels(
        resolved_state_names,
        state_labels,
    )

    instantaneous = np.asarray(
        [
            _metric_value(
                metrics,
                f"state_probe_r2_{name}",
            )
            for name in resolved_state_names
        ],
        dtype=float,
    )

    temporal = np.asarray(
        [
            _metric_value(
                metrics,
                f"temporal_state_probe_r2_{name}",
            )
            for name in resolved_state_names
        ],
        dtype=float,
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(7.5, 4.6)
        )

    x_positions = np.arange(
        len(resolved_state_names)
    )
    width = 0.36

    ax.bar(
        x_positions - width / 2.0,
        instantaneous,
        width=width,
        label=r"Instantaneous: $z_t$",
        edgecolor="black",
        linewidth=0.7,
    )

    ax.bar(
        x_positions + width / 2.0,
        temporal,
        width=width,
        label=r"Temporal: $[z_t,z_{t+1}]$",
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        x_positions,
        labels=resolved_state_labels,
    )
    ax.set_ylabel(
        r"Probe $R^2$"
    )
    ax.set_title(
        "Instantaneous versus temporally distributed state information"
    )
    ax.grid(
        axis="y",
        alpha=0.22,
    )
    ax.legend()

    return ax


def plot_cca_correlations(
    canonical_correlations: np.ndarray,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot canonical correlations between latent and physical state spaces."""
    correlations = np.asarray(
        canonical_correlations,
        dtype=float,
    ).reshape(-1)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6.4, 4.4)
        )

    x_positions = np.arange(
        1,
        correlations.size + 1,
    )

    bars = ax.bar(
        x_positions,
        correlations,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(
        0.9,
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
    )

    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1.1,
        alpha=0.65,
    )

    ax.set_xticks(
        x_positions,
        labels=[
            rf"$\rho_{index}$"
            for index in x_positions
        ],
    )

    ax.set_ylim(
        0.0,
        1.05,
    )
    ax.set_xlabel(
        "Canonical component"
    )
    ax.set_ylabel(
        "Canonical correlation"
    )
    ax.set_title(
        "Linear alignment between latent and physical state spaces"
    )
    ax.grid(
        axis="y",
        alpha=0.22,
    )

    for bar, value in zip(
        bars,
        correlations,
        strict=True,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value + 0.025,
            f"{value:.2f}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    return ax


def plot_training_history(
    history: pd.DataFrame,
    *,
    metric: str = "selection_total",
    ax: Axes | None = None,
) -> Axes:
    """Plot training and validation curves for one metric."""
    train_column = f"train_{metric}"
    validation_column = f"validation_{metric}"

    _require_keys(
        history,
        (
            "epoch",
            train_column,
            validation_column,
        ),
        context="history",
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6, 4)
        )

    ax.plot(
        history["epoch"],
        history[train_column],
        label="Train",
    )

    ax.plot(
        history["epoch"],
        history[validation_column],
        label="Validation",
    )

    ax.set_xlabel(
        "Epoch"
    )
    ax.set_ylabel(
        metric.replace(
            "_",
            " ",
        ).title()
    )
    ax.set_title(
        "Training history"
    )
    ax.grid(
        alpha=0.22,
    )
    ax.legend()

    return ax


def plot_training_components(
    history: pd.DataFrame,
    *,
    components: Sequence[str] = (
        "observation",
        "predictive",
        "temporal",
        "structural",
        "minimality",
        "invariance",
    ),
) -> Figure:
    """Plot separate raw training and validation curves for loss components."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13, 7),
    )

    if len(components) != axes.size:
        raise ValueError(
            f"Expected exactly {axes.size} components, "
            f"got {len(components)}."
        )
    for axis, component in zip(
        axes.ravel(),
        components,
        strict=True,
    ):
        train_column = (
            f"train_raw_{component}"
        )
        validation_column = (
            f"validation_raw_{component}"
        )

        if train_column not in history:
            axis.set_visible(
                False
            )
            continue

        axis.plot(
            history["epoch"],
            history[train_column],
            label="Train",
        )

        if validation_column in history:
            axis.plot(
                history["epoch"],
                history[validation_column],
                label="Validation",
            )

        axis.set_title(
            component.replace(
                "_",
                " ",
            ).title()
        )
        axis.set_xlabel(
            "Epoch"
        )
        axis.grid(
            alpha=0.22,
        )

    visible_axes = [
        axis
        for axis in axes.ravel()
        if axis.get_visible()
    ]

    if visible_axes:
        visible_axes[0].legend()

    figure.tight_layout()

    return figure


def plot_adversarial_dynamics(
    history: pd.DataFrame,
    *,
    chance_level: float | None = None,
) -> Figure:
    """Plot adversarial cross-entropy and classification accuracy."""
    required = (
        "epoch",
        "train_raw_invariance",
        "validation_raw_invariance",
        "train_nuisance_adversarial_accuracy",
        "validation_nuisance_adversarial_accuracy",
    )

    _require_keys(
        history,
        required,
        context="adversarial history",
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.3),
    )

    axes[0].plot(
        history["epoch"],
        history["train_raw_invariance"],
        label="Train",
    )

    axes[0].plot(
        history["epoch"],
        history["validation_raw_invariance"],
        label="Validation",
    )

    axes[0].set_xlabel(
        "Epoch"
    )
    axes[0].set_ylabel(
        "Adversarial cross-entropy"
    )
    axes[0].set_title(
        "Nuisance adversary loss"
    )
    axes[0].grid(
        alpha=0.22,
    )
    axes[0].legend()

    axes[1].plot(
        history["epoch"],
        history["train_nuisance_adversarial_accuracy"],
        label="Train",
    )

    axes[1].plot(
        history["epoch"],
        history["validation_nuisance_adversarial_accuracy"],
        label="Validation",
    )

    if chance_level is not None:
        axes[1].axhline(
            chance_level,
            linestyle="--",
            linewidth=1.2,
            label="Chance",
        )

    axes[1].set_xlabel(
        "Epoch"
    )
    axes[1].set_ylabel(
        "Adversarial accuracy"
    )
    axes[1].set_title(
        "Nuisance predictability during training"
    )
    axes[1].grid(
        alpha=0.22,
    )
    axes[1].legend()

    figure.tight_layout()

    return figure


# =============================================================================
# Composite publication figures
# =============================================================================

def create_environment_representation_figure(
    encoded: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, object] | None = None,
    episode_index: int = 0,
) -> Figure:
    """Create a four-panel environment and representation figure."""
    required = (
        "observation",
        "counterfactual_observation",
        "latent",
        "counterfactual_latent",
        "true_state",
        "nuisance_id",
    )

    _require_keys(
        encoded,
        required,
        context="encoded dataset",
    )

    metadata = {} if metadata is None else metadata

    state_names = _resolve_dimension_names(
        encoded["true_state"].shape[-1],
        metadata.get("state_names"),
        prefix="state",
    )
    state_labels = _resolve_dimension_labels(
        state_names,
        metadata.get("state_labels"),
    )
    observation_labels = metadata.get(
        "observation_labels"
    )
    color_by = metadata.get(
        "environment_color_by",
        state_names[0],
    )

    projection = fit_latent_projection(
        np.concatenate(
            [
                encoded["latent"],
                encoded["counterfactual_latent"],
            ],
            axis=0,
        )
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    plot_observation_views(
        encoded["observation"],
        encoded["counterfactual_observation"],
        episode_index=episode_index,
        feature_labels=observation_labels,
        ax=axes[0, 0],
    )

    plot_latent_space(
        encoded["latent"],
        encoded["true_state"],
        encoded["nuisance_id"],
        projection=projection,
        state_names=state_names,
        state_labels=state_labels,
        color_by=color_by,
        ax=axes[0, 1],
    )

    plot_counterfactual_alignment(
        encoded["latent"],
        encoded["counterfactual_latent"],
        projection=projection,
        max_pairs=1_000,
        ax=axes[1, 0],
    )

    plot_latent_trajectory(
        encoded["latent"],
        episode_index=episode_index,
        projection=projection,
        ax=axes[1, 1],
    )

    axes[0, 0].set_title(
        "A. Counterfactual measurements of one trajectory"
    )
    axes[0, 1].set_title(
        "B. Latent organization follows physical position"
    )
    axes[1, 0].set_title(
        "C. Counterfactual views align in latent space"
    )
    axes[1, 1].set_title(
        "D. Temporal organization of one latent trajectory"
    )

    # synchronizing the limits
    axis_b = axes[0, 1]
    axis_c = axes[1, 0]
    axis_d = axes[1, 1]
    xlim = axis_b.get_xlim()
    ylim = axis_b.get_ylim()
    xticks = axis_b.get_xticks()
    yticks = axis_b.get_yticks()
    box_aspect = (
        (ylim[1] - ylim[0])
        / (xlim[1] - xlim[0])
    )

    for axis in (axis_b, axis_c, axis_d):
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
        axis.set_box_aspect(box_aspect)

    figure.tight_layout()

    return figure


def create_state_analysis_figure(
    metrics: Mapping[str, MetricValue],
    cca_analysis: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, object] | None = None,
) -> Figure:
    """Create a combined state-accessibility figure."""
    _require_keys(
        cca_analysis,
        (
            "canonical_correlations",
            "state_loadings",
            "latent_loadings",
        ),
        context="CCA analysis",
    )

    metadata = {} if metadata is None else metadata

    state_dimension = cca_analysis[
        "state_loadings"
    ].shape[0]
    latent_dimension = cca_analysis[
        "latent_loadings"
    ].shape[0]

    inferred_state_names = _infer_state_names(
        metrics,
        state_dimension,
    )
    state_names = _resolve_dimension_names(
        state_dimension,
        metadata.get(
            "state_names",
            inferred_state_names,
        ),
        prefix="state",
    )
    state_labels = _resolve_dimension_labels(
        state_names,
        metadata.get("state_labels"),
    )
    latent_names = _resolve_dimension_names(
        latent_dimension,
        metadata.get("latent_names"),
        prefix="z",
    )
    latent_labels = _resolve_dimension_labels(
        latent_names,
        metadata.get("latent_labels"),
    )

    figure = plt.figure(
        figsize=(12, 9)
    )

    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(
            1.0,
            1.15,
        ),
    )

    axis_probe = figure.add_subplot(
        grid[0, 0]
    )

    axis_correlations = figure.add_subplot(
        grid[0, 1]
    )

    axis_state = figure.add_subplot(
        grid[1, 0]
    )

    axis_latent = figure.add_subplot(
        grid[1, 1]
    )

    plot_state_probe_comparison(
        metrics,
        state_names=state_names,
        state_labels=state_labels,
        ax=axis_probe,
    )

    plot_cca_correlations(
        cca_analysis[
            "canonical_correlations"
        ],
        ax=axis_correlations,
    )

    correlations = cca_analysis[
        "canonical_correlations"
    ]

    for axis, matrix, row_labels, title in (
        (
            axis_state,
            cca_analysis[
                "state_loadings"
            ],
            state_labels,
            "Physical-state composition of canonical variates",
        ),
        (
            axis_latent,
            cca_analysis[
                "latent_loadings"
            ],
            latent_labels,
            "Latent-dimension loadings on canonical variates",
        ),
    ):
        maximum = max(
            float(
                np.max(
                    np.abs(matrix)
                )
            ),
            1e-6,
        )

        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
        )

        axis.set_xticks(
            np.arange(
                matrix.shape[1]
            ),
            labels=[
                (
                    rf"$\mathrm{{CC}}_{index + 1}$"
                    "\n"
                    rf"$\rho={correlations[index]:.2f}$"
                )
                for index in range(
                    matrix.shape[1]
                )
            ],
        )

        axis.set_yticks(
            np.arange(
                matrix.shape[0]
            ),
            labels=row_labels[
                :matrix.shape[0]
            ],
        )

        axis.set_title(
            title
        )

        for row in range(
            matrix.shape[0]
        ):
            for column in range(
                matrix.shape[1]
            ):
                value = matrix[
                    row,
                    column,
                ]

                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=(
                        "white"
                        if abs(value)
                        > 0.55 * maximum
                        else "black"
                    ),
                    fontsize=8,
                )

        figure.colorbar(
            image,
            ax=axis,
            pad=0.025,
            label="Correlation loading",
        )

    axis_probe.set_title(
        "A. Instantaneous and temporal state probes"
    )
    axis_correlations.set_title(
        "B. Canonical correlations"
    )
    axis_state.set_title(
        "C. Physical-state composition of canonical variates"
    )
    axis_latent.set_title(
        "D. Latent-dimension loadings on canonical variates"
    )

    figure.tight_layout()

    return figure


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create CLSM-Toy visualizations.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    environment = subparsers.add_parser(
        "environment",
        help="Create the four-panel environment and representation figure.",
    )
    environment.add_argument(
        "--encoded",
        required=True,
    )
    environment.add_argument(
        "--output",
        required=True,
    )
    environment.add_argument(
        "--metadata",
        help="Optional JSON file describing state and observation dimensions.",
    )
    environment.add_argument(
        "--episode-index",
        type=int,
        default=0,
    )

    state = subparsers.add_parser(
        "state",
        help="Create the state accessibility and CCA figure.",
    )
    state.add_argument(
        "--metrics",
        required=True,
    )
    state.add_argument(
        "--cca",
        required=True,
    )
    state.add_argument(
        "--metadata",
        help="Optional JSON file describing state and latent dimensions.",
    )
    state.add_argument(
        "--split",
        default="test",
    )
    state.add_argument(
        "--output",
        required=True,
    )

    history = subparsers.add_parser(
        "history",
        help="Plot one training-history metric.",
    )
    history.add_argument(
        "--history",
        required=True,
    )
    history.add_argument(
        "--metric",
        default="selection_total",
    )
    history.add_argument(
        "--output",
        required=True,
    )

    components = subparsers.add_parser(
        "components",
        help="Plot separate training loss components.",
    )
    components.add_argument(
        "--history",
        required=True,
    )
    components.add_argument(
        "--output",
        required=True,
    )

    adversarial = subparsers.add_parser(
        "adversarial",
        help="Plot adversarial training dynamics.",
    )
    adversarial.add_argument(
        "--history",
        required=True,
    )
    adversarial.add_argument(
        "--chance-level",
        type=float,
        default=None,
    )
    adversarial.add_argument(
        "--output",
        required=True,
    )

    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """Dispatch the requested visualization command."""
    args = build_arg_parser().parse_args()

    # -------------------------------------------------------------------------
    # Environment representation figure
    # -------------------------------------------------------------------------

    if args.command == "environment":
        encoded = load_encoded(
            args.encoded
        )

        metadata = (
            load_metadata(args.metadata)
            if args.metadata is not None
            else None
        )

        figure = create_environment_representation_figure(
            encoded,
            metadata=metadata,
            episode_index=args.episode_index,
        )

        save_figure(
            figure,
            args.output,
        )
        plt.close(figure)

    # -------------------------------------------------------------------------
    # State accessibility figure
    # -------------------------------------------------------------------------

    elif args.command == "state":
        split_metrics = load_metrics(
            args.metrics
        )

        if args.split not in split_metrics:
            raise KeyError(
                f"Split '{args.split}' is missing."
            )

        cca_analysis = load_cca_analysis(
            args.cca
        )

        metadata = (
            load_metadata(args.metadata)
            if args.metadata is not None
            else None
        )

        figure = create_state_analysis_figure(
            split_metrics[
                args.split
            ],
            cca_analysis,
            metadata=metadata,
        )

        save_figure(
            figure,
            args.output,
        )
        plt.close(figure)

    # -------------------------------------------------------------------------
    # Training history
    # -------------------------------------------------------------------------

    elif args.command == "history":
        history = load_history(
            args.history
        )

        figure, axis = plt.subplots(
            figsize=(6, 4)
        )

        plot_training_history(
            history,
            metric=args.metric,
            ax=axis,
        )

        figure.tight_layout()

        save_figure(
            figure,
            args.output,
        )
        plt.close(figure)

    # -------------------------------------------------------------------------
    # Training components
    # -------------------------------------------------------------------------

    elif args.command == "components":
        history = load_history(
            args.history
        )

        figure = plot_training_components(
            history
        )

        save_figure(
            figure,
            args.output,
        )
        plt.close(figure)

    # -------------------------------------------------------------------------
    # Adversarial dynamics
    # -------------------------------------------------------------------------

    elif args.command == "adversarial":
        history = load_history(
            args.history
        )

        figure = plot_adversarial_dynamics(
            history,
            chance_level=args.chance_level,
        )

        save_figure(
            figure,
            args.output,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
