"""
Alignment of independently learned latent spaces.

Author: Gwenolé Quellec
Year: 2026

This module evaluates whether independently trained instances of the same
configuration recover similar latent representations, up to a single global
orthogonal transformation (or, optionally, a similarity transformation).

Analysis principles
-------------------
- use saved ``validation_encoded.npz`` and ``test_encoded.npz`` artifacts;
- align *corresponding observations*, never unordered point clouds;
- estimate all transformations on validation data only;
- quantify residual mismatch on the untouched test split in the full 4D space;
- reuse the manuscript's standardized PCA projection for visualization;
- fit that display projection once on the reference seed and reuse it throughout;
- report results for all seed pairs to reduce dependence on the reference seed.

An orthogonal Procrustes alignment preserves distances within each latent
space. Its matrix may include a reflection, which is a legitimate latent sign
symmetry. The optional ``similarity`` mode additionally allows one *global*
positive scaling factor, but not arbitrary shearing or nonlinear warping.

A small residual is evidence of *empirical reproducibility* up to the chosen
transformation class, not a mathematical proof of latent identifiability.

Example
-------

    python -m scripts.evaluate_seed_alignment \\
        --runs-dir runs \\
        --preset P29 \\
        --seeds 0 1 2 3 4 \\
        --reference-seed 0 \\
        --output-dir identifiability/P29-seed-alignment
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

DEFAULT_PRESET = "P29"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_REFERENCE_SEED = 0

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OUTPUT_DIR = Path("identifiability/P29-seed-alignment")
DEFAULT_ALIGNMENT = "orthogonal"

DEFAULT_MAX_POINTS = 2_000
DEFAULT_VISUALIZATION_SEED = 42
DEFAULT_MAX_ERROR = 1e-6

NORMALIZATION_EPSILON = 1e-12
ALIGNMENT_MODES = ("orthogonal", "similarity")

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# =============================================================================
# Analysis data structures
# =============================================================================

@dataclass(frozen=True)
class LatentDataset:
    """Saved split and the coordinates for a given trained seed."""

    seed: int
    split: str
    path: Path
    arrays: dict[str, np.ndarray]

    @property
    def latent(self) -> np.ndarray:
        """Return float64 latent trajectories with shape (N, T, D)."""
        return np.asarray(
            self.arrays["latent"],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class GlobalAlignment:
    """Row-vector transform: aligned = scale * latent @ rotation + shift."""

    rotation: np.ndarray
    shift: np.ndarray
    scale: float
    determinant: float

    def apply(
        self,
        latent: np.ndarray,
    ) -> np.ndarray:
        """Transform arrays whose final axis contains latent coordinates."""
        return (
            self.scale * latent @ self.rotation
            + self.shift
        )


# =============================================================================
# Artifact loading and correspondence checks
# =============================================================================

def load_seed_split(
    runs_dir: Path,
    preset: str,
    seed: int,
    split: str,
) -> LatentDataset:
    """Load one encoded split without loading a PyTorch checkpoint."""
    path = (
        runs_dir
        / f"{preset}-seed-{seed}"
        / "evaluation"
        / f"{split}_encoded.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split} artifact for seed {seed}: {path}. "
            "Re-run scripts.evaluation with --save-latents if needed."
        )

    arrays = load_encoded(path)
    required = (
        "latent",
        "true_state",
        "observation",
        "episode_index",
        "time_index",
    )

    missing = [
        name
        for name in required
        if name not in arrays
    ]

    if missing:
        raise KeyError(
            f"{path} is missing required keys: {missing}."
        )

    latent = np.asarray(arrays["latent"])

    if latent.ndim != 3 or latent.shape[-1] < 2:
        raise ValueError(
            f"Invalid latent shape in {path}: {latent.shape}."
        )

    if not np.isfinite(latent).all():
        raise ValueError(
            f"Non-finite latent coordinates found in {path}."
        )

    return LatentDataset(
        seed=seed,
        split=split,
        path=path,
        arrays=arrays,
    )


def validate_correspondence(
    reference: LatentDataset,
    candidate: LatentDataset,
    *,
    tolerance: float = DEFAULT_MAX_ERROR,
) -> None:
    """Check that rows in two runs correspond to identical observations."""
    if reference.split != candidate.split:
        raise ValueError(
            "Cannot align different dataset splits."
        )

    if reference.latent.shape != candidate.latent.shape:
        raise ValueError(
            f"Latent shapes differ for seeds {reference.seed} and "
            f"{candidate.seed}: {reference.latent.shape} versus "
            f"{candidate.latent.shape}."
        )

    for key in (
        "episode_index",
        "time_index",
        "nuisance_id",
    ):
        if key in reference.arrays and key in candidate.arrays:
            if not np.array_equal(
                reference.arrays[key],
                candidate.arrays[key],
            ):
                raise ValueError(
                    f"Different {key} across seeds on {reference.split}."
                )

    for key in (
        "observation",
        "true_state",
    ):
        first = reference.arrays[key]
        second = candidate.arrays[key]

        if first.shape != second.shape or not np.allclose(
            first,
            second,
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                f"{key} differs across seeds {reference.seed} and "
                f"{candidate.seed} on {reference.split}. "
                "Alignment requires exactly corresponding data."
            )


def flatten_latent(
    latent: np.ndarray,
) -> np.ndarray:
    """Flatten episode and time dimensions while preserving row order."""
    return np.asarray(
        latent,
        dtype=np.float64,
    ).reshape(
        -1,
        latent.shape[-1],
    )


# =============================================================================
# Procrustes alignment in the full latent space
# =============================================================================

def fit_global_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    mode: str = DEFAULT_ALIGNMENT,
) -> GlobalAlignment:
    """
    Fit a global rotation/reflection and translation by orthogonal Procrustes.

    ``source`` and ``target`` must contain corresponding rows. In similarity
    mode, also estimate one positive isotropic scaling factor. The SVD is
    applied to the full latent dimension, never its 2D PCA projection.
    """
    if mode not in ALIGNMENT_MODES:
        raise ValueError(
            f"Unknown alignment mode '{mode}'."
        )

    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[0] < source.shape[1]
    ):
        raise ValueError(
            "Expected matching matrices of shape (n_samples, latent_dim)."
        )

    source = np.asarray(
        source,
        dtype=np.float64,
    )
    target = np.asarray(
        target,
        dtype=np.float64,
    )

    source_mean = source.mean(
        axis=0
    )
    target_mean = target.mean(
        axis=0
    )

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_energy = float(
        np.sum(source_centered**2)
    )

    if source_energy <= NORMALIZATION_EPSILON:
        raise ValueError(
            "Source representation is effectively collapsed; "
            "the Procrustes transformation is not well defined."
        )

    cross_covariance = (
        source_centered.T @ target_centered
    )
    left, singular_values, right_transpose = np.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )

    # Reflections are permitted: changing a latent coordinate sign is a
    # legitimate reparameterization and should not be penalized here.
    rotation = left @ right_transpose

    scale = 1.0

    if mode == "similarity":
        scale = float(
            np.sum(singular_values) / source_energy
        )

    shift = target_mean - scale * source_mean @ rotation

    return GlobalAlignment(
        rotation=rotation,
        shift=shift,
        scale=scale,
        determinant=float(np.linalg.det(rotation)),
    )


def measure_alignment(
    source: np.ndarray,
    target: np.ndarray,
    transform: GlobalAlignment,
) -> dict[str, float]:
    """Measure raw and aligned mismatch in the full latent space."""
    aligned = transform.apply(source)
    reference_centered = target - target.mean(
        axis=0,
        keepdims=True,
    )

    reference_variance = float(
        np.mean(
            np.sum(reference_centered**2, axis=1)
        )
    )

    if reference_variance <= NORMALIZATION_EPSILON:
        raise ValueError(
            "Reference representation is effectively collapsed."
        )

    mse_before = float(
        np.mean(
            np.sum((source - target)**2, axis=1)
        )
    )
    mse_after = float(
        np.mean(
            np.sum((aligned - target)**2, axis=1)
        )
    )

    return {
        "rmse_before": float(np.sqrt(mse_before)),
        "rmse_after": float(np.sqrt(mse_after)),
        "nrmse_before": float(np.sqrt(mse_before / reference_variance)),
        "nrmse_after": float(np.sqrt(mse_after / reference_variance)),
        "r2_after": float(1.0 - mse_after / reference_variance),
    }


def centered_linear_cka(
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute centered linear CKA without forming sample-by-sample kernels."""
    source = source - source.mean(
        axis=0,
        keepdims=True,
    )
    target = target - target.mean(
        axis=0,
        keepdims=True,
    )

    cross = source.T @ target
    gram_source = source.T @ source
    gram_target = target.T @ target

    numerator = float(
        np.sum(cross**2)
    )
    denominator = float(
        np.linalg.norm(gram_source, ord="fro")
        * np.linalg.norm(gram_target, ord="fro")
    )

    if denominator <= NORMALIZATION_EPSILON:
        raise ValueError(
            "CKA is not defined for collapsed representations."
        )

    return numerator / denominator


# =============================================================================
# Reference projection and figure preparation
# =============================================================================

def fit_manuscript_projection(
    encoded: LatentDataset,
) -> LatentProjection:
    """Reproduce the standardized PCA used for Figure 5 of the manuscript."""
    original = encoded.latent
    counterfactual = encoded.arrays.get(
        "counterfactual_latent"
    )

    if counterfactual is None:
        raise KeyError(
            f"{encoded.path} has no 'counterfactual_latent'. "
            "The manuscript's reference PCA cannot be reconstructed."
        )

    if counterfactual.shape != original.shape:
        raise ValueError(
            "Counterfactual and original representations have "
            "incompatible shapes."
        )

    return fit_latent_projection(
        np.concatenate(
            [original, counterfactual],
            axis=0,
        ),
        n_components=2,
    )


def select_plot_indices(
    n_points: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Use the same test observations for every seed and both panels."""
    if max_points < 1:
        raise ValueError(
            "max-points must be strictly positive."
        )

    if n_points <= max_points:
        return np.arange(n_points)

    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(
            n_points,
            size=max_points,
            replace=False,
        )
    )


def plot_clouds(
    axis: Axes,
    projections: Mapping[int, np.ndarray],
    indices: np.ndarray,
    *,
    seeds: Sequence[int],
    reference_seed: int,
    colors: Mapping[int, object],
) -> None:
    """Plot matched test positions in a shared 2D reference projection."""
    for seed in seeds:
        coordinates = projections[seed][indices]

        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=7 if seed != reference_seed else 9,
            alpha=0.20 if seed != reference_seed else 0.37,
            color=colors[seed],
            linewidths=0,
            rasterized=True,
            label=f"Seed {seed}",
        )

    axis.set_xlabel("Reference PC1")
    axis.set_ylabel("Reference PC2")
    # Keep square plotting panels without distorting either PCA axis.
    # Matplotlib expands the data limits when a cloud is much wider.
    axis.set_box_aspect(1)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.12)


def create_alignment_figure(
    raw_test: Mapping[int, np.ndarray],
    aligned_test: Mapping[int, np.ndarray],
    reference_projection: LatentProjection,
    rows: Sequence[Mapping[str, float | int]],
    *,
    seeds: Sequence[int],
    reference_seed: int,
    max_points: int,
    visualization_seed: int,
    shared_limits: bool,
) -> Figure:
    """Create original/aligned scatter panels and held-out 4D errors."""
    raw_projected = {
        seed: transform_latent(
            raw_test[seed],
            reference_projection,
        ).reshape(-1, 2)
        for seed in seeds
    }
    aligned_projected = {
        seed: transform_latent(
            aligned_test[seed],
            reference_projection,
        ).reshape(-1, 2)
        for seed in seeds
    }

    indices = select_plot_indices(
        raw_projected[reference_seed].shape[0],
        max_points,
        visualization_seed,
    )

    palette = plt.get_cmap("tab10")
    colors = {
        seed: palette(index % 10)
        for index, seed in enumerate(seeds)
    }

    figure = plt.figure(
        figsize=(13, 10)
    )
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.65),
    )

    axis_before = figure.add_subplot(grid[0, 0])
    axis_after = figure.add_subplot(grid[0, 1])
    axis_error = figure.add_subplot(grid[1, :])

    plot_clouds(
        axis_before,
        raw_projected,
        indices,
        seeds=seeds,
        reference_seed=reference_seed,
        colors=colors,
    )
    plot_clouds(
        axis_after,
        aligned_projected,
        indices,
        seeds=seeds,
        reference_seed=reference_seed,
        colors=colors,
    )

    axis_before.set_title(
        "A. Independently learned latent coordinates"
    )
    axis_after.set_title(
        "B. After orthogonal alignment"
    )

    explained = reference_projection.projector.explained_variance_ratio_

    for axis in (axis_before, axis_after):
        axis.set_xlabel(
            f"Reference PC1 ({100.0 * explained[0]:.1f}%)"
        )
        axis.set_ylabel(
            f"Reference PC2 ({100.0 * explained[1]:.1f}%)"
        )

    if shared_limits:
        combined = np.concatenate(
            [
                coordinates[indices]
                for collection in (raw_projected, aligned_projected)
                for coordinates in collection.values()
            ],
            axis=0,
        )
        lower = combined.min(axis=0)
        upper = combined.max(axis=0)
        padding = 0.05 * np.maximum(upper - lower, 1e-3)

        for axis in (axis_before, axis_after):
            axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])

    comparison_rows = [
        row
        for row in rows
        if int(row["seed"]) != reference_seed
    ]
    positions = np.arange(len(comparison_rows))
    width = 0.37

    axis_error.bar(
        positions - width / 2,
        [row["nrmse_before"] for row in comparison_rows],
        width=width,
        label="Before alignment",
    )
    axis_error.bar(
        positions + width / 2,
        [row["nrmse_after"] for row in comparison_rows],
        width=width,
        label="After alignment",
    )
    axis_error.set_xticks(
        positions,
        labels=[f"Seed {int(row['seed'])}" for row in comparison_rows],
    )
    axis_error.set_ylabel("Normalized test RMSE (4D)")
    axis_error.set_title(
        "C. Held-out mismatch relative to seed "
        f"{reference_seed} (lower is better)"
    )
    axis_error.set_ylim(bottom=0.0)
    axis_error.grid(axis="y", alpha=0.2)
    axis_error.legend()

    handles, labels = axis_before.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=len(seeds),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        markerscale=3.5,
        frameon=False,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    return figure


# =============================================================================
# Report serialization
# =============================================================================

def save_rows(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Write a table with fixed column order from the first row."""
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Analysis orchestration
# =============================================================================

def analyze_seed_alignment(
    runs_dir: Path,
    preset: str,
    seeds: Sequence[int],
    reference_seed: int,
    *,
    mode: str,
) -> tuple[
    dict[int, LatentDataset],
    dict[int, np.ndarray],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    dict[int, GlobalAlignment],
]:
    """Fit alignment on validation and evaluate all seed pairs on test."""
    validation = {
        seed: load_seed_split(runs_dir, preset, seed, "validation")
        for seed in seeds
    }
    test = {
        seed: load_seed_split(runs_dir, preset, seed, "test")
        for seed in seeds
    }

    for seed in seeds:
        validate_correspondence(
            validation[reference_seed],
            validation[seed],
        )
        validate_correspondence(
            test[reference_seed],
            test[seed],
        )

    reference_validation = flatten_latent(
        validation[reference_seed].latent
    )
    reference_test = flatten_latent(
        test[reference_seed].latent
    )

    aligned_test: dict[int, np.ndarray] = {}
    transforms: dict[int, GlobalAlignment] = {}
    rows: list[dict[str, float | int]] = []

    for seed in seeds:
        source_validation = flatten_latent(
            validation[seed].latent
        )
        source_test = flatten_latent(
            test[seed].latent
        )

        alignment = fit_global_alignment(
            source_validation,
            reference_validation,
            mode=mode,
        )
        metrics = measure_alignment(
            source_test,
            reference_test,
            alignment,
        )
        metrics["cka"] = centered_linear_cka(
            source_test,
            reference_test,
        )

        aligned_test[seed] = alignment.apply(
            test[seed].latent
        )
        transforms[seed] = alignment

        rows.append({
            "seed": seed,
            "reference_seed": reference_seed,
            "scale": alignment.scale,
            "determinant": alignment.determinant,
            **metrics,
        })

    # Reference-independent perspective: independently compare all unique
    # seed pairs using the same validation/test separation.
    pairwise_rows: list[dict[str, float | int]] = []

    for source_index, source_seed in enumerate(seeds):
        for target_seed in seeds[source_index + 1:]:
            pair_alignment = fit_global_alignment(
                flatten_latent(validation[source_seed].latent),
                flatten_latent(validation[target_seed].latent),
                mode=mode,
            )
            source_test = flatten_latent(test[source_seed].latent)
            target_test = flatten_latent(test[target_seed].latent)
            metrics = measure_alignment(
                source_test,
                target_test,
                pair_alignment,
            )
            metrics["cka"] = centered_linear_cka(
                source_test,
                target_test,
            )

            pairwise_rows.append({
                "source_seed": source_seed,
                "target_seed": target_seed,
                "scale": pair_alignment.scale,
                "determinant": pair_alignment.determinant,
                **metrics,
            })

    return test, aligned_test, rows, pairwise_rows, transforms


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for seed-alignment analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare independently trained latent spaces by validation-fitted "
            "orthogonal Procrustes alignment and held-out test evaluation."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--reference-seed",
        type=int,
        default=DEFAULT_REFERENCE_SEED,
    )
    parser.add_argument(
        "--alignment",
        choices=ALIGNMENT_MODES,
        default=DEFAULT_ALIGNMENT,
        help="Orthogonal (default) or similarity (+ one isotropic scale).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum number of matched test points per seed in the plot.",
    )
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=DEFAULT_VISUALIZATION_SEED,
    )
    parser.add_argument(
        "--shared-limits",
        action="store_true",
        help="Use the same axis limits in the before and after scatter panels.",
    )
    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """Compute transformations and write the full held-out evaluation."""
    args = build_arg_parser().parse_args()

    seeds = tuple(args.seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "Provide at least two distinct seeds."
        )

    if args.reference_seed not in seeds:
        raise ValueError(
            "The reference seed must be included in --seeds."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test, aligned_test, rows, pairwise_rows, transforms = (
        analyze_seed_alignment(
            args.runs_dir,
            args.preset,
            seeds,
            args.reference_seed,
            mode=args.alignment,
        )
    )

    reference = test[args.reference_seed]
    display_projection = fit_manuscript_projection(reference)
    explained = display_projection.projector.explained_variance_ratio_

    print(
        "Reference PCA (original + counterfactual, standardized): "
        f"PC1={100.0 * explained[0]:.2f}%, "
        f"PC2={100.0 * explained[1]:.2f}%"
    )

    figure = create_alignment_figure(
        {
            seed: test[seed].latent
            for seed in seeds
        },
        aligned_test,
        display_projection,
        rows,
        seeds=seeds,
        reference_seed=args.reference_seed,
        max_points=args.max_points,
        visualization_seed=args.visualization_seed,
        shared_limits=args.shared_limits,
    )

    if args.alignment == "similarity":
        figure.axes[1].set_title(
            "B. After similarity alignment"
        )

    save_figure(
        figure,
        args.output_dir / "fig_seed_alignment.pdf",
    )
    save_figure(
        figure,
        args.output_dir / "fig_seed_alignment.png",
    )
    plt.close(figure)

    save_rows(
        rows,
        args.output_dir / "seed_alignment_metrics.csv",
    )
    save_rows(
        pairwise_rows,
        args.output_dir / "pairwise_alignment_metrics.csv",
    )

    np.savez_compressed(
        args.output_dir / "alignment_transforms.npz",
        seeds=np.asarray(seeds, dtype=np.int64),
        rotations=np.stack([transforms[seed].rotation for seed in seeds]),
        shifts=np.stack([transforms[seed].shift for seed in seeds]),
        scales=np.asarray([transforms[seed].scale for seed in seeds]),
    )

    manifest = {
        "preset": args.preset,
        "seeds": list(seeds),
        "reference_seed": args.reference_seed,
        "alignment": args.alignment,
        "fit_split": "validation",
        "evaluation_split": "test",
        "latent_dimension": reference.latent.shape[-1],
        "validation_points": int(len(flatten_latent(load_seed_split(
            args.runs_dir, args.preset, args.reference_seed, "validation"
        ).latent))),
        "test_points": int(len(flatten_latent(reference.latent))),
        "plot_max_points": args.max_points,
        "visualization_seed": args.visualization_seed,
        "reference_pca_variance_explained": explained.tolist(),
        "test_artifacts": {
            str(seed): str(test[seed].path)
            for seed in seeds
        },
        "note": (
            "Alignment reproducibility is not proof of mathematical "
            "identifiability. The display PCA is fitted on the test "
            "reference only; transformation parameters use validation only."
        ),
    }

    with (args.output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, indent=2)

    print()
    print("Held-out test alignment relative to seed", args.reference_seed)

    for row in rows:
        print(
            f"  Seed {int(row['seed'])}: "
            f"NRMSE {row['nrmse_before']:.4f} -> "
            f"{row['nrmse_after']:.4f}; "
            f"R2={row['r2_after']:.4f}; "
            f"CKA={row['cka']:.4f}; "
            f"det(Q)={row['determinant']:+.0f}"
        )

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
