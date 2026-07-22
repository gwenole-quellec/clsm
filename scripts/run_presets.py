"""
Run the complete CLSM evaluation pipeline for one or more predefined presets.

Author: Gwenolé Quellec
Year: 2026

This script trains, evaluates, and visualizes a collection of CLSM presets
using multiple random seeds and a pre-generated dataset. The training module
is supplied explicitly, while evaluation and visualization are performed by
the generic modules in the ``scripts`` package.

The pipeline generates publication-ready figures, aggregates evaluation
metrics across seeds, and saves all results in a standardized directory
structure.

Examples
--------
Run all presets defined by the toy training module:

    python -m scripts.run_presets \
        --train-module toy.train \
        --metadata toy/metadata.json \
        --data-dir data

Run only the ``full`` preset:

    python -m scripts.run_presets \
        --train-module toy.train \
        --metadata toy/metadata.json \
        --data-dir data \
        --presets full

Write runs and figures to custom directories:

    python -m scripts.run_presets \
        --train-module toy.train \
        --metadata toy/metadata.json \
        --data-dir data \
        --runs-dir outputs/runs \
        --figures-dir outputs/figures

By default, outputs are written to ``runs/`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from clsm.training import PRESET_WEIGHTS
from clsm.utils import module_command, print_banner, print_separator


# =============================================================================
# Pipeline helpers
# =============================================================================

def run_command(
    command: list[str],
    *,
    title: str,
) -> None:
    """Print and execute one pipeline command."""
    print_banner(title)
    print(
        subprocess.list2cmdline(
            command
        )
    )

    subprocess.run(
        command,
        check=True,
    )


def require_file(
    path: Path,
) -> None:
    """Raise a clear error when an expected artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected pipeline artifact was not found: {path}"
        )


# =============================================================================
# Pipeline orchestration
# =============================================================================

def run_preset_pipeline(
    preset: str,
    train_module: str,
    metadata_path: str | Path,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    data_dir: str | Path = "data",
    runs_dir: str | Path = "runs",
    figures_dir: str | Path = "figures",
    epochs: int = 100,
    train_batch_size: int = 128,
    evaluation_batch_size: int = 256,
    rollout_horizons: Iterable[int] = (1, 5, 10),
    device: str = "cuda",
    visualization_seed: int = 0,
    episode_index: int = 0,
    adversarial_chance_level: float | None = None,
) -> None:
    """
    Train, evaluate, and visualize one CLSM preset.

    The preset is defined by the supplied training module.

    The pipeline:

    1. trains the requested preset for all model seeds;
    2. evaluates every trained checkpoint on all available dataset splits;
    3. saves encoded datasets and split-specific CCA analyses;
    4. generates manuscript figures for the selected seed;
    5. generates training diagnostics;
    6. aggregates evaluation metrics across seeds.
    """

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    seeds = tuple(
        int(seed)
        for seed in seeds
    )
    if not seeds:
        raise ValueError(
            "At least one seed must be provided."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"Seeds must be unique, got {seeds}."
        )
    if visualization_seed not in seeds:
        raise ValueError(
            f"visualization_seed={visualization_seed} "
            f"is not present in seeds={seeds}."
        )

    rollout_horizons = tuple(
        int(horizon)
        for horizon in rollout_horizons
    )
    if not rollout_horizons:
        raise ValueError(
            "At least one rollout horizon must be provided."
        )
    if any(
        horizon < 1
        for horizon in rollout_horizons
    ):
        raise ValueError(
            "All rollout horizons must be at least 1."
        )

    metadata_path = Path(
        metadata_path
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file was not found: {metadata_path}"
        )

    # -------------------------------------------------------------------------
    # Output path preparation
    # -------------------------------------------------------------------------

    data_dir = Path(data_dir)
    runs_dir = Path(runs_dir)
    figures_dir = Path(figures_dir)

    preset_figures_dir = (
        figures_dir
        / preset
    )
    preset_figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name_template = (
        f"{preset}-seed-{{}}"
    )
    checkpoint_template = (
        runs_dir
        / f"{preset}-seed-{{}}"
        / "best.pt"
    )

    aggregate_output = (
        runs_dir
        / f"{preset}-aggregate-evaluation.json"
    )

    selected_run_dir = (
        runs_dir
        / f"{preset}-seed-{visualization_seed}"
    )

    evaluation_dir = (
        selected_run_dir
        / "evaluation"
    )
    metrics_path = (
        evaluation_dir
        / "evaluation_metrics.json"
    )
    history_path = (
        selected_run_dir
        / "history.csv"
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    train_command = module_command(
        train_module,
        "--preset",
        preset,
        "--run-name",
        run_name_template,
        "--seeds",
        *[
            str(seed)
            for seed in seeds
        ],
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(runs_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(train_batch_size),
        "--device",
        device,
    )

    run_command(
        train_command,
        title=(
            f"TRAINING PRESET: {preset}"
        ),
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    evaluation_command = module_command(
        "scripts.evaluation",
        "--checkpoint-template",
        str(checkpoint_template),
        "--data-dir",
        str(data_dir),
        "--seeds",
        *[
            str(seed)
            for seed in seeds
        ],
        "--split",
        "all",
        "--batch-size",
        str(evaluation_batch_size),
        "--rollout-horizons",
        *[
            str(horizon)
            for horizon in rollout_horizons
        ],
        "--save-latents",
        "--device",
        device,
    )

    # evaluation.py only creates an aggregate file when several
    # checkpoints are evaluated
    if len(seeds) > 1:
        evaluation_command.extend(
            [
                "--aggregate-output",
                str(aggregate_output),
            ]
        )

    run_command(
        evaluation_command,
        title=(
            f"EVALUATING PRESET: {preset}"
        ),
    )

    require_file(
        metrics_path
    )

    # -------------------------------------------------------------------------
    # Split-specific manuscript figures
    # -------------------------------------------------------------------------

    figure_splits = ["test"]
    if (data_dir / "ood.npz").exists():
        figure_splits.append("ood")
    for split in figure_splits:
        encoded_path = (
            evaluation_dir
            / f"{split}_encoded.npz"
        )

        cca_path = (
            evaluation_dir
            / f"{split}_cca_analysis.npz"
        )

        require_file(
            encoded_path
        )

        require_file(
            cca_path
        )

        # Environment and learned-representation figure
        environment_output = (
            preset_figures_dir
            / (
                f"{preset}_{split}_"
                "environment_representation.pdf"
            )
        )

        environment_command = module_command(
            "scripts.visualization",
            "environment",
            "--encoded",
            str(encoded_path),
            "--metadata",
            str(metadata_path),
            "--episode-index",
            str(episode_index),
            "--output",
            str(environment_output),
        )

        run_command(
            environment_command,
            title=(
                f"VISUALIZING ENVIRONMENT: "
                f"{preset} ({split.upper()})"
            ),
        )

        # State probes and CCA figure
        state_output = (
            preset_figures_dir
            / (
                f"{preset}_{split}_"
                "state_analysis.pdf"
            )
        )

        state_command = module_command(
            "scripts.visualization",
            "state",
            "--metrics",
            str(metrics_path),
            "--cca",
            str(cca_path),
            "--metadata",
            str(metadata_path),
            "--split",
            split,
            "--output",
            str(state_output),
        )

        run_command(
            state_command,
            title=(
                f"VISUALIZING STATE ANALYSIS: "
                f"{preset} ({split.upper()})"
            ),
        )

    # -------------------------------------------------------------------------
    # Training diagnostics
    # -------------------------------------------------------------------------

    require_file(
        history_path
    )

    # Selection objective
    history_output = (
        preset_figures_dir
        / (
            f"{preset}_seed-{visualization_seed}_"
            "selection_history.pdf"
        )
    )

    history_command = module_command(
        "scripts.visualization",
        "history",
        "--history",
        str(history_path),
        "--metric",
        "selection_total",
        "--output",
        str(history_output),
    )

    run_command(
        history_command,
        title=(
            f"VISUALIZING TRAINING HISTORY: "
            f"{preset}-seed-{visualization_seed}"
        ),
    )

    # Individual loss components
    components_output = (
        preset_figures_dir
        / (
            f"{preset}_seed-{visualization_seed}_"
            "training_components.pdf"
        )
    )

    components_command = module_command(
        "scripts.visualization",
        "components",
        "--history",
        str(history_path),
        "--output",
        str(components_output),
    )

    run_command(
        components_command,
        title=(
            f"VISUALIZING LOSS COMPONENTS: "
            f"{preset}-seed-{visualization_seed}"
        ),
    )

    # Adversarial dynamics, when the required columns exist
    history = pd.read_csv(
        history_path
    )

    adversarial_columns = {
        "train_raw_invariance",
        "validation_raw_invariance",
        "train_nuisance_adversarial_accuracy",
        "validation_nuisance_adversarial_accuracy",
    }

    if adversarial_columns.issubset(
        history.columns
    ):
        adversarial_output = (
            preset_figures_dir
            / (
                f"{preset}_seed-{visualization_seed}_"
                "adversarial_dynamics.pdf"
            )
        )

        adversarial_command = module_command(
            "scripts.visualization",
            "adversarial",
            "--history",
            str(history_path),
            "--output",
            str(adversarial_output),
        )

        if adversarial_chance_level is not None:
            adversarial_command.extend(
                [
                    "--chance-level",
                    str(
                        adversarial_chance_level
                    ),
                ]
            )

        run_command(
            adversarial_command,
            title=(
                f"VISUALIZING ADVERSARIAL DYNAMICS: "
                f"{preset}-seed-{visualization_seed}"
            ),
        )
    else:
        print()
        print(
            "Skipping adversarial visualization: "
            "the selected history does not contain all required columns."
        )

    # -------------------------------------------------------------------------
    # Completion report
    # -------------------------------------------------------------------------

    print()
    print_separator()
    print(
        f"PIPELINE COMPLETED: {preset}"
    )

    if len(seeds) > 1:
        print(
            f"Aggregate metrics : {aggregate_output}"
        )
    else:
        print(
            "Aggregate metrics : not generated "
            "(only one seed was evaluated)"
        )

    print(
        f"Run metrics       : {metrics_path}"
    )
    print(
        f"Figures directory : {preset_figures_dir}"
    )
    print_separator()


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CLSM pipeline for one or more predefined presets."
        )
    )

    parser.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help=(
            "Presets to evaluate. "
            "Defaults to all presets defined by the training module."
        ),
    )
    parser.add_argument(
        "--train-module",
        required=True,
        help=(
            "Training module executed with 'python -m'."
        ),
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="JSON file describing state and observation dimensions.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help=(
            "Directory containing train.npz, validation.npz, "
            "test.npz, and optionally ood.npz."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory in which checkpoints and metrics are saved.",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help=(
            "Directory in which figures are saved."
        ),
    )

    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()

    available_presets = tuple(
        PRESET_WEIGHTS.keys()
    )

    presets = (
        available_presets
        if args.presets is None
        else tuple(args.presets)
    )
    unknown = sorted(
        set(presets)
        - set(available_presets)
    )
    if unknown:
        parser.error(
            "Unknown preset(s): "
            + ", ".join(unknown)
            + ". Available presets: "
            + ", ".join(available_presets)
        )

    for preset in presets:
        run_preset_pipeline(
            preset,
            args.train_module,
            args.metadata,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            figures_dir=args.figures_dir,
        )


if __name__ == "__main__":
    main()
