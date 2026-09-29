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
structure.DEFAULT_CONFIGURATIONS_FILE = Path(
    "toy/configurations.json"
)

Examples
--------
Run all presets defined in the configuration file:

    python -m scripts.run_presets \
        --train-module toy.train \
        --metadata toy/metadata.json \
        --data-dir data

Run only the ``P5`` preset:

    python -m scripts.run_presets \
        --train-module toy.train \
        --metadata toy/metadata.json \
        --data-dir data \
        --presets P5

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
from collections.abc import Iterable
from pathlib import Path

from clsm.training import load_configurations
from clsm.utils import module_command, print_banner, print_separator


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_DATA_DIR = "data"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_FIGURES_DIR = "figures"
DEFAULT_CONFIGURATIONS_FILE = Path(
    "toy/configurations.json"
)

DEFAULT_ADVERSARY_STEPS = 15

DEFAULT_PROBE_WORKERS = 1
DEFAULT_PROBE_CACHE_DIR = ".probe-cache"
DEFAULT_PROBE_PROFILE = "full"

PROBE_PROFILES = (
    "full",
    "pareto",
)

DEFAULT_EPOCHS = 50
DEFAULT_REFRESH_PRETRAIN_EPOCHS = 40
DEFAULT_NUISANCE_PROBE_EPOCHS = 500

DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_EVALUATION_BATCH_SIZE = 256

DEFAULT_ROLLOUT_HORIZONS = (
    1,
    5,
    10,
)

DEFAULT_DEVICE = "auto"

DEFAULT_VISUALIZATION_SEED = 0
DEFAULT_EPISODE_INDEX = 0


# =============================================================================
# Pipeline helpers
# =============================================================================

def require_file(
    path: Path,
) -> None:
    """Raise a clear error when an expected artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected pipeline artifact was not found: {path}"
        )


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

    try:
        subprocess.run(
            command,
            check=True,
        )

    except subprocess.CalledProcessError as error:
        raise SystemExit(
            error.returncode
        ) from None


# =============================================================================
# Pipeline orchestration
# =============================================================================

def run_preset_pipeline(
    preset: str,
    train_module: str,
    metadata_path: str | Path,
    *,
    configurations_path: str | Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    adversary_steps: int = DEFAULT_ADVERSARY_STEPS,
    probe_workers: int = DEFAULT_PROBE_WORKERS,
    probe_cache_dir: str = DEFAULT_PROBE_CACHE_DIR,
    strong_probe_epochs: int | None = None,
    physical_probe_epochs: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
    refresh_pretrain_epochs: int = DEFAULT_REFRESH_PRETRAIN_EPOCHS,
    nuisance_probe_epochs: int = DEFAULT_NUISANCE_PROBE_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    evaluation_batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
    rollout_horizons: Iterable[int] = DEFAULT_ROLLOUT_HORIZONS,
    device: str = DEFAULT_DEVICE,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    episode_index: int = DEFAULT_EPISODE_INDEX,
    adversarial_chance_level: float | None = None,
    probe_profile: str = DEFAULT_PROBE_PROFILE,
) -> None:
    """
    Train, evaluate, and visualize one CLSM preset.

    The preset is defined in the supplied configuration file.

    The pipeline:

    1. trains the requested preset for all model seeds;
    2. evaluates every trained checkpoint on all available dataset splits;
    3. saves encoded datasets and, in the full probe profile, CCA analyses;
    4. generates the available manuscript figures for the selected seed;
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
        "--configurations-file",
        str(configurations_path),
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

    train_command.extend(
        [
            "--refresh-pretrain-epochs",
            str(refresh_pretrain_epochs),
        ]
    )

    train_command.extend(
        [
            "--adversary-steps",
            str(adversary_steps),
        ]
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

    evaluation_command.extend(
        ["--nuisance-probe-epochs",
         str(nuisance_probe_epochs)]
    )
    evaluation_command.extend(
        ["--probe-profile", str(probe_profile)]
    )
    if physical_probe_epochs is not None:
        evaluation_command.extend(
            ["--physical-probe-epochs", str(physical_probe_epochs)]
        )
    if strong_probe_epochs is not None:
        evaluation_command.extend(
            ["--strong-probe-epochs", str(strong_probe_epochs)]
        )
    if probe_cache_dir is not None:
        evaluation_command.extend(
            ["--probe-cache-dir", str(probe_cache_dir)]
        )
    if probe_workers is not None:
        evaluation_command.extend(
            ["--probe-workers", str(probe_workers)]
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
            evaluation_dir / f"{split}_encoded.npz"
        )

        require_file(encoded_path)

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

        if probe_profile == "full":
            cca_path = (
                evaluation_dir
                / f"{split}_cca_analysis.npz"
            )

            require_file(cca_path)

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
        "--configurations-file",
        type=Path,
        default=DEFAULT_CONFIGURATIONS_FILE,
        help="JSON file containing named CLSM constraint configurations.",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help=(
            "Presets to evaluate. "
            "Defaults to all presets defined in --configurations-file."
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
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing train.npz, validation.npz, "
            "test.npz, and optionally ood.npz."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="Directory in which checkpoints and metrics are saved.",
    )
    parser.add_argument(
        "--figures-dir",
        default=DEFAULT_FIGURES_DIR,
        help=(
            "Directory in which figures are saved."
        ),
    )
    parser.add_argument(
        "--probe-profile",
        choices=PROBE_PROFILES,
        default=DEFAULT_PROBE_PROFILE,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--refresh-pretrain-epochs",
        type=int,
        default=DEFAULT_REFRESH_PRETRAIN_EPOCHS,
    )
    parser.add_argument(
        "--nuisance-probe-epochs",
        type=int,
        default=DEFAULT_NUISANCE_PROBE_EPOCHS,
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--adversary-steps",
        type=int,
        default=DEFAULT_ADVERSARY_STEPS,
    )
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=DEFAULT_PROBE_WORKERS,
    )
    parser.add_argument(
        "--probe-cache-dir",
        type=str,
        default=DEFAULT_PROBE_CACHE_DIR,
    )
    parser.add_argument(
        "--strong-probe-epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--physical-probe-epochs",
        type=int,
        default=None,
    )
    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()

    configurations = load_configurations(
        args.configurations_file
    )

    available_presets = tuple(
        configurations.keys()
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
            configurations_path=args.configurations_file,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            figures_dir=args.figures_dir,
            adversary_steps=args.adversary_steps,
            probe_workers=args.probe_workers,
            probe_cache_dir=args.probe_cache_dir,
            strong_probe_epochs=args.strong_probe_epochs,
            physical_probe_epochs=args.physical_probe_epochs,
            epochs=args.epochs,
            refresh_pretrain_epochs=args.refresh_pretrain_epochs,
            nuisance_probe_epochs=args.nuisance_probe_epochs,
            device=args.device,
            seeds=args.seeds,
            visualization_seed=args.seeds[0],
            probe_profile=args.probe_profile,
        )


if __name__ == "__main__":
    main()

