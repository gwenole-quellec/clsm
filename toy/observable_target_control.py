"""Train CLSM with nuisance-independent observable prediction targets.

The model observes all 12 channels, but its prediction and (when enabled)
reconstruction losses use only channels 2:8. Other constraints, optimization,
adversarial refresh, checkpoint selection, and data loading are those of the
standard ``toy.train`` / ``clsm.training`` pipeline. Training uses one
observation sequence per trajectory and does not generate paired views.

Configuration weights are loaded by the standard trainer from
``toy/configurations.json``; P29 is the default, not a hard-coded weight set.

Examples
--------
Check the configuration without training::

    python -m toy.observable_target_control --data-dir data --dry-run

Train the five standard seeds::

    python -m toy.observable_target_control --data-dir data \\
        --preset P29 --seeds 0 1 2 3 4 --device cuda

Ablate adversarial invariance while retaining the other P29 weights::

    python -m toy.observable_target_control --data-dir data \\
        --preset P29 --no-invariance --seeds 0 1 2 3 4 --device cuda

Evaluate the supervised channels from frozen checkpoints::

    python -m toy.observable_target_control --data-dir data \\
        --preset P29 --seeds 0 1 2 3 4 --eval-only --device cuda

Evaluate those same channels from checkpoints trained on all 12 observations::

    python -m toy.observable_target_control --data-dir data \\
        --preset P29 --seeds 0 1 2 3 4 --eval-only \\
        --checkpoint-template 'runs/P29-seed-{seed}/best.pt' \\
        --eval-output runs-observable-target6/P29-original_mse6.json --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from clsm.models import CLSMModel, CLSMModelConfig
from clsm.training import (
    load_splits,
    resolve_constraint_weights,
    resolve_device,
    resolve_run_name,
    resolve_seeds,
    run_experiments,
)
from toy.train import build_arg_parser, build_model_config


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CHANNELS = (2, 3, 4, 5, 6, 7)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
EVALUATION_HORIZONS = (1, 5, 10)


# =============================================================================
# Command-line interface
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Reuse the full standard training CLI with experiment-specific defaults."""
    parser = build_arg_parser()
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.set_defaults(
        preset="P29",
        output_dir=Path("runs-observable-target6"),
        target_channels=DEFAULT_CHANNELS,
    )
    parser.add_argument(
        "--no-invariance", action="store_true",
        help="Set the preset's invariance weight to zero, leaving other weights unchanged.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate data, targets, and weights without training.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Evaluate selected-channel reconstruction and rollout MSE on existing checkpoints.",
    )
    parser.add_argument(
        "--eval-splits", nargs="+", choices=("test", "ood", "validation"),
        default=("test", "ood"),
    )
    parser.add_argument(
        "--checkpoint-template", type=str, default=None,
        help=(
            "Optional checkpoint path with a {seed} placeholder for post-hoc "
            "evaluation (e.g. 'runs/P29-seed-{seed}/best.pt'). "
            "May point to models trained on all 12 channels."
        ),
    )
    parser.add_argument(
        "--eval-output", type=Path, default=None,
        help="Output JSON path for --eval-only. Required with --checkpoint-template.",
    )
    return parser


# =============================================================================
# Evaluation
# =============================================================================

def validate_target_channels(channels: tuple[int, ...], observation_dim: int) -> None:
    """Refuse empty, duplicate, or invalid target-channel selections."""
    if (
        not channels
        or len(channels) != len(set(channels))
        or any(channel < 0 or channel >= observation_dim for channel in channels)
    ):
        raise ValueError(
            f"Target channels {channels!r} are invalid for observation_dim={observation_dim}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_channel_mse(
    model: CLSMModel,
    observations: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    channels: tuple[int, ...],
    horizons: tuple[int, ...] = EVALUATION_HORIZONS,
) -> dict[str, float]:
    """Measure errors in native units over every valid starting time."""
    if observations.ndim != 3 or len(observations) == 0:
        raise ValueError("Expected observation tensor with shape (episodes, steps, features).")
    validate_target_channels(channels, observations.shape[-1])
    if observations.shape[-1] != model.config.observation_dim:
        raise ValueError("Observation and model dimensions differ.")
    if observations.shape[1] <= max(horizons):
        raise ValueError("Trajectory length does not cover the requested horizons.")

    suffix = f"target{len(channels)}"
    observation_key = f"observation_{suffix}_mse"
    sums = {observation_key: 0.0}
    counts = {observation_key: 0}
    for horizon in horizons:
        key = f"rollout_observation_{suffix}_mse_h{horizon}"
        sums[key] = 0.0
        counts[key] = 0

    model.eval()
    for start in range(0, len(observations), batch_size):
        batch = torch.as_tensor(
            observations[start:start + batch_size], dtype=torch.float32, device=device,
        )
        outputs = model(batch, sample=False)
        latent = outputs["latent"]
        reconstruction = outputs["reconstructed_observation"]
        error = (reconstruction[..., list(channels)] - batch[..., list(channels)]).square()
        sums[observation_key] += float(error.sum().item())
        counts[observation_key] += error.numel()

        for horizon in horizons:
            starts = latent[:, :-horizon, :].reshape(-1, latent.shape[-1])
            rollout = model.rollout_latent(starts, horizon=horizon)
            prediction = model.decode(rollout[:, -1, :])
            target = batch[:, horizon:, :].reshape(-1, batch.shape[-1])
            error = (prediction[:, list(channels)] - target[:, list(channels)]).square()
            key = f"rollout_observation_{suffix}_mse_h{horizon}"
            sums[key] += float(error.sum().item())
            counts[key] += error.numel()

    return {name: sums[name] / counts[name] for name in sums}


def run_eval_only(args: argparse.Namespace) -> None:
    """Evaluate checkpoints without updating model or independent probes."""
    device = resolve_device(args.device)
    seeds = resolve_seeds(args)
    channels = tuple(args.target_channels)
    datasets: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for split in args.eval_splits:
        path = args.data_dir / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            datasets[split] = np.asarray(data["observation"], dtype=np.float32)
        hashes[split] = _sha256(path)

    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    checkpoint_provenance: dict[str, dict[str, object]] = {}
    for seed in seeds:
        if args.checkpoint_template is None:
            name = resolve_run_name(
                args.run_name, configuration=args.preset, seed=seed,
                multiple_runs=len(seeds) > 1,
            )
            path = args.output_dir / name / "best.pt"
        else:
            try:
                path = Path(args.checkpoint_template.format(seed=seed))
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "--checkpoint-template must contain a valid {seed} placeholder."
                ) from exc
            name = path.parent.name
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = CLSMModel(CLSMModelConfig(**checkpoint["model_config"])).to(device)
        saved_channels = checkpoint.get("config", {}).get("loss", {}).get("target_channels")
        # Legacy checkpoints trained on all observation channels do not carry
        # a target_channels field. That is not a reason to reject a post-hoc
        # evaluation on a strict subset of their outputs.
        if saved_channels is None:
            training_channels = tuple(range(model.config.observation_dim))
        else:
            training_channels = tuple(saved_channels)
        if args.checkpoint_template is None and training_channels != channels:
            raise ValueError(
                f"Checkpoint {path} was trained on {training_channels!r}, "
                f"not the requested evaluation channels {channels!r}. "
                "Use --checkpoint-template to explicitly evaluate a different "
                "checkpoint protocol."
            )
        checkpoint_provenance[str(seed)] = {
            "path": str(path),
            "training_target_channels_zero_based": list(training_channels),
        }
        print(
            f"Evaluating {path} (training channels={training_channels}, "
            f"scoring channels={channels})", flush=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        per_seed[str(seed)] = {}
        for split, observations in datasets.items():
            metrics = evaluate_channel_mse(
                model, observations, device=device,
                batch_size=args.batch_size, channels=channels,
            )
            per_seed[str(seed)][split] = metrics
            print(f"{name}, {split}: " + ", ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            ), flush=True)

    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in datasets:
        keys = next(iter(per_seed.values()))[split]
        aggregate[split] = {}
        for key in keys:
            values = np.asarray([per_seed[str(seed)][split][key] for seed in seeds])
            aggregate[split][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }

    payload = {
        "configuration": args.preset,
        "no_invariance": args.no_invariance,
        "target_channels_zero_based": list(channels),
        "horizons": list(EVALUATION_HORIZONS),
        "metric_units": "native squared observation units",
        "dataset_sha256": hashes,
        "checkpoint_provenance": checkpoint_provenance,
        "by_seed": per_seed,
        "aggregate": aggregate,
    }
    label = args.run_name.replace("-seed-{seed}", "")
    output = (
        args.eval_output if args.eval_output is not None
        else args.output_dir / f"{label}_mse{len(channels)}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()
    if args.no_invariance:
        if args.weight_invariance not in (None, 0.0):
            raise SystemExit("--no-invariance conflicts with a positive --weight-invariance.")
        args.weight_invariance = 0.0
    if args.seed is None and args.seeds is None:
        args.seeds = list(DEFAULT_SEEDS)
    if args.eval_only and args.dry_run:
        raise SystemExit("--eval-only and --dry-run are mutually exclusive.")
    if (args.checkpoint_template is not None or args.eval_output is not None) and not args.eval_only:
        raise SystemExit("--checkpoint-template and --eval-output require --eval-only.")
    if args.checkpoint_template is not None and args.eval_output is None:
        raise SystemExit("--eval-output is required with --checkpoint-template to avoid overwriting previous evaluations.")

    channels = tuple(args.target_channels)
    channel_tag = (
        "target6" if channels == DEFAULT_CHANNELS
        else "target" + "-".join(map(str, channels))
    )
    if args.run_name is None:
        inv_tag = "-noinv" if args.no_invariance else ""
        args.run_name = f"{args.preset}-{channel_tag}{inv_tag}-seed-{{seed}}"

    if args.dry_run:
        name, weights = resolve_constraint_weights(args)
        splits = load_splits(args.data_dir)
        validate_target_channels(channels, splits.train.observation_dim)
        print(f"Configuration: {name}")
        print(f"Weights: {weights.as_dict()}")
        print(f"Train: {splits.train.n_episodes} trajectories; "
              f"validation: {splits.validation.n_episodes} trajectories")
        print(f"Inputs: {splits.train.observation_dim} channels; "
              f"loss targets: {channels}")
        print("Training: standard CLSM procedure; single observation view")
        print(f"Model seeds: {resolve_seeds(args)}")
        print("Preflight OK; no training performed")
        return

    if args.eval_only:
        run_eval_only(args)
    else:
        run_experiments(args=args, model_config_factory=build_model_config)


if __name__ == "__main__":
    main()
