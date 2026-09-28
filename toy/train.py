"""
Train CLSM on the toy dataset.

Author: Gwenolé Quellec
Year: 2026

Examples
--------
Train configuration ``P3``:

    python -m toy.train \
        --data-dir data/ \
        --configurations-file toy/configurations.json \
        --preset P3 \
        --epochs 100

Train five models with different random seeds:

    python -m toy.train \
        --data-dir data/ \
        --configurations-file toy/configurations.json \
        --preset P3 \
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse

from clsm.models import CLSMModelConfig
from clsm.training import (
    UNRESOLVED_OBSERVATION_DIM,
    add_training_arguments,
    run_experiments,
)


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the toy experiment."""
    parser = argparse.ArgumentParser(
        description="Train CLSM on the toy dataset.",
    )

    add_training_arguments(parser)

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--encoder-hidden-dims",
        type=int,
        nargs="+",
        default=(64, 64),
    )
    parser.add_argument(
        "--decoder-hidden-dims",
        type=int,
        nargs="+",
        default=(64, 64),
    )
    parser.add_argument(
        "--transition-hidden-dims",
        type=int,
        nargs="+",
        default=(64, 64),
    )

    return parser


# =============================================================================
# Model configuration
# =============================================================================

def build_model_config(
    args: argparse.Namespace,
) -> CLSMModelConfig:
    """Build the model configuration for the toy experiment."""
    return CLSMModelConfig(
        observation_dim=UNRESOLVED_OBSERVATION_DIM,
        latent_dim=args.latent_dim,
        encoder_hidden_dims=tuple(
            args.encoder_hidden_dims
        ),
        decoder_hidden_dims=tuple(
            args.decoder_hidden_dims
        ),
        transition_hidden_dims=tuple(
            args.transition_hidden_dims
        ),
    )


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run_experiments(
            args=args,
            model_config_factory=build_model_config,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
