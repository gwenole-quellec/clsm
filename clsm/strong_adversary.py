"""
Categorical nuisance adversary for CLSM latent representations.

Author: Gwenolé Quellec
Year: 2026

This module defines the nuisance classifier used during adversarial invariance
training. The adversary predicts categorical nuisance labels from latent
representations and is optimized separately from the CLSM encoder.

The adversary uses a fixed two-layer multilayer perceptron. Its initialization
and minibatch sampling rely on dedicated random-number generators so that
adversary randomness remains reproducible and independent from the main
training random stream.

During the initial training phase, latent features are standardized within each
current minibatch before being passed to the adversary. During the later refresh
phase, the same adversary is instead applied to latents standardized with fixed
statistics computed at the start of that phase.
"""

import torch
from torch import nn


# =============================================================================
# Constants
# =============================================================================

ADVERSARY_HIDDEN_DIM = 64

MIN_LATENT_STD = 1e-6

ADVERSARY_INITIALIZATION_SEED_OFFSET = 100_000
ADVERSARY_GENERATOR_SEED_OFFSET = 200_000

ADVERSARY_BATCH_SIZE = 512


# =============================================================================
# Adversary functions
# =============================================================================

def make_adversary(
    latent_dim: int,
    n_classes: int,
    *,
    seed: int,
) -> nn.Module:
    """Build the categorical nuisance adversary."""

    with torch.random.fork_rng(
        devices=[]
    ):
        torch.manual_seed(
            ADVERSARY_INITIALIZATION_SEED_OFFSET + seed
        )

        adversary = nn.Sequential(
            nn.Linear(
                latent_dim,
                ADVERSARY_HIDDEN_DIM,
            ),
            nn.ReLU(),
            nn.Linear(
                ADVERSARY_HIDDEN_DIM,
                ADVERSARY_HIDDEN_DIM,
            ),
            nn.ReLU(),
            nn.Linear(
                ADVERSARY_HIDDEN_DIM,
                n_classes,
            ),
        )

    return adversary


def make_adversary_generator(
    seed: int,
) -> torch.Generator:
    """Create the deterministic generator used for adversary updates."""

    return torch.Generator().manual_seed(
        ADVERSARY_GENERATOR_SEED_OFFSET + seed
    )


def standardize_batch(
    z,
):
    """Standardize each latent dimension over the current batch."""

    centered = (
        z
        - z.mean(
            dim=0,
            keepdim=True,
        )
    )

    scale = (
        centered.square()
        .mean(
            dim=0,
            keepdim=True,
        )
        .sqrt()
        .clamp_min(
            MIN_LATENT_STD
        )
    )

    return (
        centered
        / scale
    )


def update_adversary(
    adversary,
    optimizer,
    features,
    labels,
    *,
    generator,
    steps: int,
    batch_size: int = ADVERSARY_BATCH_SIZE,
) -> None:
    """Update the nuisance adversary on detached latent features."""

    adversary.train()
    adversary.requires_grad_(
        True
    )

    for _ in range(steps):
        indices = torch.randint(
            len(features),
            (batch_size,),
            generator=generator,
        ).to(
            features.device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = adversary(
            features[
                indices
            ]
        )

        loss = nn.functional.cross_entropy(
            logits,
            labels[
                indices
            ],
        )

        loss.backward()
        optimizer.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    adversary.requires_grad_(
        False
    )
