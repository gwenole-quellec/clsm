"""
Adversarial refresh phase for CLSM invariance training.

Author: Gwenolé Quellec
Year: 2026

After the initial joint training phase, the nuisance adversary is refreshed
against the current encoder using latent representations standardized with
statistics fixed at the start of the refresh phase.

During each refresh epoch, the adversary is first updated on a fixed subsample
of encoded training observations, then the encoder is updated on minibatches
using the refreshed adversarial objective together with the other active CLSM
constraints.

The fixed latent mean and standard deviation prevent the adversary input
normalization from drifting during this second training phase. Dedicated random
generators are used for minibatch ordering and adversary updates to preserve
reproducibility without coupling their random streams to the rest of training.
"""

import numpy as np
import torch

from .strong_adversary import update_adversary


# =============================================================================
# Constants
# =============================================================================

MIN_LATENT_STD = 1e-6

MAX_PROBE_OBSERVATIONS = 25_000
ENCODING_CHUNK_SIZE = 4096

BATCH_GENERATOR_SEED_OFFSET = 700_000
ADVERSARY_GENERATOR_SEED_OFFSET = 800_000
PROBE_SUBSAMPLE_SEED = 42


# =============================================================================
# Adversarial refresh phase
# =============================================================================

class RefreshPhase:
    """Run the adversarial refresh phase after initial joint training."""

    def __init__(
        self,
        model,
        train,
        adversary,
        adversary_optimizer,
        config,
    ):
        self.config = config
        self.adversary = adversary
        self.adversary_optimizer = adversary_optimizer

        device = next(model.parameters()).device

        self.observations = torch.as_tensor(
            train.observation,
            dtype=torch.float32,
            device=device,
        )

        self.labels = torch.as_tensor(
            train.nuisance_id,
            dtype=torch.long,
            device=device,
        )

        flat_observations = self.observations.flatten(0, 1)

        if len(flat_observations) <= MAX_PROBE_OBSERVATIONS:
            probe_indices = np.arange(len(flat_observations))

        else:
            probe_indices = np.random.default_rng(
                PROBE_SUBSAMPLE_SEED
            ).choice(
                len(flat_observations),
                MAX_PROBE_OBSERVATIONS,
                replace=False,
            )

        probe_indices = torch.as_tensor(
            probe_indices,
            device=device,
        )

        self.probe_observations = flat_observations[
            probe_indices
        ]

        self.probe_labels = self.labels.repeat_interleave(
            train.episode_length
        )[probe_indices]

        initial_latent = self.encode(model)

        self.mean = initial_latent.mean(dim=0)

        self.std = initial_latent.std(
            dim=0,
            unbiased=False,
        ).clamp_min(MIN_LATENT_STD)

        self.batch_generator = torch.Generator().manual_seed(
            BATCH_GENERATOR_SEED_OFFSET
            + config.optimization.seed
        )

        self.generator = torch.Generator().manual_seed(
            ADVERSARY_GENERATOR_SEED_OFFSET
            + config.optimization.seed
        )

    @torch.no_grad()
    def encode(
        self,
        model,
    ):
        """Encode the fixed refresh probe observations."""

        return torch.cat(
            [
                model.encode(
                    observations,
                    sample=False,
                )["latent"]
                for observations in self.probe_observations.split(
                    ENCODING_CHUNK_SIZE
                )
            ]
        )

    def standardize(
        self,
        z,
    ):
        """Standardize latent representations using fixed refresh statistics."""

        return (z - self.mean) / self.std

    def epoch(
        self,
        model,
        optimizer,
        objective,
        loss_config,
    ):
        """Run one adversarial refresh epoch."""

        from .training import (
            MetricAccumulator,
            compute_loss_components,
        )

        model.train()

        accumulator = MetricAccumulator()
        device = self.observations.device

        permutation = torch.randperm(
            len(self.observations),
            generator=self.batch_generator,
        )

        batches = permutation.split(self.config.optimization.batch_size)

        for ids in batches:
            ids = ids.to(device)

            batch = {
                "observation": self.observations[ids],
                "nuisance_id": self.labels[ids],
            }

            with torch.no_grad():
                features = self.standardize(self.encode(model))
                labels = self.probe_labels

            update_adversary(
                self.adversary,
                self.adversary_optimizer,
                features,
                labels,
                generator=self.generator,
                steps=self.config.optimization.adversary_steps,
            )

            optimizer.zero_grad(set_to_none=True)

            components, diagnostics = compute_loss_components(
                model,
                batch,
                loss_config,
                sample_latent=False,
            )

            loss, weighted = objective(components)

            loss.backward()

            gradient_clip_norm = (
                self.config.optimization.gradient_clip_norm
            )

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )

            optimizer.step()

            non_adversarial_weighted = {
                name: value
                for name, value in weighted.items()
                if name != "invariance"
            }

            if non_adversarial_weighted:
                selection_total = torch.stack(
                    list(non_adversarial_weighted.values())
                ).sum()

            else:
                selection_total = loss

            metrics = {
                "total": loss,
                "selection_total": selection_total,
            }

            metrics.update(diagnostics)

            metrics.update(
                {
                    f"raw_{name}": value
                    for name, value in components.items()
                }
            )

            metrics.update(
                {
                    f"weighted_{name}": value
                    for name, value in weighted.items()
                }
            )

            accumulator.update(
                metrics,
                len(ids),
            )

        return accumulator.compute()
