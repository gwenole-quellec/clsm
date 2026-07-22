"""
Neural models for CLSM.

Author: Gwenolé Quellec
Year: 2026

This module defines lightweight, architecture-agnostic PyTorch components for
CLSM experiments:

- an observation encoder;
- an observation decoder;
- a latent transition model;
- an adversarial nuisance predictor with gradient reversal;
- a composite CLSMModel wrapper.

The models are intentionally based on small multilayer perceptrons. The goal is
to isolate the effect of competing representation-learning constraints rather
than the effect of architectural complexity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


# =============================================================================
# Gradient reversal utilities
# =============================================================================

class _GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass and sign reversal in the backward pass."""

    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        coefficient: float,
    ) -> Tensor:
        ctx.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(
        ctx,
        gradients: Tensor,
    ) -> tuple[Tensor, None]:
        return (
            -ctx.coefficient * gradients,
            None,
        )


def gradient_reverse(
    inputs: Tensor,
    coefficient: float = 1.0,
) -> Tensor:
    """
    Reverse gradients flowing through ``inputs``.

    The forward pass is the identity. During backpropagation, gradients are
    multiplied by ``-coefficient``. This allows a nuisance predictor to be
    trained normally while encouraging the encoder to remove nuisance-related
    information.
    """
    if coefficient < 0.0:
        raise ValueError(
            "gradient-reversal coefficient must be non-negative."
        )

    return _GradientReversalFunction.apply(
        inputs,
        float(coefficient),
    )


# =============================================================================
# Network construction
# =============================================================================

def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    *,
    activation: type[nn.Module] = nn.ReLU,
    final_activation: type[nn.Module] | None = None,
    dropout: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """
    Build a simple fully connected network.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    output_dim:
        Output feature dimension.
    hidden_dims:
        Hidden-layer dimensions.
    activation:
        Activation module class.
    final_activation:
        Optional activation after the output layer.
    dropout:
        Dropout probability after hidden activations.
    layer_norm:
        Apply layer normalization after hidden linear layers.
    """
    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive.")
    if any(dim < 1 for dim in hidden_dims):
        raise ValueError("All hidden dimensions must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1).")

    layers: list[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))

    if final_activation is not None:
        layers.append(final_activation())

    return nn.Sequential(*layers)


# =============================================================================
# Core model components
# =============================================================================

class ObservationEncoder(nn.Module):
    """
    Encode observations into latent representations.

    The final dimension must equal ``observation_dim``. All leading dimensions
    are preserved.
    """

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        *,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.observation_dim = int(observation_dim)
        self.latent_dim = int(latent_dim)

        self.network = make_mlp(
            observation_dim,
            latent_dim,
            hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, observation: Tensor) -> Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation feature dimension "
                f"{self.observation_dim}, got {observation.shape[-1]}."
            )

        original_shape = observation.shape[:-1]
        flat = observation.reshape(-1, self.observation_dim)
        latent = self.network(flat)

        return latent.reshape(*original_shape, self.latent_dim)


class VariationalObservationEncoder(nn.Module):
    """
    Variational encoder producing a diagonal Gaussian latent posterior.

    This encoder is optional and only needed when the KL form of the
    minimality loss is used.
    """

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        *,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.observation_dim = int(observation_dim)
        self.latent_dim = int(latent_dim)

        backbone_dim = hidden_dims[-1] if hidden_dims else observation_dim

        self.backbone = make_mlp(
            observation_dim,
            backbone_dim,
            hidden_dims[:-1] if hidden_dims else (),
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.mean_head = nn.Linear(backbone_dim, latent_dim)
        self.log_variance_head = nn.Linear(backbone_dim, latent_dim)

    def forward(
        self,
        observation: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation feature dimension "
                f"{self.observation_dim}, got {observation.shape[-1]}."
            )

        original_shape = observation.shape[:-1]
        flat = observation.reshape(-1, self.observation_dim)

        features = self.backbone(flat)
        mean = self.mean_head(features)
        log_variance = self.log_variance_head(features)

        if sample:
            std = torch.exp(0.5 * log_variance)
            epsilon = torch.randn_like(std)
            latent = mean + epsilon * std
        else:
            latent = mean

        output_shape = (*original_shape, self.latent_dim)

        return (
            latent.reshape(output_shape),
            mean.reshape(output_shape),
            log_variance.reshape(output_shape),
        )


class ObservationDecoder(nn.Module):
    """
    Decode latent representations back into observations.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        *,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.observation_dim = int(observation_dim)

        self.network = make_mlp(
            latent_dim,
            observation_dim,
            hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent feature dimension "
                f"{self.latent_dim}, got {latent.shape[-1]}."
            )

        original_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self.latent_dim)
        observation = self.network(flat)

        return observation.reshape(*original_shape, self.observation_dim)


class LatentTransitionModel(nn.Module):
    """
    Predict the next latent state from the current latent state.

    By default, the model predicts a residual update:

    ``z[t+1] = z[t] + delta(z[t])``

    This provides a useful inductive bias for smooth trajectories while still
    allowing nonlinear dynamics.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        *,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.residual = bool(residual)

        self.network = make_mlp(
            latent_dim,
            latent_dim,
            hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent feature dimension "
                f"{self.latent_dim}, got {latent.shape[-1]}."
            )

        original_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self.latent_dim)
        update = self.network(flat).reshape(*original_shape, self.latent_dim)

        return latent + update if self.residual else update

    def rollout(
        self,
        initial_latent: Tensor,
        horizon: int,
    ) -> Tensor:
        """
        Roll out the latent dynamics over multiple future steps.

        Parameters
        ----------
        initial_latent:
            Initial latent tensor of shape ``(..., latent_dim)``.
        horizon:
            Number of future steps.

        Returns
        -------
        Tensor
            Shape ``(..., horizon, latent_dim)``.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")

        states = []
        current = initial_latent

        for _ in range(horizon):
            current = self.forward(current)
            states.append(current)

        return torch.stack(states, dim=-2)


class NuisanceClassifier(nn.Module):
    """
    Predict nuisance identity from latent representations.

    Gradient reversal is optionally applied by :class:`CLSMModel` before the
    latent representations are passed to this classifier.
    """

    def __init__(
        self,
        latent_dim: int,
        n_nuisances: int,
        hidden_dims: Sequence[int] = (32,),
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.n_nuisances = int(n_nuisances)

        self.network = make_mlp(
            latent_dim,
            n_nuisances,
            hidden_dims,
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent feature dimension "
                f"{self.latent_dim}, got {latent.shape[-1]}."
            )

        original_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self.latent_dim)
        logits = self.network(flat)

        return logits.reshape(*original_shape, self.n_nuisances)


# =============================================================================
# Model configuration
# =============================================================================

@dataclass(frozen=True)
class CLSMModelConfig:
    """Configuration of a generic CLSM model."""

    observation_dim: int
    latent_dim: int
    n_nuisances: int | None

    encoder_hidden_dims: tuple[int, ...] = (64, 64)
    decoder_hidden_dims: tuple[int, ...] = (64, 64)
    transition_hidden_dims: tuple[int, ...] = (64, 64)
    nuisance_hidden_dims: tuple[int, ...] = (32,)

    variational: bool = False
    residual_transition: bool = True
    dropout: float = 0.0
    gradient_reversal_coefficient: float = 1.0


# =============================================================================
# Composite model
# =============================================================================

class CLSMModel(nn.Module):
    """
    Composite model implementing the generic CLSM architecture.

    Components
    ----------
    encoder:
        Observation-to-latent mapping.
    decoder:
        Latent-to-observation mapping.
    transition:
        One-step latent dynamics.
    nuisance_classifier:
        Optional latent-to-nuisance adversary trained through gradient
        reversal.
    """

    def __init__(self, config: CLSMModelConfig) -> None:
        super().__init__()

        self.config = config

        if config.gradient_reversal_coefficient < 0.0:
            raise ValueError(
                "gradient_reversal_coefficient must be non-negative."
            )

        if config.variational:
            self.encoder = VariationalObservationEncoder(
                observation_dim=config.observation_dim,
                latent_dim=config.latent_dim,
                hidden_dims=config.encoder_hidden_dims,
                dropout=config.dropout,
            )
        else:
            self.encoder = ObservationEncoder(
                observation_dim=config.observation_dim,
                latent_dim=config.latent_dim,
                hidden_dims=config.encoder_hidden_dims,
                dropout=config.dropout,
            )

        self.decoder = ObservationDecoder(
            latent_dim=config.latent_dim,
            observation_dim=config.observation_dim,
            hidden_dims=config.decoder_hidden_dims,
            dropout=config.dropout,
        )

        self.transition = LatentTransitionModel(
            latent_dim=config.latent_dim,
            hidden_dims=config.transition_hidden_dims,
            dropout=config.dropout,
            residual=config.residual_transition,
        )

        self.nuisance_classifier: NuisanceClassifier | None

        if config.n_nuisances is None:
            self.nuisance_classifier = None
        else:
            if config.n_nuisances < 2:
                raise ValueError(
                    "n_nuisances must be at least 2 when the nuisance "
                    "adversary is enabled."
                )

            self.nuisance_classifier = NuisanceClassifier(
                latent_dim=config.latent_dim,
                n_nuisances=config.n_nuisances,
                hidden_dims=config.nuisance_hidden_dims,
            )

    def encode(
        self,
        observation: Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, Tensor]:
        if isinstance(self.encoder, VariationalObservationEncoder):
            latent, mean, log_variance = self.encoder(
                observation,
                sample=sample,
            )
            return {
                "latent": latent,
                "mean": mean,
                "log_variance": log_variance,
            }

        latent = self.encoder(observation)

        return {
            "latent": latent,
            "mean": latent,
        }

    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent representations."""
        return self.decoder(latent)

    def predict_next_latent(self, latent: Tensor) -> Tensor:
        """Predict the next latent state."""
        return self.transition(latent)

    def rollout_latent(
        self,
        initial_latent: Tensor,
        horizon: int,
    ) -> Tensor:
        """Roll out latent dynamics for several steps."""
        return self.transition.rollout(initial_latent, horizon)

    def predict_nuisance(
        self,
        latent: Tensor,
        *,
        adversarial: bool = True,
        coefficient: float | None = None,
    ) -> Tensor:
        """
        Predict nuisance identity from latent representations.

        Parameters
        ----------
        latent:
            Latent representations.
        adversarial:
            If true, apply gradient reversal before the nuisance classifier.
        coefficient:
            Optional gradient-reversal coefficient used when ``adversarial`` is
            true. If omitted, the value from :class:`CLSMModelConfig` is used.
        """
        if self.nuisance_classifier is None:
            raise RuntimeError(
                "No categorical nuisance adversary is configured. "
                "Set n_nuisances to an integer greater than or equal to 2."
            )

        if not adversarial:
            return self.nuisance_classifier(latent)

        reversal_coefficient = (
            self.config.gradient_reversal_coefficient
            if coefficient is None
            else float(coefficient)
        )

        reversed_latent = gradient_reverse(
            latent,
            coefficient=reversal_coefficient,
        )

        return self.nuisance_classifier(
            reversed_latent
        )

    def forward(
        self,
        observation: Tensor,
        *,
        counterfactual_observation: Tensor | None = None,
        sample: bool = True,
        rollout_horizon: int | None = None,
        adversarial_coefficient: float | None = None,
    ) -> dict[str, Tensor]:
        """
        Run the full model on one observation sequence.

        Parameters
        ----------
        observation:
            Tensor of shape ``(batch, time, observation_dim)`` for observation
            sequences or ``(..., observation_dim)`` for independent observations.
            Temporal predictions require an explicit batch and time dimension.
        counterfactual_observation:
            Optional second nuisance view of the same latent states.
        sample:
            Sample from the variational posterior when enabled.
        rollout_horizon:
            Optional number of future latent rollout steps from the final
            encoded state.
        adversarial_coefficient:
            Optional gradient-reversal coefficient for the nuisance
            adversary. If omitted, the model configuration value is used.

        Returns
        -------
        dict
            Named tensors used by the different CLSM losses.
        """
        encoded = self.encode(observation, sample=sample)
        latent = encoded["latent"]

        output: dict[str, Tensor] = {
            **encoded,
            "reconstructed_observation": self.decode(latent),
        }

        if self.nuisance_classifier is not None:
            output["nuisance_logits"] = self.predict_nuisance(
                latent,
                adversarial=True,
                coefficient=adversarial_coefficient,
            )

        if latent.ndim >= 3 and latent.shape[-2] >= 2:
            output["predicted_next_latent"] = self.predict_next_latent(
                latent[..., :-1, :]
            )

        if counterfactual_observation is not None:
            counterfactual_encoded = self.encode(
                counterfactual_observation,
                sample=sample,
            )
            output["counterfactual_latent"] = counterfactual_encoded["latent"]
            output["counterfactual_mean"] = counterfactual_encoded["mean"]
            if "log_variance" in counterfactual_encoded:
                output["counterfactual_log_variance"] = (
                    counterfactual_encoded["log_variance"]
                )

        if rollout_horizon is not None:
            if rollout_horizon < 1:
                raise ValueError("rollout_horizon must be at least 1.")

            initial_latent = (
                latent[..., -1, :]
                if latent.ndim >= 3
                else latent
            )
            output["latent_rollout"] = self.rollout_latent(
                initial_latent,
                rollout_horizon,
            )

        return output

    def parameter_groups(self) -> dict[str, Iterable[nn.Parameter]]:
        """
        Return model parameters grouped by functional component.

        The groups can be used to configure separate optimizers, learning rates, or
        training schedules.
        """
        groups: dict[str, Iterable[nn.Parameter]] = {
            "encoder": self.encoder.parameters(),
            "decoder": self.decoder.parameters(),
            "transition": self.transition.parameters(),
        }

        if self.nuisance_classifier is not None:
            groups["nuisance_adversary"] = (
                self.nuisance_classifier.parameters()
            )

        return groups
