"""
Loss functions for CLSM.

Author: Gwenolé Quellec
Year: 2026

This module implements modular PyTorch losses corresponding to the six
constraints described in Constrained Latent State Modeling (CLSM):

1. predictive sufficiency;
2. minimality;
3. temporal coherence;
4. observation compatibility;
5. invariance to nuisance factors;
6. structural constraints.

The functions are architecture-agnostic. They operate on tensors produced by
any encoder, decoder, transition model or adversarial nuisance predictor.

Notes
-----
The losses in this file are practical surrogates for conceptual constraints.
For example, predictive sufficiency and minimality are information-theoretic
properties, but are approximated here through prediction errors, bottleneck
penalties, and optional variational regularization. Invariance is implemented
through adversarial nuisance prediction: the nuisance classifier minimizes a
classification loss, while a gradient-reversal layer in the model sends the
opposite gradient to the encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# =============================================================================
# Utility functions
# =============================================================================

def _check_same_shape(a: Tensor, b: Tensor, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, "
            f"got {tuple(a.shape)} and {tuple(b.shape)}."
        )


def _masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    Compute a mean with optional broadcasting mask.

    Parameters
    ----------
    values:
        Loss values.
    mask:
        Boolean or numeric mask broadcastable to ``values``.
    """
    if mask is None:
        return values.mean()

    mask = mask.to(device=values.device, dtype=values.dtype)
    try:
        weighted = values * mask
    except RuntimeError as error:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} is not broadcastable to "
            f"values with shape {tuple(values.shape)}."
        ) from error

    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return weighted.sum() / denominator


def _flatten_features(x: Tensor) -> Tensor:
    """
    Flatten every axis except the last feature axis.

    Examples
    --------
    ``(batch, time, features) -> (batch * time, features)``
    """
    if x.ndim < 2:
        raise ValueError("Expected a tensor with at least two dimensions.")
    return x.reshape(-1, x.shape[-1])


# =============================================================================
# Core CLSM losses
# =============================================================================

def observation_compatibility_loss(
    reconstructed_observation: Tensor,
    target_observation: Tensor,
    *,
    loss_type: str = "mse",
    mask: Tensor | None = None,
) -> Tensor:
    """
    Penalize disagreement between observations and their reconstructions.

    This is the practical surrogate for observation compatibility.

    Parameters
    ----------
    reconstructed_observation:
        Decoder output.
    target_observation:
        Ground-truth observation.
    loss_type:
        ``"mse"``, ``"l1"``, or ``"smooth_l1"``.
    mask:
        Optional mask broadcastable to the elementwise loss.
    """
    _check_same_shape(
        reconstructed_observation,
        target_observation,
        "reconstructed_observation",
        "target_observation",
    )

    if loss_type == "mse":
        values = (reconstructed_observation - target_observation).pow(2)
    elif loss_type == "l1":
        values = (reconstructed_observation - target_observation).abs()
    elif loss_type == "smooth_l1":
        values = F.smooth_l1_loss(
            reconstructed_observation,
            target_observation,
            reduction="none",
        )
    else:
        raise ValueError(
            "loss_type must be one of {'mse', 'l1', 'smooth_l1'}."
        )

    return _masked_mean(values, mask)


def predictive_sufficiency_loss(
    predicted_future: Tensor,
    target_future: Tensor,
    *,
    loss_type: str = "mse",
    horizon_weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Penalize open-loop future-prediction error.

    The penultimate tensor axis represents distinct prediction horizons
    originating from the same latent state.
    """
    _check_same_shape(
        predicted_future,
        target_future,
        "predicted_future",
        "target_future",
    )

    if loss_type == "mse":
        values = (
            predicted_future
            - target_future
        ).pow(2)

    elif loss_type == "l1":
        values = (
            predicted_future
            - target_future
        ).abs()

    elif loss_type == "smooth_l1":
        values = F.smooth_l1_loss(
            predicted_future,
            target_future,
            reduction="none",
        )

    else:
        raise ValueError(
            "loss_type must be one of "
            "{'mse', 'l1', 'smooth_l1'}."
        )

    if horizon_weights is not None:
        if values.ndim < 2:
            raise ValueError(
                "horizon_weights require a tensor with a horizon axis."
            )

        horizon = values.shape[-2]

        if (
            horizon_weights.ndim != 1
            or horizon_weights.shape[0] != horizon
        ):
            raise ValueError(
                f"horizon_weights must have shape ({horizon},), got "
                f"{tuple(horizon_weights.shape)}."
            )

        weights = horizon_weights.to(
            device=values.device,
            dtype=values.dtype,
        )

        if torch.any(
            weights < 0
        ):
            raise ValueError(
                "horizon_weights must be non-negative."
            )

        if not torch.any(
            weights > 0
        ):
            raise ValueError(
                "At least one horizon weight must be positive."
            )

        # Preserve the global scale of the objective.
        weights = weights / weights.mean()

        broadcast_shape = [
            1
        ] * values.ndim

        broadcast_shape[-2] = horizon

        values = (
            values
            * weights.reshape(
                broadcast_shape
            )
        )

    return _masked_mean(
        values,
        mask,
    )


def temporal_coherence_loss(
    latent_state: Tensor,
    *,
    predicted_next_latent: Tensor | None = None,
    mode: str = "velocity",
    mask: Tensor | None = None,
) -> Tensor:
    """Encourage coherent latent trajectories."""
    if latent_state.ndim < 3:
        raise ValueError(
            "latent_state must have shape (..., time, latent_dim)."
        )

    if mode == "velocity":
        if latent_state.shape[-2] < 2:
            raise ValueError(
                "At least two time points are required in velocity mode."
            )
        velocity = (
            latent_state[..., 1:, :]
            - latent_state[..., :-1, :]
        )
        values = velocity.pow(2)

    elif mode == "dynamics":
        if predicted_next_latent is None:
            raise ValueError(
                "predicted_next_latent is required in dynamics mode."
            )
        # Preserve gradients through the target latent trajectory so that temporal
        # consistency also shapes the encoder representation.
        target_next = latent_state[..., 1:, :]
        _check_same_shape(
            predicted_next_latent,
            target_next,
            "predicted_next_latent",
            "latent_state[..., 1:, :]",
        )
        values = (predicted_next_latent - target_next).pow(2)

    elif mode == "acceleration":
        if latent_state.shape[-2] < 3:
            raise ValueError(
                "At least three time points are required in acceleration mode."
            )
        acceleration = (
            latent_state[..., 2:, :]
            - 2.0 * latent_state[..., 1:-1, :]
            + latent_state[..., :-2, :]
        )
        values = acceleration.pow(2)

    else:
        raise ValueError(
            "mode must be 'velocity', 'dynamics', or 'acceleration'."
        )

    return _masked_mean(values, mask)


def nuisance_adversarial_loss(
    nuisance_logits: Tensor,
    nuisance_id: Tensor,
) -> Tensor:
    """
    Compute the nuisance-classification loss used by the adversary.

    The model applies gradient reversal before the nuisance classifier.
    Consequently, minimizing this cross-entropy trains the classifier to
    recover nuisance identity while encouraging the encoder to remove
    nuisance-related information.

    Parameters
    ----------
    nuisance_logits:
        Class logits of shape ``(..., n_nuisances)``. In the standard
        sequence setting, the expected shape is
        ``(batch, time, n_nuisances)``.
    nuisance_id:
        Episode-level nuisance labels. Accepted shapes are ``(batch,)`` or
        any shape matching ``nuisance_logits.shape[:-1]``.

    Returns
    -------
    Tensor
        Scalar cross-entropy loss.
    """
    if nuisance_logits.ndim < 2:
        raise ValueError(
            "nuisance_logits must have shape (..., n_nuisances)."
        )

    target_shape = nuisance_logits.shape[:-1]
    labels = nuisance_id.long()

    if labels.shape == target_shape:
        expanded_labels = labels
    elif labels.ndim == 1 and labels.shape[0] == target_shape[0]:
        view_shape = (
            labels.shape[0],
            *([1] * (len(target_shape) - 1)),
        )
        expanded_labels = labels.reshape(view_shape).expand(target_shape)
    else:
        raise ValueError(
            "nuisance_id must either match nuisance_logits.shape[:-1] "
            "or contain one episode-level label per batch element. "
            f"Got logits shape {tuple(nuisance_logits.shape)} and "
            f"label shape {tuple(labels.shape)}."
        )

    flat_logits = nuisance_logits.reshape(
        -1,
        nuisance_logits.shape[-1],
    )
    flat_labels = expanded_labels.reshape(-1)

    return F.cross_entropy(
        flat_logits,
        flat_labels,
    )


def nuisance_adversarial_accuracy(
    nuisance_logits: Tensor,
    nuisance_id: Tensor,
) -> Tensor:
    """
    Compute nuisance-classification accuracy for training diagnostics.

    This function is not part of the optimization objective. Chance-level
    accuracy indicates that nuisance identity is difficult to recover from the
    latent representation, provided that the classifier itself is adequately
    optimized.
    """
    if nuisance_logits.ndim < 2:
        raise ValueError(
            "nuisance_logits must have shape (..., n_nuisances)."
        )

    target_shape = nuisance_logits.shape[:-1]
    labels = nuisance_id.long()

    if labels.shape == target_shape:
        expanded_labels = labels
    elif labels.ndim == 1 and labels.shape[0] == target_shape[0]:
        view_shape = (
            labels.shape[0],
            *([1] * (len(target_shape) - 1)),
        )
        expanded_labels = labels.reshape(view_shape).expand(target_shape)
    else:
        raise ValueError(
            "nuisance_id shape is incompatible with nuisance_logits."
        )

    predictions = nuisance_logits.argmax(dim=-1)
    return (
        predictions == expanded_labels
    ).float().mean()


def minimality_loss(
    latent: Tensor,
    *,
    mode: str = "participation_ratio",
    mean: Tensor | None = None,
    log_variance: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Encourage low effective dimensionality or regularized latent activations.

    Modes
    -----
    ``"participation_ratio"``
        Minimize the effective latent dimensionality by penalizing the
        participation ratio of the latent covariance.

    ``"l1"``
        Mean absolute activation.

    ``"kl"``
        KL divergence from a diagonal Gaussian posterior to a unit Gaussian.
        Requires ``mean`` and ``log_variance``.
    """
    if mode == "participation_ratio":
        flat = _flatten_features(latent)
        flat = flat - flat.mean(dim=0, keepdim=True)

        latent_dim = flat.shape[-1]
        if latent_dim <= 1:
            return flat.new_zeros(())

        covariance = (
            flat.T @ flat
        ) / max(flat.shape[0] - 1, 1)

        eigenvalues = torch.linalg.eigvalsh(
            covariance
        ).clamp_min(0.0)

        total_variance = eigenvalues.sum()

        # A fully collapsed representation has zero effective dimensionality.
        if total_variance <= eps:
            return total_variance.new_zeros(())

        denominator = eigenvalues.pow(2).sum()
        if denominator <= eps:
            return denominator.new_zeros(())

        participation_ratio = total_variance.pow(2) / denominator

        normalized_participation_ratio = (
            participation_ratio - 1.0
        ) / (
            latent_dim - 1.0
        )

        return normalized_participation_ratio.clamp(
            min=0.0,
            max=1.0,
        )

    if mode == "l1":
        return latent.abs().mean()

    if mode == "kl":
        if mean is None or log_variance is None:
            raise ValueError(
                "mean and log_variance are required for KL minimality."
            )

        _check_same_shape(
            mean,
            log_variance,
            "mean",
            "log_variance",
        )

        kl_per_element = -0.5 * (
            1.0
            + log_variance
            - mean.pow(2)
            - log_variance.exp()
        )

        return kl_per_element.mean()

    raise ValueError(
        "mode must be one of "
        "{'participation_ratio', "
        "'l1', 'kl'}."
    )


def structural_constraint_loss(
    latent: Tensor,
    *,
    variance_target: float = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    epsilon: float = 1e-4,
) -> Tensor:
    """
    Encourage non-collapsed and decorrelated latent coordinates.

    A minimum variance is enforced on every latent dimension to avoid
    representation collapse, following the principle introduced in VICReg.

    The loss does not use ground-truth states. It imposes a generic
    factorized structure on the learned representation by combining:
    1. a variance-floor penalty;
    2. an off-diagonal covariance penalty.
    """
    if latent.ndim < 2:
        raise ValueError(
            "latent must have at least two dimensions."
        )

    flat_latent = latent.reshape(
        -1,
        latent.shape[-1],
    )

    centered = (
        flat_latent
        - flat_latent.mean(
            dim=0,
            keepdim=True,
        )
    )

    standard_deviation = torch.sqrt(
        centered.var(
            dim=0,
            unbiased=False,
        )
        + epsilon
    )

    variance_penalty = torch.relu(
        variance_target
        - standard_deviation
    ).pow(2).mean()

    denominator = max(
        centered.shape[0] - 1,
        1,
    )

    covariance = (
        centered.T
        @ centered
    ) / denominator

    off_diagonal = (
        covariance
        - torch.diag(
            torch.diag(covariance)
        )
    )

    covariance_penalty = (
        off_diagonal.pow(2).mean()
    )

    return (
        variance_weight
        * variance_penalty
        + covariance_weight
        * covariance_penalty
    )


# =============================================================================
# Weighted CLSM objective
# =============================================================================

@dataclass(frozen=True)
class ConstraintWeights:
    """Weights for the six CLSM constraint families."""

    predictive: float = 1.0
    minimality: float = 0.0
    temporal: float = 0.0
    observation: float = 0.0
    invariance: float = 0.0
    structural: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} weight must be non-negative.")

    def as_dict(self) -> dict[str, float]:
        return {
            "predictive": self.predictive,
            "minimality": self.minimality,
            "temporal": self.temporal,
            "observation": self.observation,
            "invariance": self.invariance,
            "structural": self.structural,
        }


class CLSMLoss(nn.Module):
    """
    Aggregate already-computed CLSM loss components.

    This class intentionally does not infer which tensors should be compared.
    The training code computes each relevant component using the functions in
    this module, then passes them here. The ``invariance`` component is the
    nuisance-adversarial cross-entropy; gradient reversal is implemented in
    the model rather than in this loss aggregator.

    Example
    -------
    >>> objective = CLSMLoss(
    ...     ConstraintWeights(
    ...         predictive=1.0,
    ...         observation=0.5,
    ...         invariance=0.2,
    ...     )
    ... )
    >>> total, weighted = objective({
    ...     "predictive": prediction_loss,
    ...     "observation": reconstruction_loss,
    ...     "invariance": invariance_term,
    ... })
    >>> weighted.keys()
    dict_keys(["predictive", "observation", "invariance"])
    """

    _VALID_COMPONENTS = {
        "predictive",
        "minimality",
        "temporal",
        "observation",
        "invariance",
        "structural",
    }

    def __init__(self, weights: ConstraintWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        components: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute the weighted sum of supplied loss components.

        Missing zero-weight components are allowed. Missing positive-weight
        components raise an error to avoid silently training a different
        objective from the configured one.
        """
        unknown = set(components) - self._VALID_COMPONENTS
        if unknown:
            raise KeyError(
                f"Unknown CLSM loss components: {sorted(unknown)}."
            )

        weights = self.weights.as_dict()
        missing = [
            name
            for name, weight in weights.items()
            if weight > 0 and name not in components
        ]
        if missing:
            raise KeyError(
                "Missing positive-weight CLSM components: "
                f"{sorted(missing)}."
            )

        if not components:
            raise ValueError("At least one loss component must be supplied.")

        reference = next(iter(components.values()))
        total = reference.new_zeros(())
        weighted_components: dict[str, Tensor] = {}

        for name, component in components.items():
            if component.ndim != 0:
                raise ValueError(
                    f"Loss component '{name}' must be scalar, got shape "
                    f"{tuple(component.shape)}."
                )
            weighted = weights[name] * component
            weighted_components[name] = weighted
            total = total + weighted

        return total, weighted_components
