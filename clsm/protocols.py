"""
Typing protocols for the generic CLSM framework.

These protocols specify the minimal interfaces expected from environments,
nuisances, and other interchangeable components. Concrete implementations
satisfy these interfaces through structural typing and do not need to inherit
from the protocol classes explicitly.
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .models import CLSMModelConfig
if TYPE_CHECKING:
    from .datasets import CounterfactualEpisode, Episode


# =============================================================================
# Type aliases
# =============================================================================

FloatArray = NDArray[np.float64]
NuisanceT = TypeVar(
    "NuisanceT",
    covariant=True,
)


# =============================================================================
# Nuisance interfaces
# =============================================================================

class NuisanceProtocol(Protocol):
    """Interface describing one nuisance configuration."""

    @property
    def nuisance_id(self) -> int:
        ...

    @property
    def is_ood(self) -> bool:
        ...

    def as_vector(self) -> FloatArray:
        ...


# =============================================================================
# Environment interfaces
# =============================================================================

class CLSMEnvironment(Protocol[NuisanceT]):
    """
    Interface of a generic CLSM environment.

    Concrete environments define their own latent dynamics, nuisance
    distributions, and observation generation while exposing a common API
    for dataset generation.
    """

    state_dim: int
    """Dimension of the physical state."""

    observation_dim: int
    """Dimension of one observation vector."""

    n_nuisances: int
    """Number of nuisance classes."""

    def reset_rng(
        self,
        seed: int | None = None,
    ) -> None:
        ...

    def generate_episode(
        self,
        length: int,
        *,
        add_noise: bool = True,
        ood: bool = False,
    ) -> "Episode[NuisanceT]":
        ...

    def generate_counterfactual_episode(
        self,
        length: int,
        *,
        add_noise: bool = True,
        ood_a: bool = False,
        ood_b: bool = False,
    ) -> "CounterfactualEpisode[NuisanceT]":
        ...


# =============================================================================
# Model interfaces
# =============================================================================

class CLSMModelProtocol(Protocol):
    """
    Interface of a generic CLSM model.

    Training, evaluation, and visualization interact with models through
    this protocol, independently of the underlying architecture.
    """
    config: CLSMModelConfig

    def encode(
        self,
        observation: Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, Tensor]:
        ...

    def decode(
        self,
        latent: Tensor,
    ) -> Tensor:
        ...

    def predict_next_latent(
        self,
        latent: Tensor,
    ) -> Tensor:
        ...

    def rollout_latent(
        self,
        initial_latent: Tensor,
        horizon: int,
    ) -> Tensor:
        ...

    def predict_nuisance(
        self,
        latent: Tensor,
        *,
        adversarial: bool = True,
        coefficient: float | None = None,
    ) -> Tensor:
        ...
