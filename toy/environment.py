"""
Toy environment.

Author: Gwenolé Quellec
Year: 2026

This module defines a controlled synthetic environment for illustrating
Constrained Latent State Modeling (CLSM). It separates three components:

1. the true latent state and its dynamics;
2. episode-level nuisance factors that affect only the measurement process;
3. observations generated from both the latent state and the nuisances.

The module intentionally contains no dataset, model, loss, or training logic.

Example
--------
python -m toy.environment \
    --metadata toy/metadata.json \
    --output-dir data/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from clsm.datasets import (
    CounterfactualEpisode,
    Episode,
    generate_splits,
    load_metadata,
)


# =============================================================================
# Type aliases
# =============================================================================

FloatArray = NDArray[np.float64]
Distribution = Literal["id", "ood"]


# =============================================================================
# Nuisance representation
# =============================================================================

@dataclass(frozen=True)
class Nuisance:
    """
    Episode-level nuisance factors.

    These variables affect the observation process but never the latent
    dynamics.

    Parameters
    ----------
    offset:
        Nuisance-dependent translation applied to the observed position.
        Shape: ``(2,)``.
    scale:
        Nuisance-dependent axis-wise scaling applied to the observed
        position. Shape: ``(2,)``.
    sensor_bias:
        Nuisance-specific bias channels added directly to the observation.
        Shape: ``(3,)``.
    nuisance_id:
        Integer identifier for the nuisance condition.
    distribution:
        Whether the nuisance belongs to the nominal in-distribution family
        or to the out-of-distribution family.
    """

    offset: FloatArray
    scale: FloatArray
    sensor_bias: FloatArray
    nuisance_id: int
    distribution: Distribution = "id"

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        sensor_bias = np.asarray(self.sensor_bias, dtype=np.float64)

        if offset.shape != (2,):
            raise ValueError(
                f"offset must have shape (2,), got {offset.shape}."
            )
        if scale.shape != (2,):
            raise ValueError(
                f"scale must have shape (2,), got {scale.shape}."
            )
        if sensor_bias.shape != (3,):
            raise ValueError(
                "sensor_bias must have shape (3,), "
                f"got {sensor_bias.shape}."
            )
        if np.any(scale <= 0.0):
            raise ValueError(
                "All scale values must be strictly positive."
            )
        if self.distribution not in ("id", "ood"):
            raise ValueError(
                "distribution must be either 'id' or 'ood'."
            )

        object.__setattr__(self, "offset", offset.copy())
        object.__setattr__(self, "scale", scale.copy())
        object.__setattr__(self, "sensor_bias", sensor_bias.copy())
        object.__setattr__(self, "nuisance_id", int(self.nuisance_id))

    @property
    def is_ood(self) -> bool:
        """Whether this nuisance is out-of-distribution."""
        return self.distribution == "ood"

    def continuous_vector(self) -> FloatArray:
        """
        Return the seven continuous nuisance parameters.

        The vector contains two offsets, two axis-wise scales, and three
        sensor-bias values.
        """
        return np.concatenate(
            (
                self.offset,
                self.scale,
                self.sensor_bias,
            )
        ).astype(np.float64)

    def as_vector(
        self,
        *,
        include_id: bool = True,
    ) -> FloatArray:
        """
        Return the nuisance factors as one flat numeric vector.

        Parameters
        ----------
        include_id:
            If true, append ``nuisance_id`` to the seven continuous nuisance
            parameters.
        """
        vector = self.continuous_vector()

        if include_id:
            vector = np.concatenate(
                (
                    vector,
                    np.array(
                        [float(self.nuisance_id)],
                        dtype=np.float64,
                    ),
                )
            )

        return vector


# =============================================================================
# Toy environment
# =============================================================================

class ToyEnvironment:
    """
    Controlled synthetic dynamical environment for the toy experiments.

    The true latent state is

    ``s_t = (x_t, y_t, vx_t, vy_t)``

    and evolves in a bounded two-dimensional domain. Small Gaussian process
    perturbations are applied to the velocity. When the trajectory reaches a
    boundary, the corresponding velocity component is reflected.

    The latent state is mapped to a 12-dimensional observation vector. The
    observation combines nuisance-distorted position measurements, velocity,
    nonlinear state features, nuisance-specific sensor-bias channels, and a
    constant bias channel. Nuisances affect only the observation process and
    never the latent dynamics.

    Parameters
    ----------
    dt:
        Integration time step.
    bounds:
        Lower and upper spatial bounds for both coordinates.
    process_noise_std:
        Standard deviation of the Gaussian perturbation applied to velocity
        at every time step.
    observation_noise_std:
        Standard deviation of additive Gaussian observation noise.
    velocity_range:
        Minimum and maximum initial speed.
    n_nuisances:
        Number of nominal nuisance conditions available for in-distribution
        generation.
    seed:
        Random seed.
    """

    state_dim: int = 4
    observation_dim: int = 12
    continuous_nuisance_dim: int = 7

    # -------------------------------------------------------------------------
    # Initialization and validation
    # -------------------------------------------------------------------------

    def __init__(
        self,
        dt: float = 0.1,
        bounds: tuple[float, float] = (-1.0, 1.0),
        process_noise_std: float = 0.01,
        observation_noise_std: float = 0.03,
        velocity_range: tuple[float, float] = (0.15, 0.45),
        n_nuisances: int = 4,
        seed: int | None = None,
    ) -> None:
        self._validate_parameters(
            dt=dt,
            bounds=bounds,
            process_noise_std=process_noise_std,
            observation_noise_std=observation_noise_std,
            velocity_range=velocity_range,
            n_nuisances=n_nuisances,
        )

        self.dt = float(dt)
        self.lower_bound = float(bounds[0])
        self.upper_bound = float(bounds[1])
        self.process_noise_std = float(process_noise_std)
        self.observation_noise_std = float(observation_noise_std)
        self.velocity_range = (
            float(velocity_range[0]),
            float(velocity_range[1]),
        )
        self.n_nuisances = int(n_nuisances)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_parameters(
        *,
        dt: float,
        bounds: tuple[float, float],
        process_noise_std: float,
        observation_noise_std: float,
        velocity_range: tuple[float, float],
        n_nuisances: int,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be strictly positive.")
        if bounds[0] >= bounds[1]:
            raise ValueError("bounds must satisfy lower < upper.")
        if process_noise_std < 0:
            raise ValueError(
                "process_noise_std must be non-negative."
            )
        if observation_noise_std < 0:
            raise ValueError(
                "observation_noise_std must be non-negative."
            )
        if (
            velocity_range[0] <= 0
            or velocity_range[0] > velocity_range[1]
        ):
            raise ValueError(
                "velocity_range must satisfy "
                "0 < min_speed <= max_speed."
            )
        if n_nuisances < 1:
            raise ValueError("n_nuisances must be at least 1.")

    def reset_rng(
        self,
        seed: int | None = None,
    ) -> None:
        """Reset the environment random-number generator."""
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Latent-state generation
    # -------------------------------------------------------------------------

    def sample_initial_state(self) -> FloatArray:
        """
        Sample an initial latent state.

        Returns
        -------
        ndarray of shape (4,)
            ``(x, y, vx, vy)``.
        """
        position = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=2,
        )

        speed = self.rng.uniform(
            self.velocity_range[0],
            self.velocity_range[1],
        )
        angle = self.rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        velocity = speed * np.array(
            [
                np.cos(angle),
                np.sin(angle),
            ],
            dtype=np.float64,
        )

        return np.concatenate(
            (
                position,
                velocity,
            )
        ).astype(np.float64)

    def step(
        self,
        latent_state: FloatArray,
    ) -> FloatArray:
        """
        Advance the latent dynamics by one time step.

        Parameters
        ----------
        latent_state:
            Array of shape ``(4,)`` containing ``(x, y, vx, vy)``.
        """
        state = self._validate_state(
            latent_state
        )
        x, y, vx, vy = state

        velocity_noise = self.rng.normal(
            loc=0.0,
            scale=self.process_noise_std,
            size=2,
        )

        vx += velocity_noise[0]
        vy += velocity_noise[1]

        x += self.dt * vx
        y += self.dt * vy

        x, vx = self._reflect(
            x,
            vx,
        )
        y, vy = self._reflect(
            y,
            vy,
        )

        return np.array(
            [
                x,
                y,
                vx,
                vy,
            ],
            dtype=np.float64,
        )

    def generate_latent_trajectory(
        self,
        length: int,
        *,
        initial_state: FloatArray | None = None,
    ) -> FloatArray:
        """
        Generate one trajectory of true latent states.

        Parameters
        ----------
        length:
            Number of time points.
        initial_state:
            Optional initial state of shape ``(4,)``.
        """
        if length < 2:
            raise ValueError(
                "length must be at least 2."
            )

        state_0 = (
            self.sample_initial_state()
            if initial_state is None
            else self._validate_state(
                initial_state
            )
        )

        trajectory = np.empty(
            (
                length,
                self.state_dim,
            ),
            dtype=np.float64,
        )
        trajectory[0] = state_0

        for time_index in range(
            1,
            length,
        ):
            trajectory[time_index] = self.step(
                trajectory[time_index - 1]
            )

        return trajectory

    # -------------------------------------------------------------------------
    # Nuisance sampling
    # -------------------------------------------------------------------------

    def sample_nuisance(
        self,
        nuisance_id: int | None = None,
        *,
        distribution: Distribution = "id",
    ) -> Nuisance:
        """
        Sample one in-distribution or out-of-distribution nuisance.

        In-distribution nuisances belong to one of ``n_nuisances`` nominal
        conditions. OOD nuisances are sampled from shifted parameter
        distributions while leaving the latent dynamics unchanged.
        """
        if distribution not in ("id", "ood"):
            raise ValueError(
                "distribution must be either 'id' or 'ood'."
            )

        if nuisance_id is None:
            nuisance_id = (
                int(
                    self.rng.integers(
                        0,
                        self.n_nuisances,
                    )
                )
                if distribution == "id"
                else self.n_nuisances
            )

        if distribution == "id":
            if not 0 <= nuisance_id < self.n_nuisances:
                raise ValueError(
                    "In-distribution nuisance_id must be in "
                    f"[0, {self.n_nuisances - 1}]."
                )

            center = np.linspace(
                -0.8,
                0.8,
                self.n_nuisances,
            )[nuisance_id]

            offset = self.rng.normal(
                loc=center,
                scale=0.15,
                size=2,
            )
            scale = self.rng.lognormal(
                mean=0.15 * center,
                sigma=0.08,
                size=2,
            )
            sensor_bias = self.rng.normal(
                loc=0.5 * center,
                scale=0.10,
                size=3,
            )
        else:
            center = float(
                self.rng.choice(
                    (-1.5, 1.5)
                )
            )

            offset = self.rng.normal(
                loc=center,
                scale=0.20,
                size=2,
            )
            scale = self.rng.lognormal(
                mean=0.35 * np.sign(center),
                sigma=0.12,
                size=2,
            )
            sensor_bias = self.rng.normal(
                loc=0.9 * np.sign(center),
                scale=0.15,
                size=3,
            )

        return Nuisance(
            offset=offset,
            scale=scale,
            sensor_bias=sensor_bias,
            nuisance_id=int(nuisance_id),
            distribution=distribution,
        )

    # -------------------------------------------------------------------------
    # Observation generation
    # -------------------------------------------------------------------------

    def measure(
        self,
        latent_state: FloatArray,
        nuisance: Nuisance,
        *,
        add_noise: bool = True,
    ) -> FloatArray:
        """
        Generate one 12-dimensional measurement vector.

        Observation channels
        --------------------
        0-1:
            Position after nuisance-dependent axis-wise scaling and
            translation.
        2-3:
            Instantaneous velocity.
        4-7:
            Nonlinear state features:
            ``sin(pi*x)``, ``cos(pi*y)``, ``x*y``, and ``x**2-y**2``.
        8-10:
            Nuisance-specific sensor-bias channels.
        11:
            Constant bias channel.

        Gaussian observation noise is added to every channel when
        ``add_noise`` is true. The nuisance identifier is metadata and is
        deliberately not inserted into the observation vector.
        """
        x, y, vx, vy = self._validate_state(
            latent_state
        )

        transformed_position = (
            nuisance.scale
            * np.array(
                [
                    x,
                    y,
                ],
                dtype=np.float64,
            )
            + nuisance.offset
        )

        nonlinear_features = np.array(
            [
                np.sin(np.pi * x),
                np.cos(np.pi * y),
                x * y,
                x**2 - y**2,
            ],
            dtype=np.float64,
        )

        observation = np.concatenate(
            (
                transformed_position,
                np.array(
                    [
                        vx,
                        vy,
                    ],
                    dtype=np.float64,
                ),
                nonlinear_features,
                nuisance.sensor_bias,
                np.array(
                    [1.0],
                    dtype=np.float64,
                ),
            )
        )

        if observation.shape != (
            self.observation_dim,
        ):
            raise RuntimeError(
                "The measurement function produced an "
                f"unexpected shape: {observation.shape}."
            )

        if add_noise:
            observation = observation + self.rng.normal(
                loc=0.0,
                scale=self.observation_noise_std,
                size=self.observation_dim,
            )

        return observation.astype(
            np.float64
        )

    def measure_trajectory(
        self,
        latent_state: FloatArray,
        nuisance: Nuisance,
        *,
        add_noise: bool = True,
    ) -> FloatArray:
        """Measure a complete latent trajectory."""
        trajectory = self._validate_trajectory(
            latent_state
        )

        return np.stack(
            [
                self.measure(
                    state,
                    nuisance,
                    add_noise=add_noise,
                )
                for state in trajectory
            ],
            axis=0,
        )

    # -------------------------------------------------------------------------
    # Episode generation
    # -------------------------------------------------------------------------

    def generate_episode(
        self,
        length: int = 50,
        *,
        initial_state: FloatArray | None = None,
        nuisance: Nuisance | None = None,
        add_noise: bool = True,
        ood: bool = False,
        distribution: Distribution | None = None,
    ) -> Episode[Nuisance]:
        """Generate one latent trajectory and one observation sequence."""
        if distribution is None:
            distribution = (
                "ood"
                if ood
                else "id"
            )

        latent_state = self.generate_latent_trajectory(
            length=length,
            initial_state=initial_state,
        )

        if nuisance is None:
            nuisance = self.sample_nuisance(
                distribution=distribution
            )

        observation = self.measure_trajectory(
            latent_state,
            nuisance,
            add_noise=add_noise,
        )

        return Episode(
            latent_state=latent_state,
            observation=observation,
            nuisance=nuisance,
            is_ood=nuisance.is_ood,
        )

    def generate_counterfactual_episode(
        self,
        length: int = 50,
        *,
        initial_state: FloatArray | None = None,
        nuisance_a: Nuisance | None = None,
        nuisance_b: Nuisance | None = None,
        add_noise: bool = True,
        ood_a: bool = False,
        ood_b: bool = False,
        distribution_a: Distribution | None = None,
        distribution_b: Distribution | None = None,
    ) -> CounterfactualEpisode[Nuisance]:
        """
        Generate two nuisance views of one latent trajectory.

        This method provides a controlled counterfactual oracle for
        evaluation. Whether generated pairs are included in a dataset is
        decided by the dataset-generation module.
        """
        if distribution_a is None:
            distribution_a = (
                "ood"
                if ood_a
                else "id"
            )
        if distribution_b is None:
            distribution_b = (
                "ood"
                if ood_b
                else "id"
            )

        latent_state = self.generate_latent_trajectory(
            length=length,
            initial_state=initial_state,
        )

        if nuisance_a is None:
            nuisance_a = self.sample_nuisance(
                distribution=distribution_a
            )

        if nuisance_b is None:
            if distribution_b == "id":
                candidate_ids = [
                    nuisance_index
                    for nuisance_index in range(
                        self.n_nuisances
                    )
                    if nuisance_index
                    != nuisance_a.nuisance_id
                ]

                nuisance_b_id = (
                    int(
                        self.rng.choice(
                            candidate_ids
                        )
                    )
                    if candidate_ids
                    else nuisance_a.nuisance_id
                )

                nuisance_b = self.sample_nuisance(
                    nuisance_id=nuisance_b_id,
                    distribution="id",
                )
            else:
                nuisance_b = self.sample_nuisance(
                    nuisance_id=self.n_nuisances + 1,
                    distribution="ood",
                )

        observation_a = self.measure_trajectory(
            latent_state,
            nuisance_a,
            add_noise=add_noise,
        )
        observation_b = self.measure_trajectory(
            latent_state,
            nuisance_b,
            add_noise=add_noise,
        )

        return CounterfactualEpisode(
            latent_state=latent_state,
            observation_a=observation_a,
            observation_b=observation_b,
            nuisance_a=nuisance_a,
            nuisance_b=nuisance_b,
            is_ood_a=nuisance_a.is_ood,
            is_ood_b=nuisance_b.is_ood,
        )

    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------

    def _validate_state(
        self,
        latent_state: FloatArray,
    ) -> FloatArray:
        state = np.asarray(
            latent_state,
            dtype=np.float64,
        )

        if state.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "latent_state must have shape "
                f"({self.state_dim},), got {state.shape}."
            )

        return state.copy()

    def _validate_trajectory(
        self,
        latent_state: FloatArray,
    ) -> FloatArray:
        trajectory = np.asarray(
            latent_state,
            dtype=np.float64,
        )

        if (
            trajectory.ndim != 2
            or trajectory.shape[1]
            != self.state_dim
        ):
            raise ValueError(
                "latent_state must have shape "
                f"(T, {self.state_dim}), "
                f"got {trajectory.shape}."
            )

        return trajectory

    def _reflect(
        self,
        position: float,
        velocity: float,
    ) -> tuple[float, float]:
        """
        Reflect one coordinate at the environment boundaries.

        A loop is used so that unusually large time steps remain well defined.
        """
        while (
            position < self.lower_bound
            or position > self.upper_bound
        ):
            if position < self.lower_bound:
                position = (
                    2.0 * self.lower_bound
                    - position
                )
                velocity = abs(
                    velocity
                )
            elif position > self.upper_bound:
                position = (
                    2.0 * self.upper_bound
                    - position
                )
                velocity = -abs(
                    velocity
                )

        return (
            float(position),
            float(velocity),
        )


# =============================================================================
# Command-line interface
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the CLSM toy dataset."
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="JSON file describing the dataset metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the dataset will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--n-validation",
        type=int,
        default=1000,
        help="Number of validation episodes.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Number of test episodes.",
    )
    parser.add_argument(
        "--n-ood",
        type=int,
        default=1000,
        help="Number of out-of-distribution test episodes.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Episode length.",
    )

    args = parser.parse_args()

    metadata = load_metadata(args.metadata)

    state_names = tuple(
        state
        for state in metadata["state_names"]
    )

    environment = ToyEnvironment(
        seed=args.seed,
    )

    splits = generate_splits(
        environment,
        state_names=state_names,
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_test=args.n_test,
        n_ood=args.n_ood,
        length=args.length,
        seed=args.seed,
    )

    splits.save(args.output_dir)


if __name__ == "__main__":
    main()
