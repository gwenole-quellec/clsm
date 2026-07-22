"""
Generic dataset utilities for CLSM experiments.

This module converts episodes produced by any compatible CLSM environment
into serializable train, validation, test, and optional OOD datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from .protocols import CLSMEnvironment


# =============================================================================
# Type aliases
# =============================================================================

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
IndexLike = slice | Sequence[int] | IntArray
NuisanceT = TypeVar("NuisanceT")


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class Episode(Generic[NuisanceT]):
    """One latent trajectory and one corresponding observation view."""

    latent_state: FloatArray
    observation: FloatArray
    nuisance: NuisanceT
    is_ood: bool


@dataclass(frozen=True)
class CounterfactualEpisode(Generic[NuisanceT]):
    """Two observation views generated from one latent trajectory."""

    latent_state: FloatArray
    observation_a: FloatArray
    observation_b: FloatArray
    nuisance_a: NuisanceT
    nuisance_b: NuisanceT
    is_ood_a: bool
    is_ood_b: bool


@dataclass(frozen=True)
class CLSMDataset:
    """
    Collection of fixed-length CLSM episodes.

    Parameters
    ----------
    latent_state:
        True latent trajectories, shape ``(N, T, state_dim)``.
    observation:
        Primary observation view, shape ``(N, T, observation_dim)``.
    nuisance:
        Continuous nuisance descriptors associated with each episode.
        Shape: (N, nuisance_dim).
    nuisance_id:
        Primary nuisance identifiers, shape ``(N,)``.
    observation_is_ood:
        Per-episode OOD flags for the primary observation, shape ``(N,)``.
    counterfactual_observation:
        Optional second observation view of the same latent trajectories.
    counterfactual_nuisance:
        Optional continuous nuisance vectors for the second view.
    counterfactual_nuisance_id:
        Optional nuisance identifiers for the second view.
    counterfactual_is_ood:
        Optional per-episode OOD flags for the second view.
    """

    latent_state: FloatArray
    observation: FloatArray
    nuisance: FloatArray
    nuisance_id: IntArray
    observation_is_ood: BoolArray
    state_names: tuple[str, ...]

    counterfactual_observation: FloatArray | None = None
    counterfactual_nuisance: FloatArray | None = None
    counterfactual_nuisance_id: IntArray | None = None
    counterfactual_is_ood: BoolArray | None = None

    def __post_init__(self) -> None:
        self._validate()

    @property
    def n_episodes(self) -> int:
        return int(self.latent_state.shape[0])

    @property
    def episode_length(self) -> int:
        return int(self.latent_state.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.latent_state.shape[2])

    @property
    def observation_dim(self) -> int:
        return int(self.observation.shape[2])

    @property
    def nuisance_dim(self) -> int:
        return int(self.nuisance.shape[1])

    @property
    def has_counterfactuals(self) -> bool:
        return self.counterfactual_observation is not None

    @property
    def is_ood(self) -> bool:
        """
        Whether every primary observation in the dataset is OOD.

        This preserves the previous dataset-level API while retaining
        per-episode OOD flags internally.
        """
        return bool(np.all(self.observation_is_ood))

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(self.n_episodes):
            yield self.episode(index)

    def episode(self, index: int) -> dict[str, object]:
        if not 0 <= index < self.n_episodes:
            raise IndexError(
                f"Episode index {index} is outside [0, {self.n_episodes - 1}]."
            )

        item: dict[str, object] = {
            "latent_state": self.latent_state[index],
            "observation": self.observation[index],
            "nuisance": self.nuisance[index],
            "nuisance_id": int(self.nuisance_id[index]),
            "observation_is_ood": bool(self.observation_is_ood[index]),
        }

        if self.has_counterfactuals:
            item.update(
                {
                    "counterfactual_observation":
                        self.counterfactual_observation[index],
                    "counterfactual_nuisance":
                        self.counterfactual_nuisance[index],
                    "counterfactual_nuisance_id":
                        int(self.counterfactual_nuisance_id[index]),
                    "counterfactual_is_ood":
                        bool(self.counterfactual_is_ood[index]),
                }
            )

        return item

    def subset(self, indices: IndexLike) -> "CLSMDataset":
        index_array = np.arange(self.n_episodes)[indices]
        index_array = np.atleast_1d(index_array).astype(np.int64)

        return CLSMDataset(
            latent_state=self.latent_state[index_array],
            observation=self.observation[index_array],
            nuisance=self.nuisance[index_array],
            nuisance_id=self.nuisance_id[index_array],
            observation_is_ood=self.observation_is_ood[index_array],
            counterfactual_observation=(
                None
                if self.counterfactual_observation is None
                else self.counterfactual_observation[index_array]
            ),
            counterfactual_nuisance=(
                None
                if self.counterfactual_nuisance is None
                else self.counterfactual_nuisance[index_array]
            ),
            counterfactual_nuisance_id=(
                None
                if self.counterfactual_nuisance_id is None
                else self.counterfactual_nuisance_id[index_array]
            ),
            counterfactual_is_ood=(
                None
                if self.counterfactual_is_ood is None
                else self.counterfactual_is_ood[index_array]
            ),
            state_names=self.state_names,
        )

    def shuffled(self, seed: int | None = None) -> "CLSMDataset":
        rng = np.random.default_rng(seed)
        return self.subset(rng.permutation(self.n_episodes))

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, object] = {
            "latent_state": self.latent_state,
            "observation": self.observation,
            "nuisance": self.nuisance,
            "nuisance_id": self.nuisance_id,
            "observation_is_ood": self.observation_is_ood,
            "has_counterfactuals": np.array(
                self.has_counterfactuals,
                dtype=np.bool_,
            ),
            "state_names": np.asarray(
                self.state_names,
                dtype="U",
            ),
        }

        if self.has_counterfactuals:
            arrays.update(
                {
                    "counterfactual_observation":
                        self.counterfactual_observation,
                    "counterfactual_nuisance":
                        self.counterfactual_nuisance,
                    "counterfactual_nuisance_id":
                        self.counterfactual_nuisance_id,
                    "counterfactual_is_ood":
                        self.counterfactual_is_ood,
                }
            )

        np.savez_compressed(output_path, **arrays)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "CLSMDataset":
        input_path = Path(path)

        with np.load(input_path, allow_pickle=False) as archive:
            has_counterfactuals = bool(
                archive["has_counterfactuals"].item()
            )

            if "observation_is_ood" in archive:
                observation_is_ood = archive[
                    "observation_is_ood"
                ].astype(np.bool_)
            else:
                raise KeyError("Missing key 'observation_is_ood'.")

            if "state_names" in archive:
                state_names = tuple(archive["state_names"].tolist())
            else:
                raise KeyError("Missing key 'state_names'.")

            if has_counterfactuals:
                if "counterfactual_is_ood" in archive:
                    counterfactual_is_ood = archive[
                        "counterfactual_is_ood"
                    ].astype(np.bool_)
                else:
                    counterfactual_is_ood = observation_is_ood.copy()
            else:
                counterfactual_is_ood = None

            return cls(
                latent_state=archive["latent_state"],
                observation=archive["observation"],
                nuisance=archive["nuisance"],
                nuisance_id=archive["nuisance_id"].astype(np.int64),
                observation_is_ood=observation_is_ood,
                counterfactual_observation=(
                    archive["counterfactual_observation"]
                    if has_counterfactuals
                    else None
                ),
                counterfactual_nuisance=(
                    archive["counterfactual_nuisance"]
                    if has_counterfactuals
                    else None
                ),
                counterfactual_nuisance_id=(
                    archive["counterfactual_nuisance_id"].astype(np.int64)
                    if has_counterfactuals
                    else None
                ),
                counterfactual_is_ood=counterfactual_is_ood,
                state_names=state_names,
            )

    def _validate(self) -> None:
        latent_state = np.asarray(self.latent_state)
        observation = np.asarray(self.observation)
        nuisance = np.asarray(self.nuisance)
        nuisance_id = np.asarray(self.nuisance_id)
        observation_is_ood = np.asarray(self.observation_is_ood)

        if latent_state.ndim != 3:
            raise ValueError(
                "latent_state must have shape (N, T, state_dim)."
            )
        if observation.ndim != 3:
            raise ValueError(
                "observation must have shape (N, T, observation_dim)."
            )
        if latent_state.shape[:2] != observation.shape[:2]:
            raise ValueError(
                "latent_state and observation must share episode and time axes."
            )

        n_episodes = latent_state.shape[0]

        if nuisance.ndim != 2 or nuisance.shape[0] != n_episodes:
            raise ValueError(
                "nuisance must have shape (N, nuisance_dim)."
            )
        if nuisance_id.shape != (n_episodes,):
            raise ValueError("nuisance_id must have shape (N,).")
        if observation_is_ood.shape != (n_episodes,):
            raise ValueError(
                "observation_is_ood must have shape (N,)."
            )

        counterfactual_fields = (
            self.counterfactual_observation,
            self.counterfactual_nuisance,
            self.counterfactual_nuisance_id,
            self.counterfactual_is_ood,
        )
        provided = tuple(
            field is not None
            for field in counterfactual_fields
        )

        if any(provided) and not all(provided):
            raise ValueError(
                "All counterfactual fields must be provided together."
            )

        if all(provided):
            if self.counterfactual_observation.shape != observation.shape:
                raise ValueError(
                    "counterfactual_observation must match observation shape."
                )
            if self.counterfactual_nuisance.shape != nuisance.shape:
                raise ValueError(
                    "counterfactual_nuisance must match nuisance shape."
                )
            if self.counterfactual_nuisance_id.shape != nuisance_id.shape:
                raise ValueError(
                    "counterfactual_nuisance_id must match nuisance_id shape."
                )
            if self.counterfactual_is_ood.shape != observation_is_ood.shape:
                raise ValueError(
                    "counterfactual_is_ood must match observation_is_ood shape."
                )

        if len(self.state_names) != self.state_dim:
            raise ValueError(
                f"state_names must contain {self.state_dim} names."
            )


@dataclass(frozen=True)
class DatasetSplits:
    """Reproducible train, validation, test, and optional OOD splits."""

    train: CLSMDataset
    validation: CLSMDataset
    test: CLSMDataset
    ood: CLSMDataset | None = None

    def save(self, directory: str | Path) -> dict[str, Path]:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "train": self.train.save(output_dir / "train.npz"),
            "validation": self.validation.save(
                output_dir / "validation.npz"
            ),
            "test": self.test.save(output_dir / "test.npz"),
        }

        if self.ood is not None:
            paths["ood"] = self.ood.save(output_dir / "ood.npz")

        return paths


@dataclass(frozen=True)
class PredictionWindows:
    """Temporal prediction windows extracted from a CLSMDataset."""

    context_observation: FloatArray
    future_observation: FloatArray
    context_latent_state: FloatArray
    future_latent_state: FloatArray
    nuisance: FloatArray
    nuisance_id: IntArray
    episode_index: IntArray
    start_index: IntArray
    counterfactual_context_observation: FloatArray | None = None
    counterfactual_future_observation: FloatArray | None = None

    def __len__(self) -> int:
        return int(self.context_observation.shape[0])


# =============================================================================
# Dataset generation
# =============================================================================

def generate_dataset(
    env: CLSMEnvironment[NuisanceT],
    state_names: tuple[str, ...],
    n_episodes: int,
    length: int,
    *,
    counterfactual: bool = False,
    ood: bool = False,
    seed: int | None = None,
    add_noise: bool = True,
) -> CLSMDataset:
    """
    Generate a CLSM dataset.

    Standard unpaired episodes are generated by default. When
    ``counterfactual=True``, a second observation sequence is generated from
    the same latent trajectory under an independently sampled nuisance. When
    both ``counterfactual=True`` and ``ood=True``, both views use OOD nuisance
    distributions.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1.")
    if length < 2:
        raise ValueError("length must be at least 2.")

    if seed is not None:
        env.reset_rng(seed)

    if counterfactual:
        episodes = [
            env.generate_counterfactual_episode(
                length=length,
                add_noise=add_noise,
                ood_a=ood,
                ood_b=ood,
            )
            for _ in range(n_episodes)
        ]
        return _stack_counterfactual_episodes(episodes, state_names)

    episodes = [
        env.generate_episode(
            length=length,
            add_noise=add_noise,
            ood=ood,
        )
        for _ in range(n_episodes)
    ]
    return _stack_episodes(episodes, state_names)


def generate_ood_dataset(
    env: CLSMEnvironment[NuisanceT],
    state_names: tuple[str, ...],
    n_episodes: int,
    length: int,
    *,
    counterfactual: bool = True,
    seed: int | None = None,
    add_noise: bool = True,
) -> CLSMDataset:
    """Generate a dataset whose primary view is genuinely OOD."""
    return generate_dataset(
        env,
        state_names,
        n_episodes=n_episodes,
        length=length,
        counterfactual=counterfactual,
        ood=True,
        seed=seed,
        add_noise=add_noise,
    )


def generate_splits(
    env: CLSMEnvironment[NuisanceT],
    state_names: tuple[str, ...],
    *,
    n_train: int,
    n_validation: int,
    n_test: int,
    n_ood: int,
    length: int,
    seed: int = 42,
    add_noise: bool = True,
) -> DatasetSplits:
    """
    Generate independent and reproducible dataset splits.

    The training split contains only unpaired observations,
    whereas validation, test, and OOD splits contain counterfactual views for
    evaluation. This separates the adversarial invariance objective used
    during training from the simulator-generated counterfactual oracle used
    to quantify invariance. Saved splits should be reused across all runs.
    """
    for name, value in (
        ("n_train", n_train),
        ("n_validation", n_validation),
        ("n_test", n_test),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1.")

    if n_ood < 0:
        raise ValueError("n_ood must be non-negative.")
    if length < 2:
        raise ValueError("length must be at least 2.")

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(4)
    split_seeds = [
        int(sequence.generate_state(1, dtype=np.uint32)[0])
        for sequence in child_sequences
    ]

    train = generate_dataset(
        env,
        state_names,
        n_episodes=n_train,
        length=length,
        counterfactual=False,
        seed=split_seeds[0],
        add_noise=add_noise,
    )
    validation = generate_dataset(
        env,
        state_names,
        n_episodes=n_validation,
        length=length,
        counterfactual=True,
        seed=split_seeds[1],
        add_noise=add_noise,
    )
    test = generate_dataset(
        env,
        state_names,
        n_episodes=n_test,
        length=length,
        counterfactual=True,
        seed=split_seeds[2],
        add_noise=add_noise,
    )

    ood = None
    if n_ood > 0:
        ood = generate_ood_dataset(
            env,
            state_names,
            n_episodes=n_ood,
            length=length,
            counterfactual=True,
            seed=split_seeds[3],
            add_noise=add_noise,
        )

    return DatasetSplits(
        train=train,
        validation=validation,
        test=test,
        ood=ood,
    )


# =============================================================================
# Dataset transformations
# =============================================================================

def make_prediction_windows(
    dataset: CLSMDataset,
    *,
    context_length: int,
    horizon: int = 1,
    stride: int = 1,
) -> PredictionWindows:
    if context_length < 1:
        raise ValueError("context_length must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    required_length = context_length + horizon
    if required_length > dataset.episode_length:
        raise ValueError(
            "context_length + horizon cannot exceed episode length."
        )

    contexts_obs = []
    futures_obs = []
    contexts_state = []
    futures_state = []
    nuisances = []
    nuisance_ids = []
    episode_indices = []
    start_indices = []

    cf_contexts_obs = [] if dataset.has_counterfactuals else None
    cf_futures_obs = [] if dataset.has_counterfactuals else None

    final_start = dataset.episode_length - required_length

    for episode_index in range(dataset.n_episodes):
        for start in range(0, final_start + 1, stride):
            split = start + context_length
            end = split + horizon

            contexts_obs.append(
                dataset.observation[episode_index, start:split]
            )
            futures_obs.append(
                dataset.observation[episode_index, split:end]
            )
            contexts_state.append(
                dataset.latent_state[episode_index, start:split]
            )
            futures_state.append(
                dataset.latent_state[episode_index, split:end]
            )
            nuisances.append(dataset.nuisance[episode_index])
            nuisance_ids.append(dataset.nuisance_id[episode_index])
            episode_indices.append(episode_index)
            start_indices.append(start)

            if dataset.has_counterfactuals:
                cf_contexts_obs.append(
                    dataset.counterfactual_observation[
                        episode_index,
                        start:split,
                    ]
                )
                cf_futures_obs.append(
                    dataset.counterfactual_observation[
                        episode_index,
                        split:end,
                    ]
                )

    return PredictionWindows(
        context_observation=np.stack(contexts_obs),
        future_observation=np.stack(futures_obs),
        context_latent_state=np.stack(contexts_state),
        future_latent_state=np.stack(futures_state),
        nuisance=np.asarray(nuisances, dtype=np.float64),
        nuisance_id=np.asarray(nuisance_ids, dtype=np.int64),
        episode_index=np.asarray(episode_indices, dtype=np.int64),
        start_index=np.asarray(start_indices, dtype=np.int64),
        counterfactual_context_observation=(
            None if cf_contexts_obs is None else np.stack(cf_contexts_obs)
        ),
        counterfactual_future_observation=(
            None if cf_futures_obs is None else np.stack(cf_futures_obs)
        ),
    )


# =============================================================================
# Metadata utilities
# =============================================================================

def load_metadata(
    path: str | Path,
) -> dict[str, object]:
    """Load dataset metadata from a JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    if not isinstance(metadata, dict):
        raise TypeError(
            "Visualization metadata must be a JSON object."
        )

    return metadata


# =============================================================================
# Internal episode aggregation
# =============================================================================

def _stack_episodes(
    episodes: Sequence[Episode],
    state_names: tuple[str, ...],
) -> CLSMDataset:
    if not episodes:
        raise ValueError("episodes cannot be empty.")

    return CLSMDataset(
        latent_state=np.stack(
            [episode.latent_state for episode in episodes]
        ),
        observation=np.stack(
            [episode.observation for episode in episodes]
        ),
        nuisance=np.stack(
            [episode.nuisance.continuous_vector() for episode in episodes]
        ),
        nuisance_id=np.asarray(
            [episode.nuisance.nuisance_id for episode in episodes],
            dtype=np.int64,
        ),
        observation_is_ood=np.asarray(
            [episode.is_ood for episode in episodes],
            dtype=np.bool_,
        ),
        state_names=state_names,
    )


def _stack_counterfactual_episodes(
    episodes: Sequence[CounterfactualEpisode],
    state_names: tuple[str, ...],
) -> CLSMDataset:
    if not episodes:
        raise ValueError("episodes cannot be empty.")

    return CLSMDataset(
        latent_state=np.stack(
            [episode.latent_state for episode in episodes]
        ),
        observation=np.stack(
            [episode.observation_a for episode in episodes]
        ),
        nuisance=np.stack(
            [
                episode.nuisance_a.continuous_vector()
                for episode in episodes
            ]
        ),
        nuisance_id=np.asarray(
            [episode.nuisance_a.nuisance_id for episode in episodes],
            dtype=np.int64,
        ),
        observation_is_ood=np.asarray(
            [episode.is_ood_a for episode in episodes],
            dtype=np.bool_,
        ),
        counterfactual_observation=np.stack(
            [episode.observation_b for episode in episodes]
        ),
        counterfactual_nuisance=np.stack(
            [
                episode.nuisance_b.continuous_vector()
                for episode in episodes
            ]
        ),
        counterfactual_nuisance_id=np.asarray(
            [episode.nuisance_b.nuisance_id for episode in episodes],
            dtype=np.int64,
        ),
        counterfactual_is_ood=np.asarray(
            [episode.is_ood_b for episode in episodes],
            dtype=np.bool_,
        ),
        state_names=state_names,
    )
