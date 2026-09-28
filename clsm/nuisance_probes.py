"""
Independent post-hoc probes for CLSM evaluation.

Author: Gwenolé Quellec
Year: 2026

Feature scaling is fit on the training split only. Nonlinear probe
selection uses the external validation split.
"""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .strong_adversary import make_adversary


# =============================================================================
# Constants
# =============================================================================

MIN_LATENT_STD = 1e-6

MAX_VALIDATION_OBSERVATIONS = 10_000
ENCODING_CHUNK_SIZE = 4096

PROBE_VALIDATION_SEED_OFFSET = 1

STRONG_PROBE_INITIALIZATION_SEED = 311
STRONG_PROBE_BATCH_SEED = 312

PROBE_PATIENCE = 25
PROBE_BATCH_SIZE = 256
PROBE_IMPROVEMENT_TOLERANCE = 1e-6

LINEAR_CLASSIFIER_MAX_ITERATIONS = 2_000

NONLINEAR_PROBE_HIDDEN_DIM = 64
NONLINEAR_PROBE_LEARNING_RATE = 1e-3
NONLINEAR_PROBE_L2_PENALTY = 1e-4
STRONG_PROBE_WEIGHT_DECAY = 1e-4


# =============================================================================
# Target and sampling utilities
# =============================================================================

def targets(dataset, task):
    """Repeat episode labels in flattened ``(episode, time)`` order."""

    values = (
        dataset.nuisance_id
        if task == "class"
        else dataset.nuisance
    )

    return np.repeat(
        values,
        dataset.episode_length,
        axis=0,
    )


def sample_indices(
    size,
    maximum,
    seed,
):
    """Select at most ``maximum`` deterministic sample indices."""

    if maximum < 1:
        raise ValueError(
            "Sample limits must be positive"
        )

    if size <= maximum:
        return np.arange(size)

    return np.random.default_rng(
        seed
    ).choice(
        size,
        maximum,
        replace=False,
    )


# =============================================================================
# Generic probe fitting
# =============================================================================

def _fit_probe_impl(
    features,
    data,
    *,
    family,
    task,
    settings,
):
    """
    Fit one linear or nonlinear probe.

    Scalers are fit on the training split only. For nonlinear probes, the
    selected epoch is determined from the external validation split.
    """

    seed = settings["seed"]

    indices = sample_indices(
        len(features["train"]),
        settings["max_samples"],
        seed,
    )

    validation_indices = sample_indices(
        len(features["validation"]),
        settings["validation_samples"],
        seed + PROBE_VALIDATION_SEED_OFFSET,
    )

    feature_scaler = StandardScaler().fit(
        features["train"][indices]
    )

    x_train = feature_scaler.transform(
        features["train"][indices]
    )
    x_validation = feature_scaler.transform(
        features["validation"][validation_indices]
    )

    train_target = targets(
        data["train"],
        task,
    )[indices]

    validation_target = targets(
        data["validation"],
        task,
    )[validation_indices]

    target_scaler = (
        StandardScaler().fit(train_target)
        if task == "continuous"
        else None
    )

    y_train = (
        target_scaler.transform(train_target)
        if target_scaler is not None
        else train_target
    )

    y_validation = (
        target_scaler.transform(validation_target)
        if target_scaler is not None
        else validation_target
    )

    classes = np.unique(
        targets(
            data["train"],
            "class",
        )
    )

    if (
        task == "class"
        and not np.isin(
            validation_target,
            classes,
        ).all()
    ):
        raise ValueError(
            "Validation contains unseen nuisance classes"
        )

    messages = []

    if family == "linear":
        probe = (
            LogisticRegression(
                max_iter=LINEAR_CLASSIFIER_MAX_ITERATIONS,
                random_state=seed,
            )
            if task == "class"
            else LinearRegression()
        )

        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "always"
            )
            probe.fit(
                x_train,
                y_train,
            )

        messages.extend(
            str(warning.message)
            for warning in caught
        )

        metadata = {
            "selected_epoch": None,
            "epochs_run": None,
            "stopped_by_patience": False,
        }

    else:
        probe_class = (
            MLPClassifier
            if task == "class"
            else MLPRegressor
        )

        probe = probe_class(
            hidden_layer_sizes=(NONLINEAR_PROBE_HIDDEN_DIM,),
            activation="relu",
            solver="adam",
            alpha=NONLINEAR_PROBE_L2_PENALTY,
            batch_size=min(
                PROBE_BATCH_SIZE,
                len(indices),
            ),
            learning_rate_init=NONLINEAR_PROBE_LEARNING_RATE,
            early_stopping=False,
            random_state=seed,
        )

        best = None
        best_loss = float("inf")
        best_epoch = 0
        wait = 0

        for epoch in range(
            1,
            settings["max_epochs"] + 1,
        ):
            with warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter(
                    "always"
                )

                if task == "class":
                    probe.partial_fit(
                        x_train,
                        y_train,
                        classes=classes,
                    )

                    loss = log_loss(
                        y_validation,
                        probe.predict_proba(
                            x_validation
                        ),
                        labels=classes,
                    )

                else:
                    probe.partial_fit(
                        x_train,
                        y_train,
                    )

                    loss = mean_squared_error(
                        y_validation,
                        probe.predict(
                            x_validation
                        ),
                    )

            messages.extend(
                str(warning.message)
                for warning in caught
            )

            if not np.isfinite(loss):
                raise ValueError(
                    "Nonfinite validation loss"
                )

            if loss < best_loss - PROBE_IMPROVEMENT_TOLERANCE:
                best = copy.deepcopy(
                    probe
                )
                best_loss = float(loss)
                best_epoch = epoch
                wait = 0

            else:
                wait += 1

            if wait >= settings["patience"]:
                break

        probe = best

        metadata = {
            "selected_epoch": best_epoch,
            "epochs_run": epoch,
            "stopped_by_patience": (
                wait >= settings["patience"]
            ),
            "validation_loss": best_loss,
        }

    metadata.update(
        train_samples=len(indices),
        validation_samples=len(
            validation_indices
        ),
        warnings=sorted(
            set(messages)
        ),
    )

    return (
        probe,
        feature_scaler,
        target_scaler,
        metadata,
    )


# =============================================================================
# Strong nuisance probe
# =============================================================================

def fit_strong(
    features,
    data,
    settings,
):
    """
    Fit the independent post-hoc strong nuisance probe.

    The classifier architecture matches the training adversary but is
    initialized and optimized independently for evaluation.
    """

    classes = np.unique(
        data["train"].nuisance_id
    )

    train_indices = sample_indices(
        len(features["train"]),
        settings["max_samples"],
        settings["seed"],
    )

    validation_indices = sample_indices(
        len(features["validation"]),
        settings["validation_samples"],
        settings["seed"] + PROBE_VALIDATION_SEED_OFFSET,
    )

    x = {
        name: torch.as_tensor(
            features[name][indices],
            dtype=torch.float32,
        )
        for name, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
        )
    }

    y = {
        name: torch.as_tensor(
            np.searchsorted(
                classes,
                targets(
                    data[name],
                    "class",
                )[indices],
            ),
            dtype=torch.long,
        )
        for name, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
        )
    }

    if not np.isin(
        targets(
            data["validation"],
            "class",
        ),
        classes,
    ).all():
        raise ValueError(
            "Validation contains unseen nuisance classes"
        )

    mean = x["train"].mean(
        dim=0
    )

    std = x["train"].std(
        dim=0,
        unbiased=False,
    ).clamp_min(MIN_LATENT_STD)

    x = {
        name: (values - mean) / std
        for name, values in x.items()
    }

    head = make_adversary(
        latent_dim=x["train"].shape[1],
        n_classes=len(classes),
        seed=STRONG_PROBE_INITIALIZATION_SEED,
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=NONLINEAR_PROBE_LEARNING_RATE,
        weight_decay=STRONG_PROBE_WEIGHT_DECAY,
    )

    generator = torch.Generator().manual_seed(
        STRONG_PROBE_BATCH_SEED
    )

    best = float("inf")
    wait = 0

    for epoch in range(
        1,
        settings["max_epochs"] + 1,
    ):
        batches = torch.randperm(
            len(x["train"]),
            generator=generator,
        ).split(PROBE_BATCH_SIZE)

        for indices in batches:
            optimizer.zero_grad(
                set_to_none=True
            )

            torch.nn.functional.cross_entropy(
                head(
                    x["train"][indices]
                ),
                y["train"][indices],
            ).backward()

            optimizer.step()

        with torch.no_grad():
            validation_loss = (
                torch.nn.functional.cross_entropy(
                    head(
                        x["validation"]
                    ),
                    y["validation"],
                ).item()
            )

        if not np.isfinite(
            validation_loss
        ):
            raise ValueError(
                "Nonfinite strong-probe validation loss"
            )

        if validation_loss < best - PROBE_IMPROVEMENT_TOLERANCE:
            best = validation_loss
            selected = epoch
            state = copy.deepcopy(
                head.state_dict()
            )
            wait = 0

        else:
            wait += 1

        if wait >= settings["patience"]:
            break

    head.load_state_dict(
        state
    )
    head.eval().requires_grad_(
        False
    )

    metadata = {
        "selected_epoch": selected,
        "epochs_run": epoch,
        "stopped_by_patience": (
            wait >= settings["patience"]
        ),
        "validation_loss": best,
        "train_samples": len(
            train_indices
        ),
        "validation_samples": len(
            validation_indices
        ),
        "classifier_seed": STRONG_PROBE_INITIALIZATION_SEED,
    }

    return (
        (
            head,
            mean,
            std,
            classes,
        ),
        metadata,
    )


# =============================================================================
# Classification metrics
# =============================================================================

def class_metrics(
    target,
    probability,
    classes,
):
    """Compute categorical nuisance metrics for known classes."""

    if not np.isin(
        target,
        classes,
    ).all():
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "log_loss": None,
        }

    prediction = classes[
        probability.argmax(
            axis=1
        )
    ]

    return {
        "accuracy": float(
            accuracy_score(
                target,
                prediction,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target,
                prediction,
            )
        ),
        "log_loss": float(
            log_loss(
                target,
                probability,
                labels=classes,
            )
        ),
    }


# =============================================================================
# Nuisance audit
# =============================================================================

class NuisanceAudit:
    """
    Fit probes once per frozen encoder, then reuse them on every split.
    """

    def __init__(
        self,
        train,
        validation,
        config,
    ):
        if config.nuisance_probe_epochs < 1:
            raise ValueError(
                "nuisance_probe_epochs must be positive"
            )

        if (
            config.strong_probe_epochs is not None
            and config.strong_probe_epochs < 1
        ):
            raise ValueError(
                "strong_probe_epochs must be positive when specified"
            )

        self.settings = {
            "seed": config.probe_seed,
            "max_samples": (
                config.nonlinear_probe_max_samples
            ),
            "validation_samples": MAX_VALIDATION_OBSERVATIONS,
            "max_epochs": (
                config.nuisance_probe_epochs
            ),
            "patience": PROBE_PATIENCE,
        }

        from types import SimpleNamespace

        self.data = {
            name: SimpleNamespace(
                nuisance=encoded.nuisance,
                nuisance_id=encoded.nuisance_id,
                episode_length=encoded.episode_length,
            )
            for name, encoded in (
                ("train", train),
                ("validation", validation),
            )
        }

        self.fits = {}
        self.metadata = {}

        self.classes = np.unique(
            train.nuisance_id
        )

        jobs = []
        keys = []

        views = (
            ("latent",)
            if config.probe_profile == "pareto"
            else (
                "latent",
                "state",
                "joint",
            )
        )

        for view in views:
            features = {
                name: self.features(
                    encoded,
                    view,
                )
                for name, encoded in (
                    ("train", train),
                    ("validation", validation),
                )
            }

            kinds = (
                ["strong_class"]
                if len(self.classes) >= 2
                else []
            )

            if config.probe_profile == "full":
                kinds += [
                    f"{family}_{task}"
                    for task in (
                        "class",
                        "continuous",
                    )
                    for family in (
                        "linear",
                        "nonlinear",
                    )
                    if not (
                        task == "class"
                        and len(self.classes) < 2
                    )
                    and not (
                        task == "continuous"
                        and train.nuisance is None
                    )
                ]

            for kind in kinds:
                settings = dict(
                    self.settings
                )

                if kind == "strong_class":
                    settings["max_epochs"] = (
                        config.strong_probe_epochs
                        or config.nuisance_probe_epochs
                    )

                keys.append(
                    (
                        view,
                        kind,
                    )
                )

                jobs.append(
                    (
                        kind,
                        features,
                        self.data,
                        settings,
                    )
                )

        fitted_jobs = execute_probe_jobs(
            jobs,
            config,
        )

        for key, (
            fitted,
            metadata,
        ) in zip(
            keys,
            fitted_jobs,
        ):
            self.fits[key] = fitted

            self.metadata[
                f"{key[0]}_{key[1]}"
            ] = metadata

        if config.output_dir is not None:
            directory = Path(
                config.output_dir
            )

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = {
                "protocol": "independent_nuisance_v2",
                "settings": self.settings,
                "fits": self.metadata,
            }

            (
                directory
                / "nuisance_probe_fits.json"
            ).write_text(
                json.dumps(
                    output,
                    indent=2,
                )
                + "\n"
            )

    @staticmethod
    def features(
        encoded,
        view,
    ):
        """Construct latent, state, or joint probe features."""

        latent = encoded.latent.reshape(
            -1,
            encoded.latent.shape[-1],
        )

        state = encoded.true_state.reshape(
            -1,
            encoded.true_state.shape[-1],
        )

        if view == "latent":
            return latent

        if view == "state":
            return state

        return np.c_[
            state,
            latent,
        ]

    def score(
        self,
        encoded,
    ):
        """Score all fitted nuisance probes on one encoded split."""

        result = {
            "nuisance_metric_version": 2.0,
            "nuisance_probe_chance": float(
                1 / len(self.classes)
            ),
        }

        for (
            view,
            kind,
        ), fitted in self.fits.items():
            x = self.features(
                encoded,
                view,
            )

            prefix = (
                f"nuisance_{view}_{kind}"
            )

            if kind == "strong_class":
                (
                    head,
                    mean,
                    std,
                    classes,
                ) = fitted

                with torch.no_grad():
                    probability = np.concatenate(
                        [
                            head(
                                (
                                    chunk
                                    - mean
                                )
                                / std
                            )
                            .softmax(
                                dim=-1
                            )
                            .numpy()
                            for chunk in torch.as_tensor(
                                x,
                                dtype=torch.float32,
                            ).split(ENCODING_CHUNK_SIZE)
                        ]
                    )

                scores = class_metrics(
                    targets(
                        encoded,
                        "class",
                    ),
                    probability,
                    classes,
                )

            else:
                (
                    probe,
                    feature_scaler,
                    target_scaler,
                ) = fitted

                with threadpool_limits(
                    limits=1
                ):
                    transformed = (
                        feature_scaler.transform(
                            x
                        )
                    )

                    if kind.endswith(
                        "_class"
                    ):
                        scores = class_metrics(
                            targets(
                                encoded,
                                "class",
                            ),
                            probe.predict_proba(
                                transformed
                            ),
                            probe.classes_,
                        )

                    else:
                        if encoded.nuisance is None:
                            continue

                        prediction = probe.predict(
                            transformed
                        )

                        target = targets(
                            encoded,
                            "continuous",
                        )

                        scores = {
                            "r2": float(
                                r2_score(
                                    target,
                                    target_scaler.inverse_transform(
                                        prediction
                                    ),
                                    multioutput="uniform_average",
                                )
                            ),
                            "normalized_mse": float(
                                mean_squared_error(
                                    target_scaler.transform(
                                        target
                                    ),
                                    prediction,
                                )
                            ),
                        }

            result.update(
                {
                    f"{prefix}_{name}": value
                    for name, value in scores.items()
                }
            )

        for family in (
            "strong",
            "linear",
            "nonlinear",
        ):
            for score in (
                "balanced_accuracy",
                "accuracy",
                "log_loss",
            ):
                joint = result.get(
                    f"nuisance_joint_{family}_class_{score}"
                )

                state = result.get(
                    f"nuisance_state_{family}_class_{score}"
                )

                result[
                    (
                        "nuisance_joint_over_state_"
                        f"{family}_{score}_gain"
                    )
                ] = (
                    None
                    if joint is None or state is None
                    else joint - state
                )

        return result


# =============================================================================
# Physical-state audit
# =============================================================================

class PhysicalAudit:
    """
    Independent ``Z(t) -> S(t+h)`` probes.

    These probes evaluate state accessibility and are not decoder rollouts.
    """

    def __init__(
        self,
        train,
        validation,
        config,
    ):
        from types import SimpleNamespace

        self.fits = {}
        metadata = {}

        settings = {
            "seed": config.probe_seed,
            "max_samples": (
                config.nonlinear_probe_max_samples
            ),
            "validation_samples": MAX_VALIDATION_OBSERVATIONS,
            "max_epochs": (
                config.nuisance_probe_epochs
            ),
            "patience": PROBE_PATIENCE,
        }

        if (
            config.physical_probe_epochs is not None
            and config.physical_probe_epochs < 1
        ):
            raise ValueError(
                "physical_probe_epochs must be positive when specified"
            )
        settings["max_epochs"] = (
            config.physical_probe_epochs
            or config.nuisance_probe_epochs
        )

        jobs = []
        horizons = []

        requested_horizons = (
            ()
            if config.probe_profile == "pareto"
            else (
                0,
                5,
            )
        )

        for horizon in requested_horizons:
            if min(
                train.episode_length,
                validation.episode_length,
            ) <= horizon:
                continue

            pairs = {
                name: self.pair(
                    encoded,
                    horizon,
                )
                for name, encoded in (
                    ("train", train),
                    ("validation", validation),
                )
            }

            features = {
                name: values[0]
                for name, values in pairs.items()
            }

            proxy = {
                name: SimpleNamespace(
                    nuisance=values[1],
                    nuisance_id=np.zeros(
                        len(values[1]),
                        dtype=int,
                    ),
                    episode_length=1,
                )
                for name, values in pairs.items()
            }

            horizons.append(
                horizon
            )

            jobs.append(
                (
                    "nonlinear_continuous",
                    features,
                    proxy,
                    settings,
                )
            )

        fitted_jobs = execute_probe_jobs(
            jobs,
            config,
        )

        for horizon, (
            fitted,
            fit_metadata,
        ) in zip(
            horizons,
            fitted_jobs,
        ):
            self.fits[horizon] = fitted
            metadata[horizon] = fit_metadata

        if config.output_dir is not None:
            (
                Path(config.output_dir)
                / "physical_probe_fits.json"
            ).write_text(
                json.dumps(
                    metadata,
                    indent=2,
                )
                + "\n"
            )

    @staticmethod
    def pair(
        encoded,
        horizon,
    ):
        """Build aligned ``Z(t)`` and ``S(t+h)`` arrays."""

        latent = (
            encoded.latent
            if horizon == 0
            else encoded.latent[
                :,
                :-horizon,
            ]
        )

        state = encoded.true_state[
            :,
            horizon:,
        ]

        return (
            latent.reshape(
                -1,
                latent.shape[-1],
            ),
            state.reshape(
                -1,
                encoded.state_dim,
            ),
        )

    def score(
        self,
        encoded,
    ):
        """Score all fitted physical probes on one encoded split."""

        metrics = {}

        with threadpool_limits(
            limits=1
        ):
            for horizon, (
                probe,
                feature_scaler,
                target_scaler,
            ) in self.fits.items():
                if encoded.episode_length <= horizon:
                    continue

                x, target = self.pair(
                    encoded,
                    horizon,
                )

                prediction = (
                    target_scaler.inverse_transform(
                        probe.predict(
                            feature_scaler.transform(
                                x
                            )
                        )
                    )
                )

                metrics[
                    f"physical_nonlinear_r2_h{horizon}"
                ] = float(
                    r2_score(
                        target,
                        prediction,
                        multioutput="variance_weighted",
                    )
                )

                metrics[
                    f"physical_nonlinear_mse_h{horizon}"
                ] = float(
                    mean_squared_error(
                        target,
                        prediction,
                    )
                )

                per_state_r2 = r2_score(
                    target,
                    prediction,
                    multioutput="raw_values",
                )

                for name, value in zip(
                    encoded.state_names,
                    per_state_r2,
                ):
                    metrics[
                        (
                            "physical_nonlinear_"
                            f"r2_h{horizon}_{name}"
                        )
                    ] = float(
                        value
                    )

        return metrics


# =============================================================================
# Probe execution and caching
# =============================================================================

def _probe_job(
    kind,
    features,
    data,
    settings,
    implementation_signature,
):
    """
    Execute one probe-fitting job.

    ``implementation_signature`` is deliberately part of the argument list so
    that it contributes to the joblib cache key.
    """

    with threadpool_limits(
        limits=1
    ):
        previous = torch.get_num_threads()
        torch.set_num_threads(
            1
        )

        try:
            if kind == "strong_class":
                return fit_strong(
                    features,
                    data,
                    settings,
                )

            family, task = kind.split(
                "_",
                1,
            )

            (
                probe,
                feature_scaler,
                target_scaler,
                metadata,
            ) = _fit_probe_impl(
                features,
                data,
                family=family,
                task=task,
                settings=settings,
            )

            return (
                (
                    probe,
                    feature_scaler,
                    target_scaler,
                ),
                metadata,
            )

        finally:
            torch.set_num_threads(
                previous
            )


def probe_signature():
    """Build the implementation/runtime signature used for probe caching."""

    import hashlib
    import platform
    import sklearn

    from . import strong_adversary

    return (
        hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        hashlib.sha256(
            Path(
                strong_adversary.__file__
            ).read_bytes()
        ).hexdigest(),
        platform.python_version(),
        np.__version__,
        sklearn.__version__,
        torch.__version__,
    )


def _cached_probe_job(
    cache_dir,
    job,
    signature,
):
    """Execute one probe job, optionally through the on-disk cache."""

    from joblib import Memory

    function = (
        Memory(
            cache_dir,
            verbose=0,
        ).cache(
            _probe_job
        )
        if cache_dir
        else _probe_job
    )

    return function(
        *job,
        signature,
    )


def execute_probe_jobs(
    jobs,
    config,
):
    """Execute probe-fitting jobs sequentially or in parallel."""

    from joblib import (
        Parallel,
        delayed,
        parallel_config,
    )

    if config.probe_workers < 1:
        raise ValueError(
            "probe_workers must be positive"
        )

    signature = probe_signature()

    if config.probe_workers == 1:
        return [
            _cached_probe_job(
                config.probe_cache_dir,
                job,
                signature,
            )
            for job in jobs
        ]

    with parallel_config(
        backend="loky",
        n_jobs=config.probe_workers,
        inner_max_num_threads=1,
    ):
        return Parallel()(
            delayed(
                _cached_probe_job
            )(
                config.probe_cache_dir,
                job,
                signature,
            )
            for job in jobs
        )
