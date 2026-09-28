# CLSM implementation

This document describes the current reference implementation of
**Constrained Latent State Modeling (CLSM)**.

The codebase is intentionally lightweight. Its purpose is to provide a compact,
reproducible implementation of the CLSM perspective, together with the toy
experiments used to study empirical interactions between representation
constraints.

The implementation separates three concerns:

- **training** of latent representations under selected CLSM constraints;
- **independent post-hoc evaluation** of the learned representations;
- **analysis and visualization** of the resulting trade-offs.

---

## 1. Repository organization

```text
clsm/
    datasets.py
    losses.py
    models.py
    nuisance_probes.py
    refresh.py
    strong_adversary.py
    training.py
    utils.py

toy/
    environment.py
    metadata.json
    train.py
    constraint_sweep_config.json
    configurations.json

scripts/
    constraint_sweep.py
    evaluation.py
    run_presets.py
    visualization.py
```

The main responsibilities are:

- `clsm.datasets` — dataset abstractions, loading, and serialization;
- `clsm.models` — encoder, decoder, latent dynamics, and composite CLSM model;
- `clsm.losses` — surrogate objectives for the CLSM constraint families;
- `clsm.training` — generic training configuration and optimization loop;
- `clsm.strong_adversary` — categorical nuisance adversary used during invariance training;
- `clsm.refresh` — second-stage adversarial refresh with fixed latent normalization;
- `clsm.nuisance_probes` — independent nuisance and physical-state probes;
- `scripts.evaluation` — deterministic post-hoc evaluation and artifact generation;
- `scripts.constraint_sweep` — reproducible six-weight search and Pareto analysis;
- `scripts.visualization` — publication-oriented and diagnostic visualizations;
- `scripts.run_presets` — execution of selected reference configurations.

---

## 2. Installation

```bash
git clone https://github.com/gwenole-quellec/clsm.git
cd clsm

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Toy dataset

The toy environment generates training, validation, test, and out-of-distribution
splits:

```bash
python -m toy.environment \
    --metadata toy/metadata.json \
    --output-dir data
```

The generated directory contains:

```text
data/
    train.npz
    validation.npz
    test.npz
    ood.npz
```

The toy environment provides:

- underlying physical states;
- nuisance-dependent observations;
- categorical nuisance identities;
- counterfactual observations of the same physical states under alternative
  nuisance conditions.

Counterfactual pairs are used as an evaluation oracle. They are not assumed to
be available in a generic deployment setting.

---

## 4. CLSM model

The reference model contains three learned components:

1. an observation encoder,
2. an observation decoder,
3. a one-step latent transition model.

The default encoder is deterministic. A variational encoder remains available
for experiments using KL-based minimality.

All reference neural components are multilayer perceptrons. This is deliberate:
the implementation is intended to expose the CLSM constraints and their
interactions without tying the framework to a particular backbone family.

---

## 5. Constraint families

The implementation supports six complementary constraint families:

| Constraint | Purpose |
|---|---|
| Predictive sufficiency | Preserve information needed for future prediction |
| Minimality | Encourage compact latent representations |
| Temporal coherence | Encourage consistent latent dynamics |
| Observation compatibility | Preserve information needed to explain observations |
| Invariance | Reduce nuisance information in the latent representation |
| Structural constraints | Encourage useful latent geometry or statistics |

These objectives are implemented as practical surrogate losses. Their empirical
interactions depend on the selected weights, parameterization, data, and
optimization procedure; the implementation does not assume that the constraints
are fundamentally incompatible.

---

## 6. Training

A single training run is launched through the environment-specific entry point:

```bash
python -m toy.train \
    --data-dir data \
    --device cuda
```

Constraint weights and optimization settings can be supplied through the
command-line interface or through a named configuration.

A typical training run uses:

- 50 total epochs;
- 40 epochs in the initial training phase;
- 10 epochs in the adversarial refresh phase when invariance is active;
- 15 adversary updates per encoder update;
- batch size 128;
- deterministic model seeds and dedicated random-number generators for
  data-loader shuffling and adversary sampling.

Training histories and checkpoints are written under `runs/`.

---

## 7. Adversarial invariance

Nuisance invariance is implemented with a **separate external adversary** rather
than with a nuisance classifier embedded in the CLSM model.

The adversary is a categorical multilayer perceptron with two hidden layers of
64 ReLU units. It receives latent representations only and predicts nuisance
identity.

Training alternates:

1. adversary updates that improve nuisance prediction from the current latent
   representation;
2. encoder updates that optimize the CLSM objective while opposing the
   adversary through the invariance term.

The adversary has its own initialization and sampling random-number generators,
which keeps its stochasticity reproducible and decoupled from the main training
random stream.

---

## 8. Adversarial warm-up

During the initial training phase, the effect of the adversarial invariance term
is progressively introduced through an adversarial coefficient.

---

## 9. Adversarial refresh

When nuisance invariance is active, the final training epochs use a dedicated
refresh phase.

At the beginning of this phase:

- a fixed probe subset is drawn from the training observations;
- the probe observations are encoded;
- latent mean and standard deviation are computed once.

These statistics define a fixed latent normalization for the complete refresh
phase.

Before each encoder minibatch update:

1. the fixed probe observations are re-encoded with the current model;
2. the current probe latents are standardized using the fixed statistics;
3. the nuisance adversary is updated for the configured number of steps;
4. the encoder is updated on the current training minibatch.

Dedicated random-number generators are used for refresh minibatch ordering and
adversary sampling.

---

## 10. Reproducibility and random-number generation

The implementation avoids coupling unrelated stochastic components through the
global PyTorch random-number generator.

In particular:

- shuffled training data loaders use dedicated generators;
- nuisance-adversary initialization uses a dedicated seed offset;
- nuisance-adversary minibatch sampling uses a dedicated generator;
- refresh-phase sampling uses dedicated generators;
- evaluation probes use independent seeds.

This matters because data-loader random-number consumption can alter the complete
optimization trajectory even when the nominal model seed is unchanged.

---

## 11. Evaluation philosophy

Evaluation is intentionally separated from training.

The learned representation is frozen and assessed with direct metrics and
independently fitted post-hoc probes.

The training split is used only to fit post-hoc mappings and probes.
Validation, test, and OOD metrics are computed on independent samples.

The main representation properties used in the large constraint sweep are:

1. **multi-step prediction** — observation MSE at rollout horizon 5;
2. **state accessibility** — linear physical-state probe \(R^2\);
3. **neighborhood preservation** — trustworthiness between physical and latent spaces;
4. **counterfactual consistency** — relative latent energy under nuisance intervention;
5. **nuisance accessibility** — balanced accuracy of an independently trained
   nuisance classifier.

Lower is better for prediction MSE, counterfactual relative energy, and nuisance
balanced accuracy. Higher is better for state accessibility and neighborhood
preservation.

---

## 12. Probe profiles

Two evaluation profiles are available.

### `pareto`

The lightweight profile used during the large weight sweep. It retains the
metrics needed for the five primary analysis axes and avoids unnecessary
expensive diagnostics.

### `full`

The full profile additionally provides deeper representation diagnostics,
including:

- nonlinear physical-state probes;
- temporal physical-state probes;
- additional nuisance probes;
- physical-state control probes;
- canonical correlation analysis;
- further latent compactness and geometry metrics.

The full profile is intended for detailed analysis of selected configurations,
not for defining the primary Pareto search.

---

## 13. Independent nuisance probes

The nuisance probes used for evaluation are independent of the adversary used
during representation learning.

Probe preprocessing is fitted on the training split only. Nonlinear model
selection uses external validation data rather than the final evaluation split.

The principal categorical nuisance metric is balanced accuracy. For the toy
problem, chance performance is 25%.

Probe fitting can be parallelized with `--probe-workers`.

A persistent probe and encoding cache can be enabled with `--probe-cache-dir`.

---

## 14. Physical-state probes

State accessibility is quantified independently from the training losses.

The primary metric is a linear regression probe from latent representation to
the underlying physical state, reported as variance-weighted \(R^2\).

The full evaluation profile additionally includes:

- nonlinear state probes;
- temporal probes using adjacent latent states;
- canonical correlation analysis between latent and physical-state spaces.

These additional analyses are diagnostics and are not part of the five-axis
Pareto definition.

---

## 15. Counterfactual evaluation

For each physical trajectory, the toy environment can provide two observations
that differ only through nuisance variables.

The encoder is applied independently to both observations. Counterfactual
consistency is then measured directly in latent space.

The main sweep metric is `counterfactual_relative_energy`, which normalizes the
counterfactual latent displacement by the scale of the latent representation.

Counterfactual observations are used only for evaluation, not as a generic
assumption of the CLSM framework.

---

## 16. Neighborhood preservation

Local geometric preservation between the physical-state manifold and the learned
latent representation is quantified with trustworthiness.

The evaluation module also provides complementary geometry metrics in the full
profile.

---

## 17. Multi-step rollout evaluation

The model's learned latent dynamics are rolled forward for configurable horizons.

The default horizons are:

```text
1, 5, 10
```

For each horizon, evaluation reports:

- latent rollout MSE;
- decoded observation rollout MSE.

The five-axis sweep uses observation rollout MSE at horizon 5.

---

## 18. Encoding and probe cache

Evaluation can persist encoded datasets and fitted probe artifacts.

The encoding-cache key includes information such as:

- checkpoint content;
- dataset content;
- runtime versions;
- evaluation source code;
- model source code.

This prevents accidental reuse of representations produced by incompatible
checkpoints or evaluation code.

---

## 19. Constraint sweep

The principal weight search is implemented in:

```text
scripts/constraint_sweep.py
```

A representative command is:

```bash
PYTHONPATH=. python -m scripts.constraint_sweep \
    --train-module toy.train \
    --data-dir data \
    --num-configurations 200 \
    --sweep-seed 12345 \
    --model-seeds 0 1 2 3 4 \
    --adversary-steps 15 \
    --epochs 50 \
    --refresh-pretrain-epochs 40 \
    --probe-profile pareto \
    --nuisance-probe-epochs 500 \
    --probe-workers 4 \
    --probe-cache-dir .probe-cache-final \
    --device cuda \
    --pipeline \
    --restart-incomplete \
    --stop-file STOP_SWEEP_FINAL \
    --runs-dir runs-weight-sweep-final \
    --output-dir analysis-weight-sweep-final
```

The search contains:

- 200 constraint-weight configurations;
- five model seeds per configuration;
- predefined anchor configurations;
- focused and broad log-uniform sampling;
- explicit quotas for zero-valued constraint weights.

This yields 1,000 model trainings for the complete sweep.

---

## 20. Weight ranges

Positive weights are sampled log-uniformly within the following ranges:

| Weight | Range |
|---|---:|
| Predictive | 0.01 – 5 |
| Minimality | \(10^{-5}\) – 0.05 |
| Temporal | 0.01 – 2 |
| Observation | 0.003 – 3 |
| Invariance | 0.01 – 5 |
| Structural | 0.001 – 1 |

Some sampled configurations explicitly set one or more weights to zero.

---

## 21. Pareto analysis

The sweep aggregates validation metrics across the five model seeds and computes
selected pairwise Pareto fronts.

The retained panels compare:

1. prediction vs counterfactual consistency;
2. prediction vs nuisance suppression;
3. neighborhood preservation vs nuisance suppression;
4. state accessibility vs neighborhood preservation.

Pareto fronts are always computed from all configurations.

Tukey-IQR filtering is visualization-only: dominated outliers may be hidden from
plots, but they do not alter the Pareto computation.

Representative points receive stable global labels (`P1`, `P2`, ...), shared
across figures and exported tables.

---

## 22. Sweep provenance

A new sweep records:

- the complete search manifest;
- command-line settings;
- sampling configuration;
- constraint-weight ranges;
- model seeds;
- data and source hashes where applicable.

The search protocol is created once and is not overwritten during later analysis
or figure regeneration.

Existing runs can be reused only when their stored optimization settings,
constraint weights, invariance protocol, and dataset hashes are compatible with
the requested experiment.

---

## 23. Resume and pipeline execution

The sweep supports:

- reuse of completed runs;
- clean restart of incomplete runs;
- controlled stopping through a stop file;
- overlapping evaluation with subsequent model training;
- analysis-only regeneration;
- figure-only regeneration.

Typical options are:

```text
--pipeline
--restart-incomplete
--stop-file <path>
--analyze-only
--figures-only
```

This allows long campaigns to be interrupted and resumed without repeating
completed work.

---

## 24. Selected configurations

Selected reference configurations can be run through `scripts.run_presets`.

The configuration file is intended to contain explicit constraint weights rather
than encode scientific conclusions in preset names. The large sweep assigns
stable `P` labels to representative Pareto configurations during analysis.

---

## 25. Visualization

`scripts.visualization` consumes saved training and evaluation artifacts.

Available visualizations include:

- environment and counterfactual observation views;
- shared PCA projections of the latent space;
- counterfactual latent alignment;
- temporally ordered latent trajectories;
- training histories and individual loss components;
- optional full-profile state-accessibility and CCA diagnostics.

Visualization functions do not silently re-fit statistical analyses that should
have been performed during evaluation.

---

## 26. Generated artifacts

A training run typically contains files such as:

```text
runs/<run-name>/
    best.pt
    config.json
    data_manifest.json
    history.csv
    ...
```

Evaluation adds an `evaluation/` directory containing artifacts such as:

```text
evaluation/
    evaluation_metrics.json
    evaluation_metrics.csv
    evaluation_protocol.json
    nuisance_probe_fits.json
    ...
```

Depending on the selected options, encoded latent datasets and CCA artifacts may
also be saved.

A complete sweep additionally produces:

```text
analysis-weight-sweep-final/
    search_protocol.json
    sweep_manifest.json
    sweep_manifest.csv
    sweep_run_results.json
    sweep_configuration_results.json
    pareto_configurations.json
    pareto_configurations.csv
    pareto_2d_summary.json
    table4.md
    table4.tex
    figures/
        ...
```

---

## 27. Terminology

- **Physical state** — underlying variables defining the simulated system.
- **Observation** — variables available to the model.
- **Latent state** — representation learned by the encoder.
- **Nuisance** — variation in the observations that should ideally not influence
  the task-relevant latent state.
- **Counterfactual view** — an alternative observation of the same physical state
  under a different nuisance condition.
- **Training adversary** — classifier used during representation learning to
  construct the invariance objective.
- **Post-hoc nuisance probe** — independently fitted classifier used only to
  evaluate nuisance accessibility.

---

## 28. Reproducibility

The exact software environment used for a particular experiment should be
recorded with the experiment artifacts rather than inferred from this document,
because package versions may evolve with the repository.

For reproducibility, retain at least:

- the Git commit;
- the sweep manifest and search protocol;
- the dataset hashes;
- the model seeds;
- the constraint weights;
- the complete training and evaluation configuration;
- the relevant Python and package versions.

---

## 29. Current scope

The repository is a reference implementation of the CLSM framework rather than a
general-purpose deep-learning library.

The current toy implementation uses MLP-based encoders, decoders, transition
models, adversaries, and probes. Alternative architectures can be introduced
without changing the conceptual definition of the six CLSM constraint families.

The purpose of the implementation is to make constraint interactions explicit,
reproducible, and measurable, not to prescribe one architecture or one universal
optimization recipe.
