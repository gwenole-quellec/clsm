# CLSM

**Constrained Latent State Modeling (CLSM)** is a lightweight PyTorch framework for learning latent state representations from sequential observations by combining complementary representation constraints.

```
toy.environment
        │
        ▼
   Dataset generation
        │
        ▼
      data/
        │
        ▼
scripts.run_presets
        │
        ├── toy.train
        ├── scripts.evaluation
        └── scripts.visualization
                │
                ▼
        runs/ and figures/
```

## Installation

```bash
git clone https://github.com/gwenole-quellec/clsm.git
cd clsm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate the toy dataset

Generate the synthetic training, validation, test, and OOD splits:

```bash
python -m toy.environment \
    --metadata toy/metadata.json \
    --output-dir data/
```

This creates the directory:

```
data/
├── train.npz
├── validation.npz
├── test.npz
└── ood.npz
```

## Train a single model

```bash
python -m toy.train \
    --data-dir data/ \
    --preset full \
    --epochs 100
```

Training outputs are written to:

```
runs/
└── <run-name>/
    ├── best.pt
    ├── last.pt
    ├── config.json
    ├── data_manifest.json
    ├── history.csv
    └── metrics.json
```

### Training presets

Each preset corresponds to a different combination of representation constraints, allowing individual CLSM principles to be studied in isolation or in combination.

The available presets are defined in `clsm.training` and can be selected using the `--preset` (or `--presets`) command-line argument.

## Complete pipeline

The following command trains, evaluates, and visualizes one or more CLSM
presets on a pre-generated dataset.

```bash
python -m scripts.run_presets \
    --train-module toy.train \
    --metadata toy/metadata.json \
    --data-dir data
```

By default, the pipeline:

- trains all predefined presets;
- evaluates every trained model on all dataset splits;
- generates publication-ready figures;
- saves checkpoints and metrics to `runs/`;
- saves figures to `figures/`.

To run only a subset of presets:

```bash
python -m scripts.run_presets \
    --train-module toy.train \
    --metadata toy/metadata.json \
    --data-dir data \
    --presets base full
```

Visualization outputs are written to:

```
figures/
└── <preset>/
    └── ...
```

## Repository structure

```
clsm/
    datasets.py               # Dataset structures, loading, and serialization
    losses.py                 # CLSM loss functions
    models.py                 # Neural network architectures
    protocols.py              # Shared interfaces and typing protocols
    training.py               # Generic training loop and preset definitions
    utils.py                  # Shared utility functions

toy/
    environment.py            # Toy environment and dataset generation
    metadata.json             # State and observation metadata
    train.py                  # Toy training entry point

scripts/
    run_presets.py            # End-to-end benchmark pipeline
    evaluation.py             # Model evaluation and metric computation
    visualization.py          # Publication-ready visualizations
    representation_heatmap.py # Latent representation comparisons
    constraint_sweep.py       # Constraint weight sensitivity analysis
```

## Design principles

The repository separates three concerns:

- **Framework** (`clsm`): generic implementation of Constrained Latent State Modeling.
- **Environment** (`toy`): application-specific data generation and training interface.
- **Scripts** (`scripts`): reusable evaluation and visualization utilities.

## Creating a new environment

A new environment typically consists of:

- an environment implementation;
- a metadata file describing states and observations;
- a training script exposing a command-line interface.

Once these components are provided, the generic evaluation and visualization
scripts can be reused without modification.

## Generated outputs

The complete pipeline produces:

- model checkpoints;
- training histories;
- evaluation metrics;
- aggregate benchmark statistics;
- learned latent representations;
- publication-ready figures.

## Terminology

- **State**: underlying latent physical variables.
- **Observation**: measured variables available to the model.
- **Latent state**: low-dimensional representation learned by CLSM.
- **Nuisance**: variation in the observations that the learned representation should ignore.

## Reproducibility

The experiments reported in the accompanying paper were performed with:

| Package | Version |
|---|---|
| Python | 3.14.4 |
| PyTorch | 2.13.0+cu129 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| Matplotlib | 3.11.0 |
| adjustText | 1.4.0 |
| tqdm | 4.68.4 |

## Current limitations and future work

The current implementation focuses on demonstrating the CLSM framework rather than providing a large collection of neural network architectures.

At present, all models are implemented as multilayer perceptrons (MLPs). This choice keeps the reference implementation compact and highlights the CLSM framework independently of the underlying backbone architecture.

A natural next step will be to make `clsm.models` more modular so that alternative backbone architectures (e.g., convolutional networks, recurrent networks, transformers, or state-space models) can be plugged into the CLSM framework with minimal code changes. The current implementation is intended as a compact reference implementation of the CLSM framework rather than as a comprehensive deep-learning library.
