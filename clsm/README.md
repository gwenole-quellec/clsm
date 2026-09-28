# CLSM framework

Generic implementation of the Constrained Latent State Modeling (CLSM) framework.

The package provides reusable components for learning latent state
representations from sequential observations under combinations of six
constraint families:

1. predictive sufficiency;
2. minimality;
3. temporal coherence;
4. observation compatibility;
5. invariance to nuisance factors;
6. structural constraints.

The implementation separates representation learning from post-hoc evaluation:
training objectives act on the latent representation, while independent probes
and representation-level diagnostics are used to assess the learned states.

## Main modules

- `datasets.py` — dataset abstractions and split handling
- `losses.py` — surrogate objectives for the six CLSM constraint families
- `models.py` — encoder, decoder, latent dynamics, and composite CLSM model
- `nuisance_probes.py` — independent post-hoc nuisance and physical-state probes
- `protocols.py` — evaluation protocols and representation-level diagnostics
- `refresh.py` — second-stage adversarial refresh with fixed latent normalization
- `strong_adversary.py` — categorical nuisance adversary used for invariance training
- `training.py` — generic CLSM training pipeline and experiment configuration
- `utils.py` — shared utility functions

## Training and evaluation

CLSM models are trained for a fixed number of epochs using the configured
combination of constraint surrogates. When invariance is active, a separately
optimized nuisance adversary is alternated with encoder updates, followed by an
optional adversarial refresh phase.

Evaluation is performed independently from the training objectives using
frozen representations, nuisance probes, physical-state probes, predictive and
reconstruction metrics, counterfactual consistency, and neighborhood
preservation diagnostics.