# CLSM

**Constrained Latent State Modeling (CLSM)** (pronounced *"clasm"*) is a framework for analyzing and designing latent representations under competing constraints.

CLSM views latent representations as **latent states** governed by multiple interacting design principles rather than a single optimization objective. The framework provides a unified perspective on representation learning by explicitly characterizing the constraints and trade-offs underlying existing approaches.

<p align="center">
  <img src="logo.png" width="220">
</p>

---

## Core constraints

CLSM characterizes latent state representations through six complementary constraints:

| Constraint | Description |
|---|---|
| 🎯 Predictive sufficiency | Preserve information necessary for prediction |
| ✂️ Minimality | Encourage compact and parsimonious representations |
| ⏱️ Temporal coherence | Ensure dynamically consistent latent trajectories |
| 👁️ Observation compatibility | Maintain consistency with observed data |
| 🛡️ Invariance to nuisance factors | Improve robustness to irrelevant variability |
| 🧩 Structural constraints | Introduce interpretable or mechanistic structure |

These constraints are intrinsically coupled through trade-offs and cannot, in general, be simultaneously optimized.

---

## CLSM perspective

Rather than categorizing models solely by architecture or training strategy, CLSM interprets them according to the constraints they prioritize and the trade-offs they implicitly resolve.

This repository provides:
- a conceptual overview of CLSM;
- model cards describing representative methods through the CLSM lens;
- a lightweight PyTorch reference implementation.

---

## Reference implementation

This repository also provides a lightweight PyTorch reference implementation of the CLSM framework.

The implementation includes:

- generic dataset, model, loss, and training abstractions;
- a synthetic environment illustrating the CLSM constraints;
- predefined constraint ablations;
- evaluation of latent-state accessibility, invariance, and structure;
- publication-oriented visualization utilities.

### Installation

```bash
git clone https://github.com/gwenole-quellec/clsm.git
cd clsm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the complete toy experiment

```bash
python -m toy.environment \
    --metadata toy/metadata.json \
    --output-dir data/

python -m scripts.run_presets \
    --train-module toy.train \
    --metadata toy/metadata.json \
    --data-dir data
```

Training outputs are written to `runs/`, and figures to `figures/`.

For detailed implementation and reproduction instructions, see [the implementation documentation](docs/implementation.md).

---

## Model cards

Representative models are described using lightweight CLSM model cards.

Example:

| Constraint | Level |
|---|---|
| 🎯 Predictive sufficiency | 🔥🔥🔥🔥 |
| ✂️ Minimality | 🔥🔥 |
| ⏱️ Temporal coherence | 🔥🔥🔥 |
| 👁️ Observation compatibility | ⚪ |
| 🛡️ Invariance | 🔥🔥 |
| 🧩 Structural constraints | ⚪ |

Intensity scale:
- ⚪ absent
- 🔥 low
- 🔥🔥 partial
- 🔥🔥🔥 strong
- 🔥🔥🔥🔥 dominant

---

## Repository structure

- [`clsm/`](clsm/) — generic CLSM framework implementation
- [`toy/`](toy/) — synthetic environment and training entry point
- [`scripts/`](scripts/) — evaluation, visualization, and experiment pipelines
- [`overview/`](overview/) — conceptual overview of CLSM
- [`model_cards/`](model_cards/) — CLSM descriptions of representative models
- [`docs/`](docs/) — implementation and reproduction documentation

---

## Paper

If you use CLSM in your research, please cite:

**Constrained latent state modeling: A unifying perspective on representation learning under competing constraints**

Preprint available on arXiv: [arXiv:2605.15995](https://arxiv.org/abs/2605.15995)

```bibtex
@misc{quellec2026clsm,
      title={Constrained latent state modeling: A unifying perspective on representation learning under competing constraints}, 
      author={Gwenol\'e Quellec},
      year={2026},
      eprint={2605.15995},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.15995}, 
}
```

## License

This project is released under the MIT License.
