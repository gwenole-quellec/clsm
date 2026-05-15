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
- a conceptual overview of CLSM,
- model cards describing representative methods through the CLSM lens.

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

- [`overview/`](overview/) — conceptual overview of CLSM
- [`model_cards/`](model_cards/) — CLSM descriptions of representative models

---

## Paper

**Constrained latent state modeling: A unifying perspective on representation learning under competing constraints**

Preprint available on arXiv: *(link coming soon)*

```bibtex
@article{quellec2026clsm,
  title={Constrained latent state modeling: A unifying perspective on representation learning under competing constraints},
  author={Quellec, Gwenol{\'e}},
  year={2026}
}
