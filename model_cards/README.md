# CLSM Model Cards

This directory contains lightweight CLSM model cards describing representative latent representation learning approaches through the lens of constrained latent state modeling (CLSM).

Rather than categorizing models solely by architecture or training paradigm, these cards characterize methods according to the constraints they prioritize and the trade-offs they implicitly resolve.

---

## Constraint legend

| Constraint | Meaning |
|---|---|
| 🎯 Predictive sufficiency | Preserve information necessary for prediction |
| ✂️ Minimality | Encourage compact and parsimonious representations |
| ⏱️ Temporal coherence | Ensure dynamically consistent latent trajectories |
| 👁️ Observation compatibility | Maintain consistency with observed data |
| 🛡️ Invariance to nuisance factors | Improve robustness to irrelevant variability |
| 🧩 Structural constraints | Introduce interpretable or mechanistic structure |

---

## Intensity scale

| Level | Meaning |
|---|---|
| ⚪ | absent |
| 🔥 | low |
| 🔥🔥 | partial |
| 🔥🔥🔥 | strong |
| 🔥🔥🔥🔥 | dominant |

These scores are qualitative and intended as conceptual summaries rather than strict quantitative evaluations.

---

## Available model cards

## Predictive representation learning
- [CPC](cpc.md) — Contrastive Predictive Coding
- [JEPA](jepa.md) — Joint Embedding Predictive Architecture

## Reconstruction and latent variable models
- [VAE](vae.md) — Variational Autoencoder

## Multimodal representation learning
- [CLIP](clip.md) — Contrastive Language-Image Pretraining

## Latent dynamical models
- [SSM](ssm.md) — State-Space Models

## Domain-specific structured models
- [SuStaIn](sustain.md) — Subtype and Stage Inference

---

## Purpose

The goal of these cards is not to provide exhaustive technical reviews, but to illustrate how different modeling approaches occupy different regions of the CLSM design space.

In particular, the cards emphasize:
- which constraints are explicitly enforced,
- which properties remain weakly controlled,
- and which trade-offs characterize each family of methods.

---

## Disclaimer

The assignments proposed here are intentionally schematic and may vary across implementations, variants, or application domains. They should be interpreted as conceptual guides rather than definitive classifications.
