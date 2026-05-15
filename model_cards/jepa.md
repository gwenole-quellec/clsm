# JEPA (Joint Embedding Predictive Architecture)

**Category:** Predictive / self-supervised learning  

## Constraint profile
- 🎯 Predictive: 🔥🔥🔥🔥  
- ✂️ Minimality: 🔥🔥 (implicit via predictive objective)  
- ⏱️ Temporal: 🔥🔥 (depends on variant)  
- 👁️ Observation: ⚪  
- 🛡️ Invariance: 🔥🔥🔥 (targeted invariances)  
- 🧩 Structure: 🔥 (weak inductive biases)  

## Intuition
JEPA predicts latent embeddings of targets rather than reconstructing pixels, explicitly aiming to discard nuisance variability.

## CLSM interpretation
Predictive sufficiency with stronger, more explicit control of invariance than classical contrastive methods; still weak observation grounding.

## References
Assran M. et al.  
*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*  
[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
