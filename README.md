<div style="background-color:#e0f3ff;border:2px solid #3399cc;border-radius:10px;padding:16px;margin:20px 0;font-family:sans-serif;">
  <div style="display:flex;align-items:flex-start;">
    <div style="font-size:36px;margin-right:12px;">⚠️</div>
    <div>
      <strong style="font-size:18px;">Heads up!</strong>
      <p style="margin:4px 0 0;">This section contains important information you shouldn't miss. Please read it carefully before proceeding.</p>
    </div>
  </div>
</div>

# MiCE (Mixture of Convex Examples)

## What is MiCE?

MiCE is a lightweight PyTorch library for building **convex** large language models via a series of novel mechanisms.  Instead of softmax routing or hard top-k gating, MiCE seeks to apply fusing, kernels, convex embedding, atlas lookups, intrinsically convex neural networks and convex norms.

Conditioned on the right data, this helps guarantee convexity, interpretability, and efficient compute without exponentials or discrete dispatch.

## Why MiCE?

- **Efficiency**  
  No softmax, no log-sum-exp,no MLP, very little to no concavity anywhere in the model.

- **Interpretability**  
  Clear regions of dominance — visualize arg-max and margins over latent parameters in any 2-D slice.  

---

## Feature Comparison

| Feature               | MiCE (MoMx)            | LogsumEXP MoE       | Hard MoE         | Standard MLP |
|-----------------------|------------------------|-------------------|------------------|--------------|
| **Routing**           | max(mean(…))           | LogsumEXP(weights)  | top-k mask       | none         |
| **Convexity**         | ✅ (vector-valued)      | ✅ (scalar only)  | ❌                | ❌            |
| **Atlas inversion**   | optional (`invert`)    | —                 | —                | —            |
| **Compute cost**      | ~2.6× MLP              | >10× (exp/log)    | ~k× experts      | baseline     |
| **Params**            | ~2.6× MLP              | high              | high             | baseline     |
| **Gradient smoothness**| high (piecewise convex)| smooth            | sparse           | smooth       |
| **Interpretability**  | high                   | medium            | low              | low          |

---

## Installation

```bash
pip install torch-mice

import torch
from torch_mice import VectorHull

# Forward-only mode (default):
hull = VectorHull(in_dim=512, petals=8, out_dim=512, invert=False)
y_fwd = hull(x)

# Full atlas mode with exact inversion:
hull_atlas = VectorHull(in_dim=512, petals=8, out_dim=512, invert=True)
y_atlas = hull_atlas(x)

```
## License

Licensed under the Gratis Public License © 2025 Joshuah Rainstar
