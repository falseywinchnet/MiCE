# MiCE (Mixture of Convex Experts)

## What is MiCE?

MiCE is a lightweight PyTorch library for building **convex** mixture-of-experts models via a novel **max-of-means atlas**.  Instead of softmax routing or hard top-k gating, MiCE fuses an ensemble of convex “petal” subnetworks with a fixed, irreducible atlas of affine charts:

1. **Forward projection** into each petal’s chart  
2. **Convex evaluation** in chart space (Input-Convex Neural Nets)  
3. **(Optional) exact inversion** back to the global frame  
4. **Overlapping mean-of-pairs + max** fusion  

This guarantees convexity, interpretability, and efficient compute without exponentials or discrete dispatch.


## Why MiCE?

- **Global convexity**  
  Each chart is convex; max-fusion preserves convexity in any dimension.

- **Two operating modes**  
  - **Forward-only** (`invert=False`): fast chart tiling with learnable shifts  
  - **Atlas mode**  (`invert=True`): full chart atlas with exact reprojection, no shifts needed

- **Efficiency**  
  No softmax, no log-sum-exp, no sparse dispatch.  Fusion is just a handful of GEMMs, adds, and a single max per group.  Compute scales ~2.6× a 2-layer MLP.

- **Interpretability**  
  Clear regions of dominance — visualize arg-max and margins over petals in any 2-D slice.  

---

## Feature Comparison

| Feature               | MiCE (MoMx)            | Softmax MoE       | Hard MoE         | Standard MLP |
|-----------------------|------------------------|-------------------|------------------|--------------|
| **Routing**           | max(mean(…))           | softmax(weights)  | top-k mask       | none         |
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
