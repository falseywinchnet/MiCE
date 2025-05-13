# MiCE(Mixture of Convex Experts)

## What is MiCE?

MiCE is a lightweight PyTorch library for building **convex** mixture-of-experts models. Instead of softmax routing or hard top-k gating, MiCE fuses networks of convex “petal” networks by **overlapping max-of-means** with learnable scalar shifts—guaranteeing convexity, interpretability, and efficient compute.

This work builds on research done at Carnegie Mellon [Input-Convex Neural Networks](https://arxiv.org/abs/1609.07152) and Johannes Kepler University [Principled Weight Initialisation for Input-Convex Neural Networks](https://arxiv.org/abs/2312.12474) and the development process empirically explored a multitude of domains of convex and nuanced recombination of results. The concise explanation is that the cascaded gating approach outcompetes Kolmogorov-Arnold Network basis interpretation, while the cascaded mean-max-shift approach outcompletes LogSumExp. Both approaches outcompete the mentioned comparable systems in efficiency as well as in loss behavior over convex and non-convex problems, although it is not by this implied that this or other convex models can efficiently approximate non-convex problems. 

## Why MiCE?

- **Convexity guarantees**  
  Every MiCE model computes a convex function of its inputs.  This ensures stable optimization, monotonic gradient behavior, and global convergence properties that standard MLPs and hard-MoE lack.

- **Efficiency**  
  No exponentials, no log-sum-exp, no discrete routing.  Max-of-means fusion costs only a handful of adds, means, and a single max per group.  Memory and FLOPs scale **~2.6×** a 2-layer MLP with 4× expansion—far cheaper than full softmax MoE.

- **Interpretability**  
  Each petal specializes in a convex region; groups overlap, shifts encode priors, and the max operation cleanly partitions input space.  You can visualize which expert wins where.

## How MiCE Differs

| Feature               | MiCE (MoMx)         | Softmax MoE            | Hard Routing MoE      | Standard MLP         |
|-----------------------|---------------------|------------------------|-----------------------|----------------------|
| **Routing**           | max(mean(…))        | softmax(weights)       | top-k expert mask     | monolithic           |
| **Convexity**         | ✅                  | ✅ (scalar only)       | ❌                     | ❌                    |
| **Compute cost**      | ~2.6× MLP           | >10× (exponentials)    | ~k× experts           | baseline             |
| **Memory footprint**  | ~2.6× params        | high (dense activations)| high (expert states)  | baseline             |
| **Gradient flow**     | dense in groups     | dense                  | sparse (top-k only)   | dense                |
| **Smoothness**        | piecewise convex    | smooth                 | non-smooth            | smooth               |
| **Interpretability**  | high                | medium                 | low                   | low                  |

## Relative Costs

- **Parameters & FLOPs**  
  MoMx uses ~2.6× the params and MACs of a 2-layer MLP (4× hidden).  
- **Vs. LSE Fusion**  
  No log/exp → 4–10× cheaper per petal.  
- **Vs. Hard-MoE**  
  No expert dispatch overhead or load balancing; single fused model.

## Solid Arguments

### Against Softmax  
- **High compute & memory**: O(P) exp/log per input.  
- **Numerical instability**: needs shift-and-scale tricks.  
- **Over-smooth**: blurs expert distinctions.

### Against Hard Routing  
- **Non-convex**: breaks convex guarantees.  
- **Sparse gradients**: only top-k experts update.  
- **Brittle**: large performance swings at boundaries.

### Against MLP  
- **Non-convex**: susceptible to poor local minima.  
- **Width & depth explosion**: needs huge hidden dims for expressivity.  
- **Opaque**: hard to interpret gradient flows.

## Quickstart

```python
pip install torch_mice
from torch_mice import VectorHull

model = VectorHull(in_dim=512, petals=8)   # convex, efficient MoE
y = model(x)                               # forward pass

```
## License

Licensed under the Gratis Public License © 2025 Joshuah Rainstar
