# QM9 unconditional generation with flow matching

Entry point: [`mains/main_qm9_gen_FM.py`](../mains/main_qm9_gen_FM.py) · config: [`configs/qm9_gen.yaml`](../configs/qm9_gen.yaml) · launcher: [`qm9_generation_FM.sh`](../qm9_generation_FM.sh)

This script replaces the EDM denoising objective of `main_qm9_gen.py` with straight-line conditional flow matching (FM). Everything else is shared with the EDM model: data loading, the PlatonicTransformer backbone, the atom-count sampler and the evaluation metrics. The FM objective is trained in the usual straight-line interpolation parameterization, but for sampling we reparameterize the FM state using the corresponding EDM noise level $\sigma=(1-t)/t$. This allows the same Heun sampler and $S_{\rm churn}$ mechanism used by the EDM baseline to be applied to the FM velocity field. A plain FM ODE sampler in $t$ is also available (see **Sampling**).

## Formulation

**Data.** Each molecule is a set of nodes with positions $x^{pos} \in \mathbb{R}^{N\times 3}$ (centre of mass removed) and scalar features $h \in \mathbb{R}^{N\times 6}$: 5 one-hot atom types divided by 4 and the formal charge divided by 8. Training applies a random SO(3) rotation to each batch.

**Path.** Draw noise $z_0 \sim \mathcal N(0, I)$, with the centre of mass removed from the position part, and data $z_1 = (h, x^{pos})$. Then

$$z_t = (1-t)\,z_0 + t\,z_1, \qquad u_t = z_1 - z_0 .$$

$t$ is drawn once per graph. It is sampled on EDM's log-normal noise schedule through the SNR-matching map $t = 1/(1+\sigma)$, with $\ln\sigma \sim \mathcal N(P_{mean}=-1.2,\ P_{std}=1.2)$.

**Preconditioned velocity.** With $\sigma_d$ = `sigma_data` = 1 and $d(t) = t^2\sigma_d^2 + (1-t)^2$:

$$
c_{in} = d^{-1/2},\quad
c_{skip} = \frac{t\sigma_d^2 - (1-t)}{d},\quad
c_{out} = \sigma_d\, d^{-1/2},\quad
c_{noise} = \tfrac14 \ln\tfrac{1-t}{t}
$$

$$
\hat v_\theta(z_t, t) = c_{skip}\, z_t + c_{out}\, F_\theta\!\big(c_{in} z_t,\ c_{noise}\big)
$$

These are EDM's coefficients re-derived for a velocity target: $c_{skip} z_t$ is the best linear predictor of $u_t$ from $z_t$. The network $F_\theta$ gets $c_{noise}$ as an extra scalar input channel and returns 6 scalars (features) and 1 vector (position velocity) per node.

**Loss.**

$$
\mathcal L = \mathbb E_{t,z_0,z_1}\Big[\lambda(t)\,\big\|\hat v_\theta(z_t,t) - (z_1 - z_0)\big\|^2\Big],\qquad \lambda(t) = 1/c_{out}^2 = d(t)/\sigma_d^2
$$

The loss is summed over the feature and position parts. With $\sigma_d = 1$, $\lambda \in [0.5, 1]$, so EDM's `max_weight` clamp is not needed.

**Sampling.** Two samplers are available, selected with `diffusion.sampler`. With $S_{churn}=0$ both solve the same probability-flow ODE; they differ in variables, step grid and starting point.

*EDM sampler* (`sampler: edm`, default). The same stochastic Heun sampler as the EDM baseline, with 50 steps on a Karras $\sigma$-schedule ($\sigma_{max}=1, \sigma_{min}=0.002, \rho=7$). We use the correspondence

$$
t=\frac{1}{1+\sigma}, \quad x=\frac{z_t}{t},
$$

so that $x=z_1+\sigma z_0$ has the standard EDM parametrization. The predicted FM velocity is converted to a denoised estimate of the endpoint through

$$
D(z_t,t)=z_t+(1-t)\hat{v}_\theta(z_t,t).
$$

This allows FM and EDM to use the same Heun and $S_{churn}$ sampling procedure. Note that $\sigma_{max}=1$ corresponds to $t=0.5$: sampling starts halfway along the path, from pure noise.

*FM ODE sampler* (`sampler: fm`). Integrates

$$
\frac{dz_t}{dt} = \hat v_\theta(z_t, t)
$$

directly on a uniform $t$ grid from `fm_t_start` (default 0, pure noise) to 1, with Heun (default) or Euler steps (`fm_integrator`) and `num_steps` steps. No noise is injected, so `S_churn` has no effect. The network is evaluated at $t$ clamped to $[10^{-4}, 1-10^{-4}]$, since $c_{noise}$ diverges at both endpoints. Setting `fm_t_start=0.5` reproduces the EDM sampler's starting point.

**Backbone.** PlatonicTransformer with the tetrahedral group, 14 layers, hidden size 1152, 72 heads and RoPE ($\sigma = 4$). Training uses AdamW (lr 5e-4, weight decay 1e-6) with a cosine schedule decaying to lr/100, batch size 96, 200 epochs, gradient clipping at 0.5 and EMA 0.999 (2000 warmup steps).

## Running

```bash
# train (then tests the last checkpoint); LR, S_CHURN and SAMPLER are env overrides
LR=5e-4 S_CHURN=0.0 SAMPLER=edm sbatch qm9_generation_FM.sh

# evaluate a checkpoint with the plain FM ODE sampler (extra flags are passed to the script)
SAMPLER=fm sbatch qm9_generation_FM.sh \
    --testing.test_ckpt=path/to/last.ckpt --testing.metrics_out=out.json \
    --diffusion.fm_integrator=heun --diffusion.fm_t_start=0.0 --diffusion.num_steps=50
```

Any config key can be overridden this way, e.g. `--diffusion.S_churn=10` for the EDM sampler.

## Results

All numbers are from 10,000 samples scored with the EDM metrics (bond-based stability; RDKit validity, uniqueness and novelty), in %.

**FM vs EDM, both with the EDM sampler.**

| Model | S_churn | Atom stab. | Mol. stab. | Valid | Unique | Novel |
|---|---:|---:|---:|---:|---:|---:|
| FM (lr 5e-4) | 0  | 97.42 | 79.71 | 93.18 | 77.16 | 71.75 |
|  | 5  | 99.22 | 92.58 | 98.37 | 93.42 | 59.99 |
|  | 10 | 99.42 | 94.39 | 98.39 | 96.52 | 57.63 |
|  | 20 | 99.36 | 94.22 | 97.52 | 97.63 | 55.92 |
| EDM (lr 5e-4) | 0  | 97.90 | 78.89 | 88.93 | 79.01 | 72.57 |
|  | 5  | 99.47 | 94.15 | 98.46 | 92.99 | 61.85 |
|  | 10 | 99.55 | 95.21 | 98.63 | 96.48 | 60.24 |
|  | 20 | 99.39 | 94.18 | 97.86 | 97.39 | 58.22 |

**Sampler comparison.** Same FM checkpoint (lr 5e-4), deterministic sampling ($S_{churn}=0$), 50 steps unless noted.

| Sampler | Start | Atom stab. | Mol. stab. | Valid | Unique | Novel |
|---|---|---:|---:|---:|---:|---:|
| EDM sampler (Heun, Karras grid) | t = 0.5 | 97.42 | 79.71 | 93.18 | 77.16 | 71.75 |
| FM ODE, Heun | t = 0.5 | 97.50 | 79.85 | 93.20 | 77.96 | 72.30 |
| FM ODE, Euler | t = 0.5 | 96.67 | 75.17 | 91.73 | 80.08 | 74.27 |
| FM ODE, Heun | t = 0 | 96.71 | 76.53 | 91.23 | 98.86 | 67.71 |
| FM ODE, Euler, 200 steps | t = 0 | 96.53 | 75.81 | 90.65 | 98.96 | 68.25 |
| FM ODE, Euler | t = 0 | 95.18 | 68.99 | 88.09 | 99.05 | 70.06 |

