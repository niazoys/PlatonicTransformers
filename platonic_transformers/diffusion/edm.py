"""Karras et al. 2022 EDM preconditioning, loss, and sampler.

Task-agnostic building blocks for equivariant diffusion over (positions,
scalar features) pairs. Designed around a denoiser ``model(x, pos, batch,
vec=None, t=..., ...)`` that matches ``PlatonicTransformer`` — any
equivariant network with that signature can be wrapped by
:class:`EDMPrecond`.

Reference: "Elucidating the Design Space of Diffusion-Based Generative
Models", Karras et al. 2022, https://arxiv.org/abs/2206.00364
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from platonic_transformers.utils.utils import subtract_mean


class EDMPrecond(torch.nn.Module):
    """Karras-style preconditioning wrapper for an equivariant denoiser.

    Implements the fixed-variance preconditioning from Karras et al. 2022:
    ``D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise(sigma))`` with
    the four scalars computed from ``sigma`` and the assumed data std
    ``sigma_data``. The wrapped ``model`` is the residual network ``F``.

    The preconditioner handles two calling conventions for ``sigma``:
      * **Per-node** (shape ``[N]`` or ``[N, 1]``) — what :class:`EDMLoss`
        emits during training. Values are expected to be constant within
        a graph (same sigma for every atom of molecule *i*); we reduce to
        a per-graph tensor with ``unique_consecutive``.
      * **Per-graph** (shape ``[B]`` or ``[B, 1]``) — what the sampler
        emits; we broadcast back to per-node via the batch index.
      * **Scalar** — expanded to per-graph.

    The per-graph sigma is passed to the model as the time/noise embedding
    ``t``; the per-node sigma scales the input / output residuals.

    Args:
        model: An equivariant denoiser (e.g. :class:`PlatonicTransformer`
            with ``time_conditioning=True``) that returns
            ``(scalars, vectors)``.
        sigma_min: Lower bound clamp for sampling schedule (default 0).
        sigma_max: Upper bound clamp for sampling schedule (default inf).
        sigma_data: Assumed standard deviation of clean data.
        avg_num_nodes: Average atom count in the target dataset. Passed
            through to the model so scatter-normalized ops know their
            scale.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 1.0,
        avg_num_nodes: float = 18.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.avg_num_nodes = avg_num_nodes

    def forward(
        self, x: Tensor, pos: Tensor, batch: Tensor, sigma: Tensor
    ) -> Tuple[Tensor, Tensor]:
        sigma = sigma.reshape(-1, 1)
        num_graphs = int(batch.max().item()) + 1
        if sigma.shape[0] == batch.shape[0]:
            sigma_per_graph = torch.unique_consecutive(sigma.squeeze(-1)).reshape(-1, 1)
        elif sigma.numel() == 1:
            sigma_per_graph = sigma.expand(num_graphs, 1)
        else:
            sigma_per_graph = sigma
        sigma_per_node = sigma_per_graph[batch]  # (N, 1)

        c_skip = self.sigma_data ** 2 / (sigma_per_node ** 2 + self.sigma_data ** 2)
        c_out = sigma_per_node * self.sigma_data / (sigma_per_node ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_per_node ** 2).sqrt()
        c_noise = sigma_per_graph.log() / 4  # (B, 1)

        x_in = c_in * x
        pos_in = c_in * pos

        # Noise is also concatenated onto the per-node scalar features.
        scalars_in = torch.cat([x_in, c_noise[batch]], dim=-1)

        scalars_out, vecs_out = self.model(
            scalars_in, pos_in, batch, vec=None,
            t=c_noise.squeeze(-1), avg_num_nodes=self.avg_num_nodes,
        )
        F_x = x_in - scalars_out
        F_pos = pos_in - vecs_out.squeeze(1)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_pos = c_skip * pos + c_out * F_pos.to(torch.float32)
        return D_x, D_pos


class EDMLoss:
    """Karras-style log-normal noise schedule + weighted MSE loss.

    Draws ``sigma`` from a log-normal per-graph, weights the denoising
    error by ``(sigma**2 + sigma_data**2) / (sigma * sigma_data)**2``,
    and sums atom-feature and position errors. Positions have their
    per-graph mean subtracted before noising to keep them centred.

    Args:
        P_mean, P_std: parameters of the log-normal sigma draw
            (log sigma ~ Normal(P_mean, P_std)).
        sigma_data: assumed data std; match the wrapper.
        normalize_x_factor: scalar dividing the atom-type one-hots before
            noising, so the scalar channel lives on the same scale as
            positions.
        normalize_charge_factor: separate divisor for the formal-charge
            channel (one integer per atom, larger dynamic range) when
            ``use_charges`` is ``True``.
        use_charges: whether the sixth feature column is a formal charge.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 1.0,
        normalize_x_factor: float = 4.0,
        normalize_charge_factor: float = 8.0,
        use_charges: bool = True,
    ) -> None:
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.normalize_x_factor = normalize_x_factor
        self.normalize_charge_factor = normalize_charge_factor
        self.use_charges = use_charges

    def __call__(self, net: EDMPrecond, inputs: dict) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pos, x, batch = inputs["pos"], inputs["x"], inputs["batch"]
        pos = subtract_mean(pos, batch)

        if self.use_charges:
            x = x.clone()
            x[:, :-1] = x[:, :-1] / self.normalize_x_factor
            x[:, -1] = x[:, -1] / self.normalize_charge_factor
        else:
            x = x / self.normalize_x_factor

        rnd_normal = torch.randn(
            [batch.max() + 1, 1], device=pos.device, dtype=torch.float32
        )[batch]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        x_noisy = x + torch.randn_like(x) * sigma
        pos_noisy = pos + subtract_mean(torch.randn_like(pos), batch) * sigma

        D_x, D_pos = net(x_noisy, pos_noisy, batch, sigma)
        error_x = (D_x - x) ** 2
        error_pos = (D_pos - pos) ** 2
        loss = (weight * error_x).mean() + (weight * error_pos).mean()
        return loss, (D_x, D_pos)


def edm_sampler(
    net: EDMPrecond,
    pos_0: Tensor,
    x_0: Tensor,
    batch: Tensor,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 20.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
    return_trajectory: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, List[Tuple[Tensor, Tensor]]]]:
    """Karras Euler-Maruyama sampler with second-order correction.

    Integrates the denoising SDE from ``sigma_max`` down to ``sigma_min``
    over ``num_steps`` steps on the Karras-style power-law schedule
    (``rho=7`` gives the standard curve). Optional Langevin churn
    (``S_churn`` on the ``[S_min, S_max]`` sigma range) adds stochasticity.

    Args:
        net: EDMPrecond-wrapped denoiser.
        pos_0: Initial positions tensor of shape ``(N, 3)``. Will be
            scaled by ``sigma_max`` at step 0.
        x_0: Initial scalar features tensor of shape ``(N, F)``. Same
            scaling.
        batch: Node-to-graph index of shape ``(N,)``; graphs are sampled
            independently but stacked in one sparse batch.
        return_trajectory: If ``True``, additionally returns a list of
            ``(x, pos)`` CPU snapshots, one per sampler step + the initial
            state, for visualization.

    Returns:
        ``(x_final, pos_final)`` — or ``(x_final, pos_final, trajectory)``
        if ``return_trajectory`` is set.
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    num_graphs = int(batch.max().item()) + 1

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=pos_0.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next, pos_next = x_0 * t_steps[0], pos_0 * t_steps[0]
    trajectory: Optional[List[Tuple[Tensor, Tensor]]] = (
        [(x_next.detach().cpu(), pos_next.detach().cpu())] if return_trajectory else None
    )

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, pos_cur = x_next, pos_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        pos_hat = pos_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(pos_cur)

        x_denoised, pos_denoised = net(x_hat, pos_hat, batch, t_hat.expand(num_graphs))
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        if i < num_steps - 1:
            # Heun's second-order correction
            x_denoised, pos_denoised = net(x_next, pos_next, batch, t_next.expand(num_graphs))
            dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

        if return_trajectory:
            trajectory.append((x_next.detach().cpu(), pos_next.detach().cpu()))

    pos_next = subtract_mean(pos_next, batch)
    if return_trajectory:
        return x_next, pos_next, trajectory  # type: ignore[return-value]
    return x_next, pos_next
