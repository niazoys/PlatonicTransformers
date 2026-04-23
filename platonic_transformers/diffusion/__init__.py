"""Diffusion-model building blocks (task-agnostic).

Currently ships a Karras-style EDM preconditioning wrapper, matched loss, and
Euler-Maruyama sampler that expect a :class:`PlatonicTransformer` backbone
(or any equivariant denoiser with the same forward signature). Imported by
the QM9 generation task; extend here for other diffusion tasks.
"""

from platonic_transformers.diffusion.edm import EDMLoss, EDMPrecond, edm_sampler

__all__ = ["EDMLoss", "EDMPrecond", "edm_sampler"]
