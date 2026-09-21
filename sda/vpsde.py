"""Schedules and samplers for score-based data assimilation.

Two reverse processes over the SAME denoiser interface ``d_fn(x, sigma) -> D``
(EDM form: ``x = x0 + sigma * n``, ``D ~ E[x0 | x]``):

* ``vp_sample`` -- GenDA's path, verbatim from Rozet & Louppe's ``VPSDE.sample``
  (``smartin98-GenDA-58cfc29/src/sda.py:66-104``): variance-preserving cosine
  schedule ``mu(t) = cos(acos(sqrt(eta)) t)^2``, ``sigma(t)^2 = 1 - mu^2 + eta^2``,
  a DDIM-style predictor and optional annealed-Langevin corrections. The EDM
  denoiser is converted to the VP epsilon with the identity from Manshausen et
  al. 2024 (GenDA ``eps_edm``): ``eps(x,t) = (mu/sigma) (x/mu - D(x/mu, sigma/mu))``.
* ``edm_sample`` -- the EDM-native alternative: Karras sigmas, Euler (or Heun)
  steps, ``d = (x - D)/sigma``. A time reparametrisation of the same guided ODE.

Guidance lives entirely inside ``d_fn`` (see ``sda.guidance``): a guided
denoiser ``D_post = D + sigma^2 grad log p(y|x)`` plugs into either sampler,
because ``eps - sigma_vp * s`` with ``s`` the VP-space likelihood score equals
the VP epsilon of ``D_post`` (``sigma_vp * s = (sigma/mu) * grad_{x/mu}``).

Nothing here knows about observations, stores or checkpoints.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# VP cosine schedule (sda.py:39,48-51)
# ---------------------------------------------------------------------------
def vp_mu(t, eta: float = 1e-3):
    return torch.cos(math.acos(math.sqrt(eta)) * t) ** 2


def vp_sigma(t, eta: float = 1e-3):
    return (1.0 - vp_mu(t, eta) ** 2 + eta ** 2).sqrt()


def vp_sigma_edm(t, eta: float = 1e-3):
    """The EDM noise level ``sigma/mu`` the denoiser is queried at, for VP time t."""
    return vp_sigma(t, eta) / vp_mu(t, eta)


# ---------------------------------------------------------------------------
# the prior denoiser, in the d_fn(x, sigma) interface
# ---------------------------------------------------------------------------
class PriorDenoiser(nn.Module):
    """``EDMPrecond`` as ``d_fn(x, sigma) -> D`` for an unconditional prior.

    * ``sigma`` may be a 0-dim tensor or a python float; it is expanded to the
      ``(B,)`` vector ``diffusion.edm.EDMPrecond.forward`` expects.
    * ``cond`` is the zero-width tensor the priors were trained with
      (``cfg.cond_channels() == 0``), built from ``x``'s geometry.
    * The network runs under bf16 autocast on CUDA (``autocast_dtype``); the
      returned ``D`` is fp32 so every downstream likelihood is fp32.
    * Weights are frozen so autograd through the network only computes the
      input gradient the guidance needs.
    """

    def __init__(self, precond: nn.Module, cond_channels: int = 0,
                 autocast_dtype: torch.dtype | None = torch.bfloat16):
        super().__init__()
        self.precond = precond
        self.cond_channels = int(cond_channels)
        self.autocast_dtype = autocast_dtype
        for p in self.precond.parameters():
            p.requires_grad_(False)
        self.precond.eval()

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        b, _, h, w = x.shape
        sig = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)
        if sig.numel() == 1:
            sig = sig.expand(b)
        sig = sig.contiguous()
        cond = x.new_zeros(b, self.cond_channels, h, w)
        enabled = x.is_cuda and self.autocast_dtype is not None
        with torch.autocast("cuda", dtype=self.autocast_dtype or torch.bfloat16,
                            enabled=enabled):
            d = self.precond(x, sig, cond)
        return d.float()


def vp_eps(d_fn, x: torch.Tensor, t, eta: float = 1e-3) -> torch.Tensor:
    """GenDA ``eps_edm.forward`` (sda.py:208-216): the VP epsilon from an EDM denoiser."""
    mu, sig = vp_mu(t, eta), vp_sigma(t, eta)
    return (mu / sig) * (x / mu - d_fn(x / mu, sig / mu))


# ---------------------------------------------------------------------------
# land replacement (GoM priors only)
# ---------------------------------------------------------------------------
def _replace_land(x, land, noise_std, generator):
    """Replacement conditioning on the known land value 0: the loader writes exact
    zeros on land (diffusion/data.py:663), so ``x_t = mu*0 + sigma*z`` there."""
    if land is None:
        return x
    z = torch.randn(x.shape, device=x.device, generator=generator) * noise_std
    return torch.where(land, z, x)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
@torch.no_grad()
def vp_sample(d_fn, shape, *, steps: int = 256, corrections: int = 0, tau: float = 0.3,
              eta: float = 1e-3, generator: torch.Generator | None = None,
              device=None, land_mask: torch.Tensor | None = None,
              land_mode: str = "free", progress=None) -> torch.Tensor:
    """Rozet & Louppe's predictor-corrector sampler over ``d_fn`` (sda.py:66-104).

    ``shape`` is ``(B, C, H, W)``. ``x_1 ~ N(0, I)``; ``time = linspace(1, 0, steps+1)``;
    predictor ``x <- r x + (sigma' - r sigma) eps(x, t)`` with ``r = mu'/mu``;
    ``corrections`` Langevin steps of amplitude ``tau`` at the new time. The
    returned ``x`` is at t = 0 where ``mu = 1`` and ``sigma = eta``, i.e. the
    normalised state (GenDA does no extra final denoise).

    ``land_mode="replace"`` re-imposes the known land value after every step;
    ``"free"`` (default) leaves land to the prior, as GenDA's land-free setup.
    """
    device = device or (land_mask.device if land_mask is not None else "cpu")
    x = torch.randn(shape, device=device, generator=generator)
    time = torch.linspace(1.0, 0.0, steps + 1, device=device)
    land = land_mask.to(device) if (land_mask is not None and land_mode == "replace") else None
    dims = tuple(range(1, x.ndim))
    for i in range(steps):
        t, tn = time[i], time[i + 1]
        mu_t, sig_t = vp_mu(t, eta), vp_sigma(t, eta)
        mu_n, sig_n = vp_mu(tn, eta), vp_sigma(tn, eta)
        # predictor (DDIM / exponential integrator)
        r = mu_n / mu_t
        x = r * x + (sig_n - r * sig_t) * vp_eps(d_fn, x, t, eta)
        x = _replace_land(x, land, sig_n, generator)
        # corrector (annealed Langevin, sda.py:97-102)
        for _ in range(corrections):
            z = torch.randn(x.shape, device=device, generator=generator)
            eps = vp_eps(d_fn, x, tn, eta)
            delta = tau / (eps.square().mean(dim=dims, keepdim=True) + 1e-6)
            x = x - (delta * eps + torch.sqrt(2 * delta) * z) * sig_n
            x = _replace_land(x, land, sig_n, generator)
        if progress is not None:
            progress(i, steps)
    return x


@torch.no_grad()
def edm_sample(d_fn, shape, sigmas: torch.Tensor, *, heun: bool = False,
               generator: torch.Generator | None = None, device=None,
               land_mask: torch.Tensor | None = None, land_mode: str = "free",
               progress=None) -> torch.Tensor:
    """EDM-native reverse process over ``d_fn`` (Karras Alg. 1 without churn).

    ``sigmas`` is the decreasing schedule with a trailing 0 (``diffusion.edm.karras_sigmas``).
    ``heun=True`` adds the 2nd-order correction with a second (guided) evaluation
    at the extrapolated point; default Euler, because the extra evaluation
    amplifies gradient noise from the likelihood term.
    """
    device = device or sigmas.device
    x = torch.randn(shape, device=device, generator=generator) * sigmas[0]
    land = land_mask.to(device) if (land_mask is not None and land_mode == "replace") else None
    n = int(sigmas.numel()) - 1
    for i in range(n):
        s, sn = sigmas[i], sigmas[i + 1]
        d = (x - d_fn(x, s)) / s
        xn = x + (sn - s) * d
        if heun and sn > 0:
            dn = (xn - d_fn(xn, sn)) / sn
            xn = x + (sn - s) * 0.5 * (d + dn)
        x = _replace_land(xn, land, sn, generator)
        if progress is not None:
            progress(i, n)
    return x
