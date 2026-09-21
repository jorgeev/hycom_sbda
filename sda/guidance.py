"""Gaussian-likelihood guidance for score-based data assimilation.

Implements GenDA's ``GaussianScore`` (``smartin98-GenDA-58cfc29/src/sda.py:119-168``,
Rozet & Louppe 2023 eq. 10-11) as a *guided denoiser* in EDM form::

    D      = d_fn(x, sigma)                        # Tweedie estimate E[x0 | x]
    err    = y - A(D)
    var    = std^2 + gamma * sigma^2               # Sigma_y + Gamma, Gamma = gamma * sigma^2 I
    log_p  = -1/2 * sum(err^2 / var)
    g      = grad_x log_p                          # autograd THROUGH the network (detach=False)
    D_post = D + sigma^2 * g

which is the posterior denoiser ``E[x0 | x, y]`` under the Gaussian
approximation. In VP time ``sigma = sigma(t)/mu(t)`` and ``x`` is ``x_t/mu``,
so ``sda.vpsde.vp_eps`` applied to ``D_post`` reproduces GenDA's
``eps - sigma(t) * s`` exactly (see the module docstring of ``sda.vpsde``).

There are NO clamps here. The parent repo's ``nemo_confusion/diffusion/assimilate.py``
clamped the correction to +-5 and the estimate to physical ranges; the result
was worse than the unguided prior. The likelihood is evaluated in fp32 even
when the network runs in bf16.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt


class GaussianGuidance(nn.Module):
    """``d_fn(x, sigma) -> D_post`` for ``p(y | x) = N(y | A(x), diag(std^2))``.

    Args:
        d_fn: the prior denoiser (``sda.vpsde.PriorDenoiser`` or a test stand-in).
        A: differentiable ``x_hat (B, C, H, W) -> (B, N)`` in normalised units.
        y: ``(N,)`` or ``(B, N)`` observations, normalised units.
        std: ``(N,)`` or scalar observation-error std, normalised units.
        gamma: scalar or ``(N,)`` inflation of the per-observation variance by
            ``gamma * sigma^2`` (GenDA uses 0.1; Rozet & Louppe's default 1e-2).
        var_fn: optional override ``(std, sigma) -> var`` (used by the selftest
            to plug in the exact Tweedie covariance of a Gaussian prior).
        detach: ``True`` drops the network Jacobian (grad only through the
            skip term ``x -> D``), the cheap approximation. GenDA uses ``False``.
    """

    def __init__(self, d_fn: Callable, A: Callable, y: torch.Tensor, std,
                 gamma=0.1, var_fn: Callable | None = None, detach: bool = False,
                 enabled: bool = True):
        super().__init__()
        self.d_fn = d_fn
        self.A = A
        self.register_buffer("y", torch.as_tensor(y, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float32))
        self.var_fn = var_fn
        self.detach = detach
        self.enabled = enabled
        self.last_log_p = None

    def variance(self, sigma) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.std.device)
        if self.var_fn is not None:
            return self.var_fn(self.std, sigma)
        return self.std ** 2 + self.gamma * sigma ** 2

    def log_likelihood(self, x_hat: torch.Tensor, sigma) -> torch.Tensor:
        """``log p(y | x)`` summed over observations AND batch (members are
        independent, so the per-member gradient is unaffected)."""
        with torch.autocast("cuda", enabled=False):
            err = self.y - self.A(x_hat.float())
            var = self.variance(sigma)
            return -(err ** 2 / var).sum() / 2

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        if not self.enabled:
            return self.d_fn(x, sigma)
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=x.device)
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            if self.detach:
                d = self.d_fn(xg.detach(), sigma_t).detach()
                # gradient only through the identity part of Tweedie's formula:
                # d/dx E[x0|x] ~ 1 in the small-sigma limit; keeps the graph tiny.
                d = d + (xg - xg.detach())
            else:
                d = self.d_fn(xg, sigma_t)
            log_p = self.log_likelihood(d, sigma_t)
            (g,) = torch.autograd.grad(log_p, xg)
        self.last_log_p = float(log_p.detach())
        return (d.detach() + sigma_t ** 2 * g).detach()

    @torch.no_grad()
    def prior_and_guided(self, x, sigma):
        """``(D, D_post)`` -- for the precision check and per-term diagnostics."""
        d = self.d_fn(x, sigma)
        return d, self.forward(x, sigma)


def per_term_gradients(guidance: GaussianGuidance, terms, ctx, x: torch.Tensor,
                       sigma) -> dict[str, float]:
    """``{term.name: ||grad_x log p_i||}`` at ``(x, sigma)`` -- which observation
    dominates the guidance. Costs one forward + one backward per term."""
    from sda.obs import concat_y  # local import: keep guidance.py free of obs at import time

    out = {}
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=x.device)
    for term in terms:
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            d = guidance.d_fn(xg, sigma_t)
            ax = term.apply(d, ctx)
            y_i, std_i, gam_i = concat_y([term], x.shape[0], x.device)
            var = std_i ** 2 + gam_i * sigma_t ** 2
            log_p = -((y_i - ax) ** 2 / var).sum() / 2
            (g,) = torch.autograd.grad(log_p, xg)
        out[term.name] = float((sigma_t ** 2 * g).norm() / x.shape[0] ** 0.5)
    return out


# ---------------------------------------------------------------------------
# activation checkpointing without editing diffusion/
# ---------------------------------------------------------------------------
def wrap_checkpoint(precond: nn.Module) -> int:
    """Rebind every ``UNetBlock.forward`` under ``precond.model.{enc,dec}`` to
    ``torch.utils.checkpoint`` (non-reentrant). Returns the number of blocks
    wrapped. Cuts saved activations ~3x at the price of one extra forward per
    block in the backward pass. Idempotent."""
    n = 0
    model = getattr(precond, "model", precond)
    for holder in ("enc", "dec"):
        blocks = getattr(model, holder, None)
        if blocks is None:
            continue
        for name, blk in blocks.items():
            if getattr(blk, "_sda_ckpt", False):
                continue
            fwd = blk.forward

            def make(fwd_):
                def wrapped(*args, **kw):
                    if torch.is_grad_enabled() and any(
                            torch.is_tensor(a) and a.requires_grad for a in args):
                        return _ckpt(fwd_, *args, use_reentrant=False, **kw)
                    return fwd_(*args, **kw)
                return wrapped

            blk.forward = make(fwd)  # type: ignore[method-assign]
            blk._sda_ckpt = True
            n += 1
    return n


def unwrap_checkpoint(precond: nn.Module) -> int:
    """Undo :func:`wrap_checkpoint` (restores the class method)."""
    n = 0
    model = getattr(precond, "model", precond)
    for holder in ("enc", "dec"):
        blocks = getattr(model, holder, None)
        if blocks is None:
            continue
        for blk in blocks.values():
            if getattr(blk, "_sda_ckpt", False):
                del blk.forward
                blk._sda_ckpt = False
                n += 1
    return n


__all__ = ["GaussianGuidance", "per_term_gradients", "wrap_checkpoint", "unwrap_checkpoint"]
