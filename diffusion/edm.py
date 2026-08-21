"""EDM (Karras et al. 2022): preconditioning, training loss, and Heun sampler.

The network predicts only the target channels; conditioning channels are passed
clean and are *not* scaled by ``c_in`` (only the noised target is). The loss is
masked to ocean pixels so land zeros don't dominate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EDMPrecond(nn.Module):
    """Karras preconditioning wrapper around a raw UNet.

    ``forward(x_noisy, sigma, cond)`` returns the denoised target ``D``.
    ``cond`` is concatenated to the (c_in-scaled) noisy target before the UNet.
    """

    def __init__(self, model: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.log().flatten() / 4.0
        model_in = torch.cat([c_in * x_noisy, cond], dim=1)
        F_x = self.model(model_in, c_noise)
        return c_skip * x_noisy + c_out * F_x


class EDMLoss:
    """Log-normal sigma sampling with Karras weighting.

    ``reduction="masked_mean"`` (default) averages over ocean pixels only, so
    land zeros don't dominate. ``reduction="sum"`` is GenDA's: sum over C,H,W and
    mean over the batch, with no mask -- valid only when the crops are already
    all-ocean (``reject_land: true``). The two differ by a constant factor of
    roughly C*H*W, which Adam's second-moment normalisation largely absorbs; the
    reduction is exposed so the (reduction, lr) pair can be matched as a package.
    """

    def __init__(self, p_mean: float = -1.2, p_std: float = 1.2,
                 sigma_data: float = 1.0, reduction: str = "masked_mean"):
        if reduction not in ("masked_mean", "sum"):
            raise ValueError(f"Unknown loss reduction {reduction!r}")
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.reduction = reduction

    def __call__(self, precond: EDMPrecond, target: torch.Tensor,
                 cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = target.shape[0]
        rnd = torch.randn(b, device=target.device)
        sigma = (self.p_mean + self.p_std * rnd).exp()
        s = sigma.reshape(-1, 1, 1, 1)
        weight = (s ** 2 + self.sigma_data ** 2) / (s * self.sigma_data) ** 2
        noise = torch.randn_like(target) * s
        denoised = precond(target + noise, sigma, cond)
        se = weight * (denoised - target) ** 2
        if self.reduction == "sum":
            return se.sum() / b
        # mask to ocean pixels (mask broadcasts over target channels)
        m = mask
        return (se * m).sum() / (m.sum() * target.shape[1] + 1e-8)


def karras_sigmas(num_steps: int, sigma_min: float, sigma_max: float,
                  rho: float, device) -> torch.Tensor:
    """EDM noise schedule (Eq. 5), with an appended sigma=0 endpoint."""
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def edm_heun_sampler(precond: EDMPrecond, cond: torch.Tensor, target_channels: int,
                     *, num_steps: int = 18, sigma_min: float = 0.002,
                     sigma_max: float = 80.0, rho: float = 7.0,
                     s_churn: float = 0.0, s_noise: float = 1.0,
                     s_tmin: float = 0.0, s_tmax: float = float("inf"),
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """2nd-order Heun sampler (EDM Algorithm 2).

    ``s_churn = 0`` recovers the deterministic Algorithm 1; ``s_churn > 0``
    re-injects noise each step (gamma capped at sqrt(2)-1), decorrelating
    ensemble members beyond their initial seeds. Per EDM Algorithm 2 the churn
    is only applied where ``s_tmin <= sigma <= s_tmax`` (defaults = all sigma,
    i.e. previous behaviour).
    ``cond`` is (B, Cc, H, W); returns the generated target (B, Ct, H, W).
    """
    b, _, h, w = cond.shape
    device = cond.device
    sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, rho, device)
    x = torch.randn(b, target_channels, h, w, device=device, generator=generator)
    x = x * sigmas[0]
    gamma_max = 2.0 ** 0.5 - 1.0

    for i in range(num_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        if s_churn > 0 and s_tmin <= s_cur <= s_tmax:
            gamma = min(s_churn / num_steps, gamma_max)
            s_hat = s_cur * (1.0 + gamma)
            eps = torch.randn(x.shape, device=device, generator=generator) * s_noise
            x = x + (s_hat ** 2 - s_cur ** 2).sqrt() * eps
        else:
            s_hat = s_cur
        sig = torch.full((b,), s_hat, device=device)
        d_cur = (x - precond(x, sig, cond)) / s_hat
        x_next = x + (s_next - s_hat) * d_cur
        if s_next > 0:  # 2nd-order correction
            sig_n = torch.full((b,), s_next, device=device)
            d_next = (x_next - precond(x_next, sig_n, cond)) / s_next
            x_next = x + (s_next - s_hat) * 0.5 * (d_cur + d_next)
        x = x_next
    return x
