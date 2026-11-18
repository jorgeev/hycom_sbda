"""EDM for multi-frame windows: preconditioning, loss, expert switch, sampler.

The video counterpart of :mod:`diffusion.edm`. The state is ``(B, T, C, H, W)``
and everything the network is told beyond the noisy state travels in one
``ctx`` dict, so call sites do not grow an argument per input:

    ctx = {
      "cond":   (B, T, Cc, H, W)  per-frame input channels, clean (optional)
      "sod":    (B, T)            UTC second of day of each frame   } calendar,
      "doy":    (B, T)            fraction of the year in [0, 1)    } when the
      "lon":    (B, H, W)         degrees east                      } model uses it
      "labels": (B, T, L)         per-frame scalar labels (optional)
    }

Two things follow GenLLC4320 rather than :mod:`diffusion.edm`:
  * the training sigma can be drawn LOG-UNIFORM on a fixed range, and one sigma
    is shared by all frames of a sample;
  * sampling can switch between two networks ("experts") at a sigma threshold.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .edm import karras_sigmas
from .networks_video import CalendarEmbedding


class VideoPrecond(nn.Module):
    """Karras preconditioning around a :class:`networks_video.VideoUNet`.

    ``forward(x_noisy, sigma, ctx)`` returns the denoised state ``D``. ``sigma``
    may be a scalar, ``(B,)`` or ``(B, T)``. Conditioning and calendar channels
    are concatenated AFTER the ``c_in`` scaling: they are never noised and never
    predicted.
    """

    def __init__(self, model: nn.Module, sigma_data: float = 1.0,
                 calendar_harmonics: int = 0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.calendar = (CalendarEmbedding(calendar_harmonics)
                         if calendar_harmonics > 0 else None)

    def forward(self, x_noisy: torch.Tensor, sigma, ctx: dict | None = None) -> torch.Tensor:
        ctx = ctx or {}
        b, t = x_noisy.shape[:2]
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x_noisy.device)
        if sigma.ndim == 0:
            sigma = sigma.expand(b)
        if sigma.ndim == 1:
            sigma = sigma[:, None].expand(b, t)
        s = sigma.reshape(b, t, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (s ** 2 + sd ** 2).sqrt()
        c_noise = sigma.log() / 4.0                       # (B, T)

        parts = [c_in * x_noisy]
        cond = ctx.get("cond")
        if cond is not None and cond.shape[2] > 0:
            parts.append(cond.to(x_noisy.dtype))
        if self.calendar is not None:
            missing = [k for k in ("sod", "doy", "lon") if ctx.get(k) is None]
            if missing:
                raise ValueError(f"this model uses calendar channels; ctx lacks {missing}")
            parts.append(self.calendar(ctx["sod"], ctx["doy"], ctx["lon"]).to(x_noisy.dtype))
        f_x = self.model(torch.cat(parts, dim=2), c_noise, ctx.get("labels"))
        return c_skip * x_noisy + c_out * f_x


class VideoEDMLoss:
    """Karras-weighted denoising loss with one sigma per sample.

    ``sigma_sampling="lognormal"`` draws ``ln sigma ~ N(p_mean, p_std)`` (EDM);
    ``"loguniform"`` draws ``ln sigma`` uniformly on
    ``[ln sigma_min, ln sigma_max]`` (GenLLC4320, one range per expert).

    ``reduction="masked_mean"`` averages over ocean pixels, frames and channels;
    ``"sum"`` is GenDA's sum over everything but the batch.
    """

    def __init__(self, sigma_sampling: str = "lognormal", p_mean: float = -1.2,
                 p_std: float = 1.2, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 sigma_data: float = 1.0, reduction: str = "masked_mean"):
        if sigma_sampling not in ("lognormal", "loguniform"):
            raise ValueError(f"Unknown sigma_sampling {sigma_sampling!r}")
        if reduction not in ("masked_mean", "sum"):
            raise ValueError(f"Unknown loss reduction {reduction!r}")
        if sigma_sampling == "loguniform" and not (0 < sigma_min < sigma_max):
            raise ValueError("loguniform needs 0 < train_sigma_min < train_sigma_max")
        self.sigma_sampling = sigma_sampling
        self.p_mean, self.p_std = p_mean, p_std
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.sigma_data = sigma_data
        self.reduction = reduction

    def sigma_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """The sigma at quantile ``u`` in (0, 1) of the training distribution."""
        if self.sigma_sampling == "loguniform":
            lo, hi = math.log(self.sigma_min), math.log(self.sigma_max)
            return torch.exp(lo + u * (hi - lo))
        z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        return torch.exp(self.p_mean + self.p_std * z)

    def sample_sigma(self, b: int, device, generator=None) -> torch.Tensor:
        if self.sigma_sampling == "loguniform":
            return self.sigma_from_u(torch.rand(b, device=device, generator=generator))
        rnd = torch.randn(b, device=device, generator=generator)
        return (self.p_mean + self.p_std * rnd).exp()

    def stratified_sigma(self, b: int, device) -> torch.Tensor:
        """``b`` sigmas at evenly spaced quantiles -- a fixed grid for validation."""
        u = (torch.arange(b, device=device, dtype=torch.float32) + 0.5) / b
        return self.sigma_from_u(u)

    def __call__(self, precond, target: torch.Tensor, ctx: dict, mask: torch.Tensor,
                 sigma: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        b, t, c = target.shape[:3]
        if sigma is None:
            sigma = self.sample_sigma(b, target.device, generator)
        s = sigma.reshape(b, 1, 1, 1, 1)                 # shared by every frame
        weight = (s ** 2 + self.sigma_data ** 2) / (s * self.sigma_data) ** 2
        noise = torch.randn(target.shape, device=target.device, generator=generator) * s
        denoised = precond(target + noise, sigma, ctx)
        se = weight * (denoised - target) ** 2
        if self.reduction == "sum":
            return se.sum() / b
        m = mask.reshape(b, 1, 1, *mask.shape[-2:])       # (B, 1, 1, H, W)
        return (se * m).sum() / (m.sum() * t * c + 1e-8)


class ExpertSwitch(nn.Module):
    """Two noise-level experts behind one ``d_fn(x, sigma) -> D``.

    The large-sigma network answers for ``sigma > sigma_divide``, the small one
    otherwise (GenLLC4320 ``eps_edm``). ``large=None`` makes it a plain wrapper
    around a single network. ``ctx`` is bound with :meth:`bind`, so samplers that
    only know ``d_fn(x, sigma)`` -- ``heun_sampler_video`` below and the ones in
    ``sda/vpsde.py`` -- work unchanged. Returns fp32.
    """

    def __init__(self, small: nn.Module, large: nn.Module | None = None,
                 sigma_divide: float = 0.5,
                 autocast_dtype: torch.dtype | None = torch.bfloat16):
        super().__init__()
        self.small = small
        self.large = large
        self.sigma_divide = float(sigma_divide)
        self.autocast_dtype = autocast_dtype
        self.ctx: dict = {}

    def bind(self, ctx: dict | None) -> "ExpertSwitch":
        self.ctx = ctx or {}
        return self

    def pick(self, sigma: float) -> nn.Module:
        if self.large is not None and sigma > self.sigma_divide:
            return self.large
        return self.small

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        sig = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)
        if sig.numel() > 1 and not torch.allclose(sig, sig[:1].expand_as(sig)):
            raise ValueError("ExpertSwitch needs one sigma for the whole batch")
        net = self.pick(float(sig[0]))
        enabled = x.is_cuda and self.autocast_dtype is not None
        with torch.autocast("cuda", dtype=self.autocast_dtype or torch.bfloat16,
                            enabled=enabled):
            d = net(x, sig[0].expand(x.shape[0]).contiguous(), self.ctx)
        return d.float()


@torch.no_grad()
def heun_sampler_video(d_fn, shape, *, num_steps: int = 18, sigma_min: float = 0.002,
                       sigma_max: float = 80.0, rho: float = 7.0, s_churn: float = 0.0,
                       s_noise: float = 1.0, s_tmin: float = 0.0,
                       s_tmax: float = float("inf"),
                       generator: torch.Generator | None = None,
                       device=None) -> torch.Tensor:
    """EDM Algorithm 2 (Heun, optional churn) over ``d_fn(x, sigma) -> D``.

    The loop of :func:`diffusion.edm.edm_heun_sampler`, with the state shape
    given explicitly instead of read off a conditioning tensor, so it serves any
    rank -- here ``(B, T, C, H, W)``.
    """
    sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, rho, device)
    x = torch.randn(shape, device=device, generator=generator) * sigmas[0]
    gamma_max = 2.0 ** 0.5 - 1.0
    for i in range(num_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        if s_churn > 0 and s_tmin <= s_cur <= s_tmax:
            gamma = min(s_churn / num_steps, gamma_max)
            s_hat = s_cur * (1.0 + gamma)
            eps = torch.randn(x.shape, device=x.device, generator=generator) * s_noise
            x = x + (s_hat ** 2 - s_cur ** 2).sqrt() * eps
        else:
            s_hat = s_cur
        d_cur = (x - d_fn(x, s_hat)) / s_hat
        x_next = x + (s_next - s_hat) * d_cur
        if s_next > 0:
            d_next = (x_next - d_fn(x_next, s_next)) / s_next
            x_next = x + (s_next - s_hat) * 0.5 * (d_cur + d_next)
        x = x_next
    return x


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
class _SpyNet(nn.Module):
    """Stands in for VideoUNet: records what it was called with, returns zeros."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.calls = []

    def forward(self, x, noise_labels, labels=None):
        self.calls.append({"in": tuple(x.shape), "noise": noise_labels.clone(),
                           "labels": None if labels is None else tuple(labels.shape)})
        return x.new_zeros(x.shape[0], x.shape[1], self.out_channels, *x.shape[-2:])


def _selftest() -> None:
    torch.manual_seed(0)
    b, t, c, h, w = 4, 5, 3, 8, 8
    ctx = {"cond": torch.randn(b, t, 2, h, w), "sod": torch.zeros(b, t),
           "doy": torch.zeros(b, t), "lon": torch.zeros(b, h, w),
           "labels": torch.randn(b, t, 2)}

    # 1. preconditioner: channel assembly and the three accepted sigma shapes
    spy = _SpyNet(c)
    pre = VideoPrecond(spy, sigma_data=1.0, calendar_harmonics=6)
    x = torch.randn(b, t, c, h, w)
    for sig in (torch.tensor(2.0), torch.full((b,), 2.0), torch.full((b, t), 2.0)):
        d = pre(x, sig, ctx)
        assert d.shape == x.shape
        # zero network -> D = c_skip * x
        assert torch.allclose(d, x / (2.0 ** 2 + 1.0), atol=1e-6)
    assert spy.calls[-1]["in"] == (b, t, c + 2 + 24, h, w)
    assert spy.calls[-1]["labels"] == (b, t, 2)
    try:
        pre(x, torch.tensor(1.0), {"cond": ctx["cond"]})
    except ValueError:
        pass
    else:
        raise AssertionError("missing calendar inputs were not caught")
    print("[ok] preconditioner: 3 state + 2 cond + 24 calendar channels; "
          "c_skip matches; missing calendar inputs refused")

    # 2. loss: log-uniform range, one sigma per sample shared across frames
    loss = VideoEDMLoss("loguniform", sigma_min=0.1, sigma_max=200.0)
    s = loss.sample_sigma(20000, "cpu")
    assert float(s.min()) >= 0.1 and float(s.max()) <= 200.0
    # uniform in log: median near the geometric mean, each decade equally likely
    assert abs(float(s.log().median()) - 0.5 * (math.log(0.1) + math.log(200.0))) < 0.08
    frac = float(((s > 1) & (s < 10)).float().mean())
    assert abs(frac - 1 / math.log10(2000)) < 0.02, frac
    spy.calls.clear()
    val = loss(pre, x, ctx, torch.ones(b, 1, h, w))
    nl = spy.calls[-1]["noise"]                                   # (B, T)
    assert torch.isfinite(val) and torch.equal(nl, nl[:, :1].expand_as(nl))
    assert nl[:, 0].unique().numel() == b
    print("[ok] log-uniform sigma stays in range, is uniform per decade, and is "
          "shared by all frames of a sample")

    # 3. lognormal quantiles and the fixed validation grid
    ln = VideoEDMLoss("lognormal", p_mean=-1.2, p_std=1.2)
    g = ln.stratified_sigma(1001, "cpu")
    assert abs(float(g[500]) - math.exp(-1.2)) < 1e-3 and bool((g[1:] > g[:-1]).all())
    g2 = loss.stratified_sigma(8, "cpu")
    assert float(g2[0]) > 0.1 and float(g2[-1]) < 200.0 and torch.equal(g2, loss.stratified_sigma(8, "cpu"))
    print("[ok] stratified validation sigmas are deterministic quantiles")

    # 4. masked loss ignores land
    mask = torch.ones(b, 1, h, w)
    mask[..., :4] = 0
    xl = x.clone()
    xl[..., :4] = 1e3                                             # garbage on land
    gen = lambda: torch.Generator().manual_seed(1)
    sig = torch.full((b,), 1.0)
    a = loss(pre, x, ctx, mask, sigma=sig, generator=gen())
    bb = loss(pre, xl, ctx, mask, sigma=sig, generator=gen())
    assert torch.allclose(a, bb, rtol=1e-5), (a, bb)
    print("[ok] masked_mean ignores land pixels")

    # 5. expert switch
    small, large = _SpyNet(c), _SpyNet(c)
    sw = ExpertSwitch(VideoPrecond(small), VideoPrecond(large), sigma_divide=0.5).bind({})
    sw(x, 0.5); sw(x, torch.tensor(0.2)); sw(x, 0.51); sw(x, torch.full((b,), 80.0))
    assert len(small.calls) == 2 and len(large.calls) == 2
    single = ExpertSwitch(VideoPrecond(small)).bind({})
    single(x, 80.0)
    assert len(small.calls) == 3
    try:
        sw(x, torch.tensor([0.1, 0.1, 5.0, 5.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("mixed sigmas were not caught")
    print("[ok] expert switch: small at sigma <= 0.5, large above; single-model mode")

    # 6. sampler. (a) It is the loop of diffusion.edm.edm_heun_sampler: on a 4-D
    #    state with the same generator the two must agree to the bit. (b) Against
    #    the exact denoiser of N(0, sd^2) data, a 5-D sample comes out with std
    #    sd once the step count is high enough for the discretisation error to
    #    vanish (at 32 steps from sigma 200 it is +2 % without churn, +5 % with).
    from .edm import edm_heun_sampler
    sd = 0.7
    d_exact = lambda xx, s_: xx * sd ** 2 / (sd ** 2 + s_ ** 2)
    ref_pre = lambda xx, sig_, cond_: d_exact(xx, sig_.reshape(-1, 1, 1, 1))
    for churn in (0.0, 10.0):
        ref = edm_heun_sampler(ref_pre, torch.zeros(6, 0, 8, 8), 3, num_steps=20,
                               sigma_max=200.0, s_churn=churn,
                               generator=torch.Generator().manual_seed(3))
        mine = heun_sampler_video(d_exact, (6, 3, 8, 8), num_steps=20, sigma_max=200.0,
                                  s_churn=churn, generator=torch.Generator().manual_seed(3))
        assert torch.equal(ref, mine)
        out = heun_sampler_video(d_exact, (64, 5, 2, 8, 8), num_steps=128, sigma_max=200.0,
                                 s_churn=churn, generator=torch.Generator().manual_seed(0))
        assert out.shape == (64, 5, 2, 8, 8)
        assert abs(float(out.std()) - sd) < 0.01, float(out.std())
    print("[ok] Heun sampler is bit-identical to edm_heun_sampler on 4-D and "
          "recovers a Gaussian prior's std on a 5-D state (churn 0 and 10)")

    # 7. the sda VP sampler accepts the same d_fn and a 5-D shape unchanged
    try:
        from sda.vpsde import vp_sample
    except Exception as e:                                         # pragma: no cover
        print(f"[skip] sda.vpsde not importable here: {e}")
    else:
        out = vp_sample(d_exact, (64, 5, 2, 8, 8), steps=64, device="cpu",
                        generator=torch.Generator().manual_seed(0))
        assert out.shape == (64, 5, 2, 8, 8) and abs(float(out.std()) - sd) < 0.03
        print("[ok] sda.vpsde.vp_sample runs on the 5-D state through the same d_fn")
    print("edm_video selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
