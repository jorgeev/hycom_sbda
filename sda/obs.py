"""Observation operators, masks and observation vectors for SDA.

GenDA (``inference/OSSE_inference.py:352-397``) builds one flat observation
vector ``y = [pointwise obs ; masked Gaussian-blurred L4 proxies]`` and an
operator ``A(x)`` returning the same layout from a state estimate. This module
is that, made descriptor-driven: every term is a YAML entry

    - name: ssh_aviso
      var: ssh                      # target channel it observes
      kind: blur | pointwise
      sigma_km: 65                  # blur only; Gaussian sigma (GenDA's convention)
      demean: true                  # subtract the mean over the term's pixels on
                                    # BOTH sides, in physical units (tide, MDT offset)
      anomaly: true                 # the product is an anomaly wrt a mean surface
                                    # (SSHA): compare with the state anomaly, i.e.
                                    # never add the climatology before blurring
      y_source: store:ssha_aviso    # or "truth"
      mask: {source: full | none | store_finite | tracks, ...}
      noise_std: 0.05               # physical units -> /std[var]; or noise_std_norm
      gamma: 0.1                    # optional per-term override
      add_noise: false              # default: true for y_source truth, false for store
      border_px: auto               # blur only; excluded rim; auto = ceil(1.5 sigma_px)

Units. The network state is normalised: ``x = (phys - clim_day) / std`` per
channel (``clim_day`` is a per-pixel field under ``norm_mode: anomaly``, a
scalar under ``zscore``; log10 variables are in log space). Blur terms convert
to physical units, blur, and convert back -- exactly GenDA's "add the means
before coarse-graining". Demeaned terms subtract the mean over the term's own
pixels in physical units on both ``A(x)`` and ``y``; that is what makes an
SSHA product (relative to an unknown mean SSH, without the tide) comparable
to a full-SSH prior state.

Land. Blurs use normalised convolution with the ocean mask (numerator and
denominator both masked), so land contributes nothing and receives zero
gradient. Every term's mask is ANDed with the ocean mask.

Everything is differentiable; ``--selftest`` checks the adjoint by autograd.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# per-day context
# ---------------------------------------------------------------------------
@dataclass
class ObsContext:
    """What every term needs to convert between normalised and physical units."""
    ocean: torch.Tensor          # (H, W) bool
    clim: torch.Tensor           # (C, H, W) float32 (broadcastable from (C,1,1))
    std: torch.Tensor            # (C,) float32
    is_log: list                 # per channel: state is log10(phys)
    dx_km: float
    target: list                 # channel names

    @property
    def device(self):
        return self.ocean.device

    def to_phys(self, x_norm: torch.Tensor, ch: int) -> torch.Tensor:
        """``(B,1,H,W)`` normalised -> physical (log10 space for log variables)."""
        return x_norm * self.std[ch] + self.clim[ch]

    def to_norm(self, phys: torch.Tensor, ch: int) -> torch.Tensor:
        return (phys - self.clim[ch]) / self.std[ch]


# ---------------------------------------------------------------------------
# land-safe Gaussian blur (from nemo_confusion/diffusion/obs_operator.py:53-92)
# ---------------------------------------------------------------------------
def gaussian_kernel_1d(sigma_px: float, device, dtype=torch.float32) -> torch.Tensor:
    if sigma_px < 1e-6:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


_BLUR_CACHE: dict = {}


def blur_matrix(n: int, sigma_px: float, device, dtype=torch.float32) -> torch.Tensor:
    """``(n, n)`` matrix of the 1-D Gaussian with REPLICATE padding: taps that
    fall outside ``[0, n)`` are clamped onto the edge pixel, so every row sums to
    one and ``M @ f`` equals ``conv1d(replicate_pad(f), kernel)`` exactly."""
    key = (n, round(float(sigma_px), 6), str(device), dtype)
    m = _BLUR_CACHE.get(key)
    if m is None:
        k1d = gaussian_kernel_1d(float(sigma_px), device, dtype)
        r = k1d.numel() // 2
        rows = torch.arange(n, device=device)[:, None].expand(n, k1d.numel())
        cols = (rows + torch.arange(-r, r + 1, device=device)[None, :]).clamp_(0, n - 1)
        m = torch.zeros(n, n, device=device, dtype=dtype)
        m.index_put_((rows.reshape(-1), cols.reshape(-1)), k1d.expand(n, -1).reshape(-1),
                     accumulate=True)
        _BLUR_CACHE[key] = m
    return m


def gaussian_blur(x: torch.Tensor, sigma_px: float, valid: torch.Tensor) -> torch.Tensor:
    """Isotropic Gaussian of std ``sigma_px`` over ``x (B,C,H,W)`` with normalised
    convolution against ``valid (H,W)`` (1 = ocean). Separable, replicate-padded,
    fully differentiable; masked pixels get zero weight and zero gradient.

    Implemented as ``M_y @ x @ M_x^T`` with the two banded blur matrices rather
    than ``conv2d``: cuDNN's backward for a 1-D kernel of 200+ taps is ~100x
    slower than its forward (5 s per call at 256^2 for the 65 km AVISO term),
    while the matrix form costs two small matmuls either way and is exact.
    """
    m = valid.to(dtype=x.dtype, device=x.device).view(1, 1, *valid.shape)
    if float(sigma_px) < 1e-6:
        return x * m
    h, w = x.shape[-2:]
    my = blur_matrix(h, sigma_px, x.device, x.dtype)
    mx = blur_matrix(w, sigma_px, x.device, x.dtype)

    def sep(a):
        return torch.matmul(torch.matmul(my, a), mx.transpose(0, 1))

    num = sep(x * m)
    den = sep(m.expand(x.shape[0], x.shape[1], -1, -1))
    return num / den.clamp_min(1e-6)


def _gaussian_blur_conv(x: torch.Tensor, sigma_px: float, valid: torch.Tensor) -> torch.Tensor:
    """Reference conv2d implementation (nemo_confusion/diffusion/obs_operator.py:62-92
    convention); kept for the selftest only."""
    k1d = gaussian_kernel_1d(float(sigma_px), x.device, x.dtype)
    m = valid.to(dtype=x.dtype, device=x.device).view(1, 1, *valid.shape)
    if k1d.numel() == 1:
        return x * m
    c = x.shape[1]
    kx = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    ky = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    pad = k1d.numel() // 2
    mc = m.expand(x.shape[0], c, -1, -1)

    def sep(a):
        a = F.conv2d(F.pad(a, (pad, pad, 0, 0), mode="replicate"), kx, groups=c)
        return F.conv2d(F.pad(a, (0, 0, pad, pad), mode="replicate"), ky, groups=c)

    num = sep(x * mc)
    den = sep(mc)
    return num / den.clamp_min(1e-6)


def masked_mean(a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``a (B,1,H,W)`` over ``mask (H,W)`` pixels -> ``(B,1,1,1)``."""
    m = mask.to(a.dtype)
    return (a * m).sum(dim=(2, 3), keepdim=True) / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# one observation term
# ---------------------------------------------------------------------------
@dataclass
class ObsTerm:
    name: str
    var: str
    channel: int
    kind: str                          # "pointwise" | "blur"
    mask: torch.Tensor                 # (H, W) bool -- observed pixels (AND ocean, AND border)
    std_norm: float                    # observation-error std, normalised units
    gamma: float = 0.1
    sigma_px: float = 0.0
    demean: bool = False
    anomaly: bool = False              # product is an anomaly wrt a mean surface ~ clim:
                                       # operate on the state anomaly, never add clim
    y_source: str = "truth"
    y_norm: torch.Tensor | None = None  # (n,) normalised, filled by build_terms
    y_grid: np.ndarray | None = None    # (H, W) float32, NaN outside mask (for panels)
    noise_realised: float = float("nan")

    @property
    def n(self) -> int:
        return int(self.mask.sum())

    def apply(self, x_hat: torch.Tensor, ctx: ObsContext) -> torch.Tensor:
        """``x_hat (B, C, H, W)`` normalised -> ``(B, n)`` normalised."""
        xc = x_hat[:, self.channel:self.channel + 1]
        sd = ctx.std[self.channel]
        # an anomaly product (SSHA) is compared with the state ANOMALY: adding the
        # climatology first would put the blurred mean dynamic topography into
        # A(x) and not into y (0.25 m RMS across a 256 patch of the Gulf Stream)
        phys = xc * sd if self.anomaly else ctx.to_phys(xc, self.channel)
        if self.kind == "blur":
            b = gaussian_blur(phys, self.sigma_px, ctx.ocean)
            if self.demean:
                out = (b - masked_mean(b, self.mask)) / sd
            elif self.anomaly:
                out = b / sd
            else:
                out = ctx.to_norm(b, self.channel)
        elif self.kind == "pointwise":
            if self.demean:
                out = (phys - masked_mean(phys, self.mask)) / sd
            else:
                out = xc
        else:
            raise ValueError(f"unknown term kind {self.kind!r}")
        return out[:, 0][:, self.mask]

    def observe(self, field_phys: torch.Tensor, ctx: ObsContext) -> torch.Tensor:
        """Physical (log10 for log vars) ``(H, W)`` field -> ``y (n,)`` normalised,
        with the SAME demeaning / normalisation as :meth:`apply` (no blur: a
        store product is already the blurred quantity)."""
        f = field_phys.view(1, 1, *field_phys.shape)
        if self.demean:
            out = (f - masked_mean(torch.nan_to_num(f), self.mask)) / ctx.std[self.channel]
        elif self.anomaly:
            out = f / ctx.std[self.channel]
        else:
            out = ctx.to_norm(f, self.channel)
        return out[0, 0][self.mask]


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def track_mask(h: int, w: int, dx_km: float, spacing_km: float, width_px: float = 1.0,
               angles_deg=(24.0, -24.0), jitter_deg: float = 5.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic nadir tracks: straight parallel lines per angle, random phase and
    jitter per (seed, day). Coverage ~ ``n_angles * width_px / (spacing_km/dx_km)``."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    s_px = spacing_km / dx_km
    m = np.zeros((h, w), dtype=bool)
    for ang in angles_deg:
        th = np.deg2rad(ang + rng.uniform(-jitter_deg, jitter_deg))
        phase = rng.uniform(0.0, s_px)
        d = (xx * np.cos(th) + yy * np.sin(th) + phase) % s_px
        m |= np.minimum(d, s_px - d) < width_px / 2.0
    return m


def swath_mask(h: int, w: int, dx_km: float, spacing_km: float, width_km: float,
               gap_km: float, angles_deg=(24.0, -24.0), jitter_deg: float = 5.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """SWOT-like two-swath pattern: two bands of ``width_km`` separated by ``gap_km``."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    s_px = spacing_km / dx_km
    half_gap, wid = gap_km / dx_km / 2.0, width_km / dx_km
    m = np.zeros((h, w), dtype=bool)
    for ang in angles_deg:
        th = np.deg2rad(ang + rng.uniform(-jitter_deg, jitter_deg))
        phase = rng.uniform(0.0, s_px)
        d = (xx * np.cos(th) + yy * np.sin(th) + phase) % s_px
        d = np.minimum(d, s_px - d)
        m |= (d > half_gap) & (d < half_gap + wid)
    return m


def border_mask(h: int, w: int, border_px: int) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    b = int(border_px)
    if b > 0 and 2 * b < min(h, w):
        m[:b, :] = False
        m[-b:, :] = False
        m[:, :b] = False
        m[:, -b:] = False
    return m


def resolve_mask(mcfg: dict, ocean: np.ndarray, dx_km: float, store_frames: dict,
                 rng: np.random.Generator) -> np.ndarray:
    """``mask:`` block -> ``(H, W)`` bool, ANDed with ocean."""
    src = (mcfg or {}).get("source", "full")
    h, w = ocean.shape
    if src == "full":
        m = np.ones((h, w), dtype=bool)
    elif src == "none":
        m = np.zeros((h, w), dtype=bool)
    elif src == "store_finite":
        var = mcfg["var"]
        if var not in store_frames or store_frames[var] is None:
            m = np.zeros((h, w), dtype=bool)
        else:
            m = np.isfinite(np.asarray(store_frames[var]))
    elif src == "tracks":
        m = track_mask(h, w, dx_km, mcfg["spacing_km"], mcfg.get("width_px", 1.0),
                       tuple(mcfg.get("angles_deg", (24.0, -24.0))),
                       mcfg.get("jitter_deg", 5.0), rng)
    elif src == "swath":
        m = swath_mask(h, w, dx_km, mcfg["spacing_km"], mcfg["width_km"], mcfg["gap_km"],
                       tuple(mcfg.get("angles_deg", (24.0, -24.0))),
                       mcfg.get("jitter_deg", 5.0), rng)
    else:
        raise ValueError(f"unknown mask source {src!r}")
    return m & ocean


# ---------------------------------------------------------------------------
# spec -> terms
# ---------------------------------------------------------------------------
def load_obs_spec(path: str) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f)
    spec.setdefault("defaults", {})
    spec.setdefault("terms", [])
    spec.setdefault("optional_terms", [])
    spec.setdefault("name", str(path).rsplit("/", 1)[-1].rsplit(".", 1)[0])
    for t in spec["terms"] + spec["optional_terms"]:
        for k in ("name", "var", "kind"):
            if k not in t:
                raise ValueError(f"obs term {t} lacks {k!r}")
        if t["kind"] == "blur" and "sigma_km" not in t and "cutoff_km" not in t:
            raise ValueError(f"blur term {t['name']} needs sigma_km (or cutoff_km)")
        if "noise_std" not in t and "noise_std_norm" not in t:
            raise ValueError(f"term {t['name']} needs noise_std or noise_std_norm")
    return spec


def store_vars_needed(spec: dict, include_optional: bool = True) -> list[str]:
    """Store variables the spec reads (for masks and for ``y_source: store:*``)."""
    out = []
    terms = spec["terms"] + (spec["optional_terms"] if include_optional else [])
    for t in terms:
        src = str(t.get("y_source", "truth"))
        if src.startswith("store:"):
            out.append(src.split(":", 1)[1])
        m = t.get("mask") or {}
        if m.get("source") == "store_finite":
            out.append(m["var"])
    return list(dict.fromkeys(out))


def select_terms(spec: dict, names: list[str] | None) -> list[dict]:
    """``--terms`` handling: ``None`` -> the required terms; a list -> those names
    from required + optional, in the order given."""
    if names is None:
        return list(spec["terms"])
    pool = {t["name"]: t for t in spec["terms"] + spec["optional_terms"]}
    missing = [n for n in names if n not in pool]
    if missing:
        raise KeyError(f"unknown term(s) {missing}; known: {sorted(pool)}")
    return [pool[n] for n in names]


def _sigma_px(tcfg: dict, dx_km: float) -> float:
    if "sigma_km" in tcfg:
        return float(tcfg["sigma_km"]) / dx_km
    return float(tcfg["cutoff_km"]) / (2.0 * math.pi) / dx_km   # parent-repo convention


def build_terms(term_cfgs: list[dict], defaults: dict, ctx: ObsContext,
                truth_norm: torch.Tensor, store_frames: dict,
                rng: np.random.Generator) -> list[ObsTerm]:
    """Instantiate the terms for one day / geometry and fill their ``y``.

    ``truth_norm`` is ``(C, H, W)`` normalised; ``store_frames`` maps store var
    -> ``(H, W)`` physical frame (NaN = gap) already cropped to the geometry.
    ``rng`` seeds synthetic masks and the observation noise, so pass one seeded
    by ``(seed, day)`` for reproducibility.
    """
    ocean_np = ctx.ocean.cpu().numpy().astype(bool)
    h, w = ocean_np.shape
    terms: list[ObsTerm] = []
    for tcfg in term_cfgs:
        var = tcfg["var"]
        if var not in ctx.target:
            raise KeyError(f"term {tcfg['name']}: var {var!r} not in target {ctx.target}")
        ch = ctx.target.index(var)
        kind = tcfg["kind"]
        gamma = float(tcfg.get("gamma", defaults.get("gamma", 0.1)))
        demean = bool(tcfg.get("demean", False))
        anomaly = bool(tcfg.get("anomaly", False))
        y_source = str(tcfg.get("y_source", "truth"))
        sigma_px = _sigma_px(tcfg, ctx.dx_km) if kind == "blur" else 0.0
        std_phys = float(ctx.std[ch])
        if "noise_std_norm" in tcfg:
            std_norm = float(tcfg["noise_std_norm"])
        else:
            std_norm = float(tcfg["noise_std"]) / std_phys
        add_noise = bool(tcfg.get("add_noise", y_source == "truth"))

        mask = resolve_mask(tcfg.get("mask"), ocean_np, ctx.dx_km, store_frames, rng)
        if kind == "blur":
            b = tcfg.get("border_px", defaults.get("border_px", "auto"))
            # auto = 1 sigma: 16% of the kernel mass falls outside the patch at the
            # rim's inner edge; that representation error is what the `truth rel`
            # column of report_sda.md measures and the noise_std absorbs. 1.5 sigma
            # left the 65 km AVISO term on 33% of a 256 patch.
            bpx = int(math.ceil(1.0 * sigma_px)) if b == "auto" else int(b)
            mask &= border_mask(h, w, bpx)
        raw = None
        if y_source.startswith("store:"):
            sv = y_source.split(":", 1)[1]
            fr = store_frames.get(sv)
            if fr is None:
                mask[:] = False
            else:
                raw = np.asarray(fr, dtype=np.float64)
                if ctx.is_log[ch]:
                    raw = np.log10(np.where(raw > 0, raw, np.nan))
                mask &= np.isfinite(raw)
        term = ObsTerm(name=tcfg["name"], var=var, channel=ch, kind=kind,
                       mask=torch.from_numpy(mask).to(ctx.device), std_norm=std_norm,
                       gamma=gamma, sigma_px=sigma_px, demean=demean, anomaly=anomaly,
                       y_source=y_source)
        if term.n == 0:
            term.y_norm = torch.zeros(0, device=ctx.device)
            term.y_grid = np.full((h, w), np.nan, dtype=np.float32)
            terms.append(term)
            continue
        with torch.no_grad():
            if raw is None:
                y = term.apply(truth_norm[None].to(ctx.device), ctx)[0]
            else:
                y = term.observe(torch.from_numpy(raw).float().to(ctx.device), ctx)
            if add_noise and std_norm > 0:
                noise = torch.from_numpy(rng.standard_normal(term.n)).float().to(ctx.device)
                y = y + std_norm * noise
                term.noise_realised = float(noise.std()) * std_norm
        term.y_norm = y.float()
        grid = np.full((h, w), np.nan, dtype=np.float32)
        grid[mask] = y.cpu().numpy()
        term.y_grid = grid
        terms.append(term)
    return terms


def concat_y(terms: list[ObsTerm], batch: int, device):
    """``(y (B, N), std (N,), gamma (N,))`` in the concatenation order of :func:`build_operator`."""
    if not terms or sum(t.n for t in terms) == 0:
        z = torch.zeros(0, device=device)
        return z.expand(batch, 0), z, z
    y = torch.cat([t.y_norm for t in terms if t.n > 0]).to(device)
    std = torch.cat([torch.full((t.n,), t.std_norm, device=device) for t in terms if t.n > 0])
    gam = torch.cat([torch.full((t.n,), t.gamma, device=device) for t in terms if t.n > 0])
    return y[None].expand(batch, -1), std, gam


def build_operator(terms: list[ObsTerm], ctx: ObsContext):
    active = [t for t in terms if t.n > 0]

    def A(x_hat: torch.Tensor) -> torch.Tensor:
        if not active:
            return x_hat.new_zeros(x_hat.shape[0], 0)
        return torch.cat([t.apply(x_hat, ctx) for t in active], dim=1)

    return A


def describe(terms: list[ObsTerm], hw) -> str:
    npx = hw[0] * hw[1]
    rows = []
    for t in terms:
        rows.append(f"    {t.name:<12s} {t.var:<6s} {t.kind:<9s} n={t.n:>7d} "
                    f"({100.0 * t.n / npx:5.1f}%)  std_norm={t.std_norm:.3g} "
                    f"gamma={t.gamma:.3g}" + (f" sigma_px={t.sigma_px:.1f}" if t.kind == "blur" else "")
                    + (" demean" if t.demean else "") + (" anomaly" if t.anomaly else "")
                    + f"  y<-{t.y_source}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _selftest() -> int:
    torch.manual_seed(0)
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    print("[sda.obs] selftest")
    h, w, c = 48, 64, 3
    ocean = np.ones((h, w), dtype=bool)
    ocean[:5, :] = False
    ocean[20:28, 30:38] = False
    clim = torch.zeros(c, h, w)
    clim[0] = torch.linspace(-1, 1, w)[None, :].expand(h, w) * 0.3   # a spatially varying clim
    ctx = ObsContext(ocean=torch.from_numpy(ocean), clim=clim,
                     std=torch.tensor([0.4, 0.7, 0.06]), is_log=[False] * c, dx_km=1.818,
                     target=["ssh", "sst", "sss"])

    # 1. kernel normalisation and blur of a constant
    k = gaussian_kernel_1d(3.3, "cpu")
    check(abs(float(k.sum()) - 1.0) < 1e-6 and k.numel() == 2 * 10 + 1, "1-D kernel sums to 1, radius ceil(3 sigma)")
    const = torch.full((1, 1, h, w), 2.5)
    b = gaussian_blur(const, 4.0, ctx.ocean)
    check(float((b[0, 0][ctx.ocean] - 2.5).abs().max()) < 1e-5,
          "normalised convolution: blur(const) == const on ocean, land-adjacent included")
    xr = torch.randn(2, 3, h, w)
    for spx in (0.7, 4.0, 13.0, 40.0):       # 40 px: kernel wider than the 48-row field
        ref = _gaussian_blur_conv(xr, spx, ctx.ocean)
        got = gaussian_blur(xr, spx, ctx.ocean)
        check(float((ref - got)[..., ctx.ocean].abs().max()) < 1e-4,
              f"matrix blur == replicate-padded conv2d at sigma_px={spx} "
              f"(max diff {float((ref - got)[..., ctx.ocean].abs().max()):.1e})")

    # 2. adjoint <Ax, v> == <x, A^T v> via autograd, and zero gradient on land
    rng = np.random.default_rng(1)
    truth = torch.randn(c, h, w)
    truth[:, ~ctx.ocean] = 0.0
    defaults = {"gamma": 0.1, "border_px": 2}
    cfgs = [
        {"name": "p", "var": "sst", "kind": "pointwise", "noise_std": 0.1,
         "mask": {"source": "tracks", "spacing_km": 20, "width_px": 1}},
        {"name": "b", "var": "ssh", "kind": "blur", "sigma_km": 6.0, "noise_std": 0.02,
         "mask": {"source": "full"}},
        {"name": "bd", "var": "ssh", "kind": "blur", "sigma_km": 6.0, "demean": True,
         "noise_std": 0.02, "mask": {"source": "full"}},
        {"name": "pd", "var": "sss", "kind": "pointwise", "demean": True, "noise_std": 0.01,
         "mask": {"source": "full"}},
        {"name": "ba", "var": "ssh", "kind": "blur", "sigma_km": 6.0, "demean": True, "anomaly": True,
         "noise_std": 0.02, "mask": {"source": "full"}},
    ]
    terms = build_terms(cfgs, defaults, ctx, truth, {}, np.random.default_rng(1))
    A = build_operator(terms, ctx)
    x = torch.randn(2, c, h, w, requires_grad=True)
    ax = A(x)
    v = torch.randn_like(ax)
    (atv,) = torch.autograd.grad((ax * v).sum(), x)
    x2 = torch.randn(2, c, h, w)
    lhs = float((A(x2) * v).sum())
    rhs = float((x2 * atv).sum())
    # A is affine (clim, demean offsets): compare on differences to remove the offset
    lhs0 = float((A(torch.zeros_like(x2)) * v).sum())
    check(abs((lhs - lhs0) - rhs) < 1e-3 * max(1.0, abs(rhs)),
          f"adjoint via autograd: <Ax,v>-<A0,v>={lhs - lhs0:.5f} vs <x,A^T v>={rhs:.5f}")
    check(float(atv[:, :, ~ctx.ocean].abs().max()) == 0.0, "land pixels receive zero gradient")

    # 3. y of truth-sourced terms reproduces A(truth) (no noise) and noise std
    terms0 = build_terms([dict(t, add_noise=False) for t in cfgs], defaults, ctx, truth, {},
                         np.random.default_rng(1))          # same seed -> same track mask
    y0, _, _ = concat_y(terms0, 1, "cpu")
    check(float((y0[0] - A(truth[None])[0]).abs().max()) < 1e-6, "y == A(truth) when add_noise is off")
    y1, std1, gam1 = concat_y(terms, 1, "cpu")
    z = (y1[0] - y0[0]) / std1
    check(abs(float(z.std()) - 1.0) < 0.1 and gam1.numel() == y1.shape[1],
          f"added noise has unit std in std_norm units ({float(z.std()):.3f})")

    # 4. blur identity: A_blur(x) == blur(x)*1 + (blur(clim)-clim)/std for anomaly clim
    tb = [t for t in terms if t.name == "b"][0]
    xn = torch.randn(1, 1, h, w)
    lhs = tb.apply(torch.cat([xn, torch.zeros(1, c - 1, h, w)], 1), ctx)
    bl = gaussian_blur(xn * ctx.std[0], tb.sigma_px, ctx.ocean) / ctx.std[0]
    off = (gaussian_blur(clim[0][None, None], tb.sigma_px, ctx.ocean) - clim[0]) / ctx.std[0]
    check(float((lhs[0] - (bl + off)[0, 0][tb.mask]).abs().max()) < 1e-4,
          "blur term = blur(x) + (blur(clim) - clim)/std  (linearity, per-pixel clim)")
    check(bool((~tb.mask[:2]).all()) and bool(tb.mask[10:12, 10:12].all()),
          "blur term excludes the border rim")

    # 5. demeaned terms are invariant to a uniform offset (the tide), store obs too
    tbd = [t for t in terms if t.name == "bd"][0]
    xa = torch.randn(1, c, h, w)
    xb = xa.clone()
    xb[:, 0] += 3.0                                   # uniform SSH offset in normalised units
    check(float((tbd.apply(xa, ctx) - tbd.apply(xb, ctx)).abs().max()) < 1e-4,
          "demeaned blur term ignores a uniform SSH offset")
    tpd = [t for t in terms if t.name == "pd"][0]
    check(float((tpd.apply(xa, ctx) - tpd.apply(xb * 0 + xa + 0.0, ctx)).abs().max()) < 1e-6
          and float((tpd.apply(xa, ctx) - tpd.apply(xa + torch.tensor([0, 0, 5.0])[None, :, None, None], ctx)).abs().max()) < 1e-4,
          "demeaned pointwise term ignores a uniform offset")
    # observe(): a store field equal to truth's physical field gives y == A(truth)
    phys0 = truth[0] * ctx.std[0] + clim[0]
    y_obs = tpd.observe(truth[2] * ctx.std[2] + clim[2], ctx)
    check(float((y_obs - tpd.apply(truth[None], ctx)[0]).abs().max()) < 1e-5,
          "observe(phys) matches apply(norm) for the demeaned pointwise term")
    tb_obs = tb.observe(gaussian_blur(phys0[None, None], tb.sigma_px, ctx.ocean)[0, 0], ctx)
    check(float((tb_obs - tb.apply(truth[None], ctx)[0]).abs().max()) < 1e-4,
          "observe(blur(phys)) matches the blur term applied to the state")

    # 5b. anomaly term: independent of the climatology field entirely
    tba = [t for t in terms if t.name == "ba"][0]
    ctx2 = ObsContext(ocean=ctx.ocean, clim=clim * 0 + 7.0, std=ctx.std, is_log=ctx.is_log,
                      dx_km=ctx.dx_km, target=ctx.target)
    check(float((tba.apply(xa, ctx) - tba.apply(xa, ctx2)).abs().max()) < 1e-5
          and float((tba.apply(xa, ctx) - tbd.apply(xa, ctx)).abs().max() > 1e-3),
          "anomaly blur term ignores the climatology; the absolute one does not")
    # 6. masks: tracks deterministic and coverage as predicted; store_finite; none
    m1 = track_mask(h, w, 1.818, 20.0, 1.0, (24, -24), 5.0, np.random.default_rng(7))
    m2 = track_mask(h, w, 1.818, 20.0, 1.0, (24, -24), 5.0, np.random.default_rng(7))
    pred = 2 * 1.0 / (20.0 / 1.818)
    cov = m1.mean()
    check(np.array_equal(m1, m2) and abs(cov - pred) / pred < 0.25,
          f"tracks deterministic per seed, coverage {cov:.3f} vs predicted {pred:.3f}")
    frame = np.random.default_rng(2).standard_normal((h, w))
    frame[10:20, :] = np.nan
    ms = resolve_mask({"source": "store_finite", "var": "sst_sat"}, ocean, 1.818,
                      {"sst_sat": frame}, rng)
    check(not ms[10:20].any() and ms[30:40].sum() == ocean[30:40].sum(),
          "store_finite mask = isfinite(frame) & ocean")
    tn = build_terms([{"name": "z", "var": "sst", "kind": "pointwise", "noise_std": 0.1,
                       "mask": {"source": "none"}}], defaults, ctx, truth, {}, rng)
    check(tn[0].n == 0 and build_operator(tn, ctx)(truth[None]).shape == (1, 0),
          "a term with no pixels is inert")
    # store-sourced y with a store frame: mask ANDed with finite pixels, y in target units
    ts = build_terms([{"name": "s", "var": "sst", "kind": "pointwise", "noise_std": 0.1,
                       "y_source": "store:sst_sat", "mask": {"source": "store_finite", "var": "sst_sat"}}],
                     defaults, ctx, truth, {"sst_sat": truth[1].numpy() * 0.7 + 0.0}, rng)
    check(ts[0].n == int((np.isfinite(truth[1].numpy()) & ocean).sum())
          and float((ts[0].y_norm - truth[1][ts[0].mask]).abs().max()) < 1e-5
          and np.isnan(ts[0].noise_realised),
          "store-sourced y converts to the target's normalised units with no added noise")

    print(f"[sda.obs] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.error("nothing to do; pass --selftest")
