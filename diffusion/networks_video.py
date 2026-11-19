"""Video UNet: a self-contained port of GenLLC4320's ``DhariwalUNetVideo``.

Source: ``GenLLC4320/src/models.py`` (``DhariwalUNetVideo``, ``UNetBlockVideo``,
``TemporalAttention``, ``Conv2dVideo``, ``GroupNormVideo``) and
``src/embedding.py`` (``CalendarEmbedding``), which themselves extend
PhysicsNeMo's ADM ``DhariwalUNet``. Nothing here imports PhysicsNeMo.

The state is a short window of frames, ``(B, T, C, H, W)``. Convolutions,
normalisation and spatial attention run per frame (time is folded into the
batch); frames exchange information only in ``TemporalAttention``, where every
pixel attends to the same pixel in the other frames.

What is kept from GenLLC4320, because it is the point of the port:
  * ADM blocks: the embedding sets a per-channel scale AND shift after the
    second GroupNorm (``adaptive_scale``), that layer and the second conv start
    at zero, skips are not rescaled;
  * attention heads of ``channels_per_head`` (64) channels;
  * attention gated on absolute resolution (``img_resolution >> level``), with
    separate lists for spatial and temporal attention; the bottleneck's first
    block always has both;
  * kaiming-uniform init scaled by sqrt(1/3), GroupNorm eps 1e-5, ``[1, 1]``
    box resampling applied before the 3x3 conv;
  * a learned relative-position bias in the temporal attention, sized by the
    number of frames -- so ``n_frames`` is fixed per checkpoint.

Deliberate departures:
  * ``temporal_attn_residual=True`` (default) adds the temporal attention to
    its input, like the spatial attention in the same block. GenLLC4320 as
    committed assigns it instead (``x = self.TimeAttention(x)``); pass ``False``
    for that behaviour.
  * the temporal projection's bias starts at zero (GenLLC4320 leaves torch's
    default), so with the residual on the block starts as a per-frame model.
  * blocks are named by LEVEL (``L3_block0``), not by resolution, so a state
    dict does not depend on the training patch and loads at another patch size
    without renaming -- provided the attention lists are scaled with the patch
    so attention stays on the same levels (see :func:`attention_levels`).
  * spatial attention uses ``F.scaled_dot_product_attention`` (same softmax
    attention, no explicit n-by-n matrix), as ``networks_genda`` does.
  * the calendar channels are built by the preconditioner
    (``edm_video.VideoPrecond``) rather than inside the UNet, which therefore
    only sees ``in_channels``.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


# ---------------------------------------------------------------------------
# init / basic layers (physicsnemo.models.diffusion: weight_init, Linear, Conv2d)
# ---------------------------------------------------------------------------
def _weight_init(shape, mode: str, fan_in: int, fan_out: int) -> torch.Tensor:
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f"Invalid init mode {mode!r}")


_INIT = dict(init_mode="kaiming_uniform", init_weight=float(np.sqrt(1 / 3)),
             init_bias=float(np.sqrt(1 / 3)))
_INIT_ZERO = dict(init_mode="kaiming_uniform", init_weight=0.0, init_bias=0.0)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                 init_mode: str = "kaiming_normal", init_weight: float = 1.0,
                 init_bias: float = 0.0):
        super().__init__()
        kw = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(
            _weight_init([out_features, in_features], **kw) * init_weight)
        self.bias = (nn.Parameter(_weight_init([out_features], **kw) * init_bias)
                     if bias else None)

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


class Conv2d(nn.Module):
    """Conv on ``(N, C, H, W)`` with optional 2x box resampling before the kernel.

    ``kernel=0`` is "resample only" (the identity skip of an up/down block whose
    channel count does not change) and has no parameters.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int, *,
                 bias: bool = True, up: bool = False, down: bool = False,
                 resample_filter=(1, 1), init_mode: str = "kaiming_normal",
                 init_weight: float = 1.0, init_bias: float = 0.0):
        if up and down:
            raise ValueError("up and down cannot both be set")
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.up, self.down = up, down
        kw = dict(mode=init_mode, fan_in=in_channels * kernel * kernel,
                  fan_out=out_channels * kernel * kernel)
        self.weight = (nn.Parameter(
            _weight_init([out_channels, in_channels, kernel, kernel], **kw) * init_weight)
            if kernel else None)
        self.bias = (nn.Parameter(_weight_init([out_channels], **kw) * init_bias)
                     if kernel and bias else None)
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if (up or down) else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        f = self.resample_filter
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if self.up:
            x = F.conv_transpose2d(
                x, f.mul(4).tile([self.in_channels, 1, 1, 1]).to(x.dtype),
                groups=self.in_channels, stride=2, padding=f_pad)
        if self.down:
            x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]).to(x.dtype),
                         groups=self.in_channels, stride=2, padding=f_pad)
        if w is not None:
            x = F.conv2d(x, w, padding=w_pad)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype).reshape(1, -1, 1, 1)
        return x


class GroupNorm(nn.Module):
    """PhysicsNeMo's group rule: ``min(32, channels // 4)`` groups."""

    def __init__(self, num_channels: int, num_groups: int = 32,
                 min_channels_per_group: int = 4, eps: float = 1e-5):
        super().__init__()
        self.num_groups = max(1, min(num_groups, num_channels // min_channels_per_group))
        if num_channels % self.num_groups:
            raise ValueError(
                f"GroupNorm: {num_channels} channels do not divide into "
                f"{self.num_groups} groups; use channel widths that are multiples "
                "of 32 (or of 4 below 128)")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight.to(x.dtype),
                            self.bias.to(x.dtype), self.eps)


class PositionalEmbedding(nn.Module):
    """ADM sinusoidal embedding: ``endpoint=False``, cos then sin."""

    def __init__(self, num_channels: int, max_positions: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x):
        half = self.num_channels // 2
        freqs = torch.arange(half, dtype=torch.float32, device=x.device) / half
        freqs = (1 / self.max_positions) ** freqs
        x = x.float().ger(freqs)
        return torch.cat([x.cos(), x.sin()], dim=1)


# ---------------------------------------------------------------------------
# calendar channels (GenLLC4320 src/embedding.py, adapted from cBottle)
# ---------------------------------------------------------------------------
class CalendarEmbedding(nn.Module):
    """Local-solar-time and day-of-year harmonics as input channels.

    ``second_of_day`` and ``year_frac`` are ``(B, T)`` (UTC seconds since
    midnight; fraction of the year in [0, 1)), ``lon`` is ``(B, H, W)`` in
    degrees east. Returns ``(B, T, 4 * harmonics, H, W)``:
    ``[cos, sin]`` of ``2 pi k local_time`` (varies per pixel through ``lon``)
    then ``[cos, sin]`` of ``2 pi k year_frac`` (constant over the patch),
    ``k = 1..harmonics``. No learned parameters.
    """

    def __init__(self, harmonics: int = 6):
        super().__init__()
        self.harmonics = int(harmonics)
        self.out_channels = 4 * self.harmonics
        self.register_buffer(
            "freqs", torch.arange(1, self.harmonics + 1, dtype=torch.float32),
            persistent=False)

    def _embed(self, phase):                      # (B, T, H, W) in [0, 1)
        ang = 2 * math.pi * phase[:, :, None] * self.freqs[None, None, :, None, None]
        return torch.cat([ang.cos(), ang.sin()], dim=2)

    def forward(self, second_of_day, year_frac, lon):
        if second_of_day.shape != year_frac.shape:
            raise ValueError("second_of_day and year_frac must both be (B, T)")
        h, w = lon.shape[-2:]
        sod = second_of_day.float()[:, :, None, None]
        local = torch.remainder(sod + lon.float()[:, None] * (86400.0 / 360.0),
                                86400.0) / 86400.0
        yf = torch.remainder(year_frac.float(), 1.0)[:, :, None, None].expand(-1, -1, h, w)
        return torch.cat([self._embed(local), self._embed(yf)], dim=2)


# ---------------------------------------------------------------------------
# temporal attention (GenLLC4320 src/models.py:80-120)
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Per-pixel self-attention across the ``T`` frames.

    Input and output are ``(B, C, T, X)`` with ``X = H * W``. The relative
    bias table has ``2 * seq_length`` entries per head (GenLLC4320's size; the
    last one is never indexed), so the module is tied to ``seq_length`` frames.
    Computed in fp32 regardless of autocast, as in the source.
    """

    def __init__(self, out_channels: int, seq_length: int, num_heads: int,
                 eps: float = 1e-5):
        super().__init__()
        self.norm2 = GroupNorm(out_channels, eps=eps)
        self.qkv = nn.Conv2d(out_channels, out_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.proj.weight)
        with torch.no_grad():
            self.proj.weight.mul_(1e-5)
            self.proj.bias.zero_()
        self.positional_embedding = nn.Parameter(
            torch.randn(num_heads, 2 * seq_length) * 1e-5)
        self.num_heads = num_heads
        self.seq_length = seq_length

    def forward(self, x):
        dtype = x.dtype
        b, c, t, n = x.shape
        if t != self.seq_length:
            raise ValueError(
                f"TemporalAttention was built for {self.seq_length} frames, got {t}; "
                "n_frames is fixed per checkpoint")
        nh = self.num_heads
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            qkv = self.qkv(self.norm2(x))                       # (B, 3*nh*ch, T, X)
            # einops "b (n heads c) t x -> n (b x) heads t c"
            qkv = qkv.reshape(b, 3, nh, c // nh, t, n).permute(1, 0, 5, 2, 4, 3)
            q, k, v = qkv.reshape(3, b * n, nh, t, c // nh).unbind(0)
            i = torch.arange(t, device=x.device)
            rel = self.positional_embedding[:, i[:, None] - i[None, :] + t - 1]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=rel.float())
            # "(b x) h c t -> b (h c) t x"
            out = out.reshape(b, n, nh, t, c // nh).permute(0, 2, 4, 3, 1)
            out = self.proj(out.reshape(b, c, t, n))
        return out.to(dtype)


# Attention-matrix budget for one call, in elements (heads * tokens^2 per frame,
# 4 bytes each -> 4 GiB). At the training patch a whole batch of frames fits in
# one call; on a full frame the token count per frame is ~30x larger and the
# frames are processed a few at a time instead of allocating one matrix for all.
_ATTN_MAX_ELEMS = 2 ** 30


def _spatial_attention(q, k, v):
    """``F.scaled_dot_product_attention`` on ``(N, heads, tokens, c)``, chunked over N."""
    n, nh, tok, _ = q.shape
    chunk = max(1, _ATTN_MAX_ELEMS // (nh * tok * tok))
    if chunk >= n:
        return F.scaled_dot_product_attention(q, k, v)
    return torch.cat([F.scaled_dot_product_attention(q[i:i + chunk], k[i:i + chunk],
                                                     v[i:i + chunk])
                      for i in range(0, n, chunk)], dim=0)


# ---------------------------------------------------------------------------
# UNet block (GenLLC4320 src/models.py:265-446)
# ---------------------------------------------------------------------------
class UNetBlockVideo(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, *,
                 up: bool = False, down: bool = False, attention: bool = False,
                 temporal_attention: bool = False, time_length: int = 1,
                 channels_per_head: int = 64, dropout: float = 0.0,
                 skip_scale: float = 1.0, eps: float = 1e-5,
                 temporal_residual: bool = True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.temporal_residual = temporal_residual
        heads = max(1, out_channels // channels_per_head)
        if out_channels % heads:
            raise ValueError(
                f"{out_channels} channels do not split into {heads} attention heads "
                f"(channels_per_head={channels_per_head})")
        self.num_heads = heads if attention else 0

        self.norm0 = GroupNorm(in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels, out_channels, 3, up=up, down=down, **_INIT)
        self.affine = Linear(emb_channels, out_channels * 2, **_INIT_ZERO)
        self.norm1 = GroupNorm(out_channels, eps=eps)
        self.conv1 = Conv2d(out_channels, out_channels, 3, **_INIT_ZERO)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if out_channels != in_channels else 0
            self.skip = Conv2d(in_channels, out_channels, kernel, up=up, down=down,
                               **_INIT)
        if self.num_heads:
            self.norm2 = GroupNorm(out_channels, eps=eps)
            self.qkv = Conv2d(out_channels, out_channels * 3, 1, **_INIT)
            self.proj = Conv2d(out_channels, out_channels, 1, **_INIT_ZERO)
        self.TimeAttention = (
            TemporalAttention(out_channels, time_length, heads, eps=eps)
            if temporal_attention else None)

    def forward(self, x, emb):
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)                                    # fold time into batch
        emb = emb.flatten(0, 1)
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        scale, shift = params.chunk(2, dim=1)
        x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            n, c, h, w = x.shape
            nh = self.num_heads
            qkv = self.qkv(self.norm2(x))
            # physicsnemo layout: (N*heads, C/heads, 3, HW)
            q, k, v = qkv.reshape(n, nh, c // nh, 3, h * w).unbind(3)
            q, k, v = (z.transpose(-1, -2) for z in (q, k, v))  # (N, heads, HW, c)
            a = _spatial_attention(q, k, v)
            a = a.transpose(-1, -2).reshape(n, c, h, w)
            x = self.proj(a) + x
            x = x * self.skip_scale

        x = x.unflatten(0, (b, t))                             # (B, T, C, H, W)
        if self.TimeAttention is not None:
            h, w = x.shape[-2:]
            y = x.flatten(-2, -1).permute(0, 2, 1, 3)          # (B, C, T, HW)
            y = self.TimeAttention(y)
            y = y.permute(0, 2, 1, 3).unflatten(-1, (h, w))
            x = x + y if self.temporal_residual else y
        return x


# ---------------------------------------------------------------------------
# attention placement
# ---------------------------------------------------------------------------
def attention_levels(img_resolution: int, n_levels: int, attn_resolutions,
                     temporal_attn_resolutions) -> dict[str, list[int]]:
    """Which 0-indexed levels carry spatial / temporal attention.

    Attention is configured by absolute resolution, but weights are stored by
    level, so two configs are weight-compatible exactly when this function
    returns the same lists for both. At twice the patch, double the resolution
    lists to keep the same levels.
    """
    res = [img_resolution >> lv for lv in range(n_levels)]
    return {
        "spatial": [lv for lv, r in enumerate(res) if r in set(attn_resolutions)],
        "temporal": [lv for lv, r in enumerate(res) if r in set(temporal_attn_resolutions)],
    }


# ---------------------------------------------------------------------------
# the UNet (GenLLC4320 src/models.py:448-705)
# ---------------------------------------------------------------------------
class VideoUNet(nn.Module):
    """``forward(x, noise_labels, labels=None)`` on ``(B, T, C, H, W)``.

    ``in_channels`` counts everything concatenated on the channel axis by the
    caller (noisy state, conditioning channels, calendar channels).
    ``noise_labels`` is ``(B, T)``; ``labels`` is ``(B, T, label_dim)`` or
    ``None`` when ``label_dim == 0``. Labels are used as given: nothing here
    rescales them, and ``map_label`` is initialised for a unit-norm vector, so
    callers hand in values of order one (:func:`video_label_dim` says how many).
    ``label_dropout`` zeroes the WHOLE label vector of a frame, tidal phases
    included.
    """

    def __init__(self, in_channels: int, out_channels: int, img_resolution: int = 256,
                 model_channels: int = 128, channel_mult=(1, 2, 2, 3, 4),
                 channel_mult_emb: int = 4, num_blocks: int = 3,
                 attn_resolutions=(32, 16), temporal_attn_resolutions=(64, 32, 16),
                 dropout: float = 0.10, label_dim: int = 0, label_dropout: float = 0.0,
                 n_frames: int = 5, channels_per_head: int = 64,
                 temporal_attn_residual: bool = True, grad_checkpoint: bool = False):
        super().__init__()
        self.img_resolution = img_resolution
        self.n_frames = int(n_frames)
        self.label_dim = int(label_dim)
        self.label_dropout = label_dropout
        self.grad_checkpoint = grad_checkpoint
        emb_channels = model_channels * channel_mult_emb
        use_temporal = self.n_frames > 1
        lv = attention_levels(img_resolution, len(channel_mult), attn_resolutions,
                              temporal_attn_resolutions)
        self.attn_levels = lv
        sp, tp = set(lv["spatial"]), set(lv["temporal"]) if use_temporal else set()
        blk = dict(emb_channels=emb_channels, channels_per_head=channels_per_head,
                   dropout=dropout, time_length=self.n_frames,
                   temporal_residual=temporal_attn_residual)

        self.map_noise = PositionalEmbedding(model_channels)
        self.map_layer0 = Linear(model_channels, emb_channels, **_INIT)
        self.map_layer1 = Linear(emb_channels, emb_channels, **_INIT)
        self.map_label = (
            Linear(label_dim, emb_channels, bias=False, init_mode="kaiming_normal",
                   init_weight=float(np.sqrt(label_dim)))
            if label_dim else None)

        # --- encoder ---
        self.enc = nn.ModuleDict()
        cout = in_channels
        skips = []
        for level, mult in enumerate(channel_mult):
            if level == 0:
                cout = model_channels * mult
                self.enc[f"L{level}_conv"] = Conv2d(in_channels, cout, 3, **_INIT)
            else:
                self.enc[f"L{level}_down"] = UNetBlockVideo(cout, cout, down=True, **blk)
            skips.append(cout)
            for idx in range(num_blocks):
                cin, cout = cout, model_channels * mult
                self.enc[f"L{level}_block{idx}"] = UNetBlockVideo(
                    cin, cout, attention=level in sp, temporal_attention=level in tp,
                    **blk)
                skips.append(cout)

        # --- decoder ---
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            if level == len(channel_mult) - 1:
                self.dec[f"L{level}_in0"] = UNetBlockVideo(
                    cout, cout, attention=True, temporal_attention=use_temporal, **blk)
                self.dec[f"L{level}_in1"] = UNetBlockVideo(cout, cout, **blk)
            else:
                self.dec[f"L{level}_up"] = UNetBlockVideo(cout, cout, up=True, **blk)
            for idx in range(num_blocks + 1):
                cin, cout = cout + skips.pop(), model_channels * mult
                self.dec[f"L{level}_block{idx}"] = UNetBlockVideo(
                    cin, cout, attention=level in sp, temporal_attention=level in tp,
                    **blk)
        self.out_norm = GroupNorm(cout)
        self.out_conv = Conv2d(cout, out_channels, 3, **_INIT_ZERO)

    def _run(self, block, x, emb):
        if self.grad_checkpoint and self.training and x.requires_grad:
            return _checkpoint(block, x, emb, use_reentrant=False)
        return block(x, emb)

    def forward(self, x, noise_labels, labels=None):
        b, t = x.shape[:2]
        emb = self.map_noise(noise_labels.reshape(b * t))
        emb = F.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            if labels is None:
                raise ValueError("this network was built with label_dim > 0; pass labels")
            lab = labels.reshape(b * t, self.label_dim).to(emb.dtype)
            if self.training and self.label_dropout:
                keep = torch.rand(b * t, 1, device=x.device) >= self.label_dropout
                lab = lab * keep.to(lab.dtype)
            emb = emb + self.map_label(lab)
        emb = F.silu(emb).unflatten(0, (b, t))

        skips = []
        for name, block in self.enc.items():
            if name.endswith("_conv"):
                x = block(x.flatten(0, 1)).unflatten(0, (b, t))
            else:
                x = self._run(block, x, emb)
            skips.append(x)
        for block in self.dec.values():
            if x.shape[2] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=2)
            x = self._run(block, x, emb)
        x = x.flatten(0, 1)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x.unflatten(0, (b, t))


def video_in_channels(cfg) -> int:
    """UNet input channels for a config: noisy state + per-frame inputs + calendar."""
    c = len(cfg.target) + len(cfg.cond_vars) + (1 if cfg.use_ocean_mask else 0)
    if cfg.use_calendar:
        c += 4 * cfg.calendar_harmonics
    return c


def video_label_dim(cfg) -> int:
    """Scalar labels per frame: one per ``label_vars`` entry, then cos and sin
    for each ``tidal_labels`` constituent (:mod:`diffusion.tides`)."""
    return len(cfg.label_vars) + 2 * len(cfg.tidal_labels)


def build_video_unet(cfg) -> VideoUNet:
    return VideoUNet(
        in_channels=video_in_channels(cfg),
        out_channels=len(cfg.target),
        img_resolution=cfg.patch,
        model_channels=cfg.model_channels,
        channel_mult=tuple(cfg.channel_mult),
        num_blocks=cfg.num_blocks,
        attn_resolutions=tuple(cfg.attn_resolutions),
        temporal_attn_resolutions=tuple(cfg.temporal_attn_resolutions),
        dropout=cfg.dropout,
        label_dim=video_label_dim(cfg),
        label_dropout=cfg.label_dropout,
        n_frames=cfg.n_frames,
        channels_per_head=cfg.channels_per_head,
        temporal_attn_residual=cfg.temporal_attn_residual,
        grad_checkpoint=cfg.grad_checkpoint,
    )


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _selftest() -> None:
    torch.manual_seed(0)
    n_par = lambda m: sum(p.numel() for p in m.parameters())

    # 1. GenLLC4320's configuration: 5 state + 24 calendar in, 5 out, 3 labels,
    #    5 frames. 159,700,801 is the count summed from GenLLC4320's layer
    #    definitions (src/models.py), so this pins the port's structure.
    big = VideoUNet(29, 5, img_resolution=256, label_dim=3, n_frames=5)
    assert n_par(big) == 159_700_801, n_par(big)
    assert big.attn_levels == {"spatial": [3, 4], "temporal": [2, 3, 4]}
    print(f"[ok] GenLLC4320 configuration: {n_par(big):,} parameters, "
          f"attention levels {big.attn_levels}")
    del big

    small = dict(model_channels=32, channel_mult=(1, 2, 2), num_blocks=1,
                 attn_resolutions=(16, 8), temporal_attn_resolutions=(16, 8),
                 channels_per_head=16, dropout=0.0)

    # 2. shapes, with and without labels, several frame / channel counts
    for t, cin, cout, lab in [(5, 31, 7, 0), (3, 9, 5, 2), (1, 4, 4, 0)]:
        net = VideoUNet(cin, cout, img_resolution=32, n_frames=t, label_dim=lab,
                        **small).eval()
        x = torch.randn(2, t, cin, 32, 32)
        y = net(x, torch.zeros(2, t), torch.randn(2, t, lab) if lab else None)
        assert y.shape == (2, t, cout, 32, 32), y.shape
    print("[ok] output shapes for T in {1, 3, 5}, with and without labels")

    # 3. zero-initialised output conv -> exactly zero at init
    assert float(y.detach().abs().max()) == 0.0
    print("[ok] network output is zero at initialisation")

    def randomise(net):
        with torch.no_grad():
            for p in net.parameters():
                p.copy_(torch.randn_like(p) * 0.05)

    # 4. without temporal attention, frame t depends only on frame t
    net = VideoUNet(6, 4, img_resolution=32, n_frames=4,
                    **{**small, "temporal_attn_resolutions": ()}).eval()
    # the bottleneck always has temporal attention when T > 1 -- remove it for
    # the independence check
    for blk in list(net.enc.values()) + list(net.dec.values()):
        if isinstance(blk, UNetBlockVideo):
            blk.TimeAttention = None
    randomise(net)
    x = torch.randn(1, 4, 6, 32, 32)
    y0 = net(x, torch.zeros(1, 4))
    x2 = x.clone()
    x2[:, 2] += 1.0
    y1 = net(x2, torch.zeros(1, 4))
    d = (y1 - y0).abs().amax(dim=(0, 2, 3, 4))
    assert d[2] > 1e-4 and float(d[[0, 1, 3]].max()) == 0.0, d
    print("[ok] no temporal attention -> frames are independent")

    # 5. with temporal attention the other frames respond; both residual modes run
    for residual in (True, False):
        net = VideoUNet(6, 4, img_resolution=32, n_frames=4,
                        temporal_attn_residual=residual, **small).eval()
        randomise(net)
        y0 = net(x, torch.zeros(1, 4))
        y1 = net(x2, torch.zeros(1, 4))
        d = (y1 - y0).abs().amax(dim=(0, 2, 3, 4))
        assert torch.isfinite(y0).all() and float(d[[0, 1, 3]].min()) > 0, d
    print("[ok] temporal attention couples frames (residual on and off)")

    # 6. wrong frame count is refused
    try:
        net(torch.randn(1, 3, 6, 32, 32), torch.zeros(1, 3))
    except ValueError:
        print("[ok] a checkpoint's n_frames is enforced")
    else:
        raise AssertionError("frame-count mismatch was not caught")

    # 7. patch-size independence of the state dict
    a = VideoUNet(6, 4, img_resolution=32, n_frames=3, **small)
    b = VideoUNet(6, 4, img_resolution=64, n_frames=3,
                  **{**small, "attn_resolutions": (32, 16),
                     "temporal_attn_resolutions": (32, 16)})
    b.load_state_dict(a.state_dict(), strict=True)
    c = VideoUNet(6, 4, img_resolution=64, n_frames=3, **small)   # lists NOT scaled
    try:
        c.load_state_dict(a.state_dict(), strict=True)
    except RuntimeError:
        print("[ok] state dict loads at 2x patch with doubled attention lists, "
              "and is refused with unscaled lists")
    else:
        raise AssertionError("unscaled attention lists loaded strictly")

    # 8. gradient checkpointing gives the same gradients
    net = VideoUNet(6, 4, img_resolution=32, n_frames=3, **small).train()
    randomise(net)
    xin = torch.randn(1, 3, 6, 32, 32)
    grads = []
    for flag in (False, True):
        net.grad_checkpoint = flag
        net.zero_grad()
        net(xin.clone().requires_grad_(True), torch.zeros(1, 3)).square().mean().backward()
        grads.append(torch.cat([p.grad.flatten() for p in net.parameters()
                                if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-7)
    print("[ok] grad_checkpoint reproduces the gradients")

    # 9. chunked spatial attention equals the single call
    global _ATTN_MAX_ELEMS
    q, k, v = (torch.randn(6, 2, 64, 8) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    keep, _ATTN_MAX_ELEMS = _ATTN_MAX_ELEMS, 2 * 64 * 64 * 2      # two frames per call
    try:
        assert torch.allclose(_spatial_attention(q, k, v), ref, atol=1e-6)
    finally:
        _ATTN_MAX_ELEMS = keep
    print("[ok] spatial attention chunked over frames matches the single call")

    # 10. calendar embedding
    cal = CalendarEmbedding(6)
    sod = torch.tensor([[0.0, 43200.0]])                 # 00 and 12 UTC
    lon = torch.tensor([[[0.0, -60.0, 180.0]]])          # (1, 1, 3)
    e = cal(sod, torch.zeros(1, 2), lon)
    assert e.shape == (1, 2, 24, 1, 3)
    # k=1 cos of local time: lon 0 at 00 UTC -> midnight (+1), at 12 UTC -> noon (-1)
    assert abs(float(e[0, 0, 0, 0, 0]) - 1) < 1e-5 and abs(float(e[0, 1, 0, 0, 0]) + 1) < 1e-5
    # lon 180 at 00 UTC is local noon; lon -60 at 12 UTC is 08:00 local
    assert abs(float(e[0, 0, 0, 0, 2]) + 1) < 1e-5
    assert abs(float(e[0, 1, 0, 0, 1]) - math.cos(2 * math.pi * 8 / 24)) < 1e-5
    # day-of-year block is constant over the patch
    assert float((e[:, :, 12:] - e[:, :, 12:, :, :1]).abs().max()) == 0.0
    print("[ok] calendar embedding: local solar time and day-of-year harmonics")
    print("networks_video selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
