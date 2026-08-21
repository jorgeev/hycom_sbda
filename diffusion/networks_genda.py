"""Faithful port of GenDA's SongUNet (DDPM++ flavour) for parity experiments.

Source: ``smartin98-GenDA-58cfc29/modulus/modulus/models/diffusion/{song_unet,
layers}.py`` (vendored Modulus 0.7.0a0), with the hyperparameters GenDA actually
runs -- ``num_heads=1``, ``skip_scale=sqrt(0.5)``, ``adaptive_scale=False``,
``resample_filter=[1,1]``, GroupNorm ``eps=1e-6``, Xavier-uniform init with
1e-5-scaled output convs and sqrt(0.2)-scaled qkv, and a positional noise
embedding with ``endpoint=True`` and swapped sin/cos halves.

This lives beside :mod:`diffusion.networks` rather than replacing it: every
checkpoint under ``runs/`` was trained with the ``SongUNet`` there, and the two
architectures differ in ways a flag cannot express cleanly (GenDA fuses
resampling *into* the residual blocks instead of using separate Downsample/
Upsample modules). Select with ``arch: genda`` in the config.

Structural differences from ``networks.py`` worth naming, since they are the
point of the port:
  * residual and attention adds are scaled by sqrt(1/2);
  * up/down sampling is a normalised 2x2 box blur-resample applied *before* the
    3x3 conv, on both the main path and the skip;
  * attention is gated on absolute resolution, and in the decoder only the last
    block of a matching level gets it (``networks.py`` attends on all of them);
  * output convs are Xavier-uniform x 1e-5, not zero.

Two deliberate departures from the reference:
  * attention uses ``F.scaled_dot_product_attention`` rather than GenDA's
    hand-written ``AttentionOp``. Same softmax attention mathematically, but
    full-frame inference at 464x528 puts 3828 tokens at the attended level and
    the explicit n-by-n matrix is needless memory. Loading GenDA's released
    weights is not a goal here.
  * gradient checkpointing is omitted -- GenDA's ``checkpoint_level=0`` makes its
    threshold unreachable at 128 px anyway, so it never fires there either.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Init (modulus/models/diffusion/utils.py:58-59)
# ---------------------------------------------------------------------------
def _xavier_uniform(shape, fan_in: int, fan_out: int) -> torch.Tensor:
    return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)


class Linear(nn.Module):
    """Dense layer with Xavier-uniform weights scaled by ``init_weight``."""

    def __init__(self, in_features: int, out_features: int, *,
                 init_weight: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        w = _xavier_uniform((out_features, in_features), in_features, out_features)
        self.weight = nn.Parameter(w * init_weight)
        b = _xavier_uniform((out_features,), in_features, out_features)
        self.bias = nn.Parameter(b * init_bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Conv2d(nn.Module):
    """Conv with optional blur-resample, matching layers.py:130-250.

    ``kernel=0`` means "resample only, no conv" (the identity skip when channels
    match but the block up/downsamples). The depthwise ``resample_filter`` is
    applied *before* the 3x3 conv, which is what makes GenDA's up/down path
    differ from a plain strided conv.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int, *,
                 up: bool = False, down: bool = False,
                 resample_filter=(1, 1), init_weight: float = 1.0,
                 init_bias: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.up, self.down = up, down
        fan_in = in_channels * kernel * kernel
        fan_out = out_channels * kernel * kernel
        if kernel:
            w = _xavier_uniform((out_channels, in_channels, kernel, kernel),
                                fan_in, fan_out)
            self.weight = nn.Parameter(w * init_weight)
            b = _xavier_uniform((out_channels,), fan_in, fan_out)
            self.bias = nn.Parameter(b * init_bias)
        else:
            self.weight = None
            self.bias = None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if (up or down) else None)

    def forward(self, x):
        w = self.weight
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
            x = F.conv2d(x, w.to(x.dtype), padding=w_pad)
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype).reshape(1, -1, 1, 1))
        return x


def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with GenDA's group rule and eps.

    Modulus caps groups at ``min(32, ch // min_channels_per_group)`` with
    ``min_channels_per_group=4``, and GenDA passes ``eps=1e-6`` everywhere. Note
    this differs from ``networks.py``'s ``min(32, ch)``: at ``model_channels=64``
    the first level gets 16 groups here, not 32.
    """
    return nn.GroupNorm(num_groups=min(32, ch // 4), num_channels=ch, eps=1e-6)


# ---------------------------------------------------------------------------
# Noise embedding (layers.py:686-694 + song_unet.py:328-333)
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """DDPM++ positional embedding: ``endpoint=True`` and cos-then-sin order.

    The ``(half - 1)`` denominator and the concat order are both load-bearing
    for parity; ``SongUNet.forward`` then flips the two halves to sin-then-cos.
    """

    def __init__(self, num_channels: int, max_positions: int = 10000,
                 endpoint: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        half = self.num_channels // 2
        freqs = torch.arange(half, dtype=torch.float32, device=x.device)
        freqs = freqs / (half - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.float().ger(freqs)
        return torch.cat([x.cos(), x.sin()], dim=1)


# ---------------------------------------------------------------------------
# UNetBlock (layers.py:343-503)
# ---------------------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, *,
                 up: bool = False, down: bool = False, attention: bool = False,
                 num_heads: int = 1, dropout: float = 0.0,
                 skip_scale: float = 1.0, resample_filter=(1, 1),
                 resample_proj: bool = True,
                 init_weight_out: float = 1e-5, init_weight_attn: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads if attention else 0
        self.dropout = dropout
        self.skip_scale = skip_scale

        self.norm0 = _gn(in_channels)
        self.conv0 = Conv2d(in_channels, out_channels, 3, up=up, down=down,
                            resample_filter=resample_filter)
        # adaptive_scale=False in GenDA -> a single shift, added before norm1.
        self.affine = Linear(emb_channels, out_channels)
        self.norm1 = _gn(out_channels)
        self.conv1 = Conv2d(out_channels, out_channels, 3,
                            init_weight=init_weight_out)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels, out_channels, kernel, up=up, down=down,
                               resample_filter=resample_filter)

        if self.num_heads:
            self.norm2 = _gn(out_channels)
            self.qkv = Conv2d(out_channels, out_channels * 3, 1,
                              init_weight=init_weight_attn)
            self.proj = Conv2d(out_channels, out_channels, 1,
                               init_weight=init_weight_out)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        x = F.silu(self.norm1(x + params))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            b, c, h, w = x.shape
            nh = self.num_heads
            qkv = self.qkv(self.norm2(x))
            q, k, v = qkv.reshape(b, nh, c // nh, 3, h * w).unbind(3)
            # (B, heads, HW, c/heads) for scaled_dot_product_attention
            q, k, v = (t.transpose(-1, -2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(-1, -2).reshape(b, c, h, w)
            x = self.proj(a) + x
            x = x * self.skip_scale
        return x


# ---------------------------------------------------------------------------
# SongUNet (song_unet.py:179-390)
# ---------------------------------------------------------------------------
class GenDASongUNet(nn.Module):
    """GenDA's SongUNet. ``forward(x, noise_labels)`` as in ``networks.SongUNet``.

    ``img_resolution`` is the *training* patch size and only affects which levels
    get attention (``res = img_resolution >> level``). Inference on a larger
    frame reuses those same blocks, exactly as GenDA's own inference does.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_resolution: int = 128,
        model_channels: int = 64,
        channel_mult=(1, 2, 2, 2),
        channel_mult_emb: int = 4,
        channel_mult_noise: int = 1,
        num_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.13,
        num_heads: int = 1,
        resample_filter=(1, 1),
    ):
        super().__init__()
        self.img_resolution = img_resolution
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        attn_resolutions = set(attn_resolutions)
        skip_scale = float(np.sqrt(0.5))
        blk = dict(emb_channels=emb_channels, num_heads=num_heads, dropout=dropout,
                   skip_scale=skip_scale, resample_filter=resample_filter,
                   init_weight_out=1e-5, init_weight_attn=float(np.sqrt(0.2)))

        self.map_noise = PositionalEmbedding(noise_channels, endpoint=True)
        self.map_layer0 = Linear(noise_channels, emb_channels)
        self.map_layer1 = Linear(emb_channels, emb_channels)

        # --- encoder ---
        self.enc = nn.ModuleDict()
        cout = in_channels
        skips = []
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin, cout = cout, model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(cin, cout, 3)
                skips.append(cout)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(cout, cout, down=True, **blk)
                skips.append(cout)
            for idx in range(num_blocks):
                cin, cout = cout, model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    cin, cout, attention=(res in attn_resolutions), **blk)
                skips.append(cout)

        # --- decoder ---
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                # bottleneck: attention here is unconditional in GenDA
                self.dec[f"{res}x{res}_in0"] = UNetBlock(cout, cout, attention=True, **blk)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(cout, cout, **blk)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(cout, cout, up=True, **blk)
            for idx in range(num_blocks + 1):
                cin, cout = cout + skips.pop(), model_channels * mult
                # only the LAST decoder block of a matching level gets attention
                attn = (idx == num_blocks) and (res in attn_resolutions)
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    cin, cout, attention=attn, **blk)
        self.out_norm = _gn(cout)
        self.out_conv = Conv2d(cout, out_channels, 3, init_weight=1e-5)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        # swap the cos/sin halves -> DDPM++ sin-then-cos ordering
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        skips = []
        for name, block in self.enc.items():
            x = block(x) if name.endswith("_conv") else block(x, emb)
            skips.append(x)

        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        return self.out_conv(F.silu(self.out_norm(x)))
