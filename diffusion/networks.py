"""Self-contained UNet backbone for the EDM diffusion model (no external deps).

A compact SongUNet-style encoder/decoder: sinusoidal noise embedding, GroupNorm+
SiLU residual blocks with a noise-conditioned bias, self-attention at the coarsest
level(s), and skip connections. Conditioning is done by the caller via channel
concatenation, so this module only needs ``in_channels``/``out_channels``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch, eps=1e-6)


class NoiseEmbedding(nn.Module):
    """Sinusoidal embedding of a continuous noise label -> MLP."""

    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device) / (half - 1)
        )
        args = x[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        nh = self.num_heads
        # (B, heads, HW, c/heads)
        q, k, v = (t.reshape(b, nh, c // nh, h * w).transpose(-1, -2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class SongUNet(nn.Module):
    """UNet with noise-conditioned residual blocks and coarse-level attention.

    ``forward(x, noise_labels)`` where ``x`` already includes any conditioning
    channels (concatenated by the caller) and ``noise_labels`` is a (B,) tensor
    (EDM ``c_noise``; pass zeros for the deterministic regression baseline).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 128,
        channel_mult=(1, 2, 2, 2),
        num_blocks: int = 2,
        attn_levels=(3,),
        dropout: float = 0.0,
    ):
        super().__init__()
        emb_dim = model_channels * 4
        self.emb = NoiseEmbedding(model_channels, emb_dim)
        attn_levels = set(attn_levels)

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # --- encoder ---
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_chs = [model_channels]
        ch = model_channels
        n_levels = len(channel_mult)
        for lvl, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(ResBlock(ch, out_ch, emb_dim, dropout))
                ch = out_ch
                attns.append(AttnBlock(ch) if lvl in attn_levels else nn.Identity())
                skip_chs.append(ch)
            self.down_blocks.append(blocks)
            self.down_attn.append(attns)
            if lvl != n_levels - 1:
                self.downsamples.append(Downsample(ch))
                skip_chs.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        # --- middle ---
        self.mid_block1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttnBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, dropout)

        # --- decoder ---
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for lvl in reversed(range(n_levels)):
            out_ch = model_channels * channel_mult[lvl]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_blocks + 1):
                blocks.append(ResBlock(ch + skip_chs.pop(), out_ch, emb_dim, dropout))
                ch = out_ch
                attns.append(AttnBlock(ch) if lvl in attn_levels else nn.Identity())
            self.up_blocks.append(blocks)
            self.up_attn.append(attns)
            self.upsamples.append(Upsample(ch) if lvl != 0 else nn.Identity())

        self.norm_out = _gn(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.emb(noise_labels)
        h = self.conv_in(x)
        skips = [h]
        n_levels = len(self.down_blocks)
        for lvl in range(n_levels):
            for block, attn in zip(self.down_blocks[lvl], self.down_attn[lvl]):
                h = attn(block(h, emb))
                skips.append(h)
            if lvl != n_levels - 1:
                h = self.downsamples[lvl](h)
                skips.append(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, emb)), emb)

        for i, lvl in enumerate(reversed(range(n_levels))):
            for block, attn in zip(self.up_blocks[i], self.up_attn[i]):
                h = torch.cat([h, skips.pop()], dim=1)
                h = attn(block(h, emb))
            h = self.upsamples[i](h)

        return self.conv_out(F.silu(self.norm_out(h)))
