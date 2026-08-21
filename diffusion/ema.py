"""Exponential moving average of model parameters (for sampling/eval)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Parameter EMA with either a constant decay or GenDA's halflife rule.

    ``halflife_kimg = 0`` (default) keeps the constant ``decay`` every existing
    checkpoint was trained with. ``halflife_kimg > 0`` switches to GenDA's rule
    (``training_loop.py:348-354``): the halflife ramps up as
    ``min(halflife_kimg*1000, cur_nimg * rampup_ratio)`` and
    ``beta = 0.5 ** (global_batch / halflife_nimg)``, so the average is short at
    the start of training and saturates (~0.99991 at batch 64 / 500 kimg).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 halflife_kimg: float = 0.0, rampup_ratio: float = 0.05):
        self.decay = decay
        self.halflife_kimg = halflife_kimg
        self.rampup_ratio = rampup_ratio
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def beta(self, cur_nimg: int = 0, global_batch: int = 0) -> float:
        if self.halflife_kimg <= 0:
            return self.decay
        halflife_nimg = self.halflife_kimg * 1000.0
        if self.rampup_ratio is not None:
            halflife_nimg = min(halflife_nimg, cur_nimg * self.rampup_ratio)
        return 0.5 ** (global_batch / max(halflife_nimg, 1e-8))

    @torch.no_grad()
    def update(self, model: nn.Module, cur_nimg: int = 0,
               global_batch: int = 0) -> None:
        d = self.beta(cur_nimg, global_batch)
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
