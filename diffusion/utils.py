"""DDP setup, seeding, and a tiny CSV scalar logger."""
from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_info() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from env (defaults for single proc)."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local


def setup_ddp() -> tuple[int, int, int, torch.device]:
    rank, world, local = ddp_info()
    if world > 1:
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return rank, world, local, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


class CSVLogger:
    """Append scalar rows to a CSV, writing the header on first write."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fields: list[str] | None = None
        if os.path.exists(path):
            with open(path) as f:
                header = f.readline().strip()
            if header:
                self._fields = header.split(",")

    def log(self, row: dict) -> None:
        if self._fields is None:
            self._fields = list(row)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self._fields)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self._fields])
