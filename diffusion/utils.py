"""DDP setup, seeding, and a tiny CSV scalar logger."""
from __future__ import annotations

import csv
import datetime
import os
import random
import time

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
    """Bring up NCCL with a collective timeout that tolerates a stalled filesystem.

    NCCL's default collective timeout is 10 min. Rank 0 alone writes checkpoints,
    and on 2026-08-22 a ~158 MB torch.save to the NFS-backed run dir blocked past
    that window: rank 1 sat in a BROADCAST, its watchdog fired at 600 s, and it
    took the whole job down at step ~248,100 (job 2354) even though rank 0 was
    merely blocked, not broken. A longer timeout lets a transient storage stall
    resolve instead of costing the run. DDP_TIMEOUT_MIN overrides.

    set_device also moves ahead of init_process_group so the PG is created with an
    explicit device_id -- otherwise NCCL guesses "GPU 0" and warns that a wrong
    guess can hang, which is a bad failure mode on a node where GPU 0 is held by
    someone else.
    """
    rank, world, local = ddp_info()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if world > 1:
        mins = int(os.environ.get("DDP_TIMEOUT_MIN", "60"))
        kw = {"timeout": datetime.timedelta(minutes=mins)}
        if torch.cuda.is_available():
            kw["device_id"] = device
        try:
            dist.init_process_group(backend="nccl", **kw)
        except TypeError:      # older torch without device_id
            kw.pop("device_id", None)
            dist.init_process_group(backend="nccl", **kw)
        if rank == 0:
            print(f"[ddp] world={world} nccl_timeout={mins}min", flush=True)
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

    # A metrics row is not worth a multi-hour run. /unity/f1 is NFS and returned
    # a transient EIO mid-append on 2026-08-22; it propagated out of log() and
    # killed rank 0 of job 2332 at step ~187,300, discarding 8h40m of GPU time.
    # The checkpoints written moments earlier were all intact and finite -- only
    # the CSV write failed. So: retry briefly, then drop the row with a warning
    # rather than raising. Every step line is also on stdout, so a dropped row
    # costs a point in log.csv and nothing else.
    _RETRIES = 4
    _BACKOFF = 0.5

    def _write(self, mode: str, rows: list[list]) -> bool:
        """Append/write rows, retrying transient OS errors. True if written."""
        for attempt in range(self._RETRIES):
            try:
                with open(self.path, mode, newline="") as f:
                    w = csv.writer(f)
                    for r in rows:
                        w.writerow(r)
                return True
            except OSError as e:
                if attempt == self._RETRIES - 1:
                    print(f"[csvlog] WARNING: dropping row after "
                          f"{self._RETRIES} attempts: {type(e).__name__}: {e}",
                          flush=True)
                    return False
                time.sleep(self._BACKOFF * (2 ** attempt))
        return False

    def log(self, row: dict) -> None:
        if self._fields is None:
            fields = list(row)
            if not self._write("w", [fields]):
                return          # header unwritten; retry the header next call
            self._fields = fields
        self._write("a", [[row.get(k, "") for k in self._fields]])
