"""Replay buffer of measurements and operators only: stores (y, A), not HQ x."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


def tensor_id(t: torch.Tensor) -> str:
    """Stable short id for a CPU tensor (replay / mask fingerprint)."""
    arr = t.detach().to("cpu").contiguous()
    digest = hashlib.sha1(arr.numpy().tobytes()).hexdigest()
    return digest[:12]


class MeasurementBuffer:
    """Capacity-N_buf reservoir of (y, A) pairs. A is the MRI mask of physics.MRI.

    Uniform sampling with replacement when drawing a replay batch.
    """

    def __init__(self, capacity: int, generator: torch.Generator | None = None):
        if capacity < 1:
            raise ValueError("N_buf / capacity must be >= 1")
        self.capacity = int(capacity)
        self.generator = generator
        self._y: list[torch.Tensor] = []
        self._mask: list[torch.Tensor] = []
        self._meta: list[dict[str, Any]] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self._y)

    def add(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        *,
        task: str | None = None,
        accel: int | None = None,
        sample_index: int | None = None,
    ) -> None:
        """Add one sample. y and mask are stored on CPU without graph."""
        y_cpu = y.detach().to("cpu").contiguous()
        mask_cpu = mask.detach().to("cpu").contiguous()
        if y_cpu.ndim == 3:
            y_cpu = y_cpu.unsqueeze(0)
        if mask_cpu.ndim == 3:
            mask_cpu = mask_cpu.unsqueeze(0)
        if y_cpu.shape[0] != 1:
            raise ValueError("add() expects a single sample; got a batch")
        rec = {
            "task": task,
            "accel": accel,
            "sample_index": sample_index,
            "y_id": tensor_id(y_cpu),
            "mask_id": tensor_id(mask_cpu),
            "mask_mean": float(mask_cpu.float().mean().item()),
        }
        self.n_seen += 1
        if len(self._y) < self.capacity:
            self._y.append(y_cpu)
            self._mask.append(mask_cpu)
            self._meta.append(rec)
            return
        idx = torch.randint(0, self.n_seen, (1,), generator=self.generator).item()
        if idx < self.capacity:
            self._y[idx] = y_cpu
            self._mask[idx] = mask_cpu
            self._meta[idx] = rec

    def fill_from_dataset(
        self,
        dataset,
        *,
        task: str,
        accel: int,
        max_items: int | None = None,
    ) -> None:
        n = len(dataset) if max_items is None else min(len(dataset), int(max_items))
        for i in range(n):
            item = dataset[i]
            if len(item) == 3:
                _x, y, params = item
                mask = params["mask"]
            else:
                raise ValueError("train dataset must return (x, y, {mask}) for (y, A) buffer")
            self.add(y, mask, task=task, accel=accel, sample_index=i)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self._y) == 0:
            raise RuntimeError("buffer is empty")
        n = len(self._y)
        idx = torch.randint(0, n, (int(batch_size),), generator=self.generator)
        ys = torch.cat([self._y[int(i)].to(device) for i in idx], dim=0)
        masks = torch.cat([self._mask[int(i)].to(device) for i in idx], dim=0)
        return ys, masks

    def summary(self) -> dict[str, Any]:
        slots = []
        pair_ids = []
        mask_ids = []
        y_ids = []
        accels = []
        tasks = []
        for i, meta in enumerate(self._meta):
            slot = {"slot": i, **meta}
            slots.append(slot)
            pair_ids.append((meta["y_id"], meta["mask_id"]))
            mask_ids.append(meta["mask_id"])
            y_ids.append(meta["y_id"])
            accels.append(meta.get("accel"))
            tasks.append(meta.get("task"))
        n_distinct_pairs = len(set(pair_ids))
        n_distinct_masks = len(set(mask_ids))
        n_distinct_y = len(set(y_ids))
        return {
            "N_buf": self.capacity,
            "buffer_size": len(self._y),
            "n_seen": self.n_seen,
            "n_distinct_ya": n_distinct_pairs,
            "n_distinct_y": n_distinct_y,
            "n_distinct_A": n_distinct_masks,
            "holds_t1_A": all(t == "T1" for t in tasks) and len(tasks) > 0,
            "accels": accels,
            "tasks": tasks,
            "full_distinct": n_distinct_pairs == self.capacity,
            "slots": slots,
        }
