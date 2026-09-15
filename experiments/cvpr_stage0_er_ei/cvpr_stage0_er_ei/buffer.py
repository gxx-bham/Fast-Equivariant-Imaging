"""Replay buffer of measurements and operators only: stores (y, A), not HQ x."""

from __future__ import annotations

import torch


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
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self._y)

    def add(self, y: torch.Tensor, mask: torch.Tensor) -> None:
        """Add one sample. y and mask are stored on CPU without graph."""
        y_cpu = y.detach().to("cpu").contiguous()
        mask_cpu = mask.detach().to("cpu").contiguous()
        if y_cpu.ndim == 3:
            y_cpu = y_cpu.unsqueeze(0)
        if mask_cpu.ndim == 3:
            mask_cpu = mask_cpu.unsqueeze(0)
        if y_cpu.shape[0] != 1:
            raise ValueError("add() expects a single sample; got a batch")
        self.n_seen += 1
        if len(self._y) < self.capacity:
            self._y.append(y_cpu)
            self._mask.append(mask_cpu)
            return
        idx = torch.randint(
            0, self.n_seen, (1,), generator=self.generator
        ).item()
        if idx < self.capacity:
            self._y[idx] = y_cpu
            self._mask[idx] = mask_cpu

    def fill_from_xy(self, y: torch.Tensor, mask: torch.Tensor) -> None:
        """Add each sample in a stacked y tensor, broadcasting a shared task mask."""
        if y.ndim == 3:
            y = y.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        for i in range(y.shape[0]):
            m = mask[i : i + 1] if mask.shape[0] == y.shape[0] else mask[:1]
            self.add(y[i : i + 1], m)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self._y) == 0:
            raise RuntimeError("buffer is empty")
        n = len(self._y)
        idx = torch.randint(
            0, n, (int(batch_size),), generator=self.generator
        )
        ys = torch.cat([self._y[int(i)].to(device) for i in idx], dim=0)
        masks = torch.cat([self._mask[int(i)].to(device) for i in idx], dim=0)
        return ys, masks
