"""deepinv MC / EI losses plus an ER replay term over stored (y, A)."""

from __future__ import annotations

import torch

import deepinv as dinv
from deepinv.loss import Loss

from .buffer import MeasurementBuffer
from .data_physics import make_mri_physics


def current_task_losses() -> list[Loss]:
    """MRI EI demo default on the current task: MC + EI(Rotate n_trans=4)."""
    return [
        dinv.loss.MCLoss(),
        dinv.loss.EILoss(dinv.transform.Rotate(n_trans=4)),
    ]


class BufferReplayLoss(Loss):
    """ER loss on a uniform sample from the (y, A) buffer.

    ER+MC: MC only on buffer. ER+EI: MC+EI on buffer.
    Replay batch size matches the current batch (current:replay mix 1:1).
    """

    def __init__(self, buffer: MeasurementBuffer, use_ei: bool):
        super().__init__()
        self.buffer = buffer
        self.use_ei = bool(use_ei)
        self.mc = dinv.loss.MCLoss()
        self.ei = (
            dinv.loss.EILoss(dinv.transform.Rotate(n_trans=4)) if use_ei else None
        )
        self._name = "ER_EI" if use_ei else "ER_MC"

    def forward(self, x_net, physics, model, y, **kwargs):
        if len(self.buffer) == 0:
            return y.new_zeros(())
        y_b, mask_b = self.buffer.sample(batch_size=int(y.shape[0]), device=y.device)
        physics_b = make_mri_physics(mask_b, device=y.device)
        x_b = model(y_b, physics_b)
        loss = self.mc(y=y_b, x_net=x_b, physics=physics_b)
        if self.ei is not None:
            loss = loss + self.ei(x_net=x_b, physics=physics_b, model=model)
        return loss


def losses_for_arm(arm: str, buffer: MeasurementBuffer | None) -> list[Loss]:
    losses: list[Loss] = list(current_task_losses())
    if arm == "finetune":
        return losses
    if buffer is None:
        raise ValueError(f"arm {arm} requires a buffer")
    if arm == "er_mc":
        losses.append(BufferReplayLoss(buffer, use_ei=False))
        return losses
    if arm == "er_ei":
        losses.append(BufferReplayLoss(buffer, use_ei=True))
        return losses
    raise ValueError(f"unknown arm {arm!r}")
