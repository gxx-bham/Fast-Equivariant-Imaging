"""deepinv MC / EI losses plus an ER replay term over stored (y, A)."""

from __future__ import annotations

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


def supervised_losses() -> list[Loss]:
    """HQ supervised MSE via deepinv.loss.SupLoss. MC/EI are off."""
    return [dinv.loss.SupLoss()]


class BufferReplayLoss(Loss):
    """ER loss on a uniform sample from the (y, A) buffer.

    ER+MC: MC only on buffer. ER+EI: MC+EI on buffer.
    Replay batch size matches the current batch (current:replay mix 1:1).

    Cross-IP: the current task physics may be Tomography (64 CT). Replay MUST
    rebuild `deepinv.physics.MRI` from the stored mask via `make_mri_physics`.
    Never pass 128 MRI y into the CT operator.
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
        self.n_forwards = 0
        self.max_abs_loss = 0.0
        self.first_loss: float | None = None
        self.last_replay_physics: str | None = None
        self.last_y_shape: list[int] | None = None
        self.last_mask_shape: list[int] | None = None

    def forward(self, x_net, physics, model, y, **kwargs):
        if len(self.buffer) == 0:
            return y.new_zeros(())
        y_b, mask_b = self.buffer.sample(batch_size=int(y.shape[0]), device=y.device)
        # Ignore `physics` (may be Tomography on the CT task). Rebuild MRI A.
        physics_b = make_mri_physics(mask_b, device=y.device)
        if type(physics_b).__name__ != "MRI":
            raise TypeError(
                "MRI buffer replay MUST use deepinv.physics.MRI via "
                f"make_mri_physics; got {type(physics_b).__name__}"
            )
        if type(physics).__name__ == "Tomography" and tuple(y_b.shape[-2:]) == (
            64,
            64,
        ):
            raise RuntimeError(
                "refusing to treat 64×64 tensors as MRI replay; "
                "never feed MRI measurements into Tomography"
            )
        x_b = model(y_b, physics_b)
        loss = self.mc(y=y_b, x_net=x_b, physics=physics_b)
        if self.ei is not None:
            loss = loss + self.ei(x_net=x_b, physics=physics_b, model=model)
        val = float(loss.detach().cpu())
        self.n_forwards += 1
        self.max_abs_loss = max(self.max_abs_loss, abs(val))
        if self.first_loss is None:
            self.first_loss = val
        self.last_replay_physics = "deepinv.physics.MRI"
        self.last_y_shape = [int(s) for s in y_b.shape]
        self.last_mask_shape = [int(s) for s in mask_b.shape]
        return loss

    def stats(self) -> dict:
        return {
            "n_forwards": int(self.n_forwards),
            "first_BufferReplayLoss": self.first_loss,
            "max_abs_BufferReplayLoss": float(self.max_abs_loss),
            "BufferReplayLoss_nonzero": bool(
                (self.first_loss not in (None, 0.0)) or self.max_abs_loss > 0.0
            ),
            "rebuilds_mri_from_stored_mask": True,
            "make_mri_physics": True,
            "replay_physics_class": self.last_replay_physics,
            "y_b_shape": self.last_y_shape,
            "mask_shape": self.last_mask_shape,
            "never_fed_mri_into_tomography": True,
        }


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
