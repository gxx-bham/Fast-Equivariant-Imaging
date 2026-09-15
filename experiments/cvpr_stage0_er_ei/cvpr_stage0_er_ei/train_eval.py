"""Compose deepinv.Trainer for sequential tasks; PSNR eval uses HQ only here."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.models import MoDL

from .config import LEARNING_RATE, WEIGHT_DECAY, SmokeConfig
from .data_physics import TaskData, XYDataset, make_mri_physics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_modl(device: torch.device) -> MoDL:
    """Locked backbone: deepinv.models.MoDL with library defaults."""
    return MoDL().to(device)


def make_loader(dataset: XYDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def _trainer_field_names() -> set[str]:
    fields = getattr(dinv.Trainer, "__dataclass_fields__", None)
    if fields:
        return set(fields)
    return set(inspect.signature(dinv.Trainer.__init__).parameters)


def make_trainer(
    model: torch.nn.Module,
    physics,
    train_loader: DataLoader,
    losses,
    device: torch.device,
    cfg: SmokeConfig,
    save_path: str | Path | None,
):
    """Build dinv.Trainer with kwargs filtered to the installed deepinv API.

    0.3.5 uses disable_train_metrics; some later wheels use compute_train_metrics.
    """
    names = _trainer_field_names()
    kwargs: dict = dict(
        model=model,
        physics=physics,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        ),
        train_dataloader=train_loader,
        losses=losses,
        epochs=cfg.epochs,
        max_batch_steps=cfg.max_batch_steps,
        device=device,
        save_path=save_path,
        metrics=dinv.metric.PSNR(complex_abs=True),
        plot_images=False,
        plot_measurements=False,
        verbose=True,
        show_progress_bar=False,
        online_measurements=False,
        ckp_interval=10**9,
        early_stop=False,
    )
    if "disable_train_metrics" in names:
        kwargs["disable_train_metrics"] = True
    elif "compute_train_metrics" in names:
        kwargs["compute_train_metrics"] = False
    kwargs = {k: v for k, v in kwargs.items() if k in names}
    return dinv.Trainer(**kwargs)


@torch.no_grad()
def eval_psnr(
    model: torch.nn.Module,
    task: TaskData,
    device: torch.device,
    batch_size: int = 1,
) -> float:
    """PSNR vs HQ on this task's eval measurements. Used only at eval."""
    model.eval()
    loader = make_loader(task.eval, batch_size=batch_size, shuffle=False)
    metric = dinv.metric.PSNR(complex_abs=True)
    scores: list[float] = []
    for batch in loader:
        if len(batch) == 3:
            x, y, params = batch
            mask = params["mask"].to(device)
        else:
            x, y = batch
            mask = task.eval.mask[: x.shape[0]].to(device)
        x = x.to(device)
        y = y.to(device)
        physics = make_mri_physics(mask, device)
        x_net = model(y, physics)
        scores.extend(metric(x_net, x).detach().cpu().flatten().tolist())
    if not scores:
        raise RuntimeError(f"empty eval loader for task {task.name}")
    return float(sum(scores) / len(scores))


def train_task(
    model: torch.nn.Module,
    task: TaskData,
    losses,
    cfg: SmokeConfig,
    device: torch.device,
    save_path: str | Path | None,
    epochs: int | None = None,
) -> None:
    if epochs is not None:
        cfg = replace(cfg, epochs=int(epochs))
    loader = make_loader(task.train, batch_size=cfg.batch_size, shuffle=True)
    trainer = make_trainer(
        model=model,
        physics=task.physics,
        train_loader=loader,
        losses=losses,
        device=device,
        cfg=cfg,
        save_path=save_path,
    )
    trainer.train()


OPTIMIZER_DEFAULTS = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
