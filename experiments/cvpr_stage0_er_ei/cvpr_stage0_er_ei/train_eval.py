"""Compose deepinv.Trainer for sequential tasks; PSNR eval uses HQ only here."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.models import MoDL

from .config import LEARNING_RATE, WEIGHT_DECAY, SmokeConfig
from .data_physics import TaskData, XYDataset


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


def make_trainer(
    model: torch.nn.Module,
    physics,
    train_loader: DataLoader,
    losses,
    device: torch.device,
    cfg: SmokeConfig,
    save_path: str | Path | None,
):
    return dinv.Trainer(
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
        disable_train_metrics=True,
        plot_images=False,
        plot_measurements=False,
        verbose=True,
        show_progress_bar=False,
        online_measurements=False,
        ckp_interval=10**9,
        early_stop=False,
    )


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
    physics = task.physics
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
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
) -> None:
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


# Re-export locked optimizer constants so callers can log them.
OPTIMIZER_DEFAULTS = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
