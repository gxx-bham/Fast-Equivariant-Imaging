"""Compose deepinv.Trainer for sequential tasks; PSNR eval uses HQ only here."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.models import ArtifactRemoval, MoDL, UNet

from .config import (
    ARTIFACT_REMOVAL_CTOR_KWARGS,
    BACKBONE_MODL,
    BACKBONE_UNET,
    LEARNING_RATE,
    UNET_CTOR_KWARGS,
    WEIGHT_DECAY,
    SmokeConfig,
)
from .data_physics import TaskData, XYDataset, make_mri_physics


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_modl(device: torch.device) -> MoDL:
    """Locked backbone: deepinv.models.MoDL with library defaults."""
    return MoDL().to(device)


def make_unet_reconstructor(device: torch.device) -> ArtifactRemoval:
    """Light backbone: UNet(scales=3) wrapped as a (y, physics) reconstructor.

    UNet is a Denoiser (forward(x, sigma)); ArtifactRemoval(mode=adjoint) maps
    measurements through A^T then the UNet. in_channels=2 matches stream-E
    MRI real/imag and CT stacked via as_modl_channels. Frozen: scales=3,
    EILoss Rotate n_trans=4 is not swept here.
    """
    denoiser = UNet(**dict(UNET_CTOR_KWARGS))
    return ArtifactRemoval(denoiser, **dict(ARTIFACT_REMOVAL_CTOR_KWARGS)).to(device)


def make_backbone(cfg: SmokeConfig, device: torch.device) -> torch.nn.Module:
    kind = str(getattr(cfg, "backbone", BACKBONE_MODL)).lower()
    if kind in (BACKBONE_MODL, "deepinv.models.modl", "deepinv.models.MoDL".lower()):
        return make_modl(device)
    if kind in (BACKBONE_UNET, "light", "deepinv.models.unet"):
        return make_unet_reconstructor(device)
    raise ValueError(
        f"unknown backbone {cfg.backbone!r}; expected {BACKBONE_MODL!r} or {BACKBONE_UNET!r}"
    )


def backbone_fingerprint(
    cfg: SmokeConfig, model: torch.nn.Module | None = None
) -> dict:
    """Exact class names + ctor kwargs for reviewer JSON (not a stale string)."""
    kind = str(getattr(cfg, "backbone", BACKBONE_MODL)).lower()
    if kind in (BACKBONE_UNET, "light", "deepinv.models.unet"):
        denoiser = None
        if model is not None and hasattr(model, "backbone_net"):
            denoiser = model.backbone_net
        return {
            "kind": BACKBONE_UNET,
            "class": "deepinv.models.UNet",
            "ctor_kwargs": dict(UNET_CTOR_KWARGS),
            "wrapper_class": "deepinv.models.ArtifactRemoval",
            "wrapper_ctor_kwargs": dict(ARTIFACT_REMOVAL_CTOR_KWARGS),
            "runtime_class": type(model).__name__ if model is not None else "ArtifactRemoval",
            "runtime_denoiser_class": (
                type(denoiser).__name__ if denoiser is not None else "UNet"
            ),
            "n_parameters": (
                int(sum(p.numel() for p in model.parameters()))
                if model is not None
                else None
            ),
        }
    return {
        "kind": BACKBONE_MODL,
        "class": "deepinv.models.MoDL",
        "ctor_kwargs": {},
        "wrapper_class": None,
        "wrapper_ctor_kwargs": None,
        "runtime_class": type(model).__name__ if model is not None else "MoDL",
        "runtime_denoiser_class": None,
        "n_parameters": (
            int(sum(p.numel() for p in model.parameters())) if model is not None else None
        ),
    }


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
    scores: list[float] = []
    for batch in loader:
        if len(batch) == 3:
            x, y, params = batch
            mask = params.get("mask") if isinstance(params, dict) else None
        else:
            x, y = batch
            mask = None
        x = x.to(device)
        y = y.to(device)
        if mask is not None:
            physics = make_mri_physics(mask.to(device), device)
        else:
            physics = task.physics.to(device) if hasattr(task.physics, "to") else task.physics
        x_net = model(y, physics)
        if getattr(task, "modality", "mri") == "ct":
            metric = dinv.metric.PSNR()
            scores.extend(
                metric(x_net[:, :1], x[:, :1]).detach().cpu().flatten().tolist()
            )
        else:
            metric = dinv.metric.PSNR(complex_abs=True)
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
