"""Sequential T1 (4x) -> T2 (8x) protocol for Fine-tune | ER+MC | ER+EI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .buffer import MeasurementBuffer
from .config import ARM_LABELS, SmokeConfig
from .data_physics import build_tasks, load_knee_slices
from .logging_utils import summary_row
from .losses import current_task_losses, losses_for_arm
from .train_eval import eval_psnr, make_modl, set_seed, train_task


def _device(cfg: SmokeConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def run_arm(arm: str, cfg: SmokeConfig, out_dir: Path | None = None) -> dict[str, Any]:
    if arm not in ARM_LABELS:
        raise ValueError(f"unknown arm {arm!r}")
    device = _device(cfg)
    set_seed(cfg.seed)
    x = load_knee_slices(
        img_size=cfg.img_size, download=cfg.download
    )
    tasks = build_tasks(x, seed=cfg.seed, device=device, img_size=cfg.img_size)
    model = make_modl(device)
    save_path = None if out_dir is None else str(out_dir / "ckpts" / arm)

    # Task T1: all arms use the MRI EI demo current-task loss (MC+EI). Buffer empty.
    train_task(
        model,
        tasks["T1"],
        losses=current_task_losses(),
        cfg=cfg,
        device=device,
        save_path=save_path,
    )
    psnr_t1_after_t1 = eval_psnr(model, tasks["T1"], device, batch_size=cfg.batch_size)
    psnr_t2_after_t1 = eval_psnr(model, tasks["T2"], device, batch_size=cfg.batch_size)

    buffer = MeasurementBuffer(
        capacity=cfg.n_buf,
        generator=torch.Generator().manual_seed(cfg.seed + 17),
    )
    buffer.fill_from_xy(tasks["T1"].train.y, tasks["T1"].mask)

    t2_losses = losses_for_arm(arm, buffer=None if arm == "finetune" else buffer)
    train_task(
        model,
        tasks["T2"],
        losses=t2_losses,
        cfg=cfg,
        device=device,
        save_path=save_path,
    )
    psnr_t1_after_t2 = eval_psnr(model, tasks["T1"], device, batch_size=cfg.batch_size)
    psnr_t2_after_t2 = eval_psnr(model, tasks["T2"], device, batch_size=cfg.batch_size)

    row = summary_row(
        arm=arm,
        n_buf=cfg.n_buf,
        seed=cfg.seed,
        psnr_t1_after_t2=psnr_t1_after_t2,
        psnr_t2_after_t2=psnr_t2_after_t2,
        psnr_t1_after_t1=psnr_t1_after_t1,
    )
    return {
        "row": row,
        "after_T1": {
            "PSNR_T1": psnr_t1_after_t1,
            "PSNR_T2": psnr_t2_after_t1,
        },
        "after_T2": {
            "PSNR_T1": psnr_t1_after_t2,
            "PSNR_T2": psnr_t2_after_t2,
            "Avg": row["Avg"],
            "Fgt": row["Fgt"],
        },
        "buffer_size": len(buffer),
        "buffer_capacity": buffer.capacity,
        "arm": arm,
        "n_train": len(tasks["T1"].train),
        "device": str(device),
    }
