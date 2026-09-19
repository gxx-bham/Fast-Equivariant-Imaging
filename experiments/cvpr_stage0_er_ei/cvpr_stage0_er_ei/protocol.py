"""Sequential T1 -> T2 protocol for Fine-tune | ER+MC | ER+EI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import deepinv as dinv
import torch

from .buffer import MeasurementBuffer
from .config import ARM_LABELS, ARM_LOSSES, MASK_FAMILY_TO_CLASS, SmokeConfig
from .data_physics import (
    build_cross_ip_tasks,
    build_tasks,
    load_anatomy_slices,
    load_ct100_tiles,
    task_operator_report,
)
from .logging_utils import summary_row
from .losses import BufferReplayLoss, current_task_losses, losses_for_arm, supervised_losses
from .train_eval import (
    backbone_fingerprint,
    eval_psnr,
    make_backbone,
    set_seed,
    train_task,
)


def _device(cfg: SmokeConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def run_arm(arm: str, cfg: SmokeConfig, out_dir: Path | None = None) -> dict[str, Any]:
    if arm not in ARM_LABELS:
        raise ValueError(f"unknown arm {arm!r}")
    if cfg.cross_ip and cfg.supervised:
        raise ValueError(
            "Gate E / cross-IP Fine-tune must use unsupervised MC/EI, not SupLoss"
        )
    device = _device(cfg)
    set_seed(cfg.seed)
    t_wall0 = time.time()
    n_eval = int(cfg.n_eval)
    n_total = int(cfg.n_train) + (n_eval if n_eval > 0 else 0)
    n_total = max(n_total, int(cfg.n_train))
    if cfg.cross_ip:
        x_t1 = load_anatomy_slices(
            anatomy=cfg.t1_anatomy,
            img_size=cfg.img_size,
            download=cfg.download,
            n_total=n_total,
        )
        x_ct = load_ct100_tiles(img_size=cfg.ct_img_size, n_total=n_total)
        tasks = build_cross_ip_tasks(
            x_t1,
            x_ct,
            seed=cfg.seed,
            device=device,
            n_train=cfg.n_train,
            n_eval=n_eval,
            t1_accel=cfg.t1_accel,
            t1_mask_family=cfg.t1_mask_family,
            t1_anatomy=cfg.t1_anatomy,
            mri_img_size=cfg.img_size,
            ct_img_size=cfg.ct_img_size,
            ct_n_angles=cfg.ct_n_angles,
        )
    else:
        x_t1 = load_anatomy_slices(
            anatomy=cfg.t1_anatomy,
            img_size=cfg.img_size,
            download=cfg.download,
            n_total=n_total,
        )
        x_t2 = None
        if str(cfg.t2_anatomy).lower() != str(cfg.t1_anatomy).lower():
            x_t2 = load_anatomy_slices(
                anatomy=cfg.t2_anatomy,
                img_size=cfg.img_size,
                download=cfg.download,
                n_total=n_total,
            )
        tasks = build_tasks(
            x_t1,
            seed=cfg.seed,
            device=device,
            img_size=cfg.img_size,
            n_train=cfg.n_train,
            n_eval=n_eval,
            t1_accel=cfg.t1_accel,
            t2_accel=cfg.t2_accel,
            t1_mask_family=cfg.t1_mask_family,
            t2_mask_family=cfg.t2_mask_family,
            t1_anatomy=cfg.t1_anatomy,
            t2_anatomy=cfg.t2_anatomy,
            x_t2=x_t2,
        )
    model = make_backbone(cfg, device)
    backbone_fp = backbone_fingerprint(cfg, model)
    save_path = None if out_dir is None else str(out_dir / "ckpts" / arm)

    t1_losses = supervised_losses() if cfg.supervised else current_task_losses()
    train_task(
        model,
        tasks["T1"],
        losses=t1_losses,
        cfg=cfg,
        device=device,
        save_path=save_path,
        epochs=cfg.t1_epochs(),
    )
    psnr_t1_after_t1 = eval_psnr(model, tasks["T1"], device, batch_size=cfg.batch_size)
    psnr_t2_after_t1 = eval_psnr(model, tasks["T2"], device, batch_size=cfg.batch_size)

    buffer = MeasurementBuffer(
        capacity=cfg.n_buf,
        generator=torch.Generator().manual_seed(cfg.seed + 17),
    )
    buffer.fill_from_dataset(tasks["T1"].train, task="T1", accel=tasks["T1"].accel)
    buffer_report = buffer.summary()
    if not cfg.tiny and buffer_report["n_distinct_ya"] < cfg.n_buf:
        raise RuntimeError(
            f"N_buf={cfg.n_buf} but buffer has only {buffer_report['n_distinct_ya']} "
            "distinct (y, A) slots. Increase --n-train so the buffer can hold "
            f"{cfg.n_buf} distinct past measurements."
        )

    if cfg.supervised:
        t2_losses = supervised_losses()
    else:
        t2_losses = losses_for_arm(arm, buffer=None if arm == "finetune" else buffer)
    replay_mod = next((loss for loss in t2_losses if isinstance(loss, BufferReplayLoss)), None)
    if replay_mod is not None:
        y0 = tasks["T2"].train.y[:1].to(device)
        replay_mod(
            x_net=None,
            physics=tasks["T2"].physics.to(device)
            if hasattr(tasks["T2"].physics, "to")
            else tasks["T2"].physics,
            model=model,
            y=y0,
        )
        print(
            "pre_T2_replay "
            f"BufferReplayLoss={replay_mod.first_loss:.8e} "
            f"physics={replay_mod.last_replay_physics} "
            f"y_b_shape={replay_mod.last_y_shape} "
            f"mask_shape={replay_mod.last_mask_shape} "
            "via make_mri_physics"
        )
    train_task(
        model,
        tasks["T2"],
        losses=t2_losses,
        cfg=cfg,
        device=device,
        save_path=save_path,
        epochs=cfg.t2_epochs(),
    )
    psnr_t1_after_t2 = eval_psnr(model, tasks["T1"], device, batch_size=cfg.batch_size)
    psnr_t2_after_t2 = eval_psnr(model, tasks["T2"], device, batch_size=cfg.batch_size)
    replay_stats = None if replay_mod is None else replay_mod.stats()
    fgt_expected = float(psnr_t1_after_t1) - float(psnr_t1_after_t2)

    row = summary_row(
        arm=arm,
        n_buf=cfg.n_buf,
        seed=cfg.seed,
        psnr_t1_after_t2=psnr_t1_after_t2,
        psnr_t2_after_t2=psnr_t2_after_t2,
        psnr_t1_after_t1=psnr_t1_after_t1,
    )
    row["arm_loss"] = (
        "deepinv.loss.SupLoss (HQ MSE; MC/EI off)"
        if cfg.supervised
        else ARM_LOSSES[arm]
    )
    wall_time_sec = float(time.time() - t_wall0)
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
        "N_buf": int(cfg.n_buf),
        "deepinv_version": getattr(dinv, "__version__", "unknown"),
        "buffer": buffer_report,
        "operators": {
            "T1": task_operator_report(tasks["T1"]),
            "T2": task_operator_report(tasks["T2"]),
        },
        "n_train": len(tasks["T1"].train),
        "n_eval": len(tasks["T1"].eval),
        "n_train_t2": len(tasks["T2"].train),
        "n_eval_t2": len(tasks["T2"].eval),
        "held_out_eval": n_eval > 0,
        "epochs_t1": cfg.t1_epochs(),
        "epochs_t2": cfg.t2_epochs(),
        "tiny": cfg.tiny,
        "arm": arm,
        "device": str(device),
        "wall_time_sec": wall_time_sec,
        "schedule": {
            "t1_anatomy": tasks["T1"].anatomy,
            "t2_anatomy": tasks["T2"].anatomy,
            "t1_accel": tasks["T1"].accel,
            "t2_accel": tasks["T2"].accel,
            "t1_mask_family": tasks["T1"].mask_family,
            "t2_mask_family": tasks["T2"].mask_family,
            "mask_generator_t1": tasks["T1"].mask_generator,
            "mask_generator_t2": tasks["T2"].mask_generator,
            "physics_class_t1": tasks["T1"].physics_class,
            "physics_class_t2": tasks["T2"].physics_class,
            "modality_t1": tasks["T1"].modality,
            "modality_t2": tasks["T2"].modality,
            "ct_n_angles": tasks["T2"].n_angles,
            "t1_img_size": tasks["T1"].img_size,
            "t2_img_size": tasks["T2"].img_size,
            "epochs_t1": cfg.t1_epochs(),
            "epochs_t2": cfg.t2_epochs(),
            "n_train": len(tasks["T1"].train),
            "n_eval": len(tasks["T1"].eval),
        },
        "gate": cfg.gate,
        "supervised": bool(cfg.supervised),
        "cross_ip": bool(cfg.cross_ip),
        "physics_class_t1": tasks["T1"].physics_class,
        "physics_class_t2": tasks["T2"].physics_class,
        "ct_n_angles": tasks["T2"].n_angles,
        "train_loss": (
            "deepinv.loss.SupLoss (HQ MSE; MC/EI off)"
            if cfg.supervised
            else ARM_LOSSES[arm]
        ),
        "arm_loss": (
            "deepinv.loss.SupLoss (HQ MSE; MC/EI off)"
            if cfg.supervised
            else ARM_LOSSES[arm]
        ),
        "unsupervised": not bool(cfg.supervised),
        "replay": replay_stats,
        "backbone": backbone_fp,
        "fgt_handcheck": {
            "formula": "after_T1.PSNR_T1 - after_T2.PSNR_T1",
            "expected": fgt_expected,
            "logged": float(row["Fgt"]),
            "ok": abs(fgt_expected - float(row["Fgt"])) < 1e-5,
        },
        "reviewer_check": {
            "source": "operators/schedule (not stale config.t2_*)",
            "cross_ip": bool(cfg.cross_ip),
            "unsupervised": not bool(cfg.supervised),
            "physics_class_t1": tasks["T1"].physics_class,
            "physics_class_t2": tasks["T2"].physics_class,
            "ct_n_angles": tasks["T2"].n_angles,
            "t2_is_tomography": tasks["T2"].physics_class == "deepinv.physics.Tomography",
            "n_angles_40": tasks["T2"].n_angles == 40,
            "fgt_handcheck_ok": abs(fgt_expected - float(row["Fgt"])) < 1e-5,
            "replay_rebuilds_mri": (
                True
                if replay_stats is None
                else bool(replay_stats.get("rebuilds_mri_from_stored_mask"))
            ),
            "BufferReplayLoss_logged": replay_stats,
            "backbone_class": backbone_fp["class"],
            "backbone_ctor_kwargs": backbone_fp["ctor_kwargs"],
            "backbone_wrapper_class": backbone_fp["wrapper_class"],
            "backbone_wrapper_ctor_kwargs": backbone_fp["wrapper_ctor_kwargs"],
            "pass": (
                bool(cfg.cross_ip)
                and not bool(cfg.supervised)
                and tasks["T2"].physics_class == "deepinv.physics.Tomography"
                and tasks["T2"].n_angles == 40
                and abs(fgt_expected - float(row["Fgt"])) < 1e-5
                and (
                    replay_stats is None
                    or (
                        bool(replay_stats.get("rebuilds_mri_from_stored_mask"))
                        and bool(replay_stats.get("BufferReplayLoss_nonzero"))
                        and replay_stats.get("replay_physics_class")
                        == "deepinv.physics.MRI"
                    )
                )
            ),
        },
        "domain_incremental": str(tasks["T1"].anatomy) != str(tasks["T2"].anatomy),
        "mask_family_to_class": dict(MASK_FAMILY_TO_CLASS),
    }
