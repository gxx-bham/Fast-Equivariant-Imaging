"""CLI for the isolated CVPR stage-0 smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import deepinv as dinv

from .config import ARMS, MASK_FAMILY_TO_CLASS, N_BUF_GRID, PINNED_DEEPINV, SmokeConfig
from .data_physics import smoke_data_physics
from .logging_utils import (
    distinct_slot_proof,
    finetune_forgetting_gate,
    go_kill_readout,
    write_csv,
    write_json,
)
from .protocol import run_arm


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CVPR stage-0 operator-incremental unsupervised MRI smoke. "
            "Composes deepinv (MoDL + MCLoss/EILoss), not FEI/SkEI plugins."
        )
    )
    parser.add_argument(
        "--unit",
        choices=("data", "finetune", "er", "nbuf1", "nbuf4", "grid"),
        default="nbuf1",
        help=(
            "Verifiable unit: (a) data, (b) finetune, (c) er, "
            "nbuf1, nbuf4 (must honor N_buf=4), or N_buf {1,4} grid."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--n-buf",
        type=int,
        default=None,
        help="Buffer capacity. Required to be 1 for --unit nbuf1 and 4 for --unit nbuf4.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="finetune,er_mc,er_ei",
        help="Comma-separated arms: finetune,er_mc,er_ei",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tiny", action="store_true", default=True)
    parser.add_argument("--no-tiny", action="store_false", dest="tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--epochs-t1", type=int, default=None)
    parser.add_argument("--epochs-t2", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=2)
    parser.add_argument(
        "--n-eval",
        type=int,
        default=0,
        help="Held-out eval slices. 0 = eval on train slices.",
    )
    parser.add_argument("--t1-accel", type=int, default=4)
    parser.add_argument("--t2-accel", type=int, default=8)
    parser.add_argument(
        "--t1-mask-family",
        choices=("gaussian", "random"),
        default="gaussian",
        help="T1 Cartesian mask family (GaussianMaskGenerator or RandomMaskGenerator).",
    )
    parser.add_argument(
        "--t2-mask-family",
        choices=("gaussian", "random"),
        default="gaussian",
        help="T2 Cartesian mask family. Gate B uses random (RandomMaskGenerator).",
    )
    parser.add_argument(
        "--t1-anatomy",
        choices=("knee", "brain"),
        default="knee",
    )
    parser.add_argument(
        "--t2-anatomy",
        choices=("knee", "brain"),
        default="knee",
        help="Gate A uses brain (domain-incremental mini RSS).",
    )
    parser.add_argument(
        "--gate",
        choices=("B", "A", "D", "F"),
        default=None,
        help=(
            "Fine-tune forgetting gate label: B=same-knee mask-family; "
            "A=knee→brain; D=domain+operator composite (A+B stacked); "
            "F=Step-3 supervised HQ Fine-tune on stream D."
        ),
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        default=False,
        help="Train with deepinv.loss.SupLoss (HQ MSE). Turns off MC/EI as the main loss.",
    )
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args(argv)


def _resolve_n_buf(args: argparse.Namespace, n_buf: int | None = None) -> int:
    """Honor unit contracts. nbuf1 cannot silently become 4; nbuf4 cannot silently become 1."""
    if args.unit == "nbuf1":
        if args.n_buf not in (None, 1):
            raise SystemExit(
                f"--unit nbuf1 requires --n-buf 1 (got {args.n_buf}). "
                "Use --unit nbuf4 --n-buf 4 for N_buf=4."
            )
        return 1
    if args.unit == "nbuf4":
        if args.n_buf not in (None, 4):
            raise SystemExit(
                f"--unit nbuf4 requires --n-buf 4 (got {args.n_buf}). "
                "This path must not silently run as N_buf=1."
            )
        return 4
    if args.unit == "grid":
        if args.n_buf is not None:
            raise SystemExit(
                "--unit grid always runs N_buf in {1, 4}; do not pass --n-buf. "
                "Use --unit nbuf4 --n-buf 4 for a single N_buf=4 cell."
            )
        if n_buf is None:
            raise SystemExit("grid cells must pass n_buf in {1, 4}")
        if int(n_buf) not in N_BUF_GRID:
            raise SystemExit(f"grid N_buf must be in {N_BUF_GRID} (got {n_buf})")
        return int(n_buf)
    if n_buf is not None:
        return int(n_buf)
    return int(args.n_buf if args.n_buf is not None else 1)


def _cfg_from_args(args: argparse.Namespace, n_buf: int | None = None) -> SmokeConfig:
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    out = args.out or str(_package_root() / "recorded_smoke" / f"seed{args.seed}")
    resolved = _resolve_n_buf(args, n_buf)
    return SmokeConfig(
        seed=args.seed,
        n_buf=resolved,
        arms=arms,
        device=args.device,
        tiny=args.tiny,
        epochs=args.epochs,
        epochs_t1=args.epochs_t1,
        epochs_t2=args.epochs_t2,
        max_batch_steps=args.max_steps if args.tiny else max(args.max_steps, 10**9),
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_eval=args.n_eval,
        t1_accel=args.t1_accel,
        t2_accel=args.t2_accel,
        t1_mask_family=args.t1_mask_family,
        t2_mask_family=args.t2_mask_family,
        t1_anatomy=args.t1_anatomy,
        t2_anatomy=args.t2_anatomy,
        gate=args.gate or "",
        supervised=bool(args.supervised),
        download=not args.no_download,
        out_dir=out,
    )


def _payload_config(cfg: SmokeConfig, arms: tuple[str, ...]) -> dict:
    return {
        "seed": cfg.seed,
        "N_buf": cfg.n_buf,
        "arms": list(arms),
        "tiny": cfg.tiny,
        "epochs": cfg.epochs,
        "epochs_t1": cfg.t1_epochs(),
        "epochs_t2": cfg.t2_epochs(),
        "max_batch_steps": cfg.max_batch_steps,
        "n_train": cfg.n_train,
        "n_eval": cfg.n_eval,
        "t1_accel": cfg.t1_accel,
        "t2_accel": cfg.t2_accel,
        "t1_mask_family": cfg.t1_mask_family,
        "t2_mask_family": cfg.t2_mask_family,
        "mask_generator_t1": MASK_FAMILY_TO_CLASS[cfg.t1_mask_family],
        "mask_generator_t2": MASK_FAMILY_TO_CLASS[cfg.t2_mask_family],
        "t1_anatomy": cfg.t1_anatomy,
        "t2_anatomy": cfg.t2_anatomy,
        "gate": cfg.gate,
        "domain_incremental": str(cfg.t1_anatomy) != str(cfg.t2_anatomy),
        "operator_shift": (
            str(cfg.t1_mask_family) != str(cfg.t2_mask_family)
            or int(cfg.t1_accel) != int(cfg.t2_accel)
        ),
        "claim_scope": (
            "Step-3 supervised Fine-tune on stream D (domain+operator composite); HQ SupLoss, MC/EI off"
            if cfg.gate == "F" or cfg.supervised
            else (
                "domain+operator composite drift (A+B stacked)"
                if cfg.gate == "D"
                else (
                    "domain-incremental (not same-knee accel-only)"
                    if cfg.gate == "A"
                    else (
                        "same-knee mask-family (not Cartesian 4x->8x fallback)"
                        if cfg.gate == "B"
                        else "operator-incremental unsupervised MRI"
                    )
                )
            )
        ),
        "supervised": bool(cfg.supervised),
        "device": cfg.device,
        "deepinv_version": getattr(dinv, "__version__", "unknown"),
        "deepinv_pinned": PINNED_DEEPINV,
        "backbone": "deepinv.models.MoDL",
        "losses_current": (
            "deepinv.loss.SupLoss (HQ MSE; MC/EI off)"
            if cfg.supervised
            else "MCLoss() + EILoss(Rotate(n_trans=4))"
        ),
        "physics": (
            f"deepinv.physics.MRI + {MASK_FAMILY_TO_CLASS[cfg.t1_mask_family]} "
            f"({cfg.t1_anatomy} {cfg.t1_accel}x) then "
            f"{MASK_FAMILY_TO_CLASS[cfg.t2_mask_family]} "
            f"({cfg.t2_anatomy} {cfg.t2_accel}x), per-sample A"
        ),
    }


def _run_arms(cfg: SmokeConfig, arms: tuple[str, ...]) -> list[dict]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    t0 = time.time()
    for arm in arms:
        print(
            f"\n=== arm={arm} N_buf={cfg.n_buf} seed={cfg.seed} "
            f"device={cfg.device} tiny={cfg.tiny} n_train={cfg.n_train} "
            f"gate={cfg.gate or '-'} supervised={cfg.supervised} "
            f"T1={cfg.t1_anatomy}/{cfg.t1_mask_family}/{cfg.t1_accel}x "
            f"T2={cfg.t2_anatomy}/{cfg.t2_mask_family}/{cfg.t2_accel}x ==="
        )
        result = run_arm(arm, cfg, out_dir=out_dir)
        if int(result["N_buf"]) != int(cfg.n_buf):
            raise RuntimeError(
                f"N_buf mismatch: cfg={cfg.n_buf} result={result['N_buf']}"
            )
        rows.append(result["row"])
        details.append(result)
        print(json.dumps(result["row"], indent=2))
        print(
            "buffer:",
            json.dumps(
                {
                    "N_buf": result["buffer"]["N_buf"],
                    "n_distinct_ya": result["buffer"]["n_distinct_ya"],
                    "n_distinct_A": result["buffer"]["n_distinct_A"],
                    "holds_t1_A": result["buffer"]["holds_t1_A"],
                    "accels": result["buffer"]["accels"],
                    "slots": [
                        {"slot": s["slot"], "y_id": s["y_id"], "mask_id": s["mask_id"]}
                        for s in result["buffer"]["slots"]
                    ],
                },
                indent=2,
            ),
        )
    elapsed_sec = time.time() - t0
    csv_path = out_dir / f"metrics_nbuf{cfg.n_buf}_seed{cfg.seed}.csv"
    json_path = out_dir / f"metrics_nbuf{cfg.n_buf}_seed{cfg.seed}.json"
    write_csv(csv_path, rows)
    proofs = [distinct_slot_proof(d["buffer"]) for d in details]
    payload = {
        "config": _payload_config(cfg, arms),
        "rows": rows,
        "details": details,
        "go_kill": go_kill_readout(rows, tiny=cfg.tiny),
        "elapsed_sec": elapsed_sec,
        "device": details[0]["device"] if details else cfg.device,
        "supervised": bool(cfg.supervised),
        "n_distinct_ya": (
            int(details[0]["buffer"]["n_distinct_ya"]) if details else None
        ),
        "distinct_slot_proof": proofs[0] if len(proofs) == 1 else proofs,
        "fgt_formula": "after_T1.PSNR_T1 - after_T2.PSNR_T1",
        "fgt_definition": (
            "Fgt = after_T1.PSNR_T1 - after_T2.PSNR_T1; "
            "Avg = 0.5 * (PSNR_T1_after_T2 + PSNR_T2_after_T2). "
            "CSV PSNR_T1/PSNR_T2 are after T2."
        ),
    }
    if cfg.gate:
        payload["forgetting_gate"] = finetune_forgetting_gate(rows)
        payload["schedule"] = details[0].get("schedule") if details else None
    write_json(json_path, payload)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print("go/kill:", json.dumps(payload["go_kill"], indent=2))
    if cfg.gate:
        print("forgetting_gate:", json.dumps(payload["forgetting_gate"], indent=2))
    print(f"device={payload['device']} elapsed_sec={elapsed_sec:.1f}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.unit == "data":
        report = smoke_data_physics(
            seed=args.seed,
            device=args.device,
            download=not args.no_download,
            n_train=args.n_train,
            n_eval=args.n_eval,
        )
        print(json.dumps(report, indent=2))
        out = Path(args.out or (_package_root() / "recorded_smoke" / "data_physics.json"))
        if out.suffix != ".json":
            out = out / "data_physics.json"
        write_json(out, report)
        print(f"Wrote {out}")
        return 0

    if args.unit == "finetune":
        cfg = _cfg_from_args(args)
        _run_arms(cfg, ("finetune",))
        return 0

    if args.unit == "er":
        cfg = _cfg_from_args(args)
        _run_arms(cfg, ("er_mc", "er_ei"))
        return 0

    if args.unit == "nbuf1":
        cfg = _cfg_from_args(args, n_buf=1)
        if cfg.n_buf != 1:
            raise SystemExit("internal error: nbuf1 must have N_buf=1")
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) or ARMS
        _run_arms(cfg, arms)
        return 0

    if args.unit == "nbuf4":
        cfg = _cfg_from_args(args, n_buf=4)
        if cfg.n_buf != 4:
            raise SystemExit("internal error: nbuf4 must have N_buf=4")
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) or ARMS
        _run_arms(cfg, arms)
        return 0

    if args.unit == "grid":
        t0 = time.time()
        all_rows = []
        for n_buf in N_BUF_GRID:
            cfg = _cfg_from_args(args, n_buf=int(n_buf))
            if cfg.n_buf != int(n_buf):
                raise SystemExit(f"grid failed to honor N_buf={n_buf} (got {cfg.n_buf})")
            cfg.out_dir = str(Path(cfg.out_dir) / f"nbuf{n_buf}")
            arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) or ARMS
            all_rows.extend(_run_arms(cfg, arms))
        elapsed_sec = time.time() - t0
        grid_dir = Path(args.out or (_package_root() / "recorded_smoke" / f"seed{args.seed}"))
        cfg0 = _cfg_from_args(args, n_buf=int(N_BUF_GRID[0]))
        write_csv(grid_dir / "metrics_grid.csv", all_rows)
        write_json(
            grid_dir / "metrics_grid.json",
            {
                "device": cfg0.device,
                "seed": cfg0.seed,
                "epochs": cfg0.epochs,
                "tiny": cfg0.tiny,
                "n_train": cfg0.n_train,
                "n_eval": cfg0.n_eval,
                "N_buf_grid": list(N_BUF_GRID),
                "deepinv_version": getattr(dinv, "__version__", "unknown"),
                "max_batch_steps": cfg0.max_batch_steps,
                "elapsed_sec": elapsed_sec,
                "rows": all_rows,
                "go_kill": go_kill_readout(all_rows, tiny=cfg0.tiny),
            },
        )
        print("grid go/kill:", json.dumps(go_kill_readout(all_rows, tiny=cfg0.tiny), indent=2))
        print(f"device={cfg0.device} elapsed_sec={elapsed_sec:.1f}")
        return 0

    raise RuntimeError(f"unhandled unit {args.unit}")


if __name__ == "__main__":
    sys.exit(main())
