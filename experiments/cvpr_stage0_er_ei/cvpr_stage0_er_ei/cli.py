"""CLI for the isolated CVPR stage-0 smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import ARMS, N_BUF_GRID, SmokeConfig
from .data_physics import smoke_data_physics
from .logging_utils import go_kill_readout, write_csv, write_json
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
        choices=("data", "finetune", "er", "nbuf1", "grid"),
        default="nbuf1",
        help=(
            "Verifiable unit: (a) data, (b) finetune, (c) er, "
            "(d) nbuf1 all three arms, or N_buf {1,4} grid."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-buf", type=int, default=1)
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
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace, n_buf: int | None = None) -> SmokeConfig:
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    out = args.out or str(_package_root() / "recorded_smoke" / f"seed{args.seed}")
    return SmokeConfig(
        seed=args.seed,
        n_buf=int(n_buf if n_buf is not None else args.n_buf),
        arms=arms,
        device=args.device,
        tiny=args.tiny,
        epochs=args.epochs,
        max_batch_steps=args.max_steps if args.tiny else max(args.max_steps, 10**9),
        batch_size=args.batch_size,
        download=not args.no_download,
        out_dir=out,
    )


def _run_arms(cfg: SmokeConfig, arms: tuple[str, ...]) -> list[dict]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    for arm in arms:
        print(f"\n=== arm={arm} N_buf={cfg.n_buf} seed={cfg.seed} device={cfg.device} ===")
        result = run_arm(arm, cfg, out_dir=out_dir)
        rows.append(result["row"])
        details.append(result)
        print(json.dumps(result["row"], indent=2))
    csv_path = out_dir / f"metrics_nbuf{cfg.n_buf}_seed{cfg.seed}.csv"
    json_path = out_dir / f"metrics_nbuf{cfg.n_buf}_seed{cfg.seed}.json"
    write_csv(csv_path, rows)
    payload = {
        "config": {
            "seed": cfg.seed,
            "N_buf": cfg.n_buf,
            "arms": list(arms),
            "tiny": cfg.tiny,
            "epochs": cfg.epochs,
            "max_batch_steps": cfg.max_batch_steps,
            "device": cfg.device,
            "backbone": "deepinv.models.MoDL",
            "losses_current": "MCLoss() + EILoss(Rotate(n_trans=4))",
            "physics": "deepinv.physics.MRI + GaussianMaskGenerator (4x then 8x)",
        },
        "rows": rows,
        "details": details,
        "go_kill": go_kill_readout(rows),
        "fgt_definition": (
            "Fgt = PSNR_T1_after_T1 - PSNR_T1_after_T2; "
            "Avg = 0.5 * (PSNR_T1_after_T2 + PSNR_T2_after_T2). "
            "CSV PSNR_T1/PSNR_T2 are after T2."
        ),
    }
    write_json(json_path, payload)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print("go/kill:", json.dumps(payload["go_kill"], indent=2))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.unit == "data":
        report = smoke_data_physics(
            seed=args.seed, device=args.device, download=not args.no_download
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
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) or ARMS
        _run_arms(cfg, arms)
        return 0

    if args.unit == "grid":
        t0 = time.time()
        all_rows = []
        for n_buf in N_BUF_GRID:
            cfg = _cfg_from_args(args, n_buf=n_buf)
            cfg.out_dir = str(Path(cfg.out_dir) / f"nbuf{n_buf}")
            arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) or ARMS
            all_rows.extend(_run_arms(cfg, arms))
        elapsed_sec = time.time() - t0
        grid_dir = Path(args.out or (_package_root() / "recorded_smoke" / f"seed{args.seed}"))
        cfg0 = _cfg_from_args(args, n_buf=N_BUF_GRID[0])
        write_csv(grid_dir / "metrics_grid.csv", all_rows)
        write_json(
            grid_dir / "metrics_grid.json",
            {
                "device": cfg0.device,
                "seed": cfg0.seed,
                "epochs": cfg0.epochs,
                "tiny": cfg0.tiny,
                "max_batch_steps": cfg0.max_batch_steps,
                "elapsed_sec": elapsed_sec,
                "rows": all_rows,
                "go_kill": go_kill_readout(all_rows),
            },
        )
        print("grid go/kill:", json.dumps(go_kill_readout(all_rows), indent=2))
        print(f"device={cfg0.device} elapsed_sec={elapsed_sec:.1f}")
        return 0

    raise RuntimeError(f"unhandled unit {args.unit}")


if __name__ == "__main__":
    sys.exit(main())
