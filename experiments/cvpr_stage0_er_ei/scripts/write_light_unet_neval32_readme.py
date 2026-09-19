#!/usr/bin/env python3
"""Aggregate light-UNet HARDEN n_eval=32 N_buf=1 seeds {1,2,3}.

Never merge with recorded_smoke/light_unet_s1_seeds/ (n_eval=2) or MoDL.
GO: ≥2/3 seeds Fgt(EI)<Fgt(MC) AND mean(Fgt_MC−Fgt_EI)>0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvpr_stage0_er_ei.logging_utils import (  # noqa: E402
    aggregate_avg_vs_nbuf,
    aggregate_fgt_vs_nbuf,
    harden_neval_go_kill,
    write_csv,
    write_json,
)

SEEDS = (1, 2, 3)
N_BUF = 1
N_TRAIN = 8
N_EVAL = 32


def _meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _load_cell(out_dir: Path, seed: int) -> dict | None:
    path = out_dir / f"seed{seed}" / f"metrics_nbuf{N_BUF}_seed{seed}.json"
    if path.is_file():
        return json.loads(path.read_text())
    return None


def _backbone(payload: dict) -> dict:
    details = payload.get("details") or []
    for d in details:
        fp = d.get("backbone") or {}
        if fp.get("n_parameters"):
            return fp
    return payload.get("backbone") or payload.get("config", {}).get("backbone") or {}


def _reviewer(payload: dict) -> dict:
    out: dict = {}
    for d in payload.get("details") or []:
        arm = str((d.get("row") or {}).get("arm") or d.get("arm"))
        rc = d.get("reviewer_check") or {}
        replay = d.get("replay") or rc.get("BufferReplayLoss_logged") or {}
        cfg = payload.get("config") or {}
        out[arm] = {
            "pass": rc.get("pass"),
            "physics_class_t2": rc.get("physics_class_t2"),
            "ct_n_angles": rc.get("ct_n_angles"),
            "n_angles_40": rc.get("n_angles_40"),
            "replay_rebuilds_mri": rc.get("replay_rebuilds_mri"),
            "BufferReplayLoss_nonzero": replay.get("BufferReplayLoss_nonzero"),
            "replay_physics_class": replay.get("replay_physics_class"),
            "n_distinct_ya": (d.get("buffer") or {}).get("n_distinct_ya"),
            "n_eval": (d.get("schedule") or {}).get("n_eval") or cfg.get("n_eval"),
            "backbone_class": rc.get("backbone_class"),
            "backbone_ctor_kwargs": rc.get("backbone_ctor_kwargs"),
            "backbone_wrapper_class": rc.get("backbone_wrapper_class"),
            "arm_loss": d.get("arm_loss"),
        }
    return out


def _fgt_triple(by: dict, seed: int) -> str:
    ft = by.get(("Fine-tune", seed))
    mc = by.get(("ER+MC", seed))
    ei = by.get(("ER+EI", seed))
    if not (ft and mc and ei):
        return "missing"
    return f"{ft['Fgt']:.3f}/{mc['Fgt']:.3f}/{ei['Fgt']:.3f}"


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if "light_unet_s1_seeds" in str(out_dir) or "stage1_crossip_grid" in str(out_dir):
        raise SystemExit(f"refusing to write HARDEN readout into {out_dir}")
    meta = _meta(out_dir / "run_meta.txt")
    cells: dict[int, dict] = {}
    rows: list[dict] = []
    missing: list[int] = []
    for seed in SEEDS:
        cell = _load_cell(out_dir, seed)
        if cell is None:
            missing.append(seed)
            continue
        cells[seed] = cell
        rows.extend(cell.get("rows") or [])

    go = harden_neval_go_kill(rows, required_seeds=SEEDS)
    agg = aggregate_fgt_vs_nbuf(rows) if rows else []
    avg_app = aggregate_avg_vs_nbuf(rows) if rows else []
    backbone = {}
    for seed in SEEDS:
        if seed in cells:
            backbone = _backbone(cells[seed])
            if backbone:
                break

    payload = {
        "label": "light-UNet HARDEN n_eval=32 N_buf=1 seeds 1-3",
        "not_modl": True,
        "separate_from": [
            "recorded_smoke/light_unet_s1_seeds/",
            "recorded_smoke/stage1_crossip_grid/",
        ],
        "do_not_merge_with_neval2": True,
        "backbone": backbone,
        "supervised": False,
        "unsupervised": True,
        "cross_ip": True,
        "physics_class_t1": "deepinv.physics.MRI",
        "physics_class_t2": "deepinv.physics.Tomography",
        "ct_n_angles": 40,
        "n_buf": N_BUF,
        "seeds": list(SEEDS),
        "seeds_present": sorted(cells),
        "missing_seeds": missing,
        "arms": ["finetune", "er_mc", "er_ei"],
        "ei": "EILoss(Rotate(n_trans=4))",
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "epochs_t1": 80,
        "epochs_t2": 160,
        "tiny": False,
        "expand_n_buf": False,
        "fgt_formula": "after_T1.PSNR_T1 - after_T2.PSNR_T1",
        "fgt_definition": (
            "Fgt = after_T1.PSNR_T1 - after_T2.PSNR_T1 on the T1 MRI test set only; "
            "T2 CT PSNR is logged separately and is not used in Fgt. Lower Fgt is better."
        ),
        "rows": rows,
        "aggregate_fgt": agg,
        "appendix_avg": avg_app,
        "go_kill": go,
        "reviewer_by_seed": {str(s): _reviewer(cells[s]) for s in cells},
        "device": meta.get("device")
        or (cells[next(iter(cells))]["device"] if cells else None),
        "elapsed_sec": meta.get("elapsed_sec"),
    }
    write_json(out_dir / "metrics_light_unet_neval32.json", payload)
    if rows:
        write_csv(out_dir / "metrics_light_unet_neval32.csv", rows)

    n_params = backbone.get("n_parameters")
    by = {(str(r["arm"]), int(r["seed"])): r for r in rows}
    lines = [
        "# Light-UNet HARDEN n_eval=32 (N_buf=1, seeds 1–3)",
        "",
        "Same fingerprint as light-UNet S1 GO: **`UNet(in_channels=2, "
        "out_channels=2, scales=3)`** + `ArtifactRemoval(mode=adjoint)`. "
        "**Not MoDL.** n_train=8 unchanged; **n_eval=32** only. "
        "Do **not** merge with `recorded_smoke/light_unet_s1_seeds/` (n_eval=2) "
        "or `stage1_crossip_grid/`.",
        "",
        "Confirmation line: [`CONFIRMATION.md`](CONFIRMATION.md).",
        "",
        "## Backbone fingerprint",
        "",
        f"- class: `{backbone.get('class', 'deepinv.models.UNet')}`",
        f"- ctor kwargs: `{json.dumps(backbone.get('ctor_kwargs', {}), sort_keys=True)}`",
        f"- wrapper: `{backbone.get('wrapper_class', 'deepinv.models.ArtifactRemoval')}`",
        f"- wrapper kwargs: `{json.dumps(backbone.get('wrapper_ctor_kwargs', {}), sort_keys=True)}`",
        f"- runtime: `{backbone.get('runtime_class')}` / denoiser `{backbone.get('runtime_denoiser_class')}`",
        f"- n_parameters: `{n_params}`",
        "",
        "## Protocol",
        "",
        "- T1: `deepinv.physics.MRI` knee Gaussian Cartesian 4× 128×128",
        "- T2: `deepinv.physics.Tomography` n_angles=40, img_width=64",
        "- unsupervised `MCLoss() + EILoss(Rotate(n_trans=4))` (n_trans not swept)",
        "- ER+MC buffer: MC only; ER+EI buffer: MC+EI",
        "- MRI replay via `make_mri_physics(stored mask)`",
        f"- N_buf=1, seeds {{1,2,3}}, n_train={N_TRAIN}, n_eval={N_EVAL}, epochs T1=80 / T2=160",
        "- Fgt = after_T1.PSNR_T1 − after_T2.PSNR_T1 on T1 MRI test only",
        "",
        "## Fgt table (N_buf=1, n_eval=32)",
        "",
        "| seed | Fine-tune | ER+MC | ER+EI | EI < MC | Δ(MC−EI) |",
        "| ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for seed in SEEDS:
        ft = by.get(("Fine-tune", seed))
        mc = by.get(("ER+MC", seed))
        ei = by.get(("ER+EI", seed))
        if not (ft and mc and ei):
            lines.append(f"| {seed} | — | — | — | missing | — |")
            continue
        delta = float(mc["Fgt"]) - float(ei["Fgt"])
        better = float(ei["Fgt"]) < float(mc["Fgt"])
        lines.append(
            f"| {seed} | **{ft['Fgt']:.3f}** | **{mc['Fgt']:.3f}** | "
            f"**{ei['Fgt']:.3f}** | **{better}** | {delta:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Mean ± std across seeds (N_buf=1, n_eval=32)",
            "",
            "| arm | n | Fgt mean | Fgt std | Fgt sem |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for rec in agg:
        if int(rec["N_buf"]) != N_BUF:
            continue
        lines.append(
            f"| {rec['arm']} | {rec['n_seeds']} | **{rec['Fgt_mean']:.3f}** | "
            f"{rec['Fgt_std']:.3f} | {rec['Fgt_sem']:.3f} |"
        )

    mean_delta = go.get("mean_delta_MC_minus_EI")
    lines.extend(
        [
            "",
            "## HARDEN go/kill",
            "",
            "Rule: GO if ≥2/3 seeds with Fgt(ER+EI)<Fgt(ER+MC) **and** "
            "mean(Fgt_MC−Fgt_EI)>0. Kill if ≥2/3 seeds EI does not win, or mean Δ≤0.",
            "",
            f"- verdict: **{go.get('verdict')}**",
            f"- reason: {go.get('reason')}",
            f"- win seeds: `{go.get('win_seeds')}` ({go.get('n_win_seeds')}/3)",
            f"- mean(Fgt_MC−Fgt_EI): `{mean_delta}`",
            f"- seeds present: `{sorted(cells)}` missing=`{missing}`",
            f"- device: `{payload.get('device')}`",
            "",
            "## Reviewer probes",
            "",
            "Per-cell JSON: `physics_class_t2` Tomography(40), `n_distinct_ya==1`, "
            "`replay_rebuilds_mri` via `make_mri_physics`, nonzero BufferReplayLoss "
            "on ER arms, UNet+ArtifactRemoval, n_eval=32.",
            "",
        ]
    )
    for seed in sorted(cells):
        probes = payload["reviewer_by_seed"][str(seed)]
        mc_p = probes.get("ER+MC") or {}
        ei_p = probes.get("ER+EI") or {}
        cfg_n_eval = (cells[seed].get("config") or {}).get("n_eval")
        lines.append(
            f"- seed={seed}: reviewer ER+MC pass=`{mc_p.get('pass')}` "
            f"ER+EI pass=`{ei_p.get('pass')}` "
            f"n_distinct_ya=`{mc_p.get('n_distinct_ya') or ei_p.get('n_distinct_ya')}` "
            f"n_eval=`{cfg_n_eval}` "
            f"T2=`{mc_p.get('physics_class_t2') or ei_p.get('physics_class_t2')}` "
            f"n_angles=`{mc_p.get('ct_n_angles') or ei_p.get('ct_n_angles')}` "
            f"replay_mri=`{mc_p.get('replay_rebuilds_mri')}` "
            f"BufferReplayLoss_nonzero MC/EI="
            f"`{mc_p.get('BufferReplayLoss_nonzero')}`/"
            f"`{ei_p.get('BufferReplayLoss_nonzero')}` "
            f"backbone=`{mc_p.get('backbone_class') or ei_p.get('backbone_class')}`"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")

    if not missing:
        conf = [
            "# Confirmation line (light UNet HARDEN n_eval=32, N_buf=1, seeds 1–3)",
            "",
            "Verified from this dir's seed1–3 JSON. Do not merge with n_eval=2.",
            "",
            "- Stream E: T1 MRI knee gaussian 4× 128×128 → T2 `deepinv.physics.Tomography(n_angles=40)` 64×64",
            "- Backbone: `deepinv.models.UNet(in_channels=2,out_channels=2,scales=3)` + `ArtifactRemoval(mode=adjoint)` — NOT MoDL",
            "- Buffer: `(y,mask)` replay via `make_mri_physics(stored mask)` (physics=`deepinv.physics.MRI`); never MRI→CT forward",
            "- `n_distinct_ya == 1` for all three seeds; N_buf=1; n_train=8; **n_eval=32**",
            "- Arm losses: ER+MC buffer = `MCLoss` only; ER+EI buffer = `MCLoss` + `EILoss(Rotate(n_trans=4))` (nonzero MC and EI probes on ER arms)",
            (
                "- Fgt table (Fine-tune/ER+MC/ER+EI): "
                f"seed1 {_fgt_triple(by, 1)}; "
                f"seed2 {_fgt_triple(by, 2)}; "
                f"seed3 {_fgt_triple(by, 3)} → "
                f"**{go.get('verdict')}** under ≥2/3 EI<MC and mean(Fgt_MC−Fgt_EI)>0 "
                f"(mean Δ={mean_delta if mean_delta is None else f'{mean_delta:.3f}'})"
            ),
            "- Claim locked: raise n_eval only; do not merge MoDL or n_eval=2 tables; no N_buf/n_trans/n_train expand without Guixian",
            "",
        ]
        (out_dir / "CONFIRMATION.md").write_text("\n".join(conf))
        print(f"Wrote {out_dir / 'CONFIRMATION.md'}")

    print(f"Wrote {out_dir / 'README.md'}")
    print(f"Wrote {out_dir / 'metrics_light_unet_neval32.json'}")
    print("go/kill:", json.dumps(go, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
