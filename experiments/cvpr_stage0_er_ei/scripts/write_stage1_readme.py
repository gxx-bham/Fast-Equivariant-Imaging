#!/usr/bin/env python3
"""Write stage-1 cross-IP matrix README from metrics_stage1.json + run_meta.txt."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _seed_table(rows: list[dict], seed: int) -> str:
    lines = [
        f"### seed={seed}",
        "",
        "| arm | N_buf | PSNR_T1 | PSNR_T2 | Avg | Fgt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    subset = [r for r in rows if int(r["seed"]) == seed]
    subset.sort(key=lambda r: (int(r["N_buf"]), str(r["arm"])))
    for row in subset:
        lines.append(
            f"| {row['arm']} | {row['N_buf']} | {row['PSNR_T1']:.3f} | "
            f"{row['PSNR_T2']:.3f} | {row['Avg']:.3f} | **{row['Fgt']:.3f}** |"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]).resolve()
    payload = json.loads((out_dir / "metrics_stage1.json").read_text())
    meta = _meta(out_dir / "run_meta.txt")
    go = payload.get("go_kill", {})
    rows = payload.get("rows", [])
    agg = payload.get("aggregate_fgt", [])
    avg_app = payload.get("appendix_avg", [])
    failed = payload.get("failed_cells", [])
    seeds = sorted({int(r["seed"]) for r in rows})

    agg_lines = [
        "| arm | N_buf | n | Fgt mean | Fgt std | Fgt sem |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rec in agg:
        agg_lines.append(
            f"| {rec['arm']} | {rec['N_buf']} | {rec['n_seeds']} | "
            f"**{rec['Fgt_mean']:.3f}** | {rec['Fgt_std']:.3f} | {rec['Fgt_sem']:.3f} |"
        )

    avg_lines = [
        "| arm | N_buf | n | Avg mean | Avg std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for rec in avg_app:
        avg_lines.append(
            f"| {rec['arm']} | {rec['N_buf']} | {rec['n_seeds']} | "
            f"{rec['Avg_mean']:.3f} | {rec['Avg_std']:.3f} |"
        )

    seed_blocks = "\n\n".join(_seed_table(rows, s) for s in seeds)

    per_seed_lines = [
        "| seed | EI better / N points | majority |",
        "| ---: | --- | --- |",
    ]
    for rec in go.get("per_seed", []):
        per_seed_lines.append(
            f"| {rec['seed']} | {rec['n_EI_better_than_MC']} / {rec['n_points']} | "
            f"{'yes' if rec['majority'] else 'no'} |"
        )

    failed_txt = "none"
    if failed:
        failed_txt = "; ".join(
            f"seed={f.get('seed')} N_buf={f.get('N_buf')}: {f.get('error')}"
            for f in failed
        )

    body = f"""# stage-1 cross-IP matrix — cross-IP continual

**{go.get('verdict', 'UNKNOWN')}**. Stream E. Unsupervised MC/EI. `N_buf ∈ {{1,2,4,8}}`, seeds `{{1,2,3}}`. Not 16. Not same-MRI.

T1 MRI knee Gaussian/Cartesian 4× (`deepinv.physics.MRI`) → T2 `deepinv.physics.Tomography` `n_angles=40` (int). Not TomographyWithAstra. Not HQ SupLoss.

| | |
| --- | --- |
| label | {payload.get('label') or meta.get('label')} |
| device | {payload.get('device') or meta.get('device')} |
| wall (s) | {meta.get('elapsed_sec', payload.get('elapsed_sec'))} |
| tiny | {payload.get('tiny')} |
| supervised | {payload.get('supervised')} |
| unsupervised | {payload.get('unsupervised')} |
| cross_ip | {payload.get('cross_ip')} |
| physics_class_t1 | `{payload.get('physics_class_t1')}` |
| physics_class_t2 | `{payload.get('physics_class_t2')}` |
| ct_n_angles | {payload.get('ct_n_angles')} |
| n_train / n_eval | {payload.get('n_train')} / {payload.get('n_eval')} (held-out) |
| epochs T1 / T2 | {payload.get('epochs_t1')} / {payload.get('epochs_t2')} |
| N_buf | {payload.get('N_buf_grid')} |
| seeds | {payload.get('seeds')} |
| failed cells | {failed_txt} |
| Fgt | {payload.get('fgt_definition') or payload.get('fgt_formula')} |

Replay rebuilds `deepinv.physics.MRI` from the stored mask (`make_mri_physics`). Never feeds 128 MRI into 64 CT Tomography.

## Per-seed tables (after T2)

{seed_blocks}

## Aggregate Fgt vs N_buf (mean ± std across seeds)

Primary metric. Lower Fgt is better.

{chr(10).join(agg_lines)}

## Stage-1 go/kill

- **Go:** ≥2/3 seeds where on a **majority** of N_buf points Fgt(ER+EI) < Fgt(ER+MC).
- **Kill:** after multi-seed, EI not stably better than MC on Fgt.

{chr(10).join(per_seed_lines)}

**Readout: {go.get('verdict')}.** `{go.get('reason')}`.

## Appendix: Avg (secondary, not used for go/kill)

{chr(10).join(avg_lines)}

Files: `metrics_stage1.json`, `metrics_stage1.csv`, per-cell JSON under `seed*/nbuf*/`, `run_meta.txt`, `run.log`.
"""
    (out_dir / "README.md").write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
