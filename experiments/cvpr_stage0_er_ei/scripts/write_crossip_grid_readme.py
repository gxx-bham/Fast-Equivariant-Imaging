#!/usr/bin/env python3
"""Write the cross-IP three-arm grid README from metrics_grid.json + run_meta.txt."""
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


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]).resolve()
    payload = json.loads((out_dir / "metrics_grid.json").read_text())
    meta = _meta(out_dir / "run_meta.txt")
    go = payload.get("go_kill", {})
    losses = payload.get("arm_losses") or {}
    rows = payload["rows"]
    table_lines = [
        "| arm | N_buf | PSNR_T1 | PSNR_T2 | Avg | Fgt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['arm']} | {row['N_buf']} | {row['PSNR_T1']:.3f} | "
            f"{row['PSNR_T2']:.3f} | {row['Avg']:.3f} | **{row['Fgt']:.3f}** |"
        )
    cmp_lines = []
    for c in go.get("comparisons", []):
        better = "yes" if c.get("ER+EI_better_Fgt_than_ER+MC") else "no"
        cmp_lines.append(
            f"| {c['N_buf']} | {c['Fgt_Fine-tune']:.3f} | {c['Fgt_ER+MC']:.3f} | "
            f"{c['Fgt_ER+EI']:.3f} | {c.get('delta_Fgt_MC_minus_EI', 0):+.3f} | {better} |"
        )
    body = f"""# stage-0b cross-IP three-arm grid — cross-IP continual

**{go.get('verdict', 'UNKNOWN')}**. Stream E. Unsupervised MC/EI. `N_buf ∈ {{1, 4}}` only (not 16).

T1 MRI knee Gaussian/Cartesian 4× (`deepinv.physics.MRI`) → T2 `deepinv.physics.Tomography` `n_angles=40` (int). Not TomographyWithAstra. Not HQ SupLoss.

| | |
| --- | --- |
| label | {payload.get('label') or meta.get('label')} |
| device | {payload.get('device') or meta.get('device')} |
| wall (s) | {meta.get('elapsed_sec', payload.get('elapsed_sec'))} |
| seed | {payload.get('seed')} |
| tiny | {payload.get('tiny')} |
| supervised | {payload.get('supervised')} |
| cross_ip | {payload.get('cross_ip')} |
| physics_class_t1 | `{payload.get('physics_class_t1')}` |
| physics_class_t2 | `{payload.get('physics_class_t2')}` |
| ct_n_angles | {payload.get('ct_n_angles')} |
| n_train / n_eval | {payload.get('n_train')} / {payload.get('n_eval')} (held-out) |
| epochs T1 / T2 | {payload.get('epochs_t1')} / {payload.get('epochs_t2')} |
| N_buf | {payload.get('N_buf_grid')} |
| Fine-tune loss | {losses.get('Fine-tune', '')} |
| ER+MC loss | {losses.get('ER+MC', '')} |
| ER+EI loss | {losses.get('ER+EI', '')} |
| Fgt | {payload.get('fgt_definition') or payload.get('fgt_formula')} |

## After T2

CSV `PSNR_T1` / `PSNR_T2` / `Avg` / `Fgt` are after T2. Fgt is on the **T1 MRI test set only** (lower is better). Fine-tune ignores the buffer; N_buf=1 and 4 Fine-tune rows share the same protocol (same seed).

{chr(10).join(table_lines)}

## Frozen go/kill

- **Go:** ER+EI clearly better Fgt than ER+MC at `N_buf=1` **or** `4` (ideally also better than Fine-tune).
- **Kill:** ER+EI not better than ER+MC on Fgt at **both** N, or ER+EI only ties Fine-tune.

| N_buf | Fgt Fine-tune | Fgt ER+MC | Fgt ER+EI | Δ (MC−EI) | EI better than MC |
| ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(cmp_lines)}

**Readout: {go.get('verdict')}.** `{go.get('reason')}`.

Files: `metrics_grid.csv`, `metrics_grid.json`, per-N_buf JSON under `nbuf1/` and `nbuf4/`, `run_meta.txt`, `run.log`.
"""
    (out_dir / "README.md").write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
