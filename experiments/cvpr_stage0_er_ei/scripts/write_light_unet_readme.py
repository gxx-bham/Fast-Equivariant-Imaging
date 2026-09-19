#!/usr/bin/env python3
"""Write light-UNet N_buf=1 seed=1 direction-smoke README from metrics JSON."""
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
    payload = json.loads((out_dir / "metrics_nbuf1_seed1.json").read_text())
    meta = _meta(out_dir / "run_meta.txt")
    rows = payload.get("rows", [])
    cfg = payload.get("config", {})
    backbone = payload.get("backbone") or cfg.get("backbone") or {}
    by_arm = {str(r["arm"]): r for r in rows}
    ft = by_arm.get("Fine-tune", {})
    mc = by_arm.get("ER+MC", {})
    ei = by_arm.get("ER+EI", {})
    ei_lt_mc = None
    if "Fgt" in ei and "Fgt" in mc:
        ei_lt_mc = float(ei["Fgt"]) < float(mc["Fgt"])

    lines = [
        "# Light-UNet single-seed direction smoke",
        "",
        "Stream E three-arm cell on **`deepinv.models.UNet(scales=3)`** wrapped by "
        "`ArtifactRemoval(mode=adjoint)`. **Not MoDL.** Separate from "
        "`recorded_smoke/stage1_crossip_grid/`. Direction only — **not** a formal "
        "stage-1 GO/KILL.",
        "",
        "## Backbone fingerprint",
        "",
        f"- class: `{backbone.get('class', 'deepinv.models.UNet')}`",
        f"- ctor kwargs: `{json.dumps(backbone.get('ctor_kwargs', {}), sort_keys=True)}`",
        f"- wrapper: `{backbone.get('wrapper_class', 'deepinv.models.ArtifactRemoval')}`",
        f"- wrapper kwargs: `{json.dumps(backbone.get('wrapper_ctor_kwargs', {}), sort_keys=True)}`",
        f"- runtime: `{backbone.get('runtime_class')}` / denoiser `{backbone.get('runtime_denoiser_class')}`",
        f"- n_parameters: `{backbone.get('n_parameters')}`",
        "",
        "## Protocol",
        "",
        "- T1: `deepinv.physics.MRI` knee Gaussian Cartesian 4×",
        "- T2: `deepinv.physics.Tomography` n_angles=40, img_width=64",
        "- unsupervised `MCLoss() + EILoss(Rotate(n_trans=4))` (n_trans not swept)",
        "- ER+EI buffer: MC+EI; MRI replay via `make_mri_physics(stored mask)`",
        "- N_buf=1, seed=1, n_train=8, n_eval=2, epochs T1=80 / T2=160",
        "- Fgt = after_T1.PSNR_T1 − after_T2.PSNR_T1 on T1 MRI test only",
        "",
        "## Fgt table (seed=1, N_buf=1)",
        "",
        "| arm | PSNR_T1 | PSNR_T2 | Avg | Fgt |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("Fine-tune", "ER+MC", "ER+EI"):
        row = by_arm.get(name)
        if not row:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {row['PSNR_T1']:.3f} | {row['PSNR_T2']:.3f} | "
            f"{row['Avg']:.3f} | **{row['Fgt']:.3f}** |"
        )
    lines.extend(
        [
            "",
            "## Direction (not formal S1 GO)",
            "",
            (
                f"- Fgt(ER+EI) < Fgt(ER+MC): **{ei_lt_mc}**"
                if ei_lt_mc is not None
                else "- Fgt(ER+EI) vs ER+MC: incomplete"
            ),
            f"- device: `{payload.get('device') or meta.get('device')}`",
            f"- elapsed_sec: `{payload.get('elapsed_sec') or meta.get('elapsed_sec')}`",
            f"- n_distinct_ya: `{payload.get('n_distinct_ya')}`",
            f"- physics T2: `{payload.get('physics_class_t2')}` n_angles=`{payload.get('ct_n_angles')}`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
