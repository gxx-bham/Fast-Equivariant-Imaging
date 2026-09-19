#!/usr/bin/env python3
"""Write a gate README from metrics JSON + run_meta.txt (called after a gate run)."""
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
    json_path = out_dir / "metrics_nbuf4_seed1.json"
    payload = json.loads(json_path.read_text())
    meta = _meta(out_dir / "run_meta.txt")
    cfg = payload.get("config", {})
    row = payload["rows"][0]
    detail = payload["details"][0]
    gate = payload.get("forgetting_gate", {})
    proof = payload.get("distinct_slot_proof", {})
    schedule = payload.get("schedule") or detail.get("schedule") or {}
    after_t1 = detail["after_T1"]
    after_t2 = detail["after_T2"]
    label = meta.get("label", "") or cfg.get("claim_scope", "")
    gate_id = cfg.get("gate") or meta.get("gate") or "?"
    if gate_id == "E" or bool(cfg.get("cross_ip")):
        scope = "cross-IP continual"
    elif gate_id == "F" or bool(cfg.get("supervised")):
        scope = "Step-3 supervised Fine-tune on stream D"
    elif gate_id == "D":
        scope = "domain+operator composite drift"
    elif bool(cfg.get("domain_incremental")):
        scope = "domain-incremental"
    else:
        scope = "same-knee mask-family"
    heading = f"# stage-0b Fine-tune forgetting gate {gate_id} — {scope}"
    verdict = gate.get("verdict", "UNKNOWN")
    t2_line = (
        f"{schedule.get('t2_anatomy')} Tomography n_angles={payload.get('ct_n_angles', schedule.get('ct_n_angles'))} "
        f"img={schedule.get('t2_img_size')} · `{payload.get('physics_class_t2') or schedule.get('physics_class_t2')}` · {schedule.get('epochs_t2')} epochs"
        if gate_id == "E" or bool(cfg.get("cross_ip"))
        else (
            f"{schedule.get('t2_anatomy')} {schedule.get('t2_mask_family')} {schedule.get('t2_accel')}× · "
            f"`{schedule.get('mask_generator_t2')}` · {schedule.get('epochs_t2')} epochs"
        )
    )
    extra_rows = ""
    if gate_id == "E" or bool(cfg.get("cross_ip")):
        extra_rows = f"""| cross_ip | {payload.get("cross_ip", cfg.get("cross_ip"))} |
| physics_class_t1 | `{payload.get("physics_class_t1") or schedule.get("physics_class_t1")}` |
| physics_class_t2 | `{payload.get("physics_class_t2") or schedule.get("physics_class_t2")}` |
| ct_n_angles | {payload.get("ct_n_angles", schedule.get("ct_n_angles"))} |
| Fgt definition | {payload.get("fgt_definition") or payload.get("fgt_formula")} |
"""
    body = f"""{heading}

**{verdict}**. Fine-tune only. One schedule try. Do **not** start the three-arm grid from this file.

{label}

| | |
| --- | --- |
| device | {payload.get("device") or cfg.get("device")} |
| supervised | {payload.get("supervised", cfg.get("supervised"))} |
| train_loss | {cfg.get("losses_current")} |
| wall (s) | {meta.get("elapsed_sec", payload.get("elapsed_sec"))} |
| n_train / n_eval | {cfg.get("n_train")} / {cfg.get("n_eval")} (held-out) |
| N_buf fill | {proof.get("n_distinct_ya")} distinct T1 (y,A); holds_t1_A={proof.get("holds_t1_A")} |
| T1 | {schedule.get("t1_anatomy")} {schedule.get("t1_mask_family")} {schedule.get("t1_accel")}× · `{schedule.get("mask_generator_t1")}` · {schedule.get("epochs_t1")} epochs |
| T2 | {t2_line} |
{extra_rows}| Fgt threshold | {gate.get("min_fgt")} (clearly positive = T1 PSNR drop) |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | {after_t1["PSNR_T1"]:.3f} | {after_t1["PSNR_T2"]:.3f} | — |
| after T2 | {after_t2["PSNR_T1"]:.3f} | {after_t2["PSNR_T2"]:.3f} | **{row["Fgt"]:.3f}** |

Gate readout: `{gate.get("reason")}`.

Distinct-slot proof: `n_distinct_ya={proof.get("n_distinct_ya")}`, `n_distinct_A={proof.get("n_distinct_A")}`, slots in `metrics_nbuf4_seed1.json`.
"""
    (out_dir / "README.md").write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
