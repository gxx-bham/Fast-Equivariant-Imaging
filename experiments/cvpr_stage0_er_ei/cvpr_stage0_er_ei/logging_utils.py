"""CSV/JSON logging with the frozen columns: arm, N_buf, seed, PSNR_T1, PSNR_T2, Avg, Fgt."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ARM_LABELS, CSV_COLUMNS


def fgt(psnr_t1_after_t1: float, psnr_t1_after_t2: float) -> float:
    """Standard CL forgetting on PSNR after T2.

    Fgt = R_T1(T1) - R_T1(T2), where R_i(j) is PSNR on task i after finishing task j.
    Positive Fgt means T1 PSNR dropped after learning T2. Lower Fgt is better.
    """
    return float(psnr_t1_after_t1) - float(psnr_t1_after_t2)


def avg_after_t2(psnr_t1_after_t2: float, psnr_t2_after_t2: float) -> float:
    return 0.5 * (float(psnr_t1_after_t2) + float(psnr_t2_after_t2))


def summary_row(
    arm: str,
    n_buf: int,
    seed: int,
    psnr_t1_after_t2: float,
    psnr_t2_after_t2: float,
    psnr_t1_after_t1: float,
) -> dict[str, Any]:
    psnr_t1 = float(psnr_t1_after_t2)
    psnr_t2 = float(psnr_t2_after_t2)
    return {
        "arm": ARM_LABELS.get(arm, arm),
        "N_buf": int(n_buf),
        "seed": int(seed),
        "PSNR_T1": psnr_t1,
        "PSNR_T2": psnr_t2,
        "Avg": avg_after_t2(psnr_t1, psnr_t2),
        "Fgt": fgt(psnr_t1_after_t1, psnr_t1),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def go_kill_readout(
    rows: Iterable[Mapping[str, Any]],
    *,
    tiny: bool = False,
) -> dict[str, Any]:
    """Apply frozen go/kill rules to after-T2 Fgt numbers.

    Go: ER+EI clearly better Fgt than ER+MC at N_buf=1 or 4.
    Kill: ER+EI <= ER+MC at both N, or only ties Fine-tune.
    Tiny stubs are pipeline checks and must not be labeled GO/KILL.
    """
    by_key: dict[tuple[str, int], float] = {}
    finetune_fgt: dict[int, float] = {}
    n_bufs: set[int] = set()
    for row in rows:
        arm = str(row["arm"])
        n_buf = int(row["N_buf"])
        n_bufs.add(n_buf)
        by_key[(arm, n_buf)] = float(row["Fgt"])
        if arm == "Fine-tune":
            finetune_fgt[n_buf] = float(row["Fgt"])

    comparisons = []
    ei_better_at: list[int] = []
    ei_not_better_at: list[int] = []
    for n_buf in sorted(n_bufs):
        ei = by_key.get(("ER+EI", n_buf))
        mc = by_key.get(("ER+MC", n_buf))
        ft = by_key.get(("Fine-tune", n_buf), finetune_fgt.get(n_buf))
        if ei is None or mc is None:
            continue
        # Lower Fgt is better (less forgetting).
        better_than_mc = ei < mc
        ties_only_finetune = ft is not None and ei >= ft and not better_than_mc
        comparisons.append(
            {
                "N_buf": n_buf,
                "Fgt_ER+EI": ei,
                "Fgt_ER+MC": mc,
                "Fgt_Fine-tune": ft,
                "ER+EI_better_Fgt_than_ER+MC": better_than_mc,
                "delta_Fgt_MC_minus_EI": mc - ei,
            }
        )
        if better_than_mc:
            ei_better_at.append(n_buf)
        else:
            ei_not_better_at.append(n_buf)

    observed = sorted(n_bufs)
    grid_complete = set(observed) >= {1, 4}
    verdict = "INCONCLUSIVE"
    reason = "Need ER+EI and ER+MC Fgt at the evaluated N_buf values."
    if comparisons:
        if ei_better_at and grid_complete and set(ei_better_at) & {1, 4}:
            verdict = "GO"
            reason = (
                "ER+EI has clearly lower Fgt than ER+MC at N_buf="
                + ",".join(str(n) for n in ei_better_at)
                + "."
            )
        elif ei_better_at and not grid_complete:
            verdict = "INCONCLUSIVE"
            reason = (
                "ER+EI lower Fgt than ER+MC at N_buf="
                + ",".join(str(n) for n in ei_better_at)
                + ", but the frozen rule needs the {1,4} grid before a go/kill call."
            )
        elif grid_complete and not ei_better_at:
            verdict = "KILL"
            reason = "ER+EI Fgt is not better than ER+MC at both N_buf=1 and N_buf=4."
        elif not ei_better_at:
            verdict = "INCONCLUSIVE"
            reason = (
                "ER+EI Fgt is not better than ER+MC on the N_buf values that were run; "
                "run N_buf=4 to apply the kill rule on both N."
            )
        if any(
            c["Fgt_Fine-tune"] is not None
            and c["Fgt_ER+EI"] >= c["Fgt_Fine-tune"]
            and not c["ER+EI_better_Fgt_than_ER+MC"]
            for c in comparisons
        ) and (grid_complete and not ei_better_at):
            verdict = "KILL"
            reason = "ER+EI does not beat ER+MC and only ties or loses to Fine-tune."

    result = {
        "verdict": verdict,
        "reason": reason,
        "comparisons": comparisons,
        "n_buf_observed": observed,
        "grid_complete": grid_complete,
        "tiny": bool(tiny),
        "note": (
            "Go/kill is defined on Fgt after T2 (lower is better). "
            "Tiny stubs must not be labeled GO/KILL."
        ),
    }
    if tiny:
        result["would_have_been"] = result["verdict"]
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            "Tiny stub (few iters) is a pipeline check, not a go/kill decision. "
            + result["reason"]
        )
    return result
