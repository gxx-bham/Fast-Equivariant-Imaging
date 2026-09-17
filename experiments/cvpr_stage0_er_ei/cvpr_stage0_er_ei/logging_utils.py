"""CSV/JSON logging with the frozen columns: arm, N_buf, seed, PSNR_T1, PSNR_T2, Avg, Fgt."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ARM_LABELS, CSV_COLUMNS, CLEAR_POSITIVE_FGT


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


def distinct_slot_proof(buffer_report: Mapping[str, Any]) -> dict[str, Any]:
    """Compact proof that the buffer stored distinct past (y, A) slots."""
    slots = []
    for rec in buffer_report.get("slots", []):
        slots.append(
            {
                "slot": rec.get("slot"),
                "y_id": rec.get("y_id"),
                "mask_id": rec.get("mask_id"),
                "accel": rec.get("accel"),
                "task": rec.get("task"),
            }
        )
    return {
        "N_buf": buffer_report.get("N_buf"),
        "n_distinct_ya": buffer_report.get("n_distinct_ya"),
        "n_distinct_A": buffer_report.get("n_distinct_A"),
        "n_distinct_y": buffer_report.get("n_distinct_y"),
        "holds_t1_A": buffer_report.get("holds_t1_A"),
        "full_distinct": buffer_report.get("full_distinct"),
        "accels": buffer_report.get("accels"),
        "slots": slots,
    }


def finetune_forgetting_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_fgt: float = CLEAR_POSITIVE_FGT,
) -> dict[str, Any]:
    """PASS iff Fine-tune Fgt is clearly positive (T1 PSNR drop after T2)."""
    fgt_val = None
    for row in rows:
        if str(row.get("arm")) in ("Fine-tune", "finetune"):
            fgt_val = float(row["Fgt"])
            break
    if fgt_val is None and rows:
        fgt_val = float(rows[0]["Fgt"])
    if fgt_val is None:
        return {
            "verdict": "FAIL",
            "Fgt": None,
            "min_fgt": float(min_fgt),
            "reason": "No Fine-tune Fgt row to score.",
        }
    passed = fgt_val > float(min_fgt)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "Fgt": fgt_val,
        "min_fgt": float(min_fgt),
        "clearly_positive": passed,
        "reason": (
            f"Fine-tune Fgt={fgt_val:.4f} {'>' if passed else '<='} {min_fgt} "
            "(need a clear T1 PSNR drop after T2)."
        ),
    }


def aggregate_fgt_vs_nbuf(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± std / sem of Fgt across seeds for each (arm, N_buf)."""
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (str(row["arm"]), int(row["N_buf"]))
        groups.setdefault(key, []).append(float(row["Fgt"]))
    out: list[dict[str, Any]] = []
    for arm, n_buf in sorted(groups, key=lambda k: (k[1], k[0])):
        vals = groups[(arm, n_buf)]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        std = var**0.5
        sem = std / (n**0.5) if n else 0.0
        out.append(
            {
                "arm": arm,
                "N_buf": n_buf,
                "n_seeds": n,
                "Fgt_mean": mean,
                "Fgt_std": std,
                "Fgt_sem": sem,
                "Fgt_values": vals,
            }
        )
    return out


def aggregate_avg_vs_nbuf(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Appendix only: mean ± std of Avg across seeds."""
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (str(row["arm"]), int(row["N_buf"]))
        groups.setdefault(key, []).append(float(row["Avg"]))
    out: list[dict[str, Any]] = []
    for arm, n_buf in sorted(groups, key=lambda k: (k[1], k[0])):
        vals = groups[(arm, n_buf)]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        std = var**0.5
        out.append(
            {
                "arm": arm,
                "N_buf": n_buf,
                "n_seeds": n,
                "Avg_mean": mean,
                "Avg_std": std,
                "Avg_values": vals,
            }
        )
    return out


def stage1_go_kill(
    rows: Iterable[Mapping[str, Any]],
    *,
    tiny: bool = False,
    required_seeds: Sequence[int] = (1, 2, 3),
    required_nbufs: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Stage-1 go/kill on Fgt (lower is better).

    Go: ≥2/3 seeds where on a majority of N_buf points Fgt(ER+EI) < Fgt(ER+MC).
    Kill: after multi-seed, EI not stably better than MC on Fgt.
    """
    required_seeds = tuple(int(s) for s in required_seeds)
    required_nbufs = tuple(int(n) for n in required_nbufs)
    by_key: dict[tuple[str, int, int], float] = {}
    seeds: set[int] = set()
    nbufs: set[int] = set()
    for row in rows:
        arm = str(row["arm"])
        n_buf = int(row["N_buf"])
        seed = int(row["seed"])
        seeds.add(seed)
        nbufs.add(n_buf)
        by_key[(arm, n_buf, seed)] = float(row["Fgt"])

    per_seed: list[dict[str, Any]] = []
    majority_seeds: list[int] = []
    for seed in sorted(seeds):
        points = []
        n_better = 0
        for n_buf in sorted(nbufs):
            ei = by_key.get(("ER+EI", n_buf, seed))
            mc = by_key.get(("ER+MC", n_buf, seed))
            if ei is None or mc is None:
                continue
            better = ei < mc
            points.append(
                {
                    "N_buf": n_buf,
                    "Fgt_ER+EI": ei,
                    "Fgt_ER+MC": mc,
                    "Fgt_Fine-tune": by_key.get(("Fine-tune", n_buf, seed)),
                    "ER+EI_better": better,
                    "delta_MC_minus_EI": mc - ei,
                }
            )
            if better:
                n_better += 1
        n_points = len(points)
        majority = n_points > 0 and n_better > (n_points / 2.0)
        if majority:
            majority_seeds.append(seed)
        per_seed.append(
            {
                "seed": seed,
                "n_points": n_points,
                "n_EI_better_than_MC": n_better,
                "majority": majority,
                "points": points,
            }
        )

    complete = set(seeds) >= set(required_seeds) and set(nbufs) >= set(required_nbufs)
    n_majority = len(majority_seeds)
    verdict = "INCONCLUSIVE"
    reason = "Need all seeds {1,2,3} and N_buf {1,2,4,8} before a stage-1 go/kill call."
    if complete:
        if n_majority >= 2:
            verdict = "GO"
            reason = (
                f"{n_majority}/3 seeds have Fgt(ER+EI)<Fgt(ER+MC) on a majority "
                f"of N_buf points (seeds {majority_seeds})."
            )
        else:
            verdict = "KILL"
            reason = (
                f"After multi-seed, ER+EI is not stably better than ER+MC on Fgt "
                f"({n_majority}/3 seeds with a majority of N points; need ≥2/3)."
            )

    result = {
        "verdict": verdict,
        "reason": reason,
        "rule": (
            "Go: ≥2/3 seeds where on a majority of N_buf points "
            "Fgt(ER+EI)<Fgt(ER+MC). Kill: EI not stably better than MC on Fgt."
        ),
        "per_seed": per_seed,
        "majority_seeds": majority_seeds,
        "n_majority_seeds": n_majority,
        "seeds_observed": sorted(seeds),
        "n_buf_observed": sorted(nbufs),
        "grid_complete": complete,
        "tiny": bool(tiny),
        "primary": "Fgt on T1 MRI test (after_T1.PSNR_T1 - after_T2.PSNR_T1)",
        "secondary": "Avg is appendix only",
    }
    if tiny:
        result["would_have_been"] = result["verdict"]
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            "Tiny stub is a pipeline check, not a stage-1 go/kill decision. "
            + result["reason"]
        )
    return result
