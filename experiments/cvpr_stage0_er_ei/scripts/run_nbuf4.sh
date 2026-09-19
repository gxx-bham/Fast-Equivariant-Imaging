#!/usr/bin/env bash
# N_buf=4 cell. Must use --unit nbuf4 (never nbuf1) so N_buf cannot silently become 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python3 -m cvpr_stage0_er_ei --unit nbuf4 --n-buf 4 --tiny --seed 1 \
  --n-train 4 --n-eval 2 \
  --arms finetune,er_mc,er_ei --device cpu \
  --out "${ROOT}/recorded_smoke/seed1_nbuf4"
