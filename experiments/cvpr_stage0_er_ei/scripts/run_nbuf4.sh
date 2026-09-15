#!/usr/bin/env bash
# N_buf=4 grid cell (same three arms). Documented; not the default debug command.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python3 -m cvpr_stage0_er_ei --unit nbuf1 --tiny --seed 1 --n-buf 4 \
  --arms finetune,er_mc,er_ei --device cpu \
  --out "${ROOT}/recorded_smoke/seed1_nbuf4"
