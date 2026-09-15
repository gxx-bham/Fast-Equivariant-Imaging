#!/usr/bin/env bash
# Single command: all three arms at N_buf=1, seed=1, CPU tiny smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python3 -m cvpr_stage0_er_ei --unit nbuf1 --tiny --seed 1 --n-buf 1 \
  --arms finetune,er_mc,er_ei --device cpu \
  --out "${ROOT}/recorded_smoke/seed1_nbuf1"
