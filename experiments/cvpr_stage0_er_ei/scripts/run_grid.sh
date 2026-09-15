#!/usr/bin/env bash
# Full N_buf ∈ {1, 4} go/kill grid. Longer than the N_buf=1 smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python3 -m cvpr_stage0_er_ei --unit grid --tiny --seed 1 \
  --arms finetune,er_mc,er_ei --device cpu \
  --out "${ROOT}/recorded_smoke/seed1_grid"
