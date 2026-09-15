#!/usr/bin/env bash
# Full N_buf ∈ {1, 4} go/kill grid at demo-scale epochs (not the few-iter stub).
# deepinv MRI EI demo from-scratch note: 150 epochs. Mini set = 2 slices, batch 1
# → 300 steps/task. GPU if torch.cuda.is_available(), else CPU (labeled in logs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="cpu"
if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cuda"
fi

OUT="${ROOT}/recorded_smoke/seed1_grid"
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 -m cvpr_stage0_er_ei --unit grid --no-tiny --epochs 150 --seed 1 \
  --arms finetune,er_mc,er_ei --device "${DEVICE}" \
  --out "${OUT}" 2>&1 | tee "${OUT}/run.log"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "device=${DEVICE}"
  echo "epochs=150"
  echo "tiny=false"
  echo "seed=1"
  echo "n_buf_grid=1,4"
  echo "start_utc=${START_UTC}"
  echo "end_utc=${END_UTC}"
  echo "elapsed_sec=${ELAPSED}"
} | tee "${OUT}/run_meta.txt"
