#!/usr/bin/env bash
# Gate B: Fine-tune-only, same-knee, T1 Gaussian Cartesian 4× → T2 RandomMaskGenerator 8×.
# One schedule try. Not a three-arm grid. Not same-family Cartesian 4×→8×.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

DEVICE="cpu"
if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cuda"
fi

OUT="${1:-${ROOT}/recorded_smoke/stage0b_gate_B}"
T1="${2:-80}"
T2="${3:-160}"
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 -m cvpr_stage0_er_ei --unit finetune --no-tiny --seed 1 --gate B \
  --n-buf 4 --n-train 6 --n-eval 2 \
  --t1-accel 4 --t2-accel 8 \
  --t1-mask-family gaussian --t2-mask-family random \
  --t1-anatomy knee --t2-anatomy knee \
  --epochs-t1 "${T1}" --epochs-t2 "${T2}" \
  --device "${DEVICE}" \
  --out "${OUT}" 2>&1 | tee "${OUT}/run.log"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "gate=B"
  echo "label=same-knee mask-family (not Cartesian 4x->8x fallback)"
  echo "device=${DEVICE}"
  echo "arm=finetune"
  echo "n_train=6"
  echo "n_eval=2"
  echo "n_buf=4"
  echo "t1_anatomy=knee"
  echo "t2_anatomy=knee"
  echo "t1_mask_family=gaussian"
  echo "t2_mask_family=random"
  echo "mask_generator_t1=deepinv.physics.generator.GaussianMaskGenerator"
  echo "mask_generator_t2=deepinv.physics.generator.RandomMaskGenerator"
  echo "t1_accel=4"
  echo "t2_accel=8"
  echo "epochs_t1=${T1}"
  echo "epochs_t2=${T2}"
  echo "tiny=false"
  echo "start_utc=${START_UTC}"
  echo "end_utc=${END_UTC}"
  echo "elapsed_sec=${ELAPSED}"
} | tee "${OUT}/run_meta.txt"
