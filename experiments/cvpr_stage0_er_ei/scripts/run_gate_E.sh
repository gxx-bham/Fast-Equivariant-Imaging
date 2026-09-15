#!/usr/bin/env bash
# Gate E: Fine-tune-only, cross-IP continual. One try.
# T1 MRI knee Gaussian Cartesian 4× → T2 deepinv Tomography (physics-tour 20-view 64×64).
# Unsupervised MC+EI (same as three-arm current-task loss). NOT SupLoss/HQ.
# Do not start the three-arm grid from this script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

DEVICE="cpu"
if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cuda"
fi

OUT="${1:-${ROOT}/recorded_smoke/stage0b_gate_E}"
T1="${2:-80}"
T2="${3:-160}"
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 -m cvpr_stage0_er_ei --unit finetune --no-tiny --seed 1 --gate E --cross-ip \
  --n-buf 4 --n-train 6 --n-eval 2 \
  --t1-accel 4 --t1-mask-family gaussian --t1-anatomy knee \
  --epochs-t1 "${T1}" --epochs-t2 "${T2}" \
  --device "${DEVICE}" \
  --out "${OUT}" 2>&1 | tee "${OUT}/run.log"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "gate=E"
  echo "label=cross-IP continual"
  echo "supervised=false"
  echo "cross_ip=true"
  echo "train_loss=MCLoss() + EILoss(Rotate(n_trans=4))"
  echo "physics_class_t1=deepinv.physics.MRI"
  echo "physics_class_t2=deepinv.physics.Tomography"
  echo "ct_n_angles=20"
  echo "ct_img_size=64"
  echo "ct_demo_source=deepinv==0.3.5 examples/physics/demo_physics_tour.py (Tomography, angles=20, img_width=64)"
  echo "stream=T1 knee Gaussian Cartesian 4x -> T2 CT Tomography 20-view 64x64"
  echo "fgt=after_T1.PSNR_T1 - after_T2.PSNR_T1 (T1 MRI test only)"
  echo "device=${DEVICE}"
  echo "arm=finetune"
  echo "n_train=6"
  echo "n_eval=2"
  echo "n_buf=4"
  echo "epochs_t1=${T1}"
  echo "epochs_t2=${T2}"
  echo "tiny=false"
  echo "start_utc=${START_UTC}"
  echo "end_utc=${END_UTC}"
  echo "elapsed_sec=${ELAPSED}"
} | tee "${OUT}/run_meta.txt"
