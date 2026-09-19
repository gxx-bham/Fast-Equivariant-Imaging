#!/usr/bin/env bash
# Cross-IP three-arm go/kill grid on stream E. N_buf ∈ {1, 4} only (not 16).
# T1 MRI knee Gaussian Cartesian 4× → T2 deepinv.physics.Tomography (angles=40 int).
# Unsupervised MC / MC+EI on buffer. NOT HQ SupLoss.
# Demo-scale epochs match Gate E (80 / 160), not the tiny stub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

DEVICE="cpu"
if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cuda"
fi

OUT="${1:-${ROOT}/recorded_smoke/stage0b_crossip_grid}"
T1="${2:-80}"
T2="${3:-160}"
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 -m cvpr_stage0_er_ei --unit grid --no-tiny --seed 1 --cross-ip \
  --n-train 6 --n-eval 2 \
  --t1-accel 4 --t1-mask-family gaussian --t1-anatomy knee \
  --arms finetune,er_mc,er_ei \
  --epochs-t1 "${T1}" --epochs-t2 "${T2}" \
  --device "${DEVICE}" \
  --out "${OUT}" 2>&1 | tee "${OUT}/run.log"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "label=cross-IP continual"
  echo "supervised=false"
  echo "cross_ip=true"
  echo "physics_class_t1=deepinv.physics.MRI"
  echo "physics_class_t2=deepinv.physics.Tomography"
  echo "ct_n_angles=40"
  echo "ct_img_size=64"
  echo "n_buf_grid=1,4"
  echo "arms=finetune,er_mc,er_ei"
  echo "arm_loss_finetune=current: MCLoss() + EILoss(Rotate(n_trans=4)); no buffer replay"
  echo "arm_loss_er_mc=current: MCLoss() + EILoss(Rotate(n_trans=4)); buffer: MCLoss only"
  echo "arm_loss_er_ei=current: MCLoss() + EILoss(Rotate(n_trans=4)); buffer: MCLoss() + EILoss(Rotate(n_trans=4))"
  echo "stream=T1 knee Gaussian Cartesian 4x -> T2 CT Tomography 40-view 64x64"
  echo "fgt=after_T1.PSNR_T1 - after_T2.PSNR_T1 (T1 MRI test only; lower is better)"
  echo "device=${DEVICE}"
  echo "n_train=6"
  echo "n_eval=2"
  echo "epochs_t1=${T1}"
  echo "epochs_t2=${T2}"
  echo "tiny=false"
  echo "start_utc=${START_UTC}"
  echo "end_utc=${END_UTC}"
  echo "elapsed_sec=${ELAPSED}"
} | tee "${OUT}/run_meta.txt"

python3 "${ROOT}/scripts/write_crossip_grid_readme.py" "${OUT}"
