#!/usr/bin/env bash
# Stage-1 cross-IP three-arm matrix on stream E.
# N_buf ∈ {1,2,4,8} × seeds {1,2,3} × Fine-tune | ER+MC | ER+EI.
# T1 MRI knee Gaussian Cartesian 4× → T2 deepinv.physics.Tomography (angles=40 int).
# Unsupervised MC / MC+EI on buffer. NOT HQ SupLoss. Not 16. Not same-MRI.
# Demo-scale epochs match Gate E / stage-0 E grid (80 / 160), not the tiny stub.
# Resume-safe: existing per-cell JSON is skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PATH="${HOME}/.local/bin:${PATH}"

DEVICE="cpu"
if python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cuda"
fi

OUT="${1:-${ROOT}/recorded_smoke/stage1_crossip_grid}"
T1="${2:-80}"
T2="${3:-160}"
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 -m cvpr_stage0_er_ei --unit stage1 --no-tiny --cross-ip \
  --seeds 1,2,3 \
  --n-train 8 --n-eval 2 \
  --t1-accel 4 --t1-mask-family gaussian --t1-anatomy knee \
  --arms finetune,er_mc,er_ei \
  --epochs-t1 "${T1}" --epochs-t2 "${T2}" \
  --device "${DEVICE}" \
  --out "${OUT}" 2>&1 | tee -a "${OUT}/run.log"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "label=cross-IP continual"
  echo "stage=stage-1"
  echo "supervised=false"
  echo "unsupervised=true"
  echo "cross_ip=true"
  echo "physics_class_t1=deepinv.physics.MRI"
  echo "physics_class_t2=deepinv.physics.Tomography"
  echo "ct_n_angles=40"
  echo "n_buf_grid=1,2,4,8"
  echo "seeds=1,2,3"
  echo "arms=finetune,er_mc,er_ei"
  echo "stream=T1 knee Gaussian Cartesian 4x -> T2 CT Tomography 40-view 64x64"
  echo "fgt=after_T1.PSNR_T1 - after_T2.PSNR_T1 (T1 MRI test only; lower is better)"
  echo "device=${DEVICE}"
  echo "n_train=8"
  echo "n_eval=2"
  echo "epochs_t1=${T1}"
  echo "epochs_t2=${T2}"
  echo "tiny=false"
  echo "start_utc=${START_UTC}"
  echo "end_utc=${END_UTC}"
  echo "elapsed_sec=${ELAPSED}"
} | tee "${OUT}/run_meta.txt"

python3 "${ROOT}/scripts/write_stage1_readme.py" "${OUT}"
