#!/usr/bin/env bash
# Light-backbone single-seed direction smoke (NOT MoDL stage-1).
# Stream E: T1 MRI knee Gaussian 4× → T2 Tomography n_angles=40, unsupervised.
# Backbone: deepinv.models.UNet(scales=3) + ArtifactRemoval(mode=adjoint).
# Arms: Fine-tune | ER+MC | ER+EI. N_buf=1 seed=1 only. Do not write into
# recorded_smoke/stage1_crossip_grid/.
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

OUT="${1:-${ROOT}/recorded_smoke/light_unet_s1_seed1}"
T1="${2:-80}"
T2="${3:-160}"
if [[ "${OUT}" == *stage1_crossip_grid* ]]; then
  echo "refusing to write light-UNet smoke into MoDL stage1_crossip_grid" >&2
  exit 1
fi
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 -m cvpr_stage0_er_ei --unit nbuf1 --no-tiny --cross-ip \
  --backbone unet \
  --seed 1 --n-buf 1 \
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
  echo "label=light-UNet single-seed direction smoke"
  echo "backbone=deepinv.models.UNet"
  echo "backbone_ctor_kwargs=in_channels=2,out_channels=2,residual=True,circular_padding=False,cat=True,bias=True,batch_norm=True,scales=3"
  echo "wrapper=deepinv.models.ArtifactRemoval"
  echo "wrapper_ctor_kwargs=mode=adjoint"
  echo "supervised=false"
  echo "unsupervised=true"
  echo "cross_ip=true"
  echo "physics_class_t1=deepinv.physics.MRI"
  echo "physics_class_t2=deepinv.physics.Tomography"
  echo "ct_n_angles=40"
  echo "n_buf=1"
  echo "seed=1"
  echo "arms=finetune,er_mc,er_ei"
  echo "ei=EILoss(Rotate(n_trans=4))"
  echo "stream=T1 knee Gaussian Cartesian 4x -> T2 CT Tomography 40-view 64x64"
  echo "fgt=after_T1.PSNR_T1 - after_T2.PSNR_T1 (T1 MRI test only; lower is better)"
  echo "direction_only=true"
  echo "formal_stage1_go_kill=false"
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

python3 "${ROOT}/scripts/write_light_unet_readme.py" "${OUT}"
