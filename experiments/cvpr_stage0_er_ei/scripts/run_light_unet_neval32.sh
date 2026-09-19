#!/usr/bin/env bash
# HARDEN cut: light-UNet N_buf=1 seeds {1,2,3}, n_train=8, n_eval=32 (NEW).
# Same fingerprint as light-UNet S1 GO. NOT MoDL. Do not write into
# recorded_smoke/light_unet_s1_seeds/ (n_eval=2) or stage1_crossip_grid/.
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

OUT="${1:-${ROOT}/recorded_smoke/light_unet_neval32_nbuf1}"
T1="${2:-80}"
T2="${3:-160}"
SEEDS_CSV="${4:-1,2,3}"

case "${OUT}" in
  *stage1_crossip_grid*|*light_unet_s1_seeds*|*light_unet_s1_seed1*)
    echo "refusing to write HARDEN n_eval=32 into 8/2 or MoDL dirs: ${OUT}" >&2
    exit 1
    ;;
esac
mkdir -p "${OUT}"
START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

for seed in "${SEEDS[@]}"; do
  seed="$(echo "${seed}" | tr -d '[:space:]')"
  [[ -z "${seed}" ]] && continue
  cell="${OUT}/seed${seed}"
  json="${cell}/metrics_nbuf1_seed${seed}.json"
  mkdir -p "${cell}"
  if [[ -f "${json}" ]]; then
    echo "skip existing ${json}"
    continue
  fi
  echo "=== HARDEN light-UNet n_eval=32 N_buf=1 seed=${seed} backbone=unet device=${DEVICE} ==="
  python3 -m cvpr_stage0_er_ei --unit nbuf1 --no-tiny --cross-ip \
    --backbone unet \
    --seed "${seed}" --n-buf 1 \
    --n-train 8 --n-eval 32 \
    --t1-accel 4 --t1-mask-family gaussian --t1-anatomy knee \
    --arms finetune,er_mc,er_ei \
    --epochs-t1 "${T1}" --epochs-t2 "${T2}" \
    --device "${DEVICE}" \
    --out "${cell}" 2>&1 | tee -a "${OUT}/run.log" | tee -a "${cell}/run.log"
done

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
  echo "label=light-UNet HARDEN n_eval=32 N_buf=1 seeds 1-3"
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
  echo "seeds=${SEEDS_CSV}"
  echo "arms=finetune,er_mc,er_ei"
  echo "ei=EILoss(Rotate(n_trans=4))"
  echo "stream=T1 knee Gaussian Cartesian 4x 128x128 -> T2 CT Tomography 40-view 64x64"
  echo "fgt=after_T1.PSNR_T1 - after_T2.PSNR_T1 (T1 MRI test only; lower is better)"
  echo "n_train=8"
  echo "n_eval=32"
  echo "do_not_merge_with=recorded_smoke/light_unet_s1_seeds/"
  echo "not_modl=true"
  echo "expand_n_buf=false"
  echo "device=${DEVICE}"
  echo "epochs_t1=${T1}"
  echo "epochs_t2=${T2}"
  echo "tiny=false"
  echo "start_utc=${START_UTC}"
  echo "end_utc=${END_UTC}"
  echo "elapsed_sec=${ELAPSED}"
} | tee "${OUT}/run_meta.txt"

python3 "${ROOT}/scripts/write_light_unet_neval32_readme.py" "${OUT}"
