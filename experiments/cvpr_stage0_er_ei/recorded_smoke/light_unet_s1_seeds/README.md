# Light-UNet N_buf=1 seeds 1–3 (formal S1 at N_buf=1)

Stream E three-arm cells on **`deepinv.models.UNet(in_channels=2, out_channels=2, scales=3)`** wrapped by `ArtifactRemoval(mode=adjoint)`. **Not MoDL.** Separate from `recorded_smoke/stage1_crossip_grid/`. N_buf=1 only — **do not expand the N_buf grid**.

Code Reviewer formal GO is conditional on the confirmation line in [`CONFIRMATION.md`](CONFIRMATION.md).

## Backbone fingerprint

- class: `deepinv.models.UNet`
- ctor kwargs: `{"batch_norm": true, "bias": true, "cat": true, "circular_padding": false, "in_channels": 2, "out_channels": 2, "residual": true, "scales": 3}`
- wrapper: `deepinv.models.ArtifactRemoval`
- wrapper kwargs: `{"mode": "adjoint"}`
- runtime: `ArtifactRemoval` / denoiser `UNet`
- n_parameters: `2070082`

## Protocol

- T1: `deepinv.physics.MRI` knee Gaussian Cartesian 4×
- T2: `deepinv.physics.Tomography` n_angles=40, img_width=64
- unsupervised `MCLoss() + EILoss(Rotate(n_trans=4))` (n_trans not swept)
- ER+MC buffer: MC only; ER+EI buffer: MC+EI
- MRI replay via `make_mri_physics(stored mask)` (pre_T2_replay probe)
- N_buf=1, seeds {1,2,3}, n_train=8, n_eval=2, epochs T1=80 / T2=160
- Fgt = after_T1.PSNR_T1 − after_T2.PSNR_T1 on T1 MRI test only

## Fgt table (N_buf=1)

| seed | Fine-tune | ER+MC | ER+EI | EI < MC | Δ(MC−EI) |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | **25.171** | **7.188** | **0.847** | **True** | 6.341 |
| 2 | **12.034** | **0.601** | **0.425** | **True** | 0.176 |
| 3 | **14.083** | **2.948** | **0.726** | **True** | 2.222 |

## Mean ± std across seeds (N_buf=1)

| arm | n | Fgt mean | Fgt std | Fgt sem |
| --- | ---: | ---: | ---: | ---: |
| ER+EI | 3 | **0.666** | 0.217 | 0.125 |
| ER+MC | 3 | **3.579** | 3.339 | 1.927 |
| Fine-tune | 3 | **17.096** | 7.068 | 4.081 |

## Formal S1 go/kill (N_buf=1 only)

Rule: ≥2/3 seeds with Fgt(ER+EI) < Fgt(ER+MC). With a single N_buf point, majority-of-N is that one point. Not an N_buf∈{1,2,4,8} matrix.

- verdict: **GO**
- reason: 3/3 seeds have Fgt(ER+EI)<Fgt(ER+MC) on a majority of N_buf points (seeds [1, 2, 3]).
- majority seeds: `[1, 2, 3]` (3/3)
- seeds present: `[1, 2, 3]` missing=`[]`
- device: `cpu`

## Reviewer probes

Per-cell JSON must keep: `physics_class_t2` Tomography(40), `n_distinct_ya==1`, `replay_rebuilds_mri` via `make_mri_physics`, nonzero `BufferReplayLoss` on ER arms, backbone UNet+ArtifactRemoval.

- seed=1: reviewer ER+MC pass=`True` ER+EI pass=`True` n_distinct_ya=`1` T2=`deepinv.physics.Tomography` n_angles=`40` replay_mri=`True` BufferReplayLoss_nonzero MC/EI=`True`/`True` backbone=`deepinv.models.UNet`
- seed=2: reviewer ER+MC pass=`True` ER+EI pass=`True` n_distinct_ya=`1` T2=`deepinv.physics.Tomography` n_angles=`40` replay_mri=`True` BufferReplayLoss_nonzero MC/EI=`True`/`True` backbone=`deepinv.models.UNet`
- seed=3: reviewer ER+MC pass=`True` ER+EI pass=`True` n_distinct_ya=`1` T2=`deepinv.physics.Tomography` n_angles=`40` replay_mri=`True` BufferReplayLoss_nonzero MC/EI=`True`/`True` backbone=`deepinv.models.UNet`

