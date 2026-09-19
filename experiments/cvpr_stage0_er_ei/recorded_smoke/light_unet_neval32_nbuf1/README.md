# Light-UNet HARDEN n_eval=32 (N_buf=1, seeds 1–3)

Same fingerprint as light-UNet S1 GO: **`UNet(in_channels=2, out_channels=2, scales=3)`** + `ArtifactRemoval(mode=adjoint)`. **Not MoDL.** n_train=8 unchanged; **n_eval=32** only. Do **not** merge with `recorded_smoke/light_unet_s1_seeds/` (n_eval=2) or `stage1_crossip_grid/`.

Confirmation line: [`CONFIRMATION.md`](CONFIRMATION.md).

## Backbone fingerprint

- class: `deepinv.models.UNet`
- ctor kwargs: `{"batch_norm": true, "bias": true, "cat": true, "circular_padding": false, "in_channels": 2, "out_channels": 2, "residual": true, "scales": 3}`
- wrapper: `deepinv.models.ArtifactRemoval`
- wrapper kwargs: `{"mode": "adjoint"}`
- runtime: `ArtifactRemoval` / denoiser `UNet`
- n_parameters: `2070082`

## Protocol

- T1: `deepinv.physics.MRI` knee Gaussian Cartesian 4× 128×128
- T2: `deepinv.physics.Tomography` n_angles=40, img_width=64
- unsupervised `MCLoss() + EILoss(Rotate(n_trans=4))` (n_trans not swept)
- ER+MC buffer: MC only; ER+EI buffer: MC+EI
- MRI replay via `make_mri_physics(stored mask)`
- N_buf=1, seeds {1,2,3}, n_train=8, n_eval=32, epochs T1=80 / T2=160
- Fgt = after_T1.PSNR_T1 − after_T2.PSNR_T1 on T1 MRI test only

## Fgt table (N_buf=1, n_eval=32)

| seed | Fine-tune | ER+MC | ER+EI | EI < MC | Δ(MC−EI) |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | **28.293** | **5.744** | **1.068** | **True** | 4.676 |
| 2 | **11.739** | **1.142** | **0.354** | **True** | 0.788 |
| 3 | **13.458** | **2.546** | **0.608** | **True** | 1.938 |

## Mean ± std across seeds (N_buf=1, n_eval=32)

| arm | n | Fgt mean | Fgt std | Fgt sem |
| --- | ---: | ---: | ---: | ---: |
| ER+EI | 3 | **0.677** | 0.362 | 0.209 |
| ER+MC | 3 | **3.144** | 2.358 | 1.362 |
| Fine-tune | 3 | **17.830** | 9.102 | 5.255 |

## HARDEN go/kill

Rule: GO if ≥2/3 seeds with Fgt(ER+EI)<Fgt(ER+MC) **and** mean(Fgt_MC−Fgt_EI)>0. Kill if ≥2/3 seeds EI does not win, or mean Δ≤0.

- verdict: **GO**
- reason: 3/3 seeds have Fgt(ER+EI)<Fgt(ER+MC) and mean(Fgt_MC−Fgt_EI)=2.468>0 (seeds [1, 2, 3]).
- win seeds: `[1, 2, 3]` (3/3)
- mean(Fgt_MC−Fgt_EI): `2.467511256535848`
- seeds present: `[1, 2, 3]` missing=`[]`
- device: `cpu`

## Reviewer probes

Per-cell JSON: `physics_class_t2` Tomography(40), `n_distinct_ya==1`, `replay_rebuilds_mri` via `make_mri_physics`, nonzero BufferReplayLoss on ER arms, UNet+ArtifactRemoval, n_eval=32.

- seed=1: reviewer ER+MC pass=`True` ER+EI pass=`True` n_distinct_ya=`1` n_eval=`32` T2=`deepinv.physics.Tomography` n_angles=`40` replay_mri=`True` BufferReplayLoss_nonzero MC/EI=`True`/`True` backbone=`deepinv.models.UNet`
- seed=2: reviewer ER+MC pass=`True` ER+EI pass=`True` n_distinct_ya=`1` n_eval=`32` T2=`deepinv.physics.Tomography` n_angles=`40` replay_mri=`True` BufferReplayLoss_nonzero MC/EI=`True`/`True` backbone=`deepinv.models.UNet`
- seed=3: reviewer ER+MC pass=`True` ER+EI pass=`True` n_distinct_ya=`1` n_eval=`32` T2=`deepinv.physics.Tomography` n_angles=`40` replay_mri=`True` BufferReplayLoss_nonzero MC/EI=`True`/`True` backbone=`deepinv.models.UNet`

