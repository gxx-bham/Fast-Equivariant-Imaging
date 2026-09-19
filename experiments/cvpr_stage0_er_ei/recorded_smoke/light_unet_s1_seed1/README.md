# Light-UNet single-seed direction smoke

Stream E three-arm cell on **`deepinv.models.UNet(scales=3)`** wrapped by `ArtifactRemoval(mode=adjoint)`. **Not MoDL.** Separate from `recorded_smoke/stage1_crossip_grid/`. Direction only — **not** a formal stage-1 GO/KILL.

## Backbone fingerprint

- class: `deepinv.models.UNet`
- ctor kwargs: `{"batch_norm": true, "bias": true, "cat": true, "circular_padding": false, "in_channels": 2, "out_channels": 2, "residual": true, "scales": 3}`
- wrapper: `deepinv.models.ArtifactRemoval`
- wrapper kwargs: `{"mode": "adjoint"}`
- runtime: `ArtifactRemoval` / denoiser `UNet`
- n_parameters: `2070082` (runtime; top-level payload fingerprint is cfg-only)

## Protocol

- T1: `deepinv.physics.MRI` knee Gaussian Cartesian 4×
- T2: `deepinv.physics.Tomography` n_angles=40, img_width=64
- unsupervised `MCLoss() + EILoss(Rotate(n_trans=4))` (n_trans not swept)
- ER+EI buffer: MC+EI; MRI replay via `make_mri_physics(stored mask)`
- N_buf=1, seed=1, n_train=8, n_eval=2, epochs T1=80 / T2=160
- Fgt = after_T1.PSNR_T1 − after_T2.PSNR_T1 on T1 MRI test only

## Fgt table (seed=1, N_buf=1)

| arm | PSNR_T1 | PSNR_T2 | Avg | Fgt |
| --- | ---: | ---: | ---: | ---: |
| Fine-tune | -0.133 | 12.209 | 6.038 | **25.171** |
| ER+MC | 17.850 | 11.842 | 14.846 | **7.188** |
| ER+EI | 24.191 | 14.742 | 19.467 | **0.847** |

## Direction (not formal S1 GO)

- Fgt(ER+EI) < Fgt(ER+MC): **True**
- device: `cpu`
- elapsed_sec: `1911.3324432373047`
- n_distinct_ya: `1`
- physics T2: `deepinv.physics.Tomography` n_angles=`40`
