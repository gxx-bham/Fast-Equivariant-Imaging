# stage-0b Fine-tune forgetting gate E — cross-IP continual

**PASS**. Fine-tune only. One schedule try. Do **not** start the three-arm grid from this file.

cross-IP continual

| | |
| --- | --- |
| device | cpu |
| supervised | False |
| train_loss | MCLoss() + EILoss(Rotate(n_trans=4)) |
| wall (s) | 1150 |
| n_train / n_eval | 6 / 2 (held-out) |
| N_buf fill | 4 distinct T1 (y,A); holds_t1_A=True |
| T1 | knee gaussian 4× · `deepinv.physics.generator.GaussianMaskGenerator` · 80 epochs |
| T2 | ct100 Tomography n_angles=40 img=64 · `deepinv.physics.Tomography` · 160 epochs |
| cross_ip | True |
| physics_class_t1 | `deepinv.physics.MRI` |
| physics_class_t2 | `deepinv.physics.Tomography` |
| ct_n_angles | 40 |
| Fgt definition | Fgt = after_T1.PSNR_T1 - after_T2.PSNR_T1 on the T1 MRI test set only; T2 CT PSNR is logged separately and is not used in Fgt. Avg = 0.5 * (PSNR_T1_after_T2 + PSNR_T2_after_T2). CSV PSNR_T1/PSNR_T2 are after T2. |
| Fgt threshold | 0.2 (clearly positive = T1 PSNR drop) |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 24.746 | 19.410 | — |
| after T2 | 22.169 | 28.353 | **2.577** |

Gate readout: `Fine-tune Fgt=2.5765 > 0.2 (need a clear T1 PSNR drop after T2).`.

Distinct-slot proof: `n_distinct_ya=4`, `n_distinct_A=4`, slots in `metrics_nbuf4_seed1.json`.

Exact Fgt = **+2.577 dB** (`after_T1.PSNR_T1` 24.746 − `after_T2.PSNR_T1` 22.169) on the **T1 MRI test set only**. Clearly > 0.2. Physics = `deepinv.physics.Tomography` with **n_angles=40** (not TomographyWithAstra). Unsupervised MC+EI; `supervised=false`.

**Stop.** Wait for room go-ahead before Fine-tune | ER+MC | ER+EI × `N_buf∈{1,4}`. Do not retune angles/datasets.

`superseded_angles20/` is the 0.3.5-tag 20-view try (Fgt=+2.290); Gap Scout froze 40 while that run was in flight. Official readout is this 40-view file.
