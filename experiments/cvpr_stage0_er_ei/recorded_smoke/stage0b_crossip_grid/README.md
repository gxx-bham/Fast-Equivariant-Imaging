# stage-0b cross-IP three-arm grid — cross-IP continual

**GO**. Stream E. Unsupervised MC/EI. `N_buf ∈ {1, 4}` only (not 16).

T1 MRI knee Gaussian/Cartesian 4× (`deepinv.physics.MRI`) → T2 `deepinv.physics.Tomography` `n_angles=40` (int). Not TomographyWithAstra. Not HQ SupLoss.

| | |
| --- | --- |
| label | cross-IP continual |
| device | cpu |
| wall (s) | 12595 |
| seed | 1 |
| tiny | False |
| supervised | False |
| cross_ip | True |
| physics_class_t1 | `deepinv.physics.MRI` |
| physics_class_t2 | `deepinv.physics.Tomography` |
| ct_n_angles | 40 |
| n_train / n_eval | 6 / 2 (held-out) |
| epochs T1 / T2 | 80 / 160 |
| N_buf | [1, 4] |
| Fine-tune loss | current: MCLoss() + EILoss(Rotate(n_trans=4)); no buffer replay |
| ER+MC loss | current: MCLoss() + EILoss(Rotate(n_trans=4)); buffer: MCLoss only |
| ER+EI loss | current: MCLoss() + EILoss(Rotate(n_trans=4)); buffer: MCLoss() + EILoss(Rotate(n_trans=4)) |
| Fgt | Fgt = after_T1.PSNR_T1 - after_T2.PSNR_T1 on the T1 MRI test set only; T2 CT PSNR is logged separately and is not used in Fgt. Avg = 0.5 * (PSNR_T1_after_T2 + PSNR_T2_after_T2). CSV PSNR_T1/PSNR_T2 are after T2. Lower Fgt is better. |

## After T2

CSV `PSNR_T1` / `PSNR_T2` / `Avg` / `Fgt` are after T2. Fgt is on the **T1 MRI test set only** (lower is better). Fine-tune ignores the buffer; N_buf=1 and 4 Fine-tune rows share the same protocol (same seed).

| arm | N_buf | PSNR_T1 | PSNR_T2 | Avg | Fgt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fine-tune | 1 | 22.169 | 28.353 | 25.261 | **2.577** |
| ER+MC | 1 | 24.430 | 27.271 | 25.851 | **0.315** |
| ER+EI | 1 | 24.709 | 25.389 | 25.049 | **0.036** |
| Fine-tune | 4 | 22.169 | 28.353 | 25.261 | **2.577** |
| ER+MC | 4 | 24.419 | 25.149 | 24.784 | **0.327** |
| ER+EI | 4 | 25.012 | 23.010 | 24.011 | **-0.266** |

## Frozen go/kill

- **Go:** ER+EI clearly better Fgt than ER+MC at `N_buf=1` **or** `4` (ideally also better than Fine-tune).
- **Kill:** ER+EI not better than ER+MC on Fgt at **both** N, or ER+EI only ties Fine-tune.

| N_buf | Fgt Fine-tune | Fgt ER+MC | Fgt ER+EI | Δ (MC−EI) | EI better than MC |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2.577 | 0.315 | 0.036 | +0.279 | yes |
| 4 | 2.577 | 0.327 | -0.266 | +0.593 | yes |

**Readout: GO.** `ER+EI has clearly lower Fgt than ER+MC at N_buf=1,4.`. ER+EI also beats Fine-tune on Fgt at both N. Avg is secondary (Fine-tune keeps a higher T2 CT PSNR).

Files: `metrics_grid.csv`, `metrics_grid.json`, per-N_buf JSON under `nbuf1/` and `nbuf4/`, `run_meta.txt`, `run.log`.
