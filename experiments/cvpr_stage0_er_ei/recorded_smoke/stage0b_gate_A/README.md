# stage-0b Fine-tune forgetting gate A — domain-incremental

**FAIL**. Fine-tune only. One schedule try. Do **not** start the three-arm grid from this file.

domain-incremental knee->brain (mini RSS); not same-knee accel-only

| | |
| --- | --- |
| device | cpu |
| wall (s) | 577 |
| n_train / n_eval | 6 / 2 (held-out) |
| N_buf fill | 4 distinct T1 (y,A); holds_t1_A=True |
| T1 | knee gaussian 4× · `deepinv.physics.generator.GaussianMaskGenerator` · 80 epochs |
| T2 | brain gaussian 4× · `deepinv.physics.generator.GaussianMaskGenerator` · 160 epochs |
| Fgt threshold | 0.2 (clearly positive = T1 PSNR drop) |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 24.407 | 33.128 | — |
| after T2 | 25.491 | 39.501 | **-1.085** |

Gate readout: `Fine-tune Fgt=-1.0846 <= 0.2 (need a clear T1 PSNR drop after T2).`.

Distinct-slot proof: `n_distinct_ya=4`, `n_distinct_A=4`, slots in `metrics_nbuf4_seed1.json`.

T2 brain Fine-tune **raised** held-out knee T1 PSNR by 1.08 dB (positive transfer on the shared 4× Cartesian operator). Brain RSS eval PSNR is high because the mini brain slices are real-valued (imag=0) and stay at 4×. This is **domain-incremental**, not same-knee accel-only.
