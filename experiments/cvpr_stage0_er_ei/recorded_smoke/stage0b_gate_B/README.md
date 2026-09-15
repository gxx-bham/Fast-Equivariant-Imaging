# stage-0b Fine-tune forgetting gate B — same-knee mask-family

**FAIL**. Fine-tune only. One schedule try. Do **not** start the three-arm grid from this file.

same-knee mask-family (not Cartesian 4x->8x fallback)

| | |
| --- | --- |
| device | cpu |
| wall (s) | 604 |
| n_train / n_eval | 6 / 2 (held-out) |
| N_buf fill | 4 distinct T1 (y,A); holds_t1_A=True |
| T1 | knee gaussian 4× · `deepinv.physics.generator.GaussianMaskGenerator` · 80 epochs |
| T2 | knee random 8× · `deepinv.physics.generator.RandomMaskGenerator` · 160 epochs |
| Fgt threshold | 0.2 (clearly positive = T1 PSNR drop) |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 24.407 | 21.914 | — |
| after T2 | 24.326 | 21.701 | **0.081** |

Gate readout: `Fine-tune Fgt=0.0810 <= 0.2 (need a clear T1 PSNR drop after T2).`.

Distinct-slot proof: `n_distinct_ya=4`, `n_distinct_A=4`, slots in `metrics_nbuf4_seed1.json`.
