# stage-0b Fine-tune forgetting gate D — domain+operator composite drift

**FAIL**. Fine-tune only. One schedule try. Do **not** start the three-arm grid from this file.

domain+operator composite drift (A+B stacked); not A-or-B retune

| | |
| --- | --- |
| device | cpu |
| wall (s) | 632 |
| n_train / n_eval | 6 / 2 (held-out) |
| N_buf fill | 4 distinct T1 (y,A); holds_t1_A=True |
| T1 | knee gaussian 4× · `deepinv.physics.generator.GaussianMaskGenerator` · 80 epochs |
| T2 | brain random 8× · `deepinv.physics.generator.RandomMaskGenerator` · 160 epochs |
| Fgt threshold | 0.2 (clearly positive = T1 PSNR drop) |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 24.407 | 28.947 | — |
| after T2 | 24.245 | 31.269 | **0.162** |

Gate readout: `Fine-tune Fgt=0.1618 <= 0.2 (need a clear T1 PSNR drop after T2).`.

Distinct-slot proof: `n_distinct_ya=4`, `n_distinct_A=4`, slots in `metrics_nbuf4_seed1.json`.
