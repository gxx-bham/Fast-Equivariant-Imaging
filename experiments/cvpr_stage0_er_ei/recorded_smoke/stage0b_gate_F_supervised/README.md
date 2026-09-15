# stage-0b Fine-tune forgetting gate F — Step-3 supervised Fine-tune on stream D

**PASS**. Fine-tune only. One schedule try. Do **not** start the three-arm grid from this file.

Step-3 supervised Fine-tune on stream D (domain+operator composite); HQ SupLoss, MC/EI off

| | |
| --- | --- |
| device | cpu |
| supervised | True |
| train_loss | deepinv.loss.SupLoss (HQ MSE; MC/EI off) |
| wall (s) | 135 |
| n_train / n_eval | 6 / 2 (held-out) |
| N_buf fill | 4 distinct T1 (y,A); holds_t1_A=True |
| T1 | knee gaussian 4× · `deepinv.physics.generator.GaussianMaskGenerator` · 80 epochs |
| T2 | brain random 8× · `deepinv.physics.generator.RandomMaskGenerator` · 160 epochs |
| Fgt threshold | 0.2 (clearly positive = T1 PSNR drop) |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 25.848 | 28.240 | — |
| after T2 | 22.368 | 28.918 | **3.480** |

Gate readout: `Fine-tune Fgt=3.4800 > 0.2 (need a clear T1 PSNR drop after T2).`.

Distinct-slot proof: `n_distinct_ya=4`, `n_distinct_A=4`, slots in `metrics_nbuf4_seed1.json`.
