Fine-tune-only forgetting gate, try 2 (shorter T1, longer T2).

| | |
| --- | --- |
| device | cpu |
| wall | 559 s |
| n_train / n_eval | 6 / 2 |
| epochs T1 → T2 | 25 → 200 |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 22.229 | 21.022 | — |
| after T2 | 24.660 | 22.182 | **−2.431** |

FAIL: Fgt more negative than try 1. T1 was undertrained so T2 still raised T1 PSNR.
