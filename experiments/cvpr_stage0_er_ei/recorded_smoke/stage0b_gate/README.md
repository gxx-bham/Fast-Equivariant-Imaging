# stage-0b Fine-tune forgetting gate — FAIL

Stop here. Do **not** run the three-arm `{1,4}` grid.

Device: **cpu**. Seed 1. Same 4×→8× Gaussian Cartesian family. `n_train=6`, `n_eval=2` (held-out). Buffer fill on Fine-tune is unused for the loss but was 4 distinct T1 `(y,A)` on both tries.

## Tries (max 2)

| try | T1 epochs | T2 epochs | wall (s) | PSNR_T1 after T1 | PSNR_T1 after T2 | **Fgt** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 | 150 | 681 | 24.407 | 24.590 | **−0.184** |
| 2 | 25 | 200 | 559 | 22.229 | 24.660 | **−2.431** |

Gate PASS requires Fine-tune Fgt **clearly > 0** (T1 PSNR drop after T2). Both tries stayed **non-positive**.

## Why Fgt stayed ≤ 0

- Try 1: T1 was near a plateau on held-out 4× (~24.4 dB). Learning 8× for 150 epochs still nudged T1 PSNR **up** 0.18 dB. T2 eval PSNR did not rise (22.27 → 22.19).
- Try 2: shorter T1 left T1 undertrained; longer T2 then **improved** T1 by 2.43 dB (same failure mode as the old n=2 equal-epoch run).

So: stronger 8× Fine-tune on this mini 128×128 crop set does not produce a T1 PSNR drop. The 4× and 8× operators still share enough MoDL features that T2 updates help T1.

Logs: `try1/`, `try2/` (`metrics_nbuf4_seed1.json`, `run_meta.txt`, `run.log`).
