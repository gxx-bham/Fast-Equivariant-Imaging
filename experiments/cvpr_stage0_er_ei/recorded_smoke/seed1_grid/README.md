# seed=1 go/kill grid (demo-scale)

Not the few-iter stub. Same 128×128 mini knee set, `MoDL` + MC/EI, T1 4× → T2 8×.

| | |
| --- | --- |
| device | **cpu** (`torch.cuda.is_available() == False`) |
| wall time | **1804 s** (~30.1 min); `2026-09-15T11:06:05Z` → `11:36:09Z` |
| seed | 1 (all cells) |
| epochs | 150 (`--no-tiny`); 2 slices, batch 1 → **300 steps/task** |
| N_buf | {1, 4} |
| arms | Fine-tune \| ER+MC \| ER+EI |

deepinv MRI EI demo from-scratch note is 150 epochs. This run uses that epoch count on the demo mini set (2 slices), not 900 images and not 320².

## After T2

`PSNR_T1` / `PSNR_T2` / `Avg` / `Fgt` are after T2. `Fgt = PSNR_T1_after_T1 − PSNR_T1_after_T2` (lower is better). After every T1 block, `PSNR_T1_after_T1 = 26.625`.

| arm | N_buf | PSNR_T1 | PSNR_T2 | Avg | Fgt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fine-tune | 1 | 28.458 | 26.224 | 27.341 | −1.833 |
| ER+MC | 1 | 29.472 | 26.640 | 28.056 | −2.847 |
| ER+EI | 1 | 29.190 | 26.554 | 27.872 | −2.565 |
| Fine-tune | 4 | 28.458 | 26.224 | 27.341 | −1.833 |
| ER+MC | 4 | 29.580 | 26.691 | 28.136 | −2.955 |
| ER+EI | 4 | 29.252 | 26.705 | 27.978 | −2.627 |

Fine-tune ignores the buffer; N_buf=1 and 4 Fine-tune rows are the same run protocol (same seed).

## Frozen go/kill

- **Go:** ER+EI clearly better Fgt than ER+MC at N_buf=1 **or** 4.
- **Kill:** ER+EI ≤ ER+MC on Fgt at **both** N, or ER+EI only ties Fine-tune.

ER+EI Fgt is **worse** than ER+MC at N_buf=1 (−2.565 vs −2.847) and at N_buf=4 (−2.627 vs −2.955). Deltas ≈ 0.28–0.33 dB. ER+EI **does** beat Fine-tune on Fgt at both N (not “only ties Fine-tune”).

**Readout: KILL.** Not GO.

Files: `metrics_grid.csv`, `metrics_grid.json`, per-N_buf JSON under `nbuf1/` and `nbuf4/`, `run_meta.txt`, `run.log`.

## Caveats (do not turn this into a GO)

- Fgt is **negative on every arm**: T1 PSNR still rose during T2. This is not a positive-forgetting stress case; it is a ranking of how much T1 PSNR moved while learning T2.
- Mini set has **2 unique slices**. N_buf=1 stored 1 pair; N_buf=4 stored **2** unique `(y, A)` (capacity 4, not filled).
- Train and eval are those same 2 slices. Loss at epoch 149 is ~0; PSNR ~26–29 is below the deepinv MRI EI demo’s reported ~37 PSNR on a larger set.
- One seed, CPU only. No extra foils, no 320², no N_buf=16.
