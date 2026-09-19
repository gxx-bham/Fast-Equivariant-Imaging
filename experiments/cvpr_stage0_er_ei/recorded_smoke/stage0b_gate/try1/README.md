Fine-tune-only forgetting gate, try 1.

| | |
| --- | --- |
| device | cpu |
| wall | 681 s |
| n_train / n_eval | 6 / 2 (held-out) |
| epochs T1 → T2 | 80 → 150 |
| N_buf fill | 4 distinct T1 (y,A), holds_t1_A |

| | PSNR_T1 | PSNR_T2 | Fgt |
| --- | ---: | ---: | ---: |
| after T1 | 24.407 | 22.269 | — |
| after T2 | 24.590 | 22.190 | **−0.184** |

Gate: FAIL this try (Fgt not clearly > 0). T1 PSNR still rose slightly on held-out 4× after learning 8×.
