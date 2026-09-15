Tiny seed=1 CPU stub (few iters) and demo-scale go/kill grid:

- `data_physics.json` — unit (a): shapes + one untrained MoDL recon step
- `seed1_nbuf1/` — few-iter stub, all three arms at N_buf=1 (pipeline only; INCONCLUSIVE)
- `nbuf4_path_check/` — proves `--unit nbuf4` stores 4 distinct T1 `(y, A)`
- `seed1_grid/` — 150-epoch CPU grid on the **old n=2 path** (pre-fix; not a valid N_buf=4 distinct-A run)
- `stage0b_gate/` — Fine-tune-only forgetting gate, **same-family Cartesian 4×→8×** (**FAIL**, Fgt stayed ≤ 0; no three-arm re-grid)
- `stage0b_gate_B/` — Fine-tune-only gate B: same-knee, T1 Gaussian Cartesian 4× → T2 `RandomMaskGenerator` (one try)
- `stage0b_gate_A/` — Fine-tune-only gate A: **domain-incremental** knee → brain mini RSS (only if B Fgt is not clearly > 0)

