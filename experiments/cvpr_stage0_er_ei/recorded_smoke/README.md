Tiny seed=1 CPU stub (few iters) and demo-scale go/kill grid:

- `data_physics.json` — unit (a): shapes + one untrained MoDL recon step
- `seed1_nbuf1/` — few-iter stub, all three arms at N_buf=1 (pipeline only; INCONCLUSIVE)
- `nbuf4_path_check/` — proves `--unit nbuf4` stores 4 distinct T1 `(y, A)`
- `seed1_grid/` — 150-epoch CPU grid on the **old n=2 path** (pre-fix; not a valid N_buf=4 distinct-A run)
- `stage0b_gate/` — Fine-tune-only forgetting gate, **same-family Cartesian 4×→8×** (**FAIL**, Fgt stayed ≤ 0)
- `stage0b_gate_B/` — Gate B same-knee T1 Gaussian 4× → T2 `RandomMaskGenerator` 8× (**FAIL**, Fgt=+0.081, not clearly > 0)
- `stage0b_gate_A/` — Gate A **domain-incremental** knee → brain mini RSS (**FAIL**, Fgt=−1.085)
- `stage0b_gate_D/` — Gate D **domain+operator composite** (knee Gaussian 4× → brain `RandomMaskGenerator` 8×) (**FAIL**, Fgt=+0.162, not clearly > 0.2). Last unsupervised Fgt gate.
- `stage0b_gate_F_supervised/` — Step 3 supervised Fine-tune on stream D (HQ `SupLoss`, MC/EI off). **PASS**, Fgt=**+3.480 dB**. No E, no three-arm.
- `stage0b_gate_E/` — Gate E **cross-IP continual**: T1 MRI knee Gaussian 4× → T2 `deepinv.physics.Tomography` (physics-tour 20-view 64×64). Unsupervised MC/EI. No three-arm.

