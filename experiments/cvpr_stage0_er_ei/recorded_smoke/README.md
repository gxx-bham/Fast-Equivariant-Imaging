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
- `stage0b_gate_E/` — Gate E **cross-IP continual**: T1 MRI knee Gaussian 4× → T2 `deepinv.physics.Tomography` (physics-tour **40**-view 64×64; not TomographyWithAstra). Unsupervised MC/EI. **PASS**, Fgt=**+2.577 dB**. Stop three-arm. `superseded_angles20/` is the 0.3.5-tag 20-view try (Fgt=+2.290; not the readout).
- `stage0b_crossip_grid/` — Stream-E three-arm go/kill grid (Fine-tune | ER+MC | ER+EI × `N_buf∈{1,4}`). Cross-IP continual. Unsupervised. **GO** (ER+EI lower Fgt than ER+MC at both N). Not 16.
- `stage1_crossip_grid/` — Stage-1 MoDL matrix: stream E, N_buf∈{1,2,4,8} × seeds {1,2,3}. **Remainder aborted per Guixian** (seed=3 N_buf∈{4,8} not run). See `ABORT.md`. Do not merge with light-UNet.
- `light_unet_s1_seed1/` — Light backbone direction smoke: `UNet(scales=3)` + `ArtifactRemoval`, stream E, N_buf=1 seed=1, Fine-tune | ER+MC | ER+EI. Not MoDL. Seed=1 cell kept; formal 3-seed readout lives in `light_unet_s1_seeds/`.
- `light_unet_s1_seeds/` — Light-UNet N_buf=1 seeds {1,2,3}. Same fingerprint as seed1. **Not MoDL.** Formal S1 go/kill at N_buf=1 only (do not expand N_buf).

