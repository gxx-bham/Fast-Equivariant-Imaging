# Confirmation line (light UNet HARDEN n_eval=32, N_buf=1, seeds 1–3)

Verified from this dir's seed1–3 JSON. Do not merge with n_eval=2.

- Stream E: T1 MRI knee gaussian 4× 128×128 → T2 `deepinv.physics.Tomography(n_angles=40)` 64×64
- Backbone: `deepinv.models.UNet(in_channels=2,out_channels=2,scales=3)` + `ArtifactRemoval(mode=adjoint)` — NOT MoDL
- Buffer: `(y,mask)` replay via `make_mri_physics(stored mask)` (physics=`deepinv.physics.MRI`); never MRI→CT forward
- `n_distinct_ya == 1` for all three seeds; N_buf=1; n_train=8; **n_eval=32**
- Arm losses: ER+MC buffer = `MCLoss` only; ER+EI buffer = `MCLoss` + `EILoss(Rotate(n_trans=4))` (nonzero MC and EI probes on ER arms)
- Fgt table (Fine-tune/ER+MC/ER+EI): seed1 28.293/5.744/1.068; seed2 11.739/1.142/0.354; seed3 13.458/2.546/0.608 → **GO** under ≥2/3 EI<MC and mean(Fgt_MC−Fgt_EI)>0 (mean Δ=2.468)
- Claim locked: raise n_eval only; do not merge MoDL or n_eval=2 tables; no N_buf/n_trans/n_train expand without Guixian
