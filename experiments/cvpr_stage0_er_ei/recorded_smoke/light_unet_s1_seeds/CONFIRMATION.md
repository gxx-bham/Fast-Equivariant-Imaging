# Confirmation line (light UNet S1, N_buf=1, seeds 1–3)

Verified from existing seed1–3 JSON on this PR. No new experiments.

- Stream E: T1 MRI knee gaussian 4× → T2 `deepinv.physics.Tomography(n_angles=40)`
- Backbone: `deepinv.models.UNet(in_channels=2,out_channels=2,scales=3)` + `ArtifactRemoval(mode=adjoint)` — NOT MoDL
- Buffer: `(y,mask)` replay via `make_mri_physics(stored mask)` (physics=`deepinv.physics.MRI`); never MRI→CT forward
- `n_distinct_ya == 1` for all three seeds
- Arm losses: ER+MC buffer = `MCLoss` only; ER+EI buffer = `MCLoss` + `EILoss(Rotate(n_trans=4))` (nonzero MC and EI probes on ER arms)
- Fgt table (already on PR): seed1 25.171/7.188/0.847; seed2 12.034/0.601/0.425; seed3 14.083/2.948/0.726 → GO 3/3 under ≥2/3 EI<MC
- Claim locked: extreme-small-buffer EI amplifies replay; do not merge MoDL tables; no N_buf/n_trans expand without Guixian
