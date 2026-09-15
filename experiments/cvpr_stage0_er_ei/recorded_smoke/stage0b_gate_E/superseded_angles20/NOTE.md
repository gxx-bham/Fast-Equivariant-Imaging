# Not the Gate E readout

This folder is the Fine-tune run that copied **deepinv==0.3.5** `examples/physics/demo_physics_tour.py` (`angles=20`, `img_width=64`).

Gap Scout froze **`n_angles=40`** (current physics-tour default; `deepinv.physics.Tomography`, not `TomographyWithAstra`) while that run was in flight. Official Gate E logs live in the parent directory.

That 20-view try was Fine-tune-only, unsupervised MC+EI, Fgt=+2.290 dB, **PASS** under the 0.2 dB bar. Do not treat it as the frozen geometry.
