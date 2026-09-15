# CVPR stage-0 smoke (isolated package)

Operator-incremental **unsupervised MRI recon** on the same knee domain; tasks are acceleration/mask operators.

This directory is a **self-contained experiment**. It **composes deepinv** (`MoDL` + `MCLoss` / `EILoss` + `physics.MRI` + `Trainer`). It does **not** use Fast Equivariant Imaging (FEI) or SkEI plugin code from the rest of this repository, and it does not rewrite those plugins.

Frozen claim (do not redesign): ER with EI as the buffer loss should match ER+MC forgetting at a smaller `N_buf`; `N_buf=1` is the stress case.

This package does not claim novelty and does not invent citations.

## Locked defaults

| Item | Value |
| --- | --- |
| Backbone | `deepinv.models.MoDL` (`MoDL()`, library defaults) |
| Current-task loss (all arms) | `MCLoss()` + `EILoss(Rotate(n_trans=4))` (deepinv MRI EI demo) |
| Physics | `deepinv.physics.MRI` |
| Mask family | `deepinv.physics.generator.GaussianMaskGenerator` |
| Task stream | T1 accel **4×** → T2 accel **8×**, sequential train, eval **both** after each task |
| Arms | Fine-tune \| ER+MC \| ER+EI |
| Buffer | stores **(y, A) only** (measurement + MRI mask); uniform sampling; current:replay mix **1:1** |
| ER+MC | MC only on buffer |
| ER+EI | MC+EI on buffer |
| `N_buf` grid | `{1, 4}` |
| Data | 128×128 single-coil knee, deepinv MRI EI demo mini FastMRI subset |
| HQ | used to simulate retrospective undersampling; **PSNR uses HQ only at eval** |
| Metric | PSNR (complex magnitude, `PSNR(complex_abs=True)`); log `N_buf`, arm, seed |
| Seeds | `1` for debug; then 3 if curves look real |

## How Fgt and Avg are defined

After finishing task \(j\), let \(R_i(j)\) be **PSNR on task \(i\)**.

After T1: record \(R_{\mathrm{T1}}(\mathrm{T1})\) (and \(R_{\mathrm{T2}}(\mathrm{T1})\) for the log).

After T2: CSV `PSNR_T1` = \(R_{\mathrm{T1}}(\mathrm{T2})\), `PSNR_T2` = \(R_{\mathrm{T2}}(\mathrm{T2})\).

\[
\mathrm{Avg} = \tfrac{1}{2}\big(R_{\mathrm{T1}}(\mathrm{T2}) + R_{\mathrm{T2}}(\mathrm{T2})\big)
\]

\[
\mathrm{Fgt} = R_{\mathrm{T1}}(\mathrm{T1}) - R_{\mathrm{T1}}(\mathrm{T2})
\]

This is standard continual-learning forgetting on PSNR after T2. **Positive Fgt** means T1 PSNR dropped after learning T2. **Lower Fgt is better.**

## Go / kill (from the frozen handoff)

- **Go:** ER+EI clearly better Fgt than ER+MC at `N_buf=1` **or** `N_buf=4`.
- **Kill:** ER+EI \(\le\) ER+MC at **both** \(N\), or ER+EI only ties Fine-tune.

The `--unit grid` readout prints this table. A **tiny CPU smoke is a pipeline check**, not a science decision. Treat go/kill as defined on a non-tiny `{1,4}` grid once the curves look real (seed 1, then 3).

## Install

From this directory (isolated from the parent FEI `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# optional: python3 -m pip install -e .
```

CPU is enough for the tiny smoke. GPU is optional (`--device cuda`).

By using the mini FastMRI subset that deepinv downloads, you confirm that you have agreed to the FastMRI data use agreement.

## How to run

All commands below are from `experiments/cvpr_stage0_er_ei/` with `PYTHONPATH=.` (the scripts set this).

Verifiable units, in order:

```bash
# (a) data + physics: forward, shapes, one MoDL recon step
PYTHONPATH=. python3 -m cvpr_stage0_er_ei --unit data --seed 1 --device cpu

# (b) Fine-tune, tiny epochs / few steps
PYTHONPATH=. python3 -m cvpr_stage0_er_ei --unit finetune --tiny --seed 1 --device cpu

# (c) ER+MC and ER+EI with (y, A) buffer
PYTHONPATH=. python3 -m cvpr_stage0_er_ei --unit er --tiny --seed 1 --n-buf 1 --device cpu
```

**Single command** (all three arms at `N_buf=1`, seed 1, CPU tiny smoke):

```bash
bash scripts/run_nbuf1.sh
```

Equivalent:

```bash
PYTHONPATH=. python3 -m cvpr_stage0_er_ei --unit nbuf1 --tiny --seed 1 --n-buf 1 \
  --arms finetune,er_mc,er_ei --device cpu --out recorded_smoke/seed1_nbuf1
```

Documented `N_buf=4` cell (same three arms):

```bash
bash scripts/run_nbuf4.sh
```

Full `{1,4}` go/kill grid (demo-scale: 150 epochs, `--no-tiny`; GPU if available, else CPU):

```bash
bash scripts/run_grid.sh
```

Tiny mode uses `--epochs 1 --max-steps 2` so the CPU path finishes. `run_grid.sh` is the go/kill run: 150 epochs (deepinv MRI EI demo from-scratch order of magnitude) on the same 128×128 mini knee set. GPU is used when `torch.cuda.is_available()`, otherwise CPU, and the device is labeled in the log.

## Logs

CSV/JSON columns after T2:

`arm, N_buf, seed, PSNR_T1, PSNR_T2, Avg, Fgt`

JSON also stores after-T1 PSNR on both tasks, buffer fill size, and a go/kill readout. Example tiny seed=1 output lives in `recorded_smoke/seed1_nbuf1/`.

Fine-tune ignores the buffer; `N_buf` is still logged so the row sits on the same grid.

## What is composed vs written here

Composed from deepinv (MRI EI demo / MRI tour blocks):

- `SimpleFastMRISliceDataset` (knee, resize 128)
- `physics.MRI`
- `models.MoDL`
- `loss.MCLoss`, `loss.EILoss(transform.Rotate(n_trans=4))`
- `Trainer` for the train loop
- `metric.PSNR(complex_abs=True)` at eval

Written only for the stage-0 protocol:

- two-task 4×→8× stream
- `(y, A)` reservoir buffer
- ER replay loss that calls the same MC/EI objects on buffer samples
- Avg / Fgt logging

## REI parity gap (deepinv preferred)

If deepinv and [REI](https://github.com/edongdongchen/REI) paths conflict, **this smoke uses the deepinv API**.

| Topic | This smoke (deepinv) | REI (reference only) |
| --- | --- | --- |
| Images | Mini SimpleFastMRI knee, **128×128** (MRI EI demo) | Full FastMRI knee k-space / REI MRI loader |
| Mask family | `GaussianMaskGenerator` Cartesian lines, ACS 0.08 @ 4× and 0.04 @ 8× | `fastmri.data.subsample.RandomMaskFunc` (uniform random Cartesian, same ACS sizes) |
| Operator | `deepinv.physics.MRI` | REI MRI forward |
| Train loop | `deepinv.Trainer` + MC/EI losses | REI closures |

ACS line fractions match the usual 4×/8× convention; the **sampling pdf is Gaussian (deepinv) vs uniform-random (REI)**. Data scale and resolution follow the deepinv MRI EI demo, not REI’s full FastMRI setting.

## Out of scope

SkEI / FEI plugins, KD, EWC / MOST, cross-anatomy, architecture search, hyperparameter sweeps beyond LR sanity, paper writing, literature reviews.

## License

Same as the parent snapshot (GPL-3.0-only). deepinv remains a BSD-3-Clause dependency.
