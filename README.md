# Fast Equivariant Imaging

Code accompanying:

> Guixian Xu, Jinglai Li, and Junqi Tang, **Fast Equivariant Imaging: Accelerating Unsupervised Learning and Model Adaptation via Inexact Splitting**, [arXiv:2507.06764v5](https://arxiv.org/abs/2507.06764v5).

## Scope

This repository provides the training and evaluation implementations for the offline unsupervised experiments in Sections 5.1 and 5.2 of the paper:

- sparse-view CT reconstruction;
- Urban100 image inpainting;
- EI;
- FEI-O1 and FEI-O2;
- PnP-FEI-O1 and PnP-FEI-O2.

The snapshot does not contain datasets, trained checkpoints, result files, manuscript figures or tables, or the test-time adaptation implementation. The small execution checks described below produce diagnostic outputs only; no values from those checks are reported in the paper.

The experiment settings encoded by the supplied configurations are summarized in [docs/EXPERIMENT_CONFIGURATION.md](docs/EXPERIMENT_CONFIGURATION.md).

## Environment

The code was checked with Python 3.12.7, PyTorch 2.5.1+cu121, TorchVision 0.20.1+cu121, and DeepInv 0.3.5.

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a specific CUDA or CPU build, install the matching PyTorch 2.5.1 and TorchVision 0.20.1 wheels from the official PyTorch index before installing the remaining requirements.

## Data layout

Data is not redistributed in this repository. Place independently obtained data at:

```text
CT/
  CT100_128x128.mat       # MATLAB key: DATA; expected shape: (100, 128, 128)

Urban100/
  all/
    all/
      img_001.png
      ...
      img_100.png
```

The expected SHA-256 of the CT file used to check this snapshot is:

```text
265fb8938fb3dff9e6bf80a1c221b4758b7a0ae3731f38f804f93aa262867871
```

The repository does not provide CT100 preprocessing or dataset download utilities. Users are responsible for obtaining the datasets and complying with their terms of use.

For both tasks, the first 90 samples are used for training and the last 10 are reserved for evaluation.

## Configuration check

Resolve a configuration and inspect its run metadata without loading data or starting training:

```bash
python train_FEI.py --config configs/experiments/ct_ei.yaml --device cpu --output-dir outputs_dry_run --dry-run-config
```

## Reduced-scale execution checks

The following commands use one epoch and small data caps to check data loading, forward and backward passes, and checkpoint save/load. Outputs are written to ignored directories.

```bash
python train_FEI.py --config configs/experiments/ct_pnp_fei_o2.yaml --device cpu --output-dir outputs_smoke --smoke-test --max-epochs 1 --max-batches 1 --max-train-samples 1

python train_FEI.py --config configs/experiments/urban100_pnp_fei_o1.yaml --device cpu --output-dir outputs_smoke --smoke-test --max-epochs 1 --max-batches 1 --max-train-samples 2
```

The default PnP configurations use DeepInv's official pretrained DnCNN weights, which DeepInv downloads on first use and caches through PyTorch. Other optional denoisers in `functions/utils_dinv.py` require user-supplied weights under `weights/` and are not used by the supplied experiment configurations.

## Training

Run one method directly:

```bash
python train_FEI.py --config configs/experiments/ct_fei_o1.yaml --seed 0 --device cuda:0 --output-dir outputs --log-interval 50
```

The scripts under `scripts/experiments/` contain the corresponding commands for all five methods and both tasks. The supplied configurations use fixed epoch schedules and save the last training state as `final_model.pth.tar`.

## Evaluation

Evaluate a CT checkpoint:

```bash
python test_FEI_CT.py --config configs/experiments/ct_fei_o1.yaml --checkpoint outputs/ct/fei_o1/seed0/checkpoints/final_model.pth.tar --device cuda:0 --output-dir outputs/ct/fei_o1/seed0/evaluation
```

Evaluate an Urban100 checkpoint:

```bash
python test_FEI_Inpainting.py --config configs/experiments/urban100_fei_o1.yaml --checkpoint outputs/urban100/fei_o1/seed0/checkpoints/final_model.pth.tar --device cuda:0 --output-dir outputs/urban100/fei_o1/seed0/evaluation
```

## License and third-party code

This repository is distributed under the GNU General Public License v3.0 only. The CT Radon helper implementation in `functions/radon.py`, `functions/filters.py`, and `functions/utils.py` is derived from [TorchRadon](https://github.com/matteo-ronchetti/torch-radon), which is GPL-3.0 licensed. DeepInv is used as an external dependency under its BSD-3-Clause license.

## Citation

```bibtex
@article{xu2025fast,
  title   = {Fast Equivariant Imaging: Accelerating Unsupervised Learning and Model Adaptation via Inexact Splitting},
  author  = {Xu, Guixian and Li, Jinglai and Tang, Junqi},
  journal = {arXiv preprint arXiv:2507.06764},
  year    = {2025}
}
```
