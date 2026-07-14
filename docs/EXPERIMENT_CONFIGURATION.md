# Experiment Configuration Reference

This note summarizes the settings encoded by the experiment configurations included in this repository.

## Methods

- EI
- FEI-O1
- FEI-O2
- PnP-FEI-O1
- PnP-FEI-O2

The release covers the offline unsupervised training experiments in Sections 5.1 and 5.2 of the paper.

## Dataset splits

Both datasets contain 100 samples. The first 90 samples are used for training and the last 10 for evaluation:

- CT: indices `0:90` for training and `90:100` for evaluation;
- Urban100: `img_001.png` through `img_090.png` for training and `img_091.png` through `img_100.png` for evaluation.

Training updates use only the training subset. Evaluation is run separately using `test_FEI_CT.py` or `test_FEI_Inpainting.py` and `final_model.pth.tar`.

## Sparse-view CT reconstruction

- Data: `CT/CT100_128x128.mat`, MATLAB key `DATA`
- Image size: `128 x 128`
- Geometry: 50 projection views, `circle = false`
- Transformation group: five randomly sampled in-plane rotations
- Loss: L2
- Batch size: 8
- Epochs: 5000
- EI: `alpha = 1000`
- FEI and PnP-FEI: `lambda = 1`, `beta = 0.1`, `eta = 0.01`, `J = 10`
- PnP denoiser: pretrained DnCNN, `sigma = 0.01`
- Evaluation: PSNR, SSIM, and EQUIV; FBP is reported as the physics baseline

## Urban100 image inpainting

- Data: `Urban100/all/all`
- Preprocessing: center crop to `512 x 512`, then resize to `256 x 256`
- Missing-pixel rate: 0.6 (60% of pixels removed)
- Transformation group: three randomly sampled two-dimensional circular shifts
- Loss: L2
- Batch size: 4
- Epochs: 2000
- EI: `alpha = 1`
- FEI and PnP-FEI: `lambda = 0.1`, `beta = 0.9`, `eta = 0.09`, `J = 10`
- PnP denoiser: pretrained DnCNN, `sigma = 0.01`
- Evaluation: PSNR and SSIM; the masked input is reported as the physics baseline

## Outputs

The default seed is 0 and can be overridden from the command line. Each run records the resolved configuration, random seed, environment information, Git metadata when available, training progress, checkpoints, and reconstruction previews. The final training state is saved as `final_model.pth.tar` and is the checkpoint used by the supplied evaluation commands.

Reduced-scale execution checks save the same classes of artifacts under ignored output directories and label their metadata with `debug_only: true`.
