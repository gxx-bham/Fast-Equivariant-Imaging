"""Locked defaults from the frozen stage-0 handoff. Do not redesign the claim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# Claim (frozen): operator-incremental unsupervised MRI recon, same knee domain;
# tasks = accel/mask. ER with EI as buffer loss should match ER+MC forgetting at
# smaller N_buf; N_buf=1 is the stress case.
IMG_SIZE = 128
ANATOMY = "knee"
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-8
CURRENT_REPLAY_MIX = (1, 1)  # current:replay sample counts
N_BUF_GRID = (1, 4)
ARMS = ("finetune", "er_mc", "er_ei")
ARM_LABELS = {
    "finetune": "Fine-tune",
    "er_mc": "ER+MC",
    "er_ei": "ER+EI",
}
TASKS = (
    {"name": "T1", "accel": 4},
    {"name": "T2", "accel": 8},
)
MASK_FAMILIES = ("gaussian", "random")
ANATOMIES = ("knee", "brain")
MASK_FAMILY_TO_CLASS = {
    "gaussian": "deepinv.physics.generator.GaussianMaskGenerator",
    "random": "deepinv.physics.generator.RandomMaskGenerator",
}
# Fine-tune forgetting gate: Fgt must be clearly > 0 (T1 PSNR drop after T2).
# 0.2 dB is above the ~0.18 opposite-direction wiggle on the failed Cartesian 4×→8× try.
CLEAR_POSITIVE_FGT = 0.2
# Physics-tour CT geometry. Class MUST be deepinv.physics.Tomography
# (not TomographyWithAstra). Pass angles as int; Tomography samples uniformly.
# Current tour default is 40 views @ 64×64, normalize=True. Pinned wheel
# deepinv==0.3.5 tagged demo_physics_tour.py used angles=20; Gap Scout froze 40.
CT_DEMO_N_ANGLES = 40
CT_DEMO_IMG_SIZE = 64
CT_DEMO_SOURCE = (
    "deepinv.physics.Tomography physics-tour defaults: angles=40, img_width=64, "
    "normalize=True (int angles; not TomographyWithAstra)"
)
CSV_COLUMNS = ("arm", "N_buf", "seed", "PSNR_T1", "PSNR_T2", "Avg", "Fgt")
PINNED_DEEPINV = "0.3.5"


@dataclass
class SmokeConfig:
    seed: int = 1
    n_buf: int = 1
    arms: Sequence[str] = field(default_factory=lambda: ARMS)
    device: str = "cpu"
    tiny: bool = True
    epochs: int = 1
    epochs_t1: int | None = None
    epochs_t2: int | None = None
    max_batch_steps: int = 2
    batch_size: int = 1
    img_size: int = IMG_SIZE
    n_train: int = 2
    n_eval: int = 0  # 0 = eval on the train slices (tiny default)
    t1_accel: int = 4
    t2_accel: int = 8
    t1_mask_family: str = "gaussian"
    t2_mask_family: str = "gaussian"
    t1_anatomy: str = ANATOMY
    t2_anatomy: str = ANATOMY
    gate: str = ""  # B/A/D/F/E
    supervised: bool = False  # True: HQ SupLoss only (MC/EI off) for this gate
    cross_ip: bool = False  # Gate E: T1 MRI → T2 CT Tomography
    ct_n_angles: int = CT_DEMO_N_ANGLES
    ct_img_size: int = CT_DEMO_IMG_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    download: bool = True
    out_dir: str = "recorded_smoke"

    def t1_epochs(self) -> int:
        return int(self.epochs if self.epochs_t1 is None else self.epochs_t1)

    def t2_epochs(self) -> int:
        return int(self.epochs if self.epochs_t2 is None else self.epochs_t2)
