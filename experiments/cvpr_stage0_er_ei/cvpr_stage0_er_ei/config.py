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
CSV_COLUMNS = ("arm", "N_buf", "seed", "PSNR_T1", "PSNR_T2", "Avg", "Fgt")


@dataclass
class SmokeConfig:
    seed: int = 1
    n_buf: int = 1
    arms: Sequence[str] = field(default_factory=lambda: ARMS)
    device: str = "cpu"
    tiny: bool = True
    epochs: int = 1
    max_batch_steps: int = 2
    batch_size: int = 1
    img_size: int = IMG_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    download: bool = True
    out_dir: str = "recorded_smoke"
