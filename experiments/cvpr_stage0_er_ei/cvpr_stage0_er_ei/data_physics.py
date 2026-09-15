"""Data + MRI physics smoke, following the deepinv MRI EI demo conventions.

HQ images are used to simulate undersampled measurements (retrospective) and
are used for PSNR only at eval. Training losses are MC/EI on (y, A).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import deepinv as dinv
from deepinv.datasets import SimpleFastMRISliceDataset
from deepinv.models import MoDL
from deepinv.physics.generator import GaussianMaskGenerator


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / ".cache"


def _stack_images(dataset: Dataset) -> torch.Tensor:
    xs = []
    for i in range(len(dataset)):
        item = dataset[i]
        x = item[0] if isinstance(item, (tuple, list)) else item
        xs.append(torch.as_tensor(x))
    return torch.stack(xs, dim=0)


def load_knee_slices(
    img_size: int = 128,
    download: bool = True,
    root: Path | None = None,
) -> torch.Tensor:
    """Load the deepinv mini single-coil FastMRI knee subset, resized to img_size.

    Uses both mini slices (length 2). Requires the FastMRI data use agreement
    when downloading the demo subset shipped by deepinv.
    """
    root = Path(root) if root is not None else _cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Resize(img_size)])
    dataset = SimpleFastMRISliceDataset(
        root,
        anatomy="knee",
        train=True,
        train_percent=1.0,
        transform=transform,
        download=download,
    )
    x = _stack_images(dataset)
    if x.ndim != 4 or x.shape[1] != 2:
        raise RuntimeError(
            f"Expected stacked complex MRI images (N, 2, H, W); got {tuple(x.shape)}"
        )
    return x


def make_gaussian_mask(
    img_size: int,
    acceleration: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """One fixed Cartesian Gaussian mask per task (same mask family, different accel).

    deepinv's GaussianMaskGenerator uses ACS center_fraction 0.08 for accel<8
    and 0.04 for accel>=8 (fastMRI-style ACS sizes).
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed) + int(acceleration))
    generator = GaussianMaskGenerator(
        img_size=(2, img_size, img_size),
        acceleration=int(acceleration),
        rng=rng,
        device="cpu",
    )
    mask = generator.step(batch_size=1)["mask"]
    return mask.to(device=device)


def make_mri_physics(mask: torch.Tensor, device: torch.device) -> dinv.physics.MRI:
    return dinv.physics.MRI(mask=mask.to(device), device=device)


class XYDataset(Dataset):
    """Offline (x, y) pairs. x is HQ and must not enter the training losses."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must share the batch/sample dimension")
        self.x = x.detach().cpu().contiguous()
        self.y = y.detach().cpu().contiguous()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class TaskData:
    name: str
    accel: int
    physics: dinv.physics.MRI
    mask: torch.Tensor
    train: XYDataset
    eval: XYDataset


def build_tasks(
    x: torch.Tensor,
    seed: int,
    device: torch.device,
    img_size: int = 128,
) -> dict[str, TaskData]:
    """T1 accel 4x then T2 accel 8x, same Gaussian Cartesian mask family."""
    device = _resolve_device(device)
    x = x.to(device)
    tasks: dict[str, TaskData] = {}
    for name, accel in (("T1", 4), ("T2", 8)):
        mask = make_gaussian_mask(img_size, accel, seed=seed, device=device)
        physics = make_mri_physics(mask, device)
        y = physics(x)
        dataset = XYDataset(x.cpu(), y.cpu())
        tasks[name] = TaskData(
            name=name,
            accel=accel,
            physics=physics,
            mask=mask.detach().cpu().clone(),
            train=dataset,
            eval=dataset,
        )
    return tasks


def smoke_data_physics(
    seed: int = 1,
    device: str | torch.device | None = "cpu",
    img_size: int = 128,
    download: bool = True,
) -> dict[str, Any]:
    """Verifiable unit (a): forward, shapes, one MoDL recon step. No training."""
    device = _resolve_device(device)
    torch.manual_seed(seed)
    x = load_knee_slices(img_size=img_size, download=download).to(device)
    tasks = build_tasks(x, seed=seed, device=device, img_size=img_size)
    model = MoDL().to(device)
    model.eval()
    recon_psnr = None
    with torch.no_grad():
        y = tasks["T1"].train.y.to(device)
        x0 = tasks["T1"].train.x.to(device)
        x_hat = model(y[:1], tasks["T1"].physics)
        psnr = dinv.metric.PSNR(complex_abs=True)
        recon_psnr = float(psnr(x_hat, x0[:1]).mean().item())
    report = {
        "x_shape": list(x.shape),
        "T1_y_shape": list(tasks["T1"].train.y.shape),
        "T2_y_shape": list(tasks["T2"].train.y.shape),
        "T1_mask_shape": list(tasks["T1"].mask.shape),
        "T2_mask_shape": list(tasks["T2"].mask.shape),
        "T1_accel": tasks["T1"].accel,
        "T2_accel": tasks["T2"].accel,
        "recon_shape": list(x_hat.shape),
        "one_step_psnr_t1": recon_psnr,
        "device": str(device),
        "n_images": int(x.shape[0]),
    }
    return report
