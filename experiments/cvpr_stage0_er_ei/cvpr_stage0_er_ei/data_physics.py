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

from .buffer import tensor_id


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


def _corner_crops(x: torch.Tensor, size: int) -> torch.Tensor:
    """Four 128×128 corner crops per native slice (still 128 recon, not 320²)."""
    _n, _c, h, w = x.shape
    if h == size and w == size:
        return x
    if h < size or w < size:
        return torch.nn.functional.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False
        )
    origins = [(0, 0), (0, w - size), (h - size, 0), (h - size, w - size)]
    crops = [x[:, :, y0 : y0 + size, x0 : x0 + size] for y0, x0 in origins]
    return torch.cat(crops, dim=0)


def load_knee_slices(
    img_size: int = 128,
    download: bool = True,
    root: Path | None = None,
    n_total: int = 2,
) -> torch.Tensor:
    """Load the deepinv mini single-coil FastMRI knee subset.

    n_total <= 2: resize both demo slices to img_size (tiny path).
    n_total > 2: 128×128 corner crops from native 320×320 (up to 8 slices)
    so N_buf=4 can store 4 distinct (y, A). Recon stays 128×128.
    """
    root = Path(root) if root is not None else _cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    if n_total <= 2:
        transform = transforms.Compose([transforms.Resize(img_size)])
        dataset = SimpleFastMRISliceDataset(
            root,
            anatomy="knee",
            train=True,
            train_percent=1.0,
            transform=transform,
            download=download,
        )
        x = _stack_images(dataset)[:n_total]
    else:
        dataset = SimpleFastMRISliceDataset(
            root,
            anatomy="knee",
            train=True,
            train_percent=1.0,
            transform=None,
            download=download,
        )
        x = _corner_crops(_stack_images(dataset), img_size)[:n_total]
    if x.ndim != 4 or x.shape[1] != 2:
        raise RuntimeError(
            f"Expected stacked complex MRI images (N, 2, H, W); got {tuple(x.shape)}"
        )
    if x.shape[0] < n_total:
        raise RuntimeError(
            f"Requested n_total={n_total} slices but only loaded {x.shape[0]}"
        )
    return x


def make_gaussian_masks(
    img_size: int,
    acceleration: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """One Cartesian Gaussian mask per sample (same family, distinct A)."""
    rng = torch.Generator(device="cpu").manual_seed(int(seed) + int(acceleration) * 1009)
    generator = GaussianMaskGenerator(
        img_size=(2, img_size, img_size),
        acceleration=int(acceleration),
        rng=rng,
        device="cpu",
    )
    mask = generator.step(batch_size=int(batch_size))["mask"]
    return mask.to(device=device)


def make_mri_physics(mask: torch.Tensor, device: torch.device) -> dinv.physics.MRI:
    return dinv.physics.MRI(mask=mask.to(device), device=device)


class XYDataset(Dataset):
    """Offline (x, y, {mask}) triples. x is HQ and must not enter training losses."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        if x.shape[0] != y.shape[0] or x.shape[0] != mask.shape[0]:
            raise ValueError("x, y, mask must share the sample dimension")
        self.x = x.detach().cpu().contiguous()
        self.y = y.detach().cpu().contiguous()
        self.mask = mask.detach().cpu().contiguous()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], {"mask": self.mask[index]}


@dataclass
class TaskData:
    name: str
    accel: int
    physics: dinv.physics.MRI
    train: XYDataset
    eval: XYDataset


def build_tasks(
    x: torch.Tensor,
    seed: int,
    device: torch.device,
    img_size: int = 128,
    n_train: int | None = None,
    n_eval: int = 0,
) -> dict[str, TaskData]:
    """T1 accel 4x then T2 accel 8x, same Gaussian Cartesian mask family.

    Per-sample masks so buffer slots can be distinct (y, A). If n_eval>0,
    eval slices are held out from train.
    """
    device = _resolve_device(device)
    x = x.to(device)
    n = int(x.shape[0])
    if n_train is None:
        n_train = n if n_eval <= 0 else n - int(n_eval)
    n_train = int(n_train)
    n_eval = int(n_eval)
    if n_eval > 0:
        if n_train + n_eval > n:
            raise ValueError(
                f"n_train ({n_train}) + n_eval ({n_eval}) exceeds n_images ({n})"
            )
        x_train, x_eval = x[:n_train], x[n_train : n_train + n_eval]
    else:
        x_train = x[:n_train]
        x_eval = x_train
    tasks: dict[str, TaskData] = {}
    for name, accel in (("T1", 4), ("T2", 8)):
        train_mask = make_gaussian_masks(
            img_size, accel, seed=seed, device=device, batch_size=x_train.shape[0]
        )
        eval_mask = make_gaussian_masks(
            img_size,
            accel,
            seed=seed + 7919,
            device=device,
            batch_size=x_eval.shape[0],
        )
        physics = make_mri_physics(train_mask[:1], device)
        y_train = torch.stack(
            [
                physics(x_train[i : i + 1], mask=train_mask[i : i + 1])[0]
                for i in range(x_train.shape[0])
            ],
            dim=0,
        )
        y_eval = torch.stack(
            [
                physics(x_eval[i : i + 1], mask=eval_mask[i : i + 1])[0]
                for i in range(x_eval.shape[0])
            ],
            dim=0,
        )
        tasks[name] = TaskData(
            name=name,
            accel=accel,
            physics=physics,
            train=XYDataset(x_train.cpu(), y_train.cpu(), train_mask.cpu()),
            eval=XYDataset(x_eval.cpu(), y_eval.cpu(), eval_mask.cpu()),
        )
    return tasks


def task_operator_report(task: TaskData) -> dict[str, Any]:
    train_mask_ids = [tensor_id(task.train.mask[i]) for i in range(len(task.train))]
    return {
        "task": task.name,
        "accel": task.accel,
        "n_train": len(task.train),
        "n_eval": len(task.eval),
        "train_mask_ids": train_mask_ids,
        "n_distinct_train_A": len(set(train_mask_ids)),
        "held_out_eval": len(task.eval) > 0
        and (
            len(task.eval) != len(task.train)
            or not torch.equal(task.train.x, task.eval.x)
        ),
    }


def smoke_data_physics(
    seed: int = 1,
    device: str | torch.device | None = "cpu",
    img_size: int = 128,
    download: bool = True,
    n_train: int = 2,
    n_eval: int = 0,
) -> dict[str, Any]:
    """Verifiable unit (a): forward, shapes, one MoDL recon step. No training."""
    device = _resolve_device(device)
    torch.manual_seed(seed)
    n_total = n_train + (n_eval if n_eval > 0 else 0)
    n_total = max(n_total, 2)
    x = load_knee_slices(
        img_size=img_size, download=download, n_total=n_total
    ).to(device)
    tasks = build_tasks(
        x, seed=seed, device=device, img_size=img_size, n_train=n_train, n_eval=n_eval
    )
    model = MoDL().to(device)
    model.eval()
    with torch.no_grad():
        y0 = tasks["T1"].train.y[:1].to(device)
        x0 = tasks["T1"].train.x[:1].to(device)
        m0 = tasks["T1"].train.mask[:1].to(device)
        physics = make_mri_physics(m0, device)
        x_hat = model(y0, physics)
        psnr = dinv.metric.PSNR(complex_abs=True)
        recon_psnr = float(psnr(x_hat, x0).mean().item())
    report = {
        "x_shape": list(x.shape),
        "T1_y_shape": list(tasks["T1"].train.y.shape),
        "T2_y_shape": list(tasks["T2"].train.y.shape),
        "T1_mask_shape": list(tasks["T1"].train.mask.shape),
        "T2_mask_shape": list(tasks["T2"].train.mask.shape),
        "T1_accel": tasks["T1"].accel,
        "T2_accel": tasks["T2"].accel,
        "recon_shape": list(x_hat.shape),
        "one_step_psnr_t1": recon_psnr,
        "device": str(device),
        "n_images": int(x.shape[0]),
        "n_train": n_train,
        "n_eval": n_eval if n_eval > 0 else n_train,
        "operators": {
            "T1": task_operator_report(tasks["T1"]),
            "T2": task_operator_report(tasks["T2"]),
        },
        "deepinv_version": getattr(dinv, "__version__", "unknown"),
    }
    return report
