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
from deepinv.physics.generator import GaussianMaskGenerator, RandomMaskGenerator
from deepinv.utils.demo import load_example

from .buffer import tensor_id
from .config import (
    CT_DEMO_IMG_SIZE,
    CT_DEMO_N_ANGLES,
    CT_DEMO_SOURCE,
    MASK_FAMILY_TO_CLASS,
)


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


def _spatial_crops(x: torch.Tensor, size: int) -> torch.Tensor:
    """3×3 windows of `size` on a larger native slice (still 128 recon).

    Corner crops yield at most 8 tiles from 2 native 320 slices. Stage-1 N_buf=8
    plus held-out eval needs 10, so include the center window as well.
    """
    _n, _c, h, w = x.shape
    if h == size and w == size:
        return x
    if h < size or w < size:
        return torch.nn.functional.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False
        )
    ys = sorted({0, max(0, (h - size) // 2), max(0, h - size)})
    xs = sorted({0, max(0, (w - size) // 2), max(0, w - size)})
    crops = [x[:, :, y0 : y0 + size, x0 : x0 + size] for y0 in ys for x0 in xs]
    return torch.cat(crops, dim=0)


def load_anatomy_slices(
    anatomy: str = "knee",
    img_size: int = 128,
    download: bool = True,
    root: Path | None = None,
    n_total: int = 2,
) -> torch.Tensor:
    """Load the deepinv mini FastMRI subset for one anatomy (knee or brain RSS).

    n_total <= 2: resize both demo slices to img_size (tiny path).
    2 < n_total <= 8: 128×128 corner crops from native 320×320 (up to 8 slices).
    n_total > 8: 3×3 128 windows (up to 18) so N_buf=8 plus held-out eval fit.
    Recon stays 128×128. Not a 320² architecture change.
    """
    anatomy = str(anatomy).lower()
    if anatomy not in ("knee", "brain"):
        raise ValueError(f"anatomy must be 'knee' or 'brain' (got {anatomy!r})")
    root = Path(root) if root is not None else _cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    if n_total <= 2:
        transform = transforms.Compose([transforms.Resize(img_size)])
        dataset = SimpleFastMRISliceDataset(
            root,
            anatomy=anatomy,
            train=True,
            train_percent=1.0,
            transform=transform,
            download=download,
        )
        x = _stack_images(dataset)[:n_total]
    else:
        dataset = SimpleFastMRISliceDataset(
            root,
            anatomy=anatomy,
            train=True,
            train_percent=1.0,
            transform=None,
            download=download,
        )
        x = _stack_images(dataset)
        if n_total > 8:
            x = _spatial_crops(x, img_size)[:n_total]
        else:
            x = _corner_crops(x, img_size)[:n_total]
    if x.ndim != 4 or x.shape[1] != 2:
        raise RuntimeError(
            f"Expected stacked complex MRI images (N, 2, H, W); got {tuple(x.shape)}"
        )
    if x.shape[0] < n_total:
        raise RuntimeError(
            f"Requested n_total={n_total} slices but only loaded {x.shape[0]} "
            f"for anatomy={anatomy}"
        )
    return x


def load_knee_slices(
    img_size: int = 128,
    download: bool = True,
    root: Path | None = None,
    n_total: int = 2,
) -> torch.Tensor:
    """Load the deepinv mini single-coil FastMRI knee subset."""
    return load_anatomy_slices(
        anatomy="knee",
        img_size=img_size,
        download=download,
        root=root,
        n_total=n_total,
    )


_MASK_GENERATORS = {
    "gaussian": GaussianMaskGenerator,
    "random": RandomMaskGenerator,
}
_FAMILY_SEED = {"gaussian": 0, "random": 4243}
_ANATOMY_SEED = {"knee": 0, "brain": 777}
_TASK_SEED = {"T1": 0, "T2": 9001}


def mask_seed(
    seed: int,
    acceleration: int,
    family: str,
    *,
    anatomy: str = "knee",
    task: str = "T1",
    eval_split: bool = False,
) -> int:
    """Deterministic mask RNG seed; family/task/anatomy must not collide."""
    family = str(family).lower()
    anatomy = str(anatomy).lower()
    extra = 7919 if eval_split else 0
    return (
        int(seed)
        + int(acceleration) * 1009
        + int(_FAMILY_SEED.get(family, 0))
        + int(_ANATOMY_SEED.get(anatomy, 0))
        + int(_TASK_SEED.get(task, 0))
        + extra
    )


def make_masks(
    img_size: int,
    acceleration: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    family: str = "gaussian",
) -> torch.Tensor:
    """One Cartesian mask per sample. family=gaussian | random (RandomMaskGenerator)."""
    family = str(family).lower()
    cls = _MASK_GENERATORS.get(family)
    if cls is None:
        raise ValueError(
            f"mask family must be one of {sorted(_MASK_GENERATORS)} (got {family!r})"
        )
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    generator = cls(
        img_size=(2, img_size, img_size),
        acceleration=int(acceleration),
        rng=rng,
        device="cpu",
    )
    mask = generator.step(batch_size=int(batch_size))["mask"]
    return mask.to(device=device)


def make_gaussian_masks(
    img_size: int,
    acceleration: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """One Cartesian Gaussian mask per sample (same family, distinct A)."""
    return make_masks(
        img_size,
        acceleration,
        seed=int(seed) + int(acceleration) * 1009,
        device=device,
        batch_size=batch_size,
        family="gaussian",
    )


def as_modl_channels(x: torch.Tensor) -> torch.Tensor:
    """MoDL DnCNN is 2-channel (real/imag). Real CT is stacked as (real, 0)."""
    if x.ndim != 4:
        raise ValueError(f"expected NCHW, got {tuple(x.shape)}")
    if x.shape[1] == 2:
        return x
    if x.shape[1] == 1:
        return torch.cat([x, torch.zeros_like(x)], dim=1)
    raise ValueError(f"expected 1 or 2 channels, got {tuple(x.shape)}")


def _nonoverlap_tiles(x: torch.Tensor, size: int) -> torch.Tensor:
    _n, _c, h, w = x.shape
    if h == size and w == size:
        return x
    if h < size or w < size:
        return torch.nn.functional.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False
        )
    tiles = [
        x[:, :, y0 : y0 + size, x0 : x0 + size]
        for y0 in range(0, h - size + 1, size)
        for x0 in range(0, w - size + 1, size)
    ]
    return torch.cat(tiles, dim=0)


def load_ct100_tiles(
    img_size: int = CT_DEMO_IMG_SIZE,
    n_total: int = 8,
) -> torch.Tensor:
    """deepinv demo CT100 slice, tiled to the CT demo spatial size.

    Source image: `CT100_256x256_0.pt` from deepinv.utils.load_example.
    Physics-tour demo uses 64×64; 256 tiles into 16 non-overlapping 64 crops.
    """
    x = load_example("CT100_256x256_0.pt")
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = _nonoverlap_tiles(x, int(img_size))[: int(n_total)]
    if x.shape[0] < int(n_total):
        raise RuntimeError(
            f"Requested n_total={n_total} CT tiles but only loaded {x.shape[0]}"
        )
    return as_modl_channels(x)


def make_mri_physics(mask: torch.Tensor, device: torch.device) -> dinv.physics.MRI:
    return dinv.physics.MRI(mask=mask.to(device), device=device)


def make_ct_physics(
    img_width: int,
    n_angles: int,
    device: torch.device,
) -> dinv.physics.Tomography:
    """Copy physics-tour sparse-view geometry: Tomography(angles=int, img_width)."""
    physics = dinv.physics.Tomography(
        img_width=int(img_width),
        angles=int(n_angles),
        device=device,
        normalize=True,
    )
    if type(physics).__name__ != "Tomography":
        raise TypeError(
            "CT physics MUST be deepinv.physics.Tomography "
            f"(not TomographyWithAstra); got {type(physics).__name__}"
        )
    return physics


class XYDataset(Dataset):
    """Offline (x, y[, {mask}]) triples. x is HQ and must not enter unsupervised losses."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must share the sample dimension")
        if mask is not None and x.shape[0] != mask.shape[0]:
            raise ValueError("mask must share the sample dimension")
        self.x = x.detach().cpu().contiguous()
        self.y = y.detach().cpu().contiguous()
        self.mask = None if mask is None else mask.detach().cpu().contiguous()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        if self.mask is None:
            return self.x[index], self.y[index]
        return self.x[index], self.y[index], {"mask": self.mask[index]}


@dataclass
class TaskData:
    name: str
    accel: int
    physics: dinv.physics.Physics
    train: XYDataset
    eval: XYDataset
    mask_family: str = "gaussian"
    anatomy: str = "knee"
    modality: str = "mri"
    physics_class: str = "deepinv.physics.MRI"
    n_angles: int | None = None
    img_size: int | None = None

    @property
    def mask_generator(self) -> str:
        if self.modality == "ct":
            return self.physics_class
        return MASK_FAMILY_TO_CLASS.get(
            str(self.mask_family).lower(), str(self.mask_family)
        )


def _split_train_eval(
    x: torch.Tensor,
    n_train: int | None,
    n_eval: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        return x[:n_train], x[n_train : n_train + n_eval]
    return x[:n_train], x[:n_train]


def _simulate_task(
    name: str,
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    accel: int,
    family: str,
    anatomy: str,
    seed: int,
    device: torch.device,
    img_size: int,
) -> TaskData:
    train_mask = make_masks(
        img_size,
        accel,
        seed=mask_seed(seed, accel, family, anatomy=anatomy, task=name),
        device=device,
        batch_size=x_train.shape[0],
        family=family,
    )
    eval_mask = make_masks(
        img_size,
        accel,
        seed=mask_seed(
            seed, accel, family, anatomy=anatomy, task=name, eval_split=True
        ),
        device=device,
        batch_size=x_eval.shape[0],
        family=family,
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
    return TaskData(
        name=name,
        accel=int(accel),
        physics=physics,
        train=XYDataset(x_train.cpu(), y_train.cpu(), train_mask.cpu()),
        eval=XYDataset(x_eval.cpu(), y_eval.cpu(), eval_mask.cpu()),
        mask_family=str(family).lower(),
        anatomy=str(anatomy).lower(),
        modality="mri",
        physics_class="deepinv.physics.MRI",
        img_size=int(img_size),
    )


def build_tasks(
    x: torch.Tensor,
    seed: int,
    device: torch.device,
    img_size: int = 128,
    n_train: int | None = None,
    n_eval: int = 0,
    t1_accel: int = 4,
    t2_accel: int = 8,
    t1_mask_family: str = "gaussian",
    t2_mask_family: str = "gaussian",
    t1_anatomy: str = "knee",
    t2_anatomy: str = "knee",
    x_t2: torch.Tensor | None = None,
) -> dict[str, TaskData]:
    """T1 then T2. Default: Gaussian Cartesian 4× → 8× on the same images.

    Per-sample masks so buffer slots can be distinct (y, A). If n_eval>0,
    eval slices are held out from train. Pass x_t2 for a different anatomy.
    """
    device = _resolve_device(device)
    x = x.to(device)
    x_train, x_eval = _split_train_eval(x, n_train, n_eval)
    if x_t2 is None:
        x2_train, x2_eval = x_train, x_eval
    else:
        x2_train, x2_eval = _split_train_eval(x_t2.to(device), n_train, n_eval)
    return {
        "T1": _simulate_task(
            "T1",
            x_train,
            x_eval,
            accel=int(t1_accel),
            family=t1_mask_family,
            anatomy=t1_anatomy,
            seed=seed,
            device=device,
            img_size=img_size,
        ),
        "T2": _simulate_task(
            "T2",
            x2_train,
            x2_eval,
            accel=int(t2_accel),
            family=t2_mask_family,
            anatomy=t2_anatomy,
            seed=seed,
            device=device,
            img_size=img_size,
        ),
    }


def simulate_ct_task(
    x: torch.Tensor,
    *,
    n_train: int,
    n_eval: int,
    n_angles: int,
    img_size: int,
    device: torch.device,
    name: str = "T2",
) -> TaskData:
    """Sparse-view CT task. Geometry copied from deepinv physics-tour Tomography."""
    device = _resolve_device(device)
    x = as_modl_channels(x.to(device))
    x_train, x_eval = _split_train_eval(x, n_train, n_eval)
    physics = make_ct_physics(img_width=img_size, n_angles=n_angles, device=device)
    y_train = torch.stack(
        [physics(x_train[i : i + 1])[0] for i in range(x_train.shape[0])], dim=0
    )
    y_eval = torch.stack(
        [physics(x_eval[i : i + 1])[0] for i in range(x_eval.shape[0])], dim=0
    )
    return TaskData(
        name=name,
        accel=int(n_angles),
        physics=physics,
        train=XYDataset(x_train.cpu(), y_train.cpu(), mask=None),
        eval=XYDataset(x_eval.cpu(), y_eval.cpu(), mask=None),
        mask_family="tomography",
        anatomy="ct100",
        modality="ct",
        physics_class="deepinv.physics.Tomography",
        n_angles=int(n_angles),
        img_size=int(img_size),
    )


def build_cross_ip_tasks(
    x_mri: torch.Tensor,
    x_ct: torch.Tensor,
    seed: int,
    device: torch.device,
    *,
    n_train: int,
    n_eval: int,
    t1_accel: int = 4,
    t1_mask_family: str = "gaussian",
    t1_anatomy: str = "knee",
    mri_img_size: int = 128,
    ct_img_size: int = CT_DEMO_IMG_SIZE,
    ct_n_angles: int = CT_DEMO_N_ANGLES,
) -> dict[str, TaskData]:
    """T1 MRI (D-aligned) then T2 sparse-view CT. Eval sets are separate."""
    device = _resolve_device(device)
    t1 = _simulate_task(
        "T1",
        *_split_train_eval(x_mri.to(device), n_train, n_eval),
        accel=int(t1_accel),
        family=t1_mask_family,
        anatomy=t1_anatomy,
        seed=seed,
        device=device,
        img_size=mri_img_size,
    )
    t2 = simulate_ct_task(
        x_ct,
        n_train=n_train,
        n_eval=n_eval,
        n_angles=ct_n_angles,
        img_size=ct_img_size,
        device=device,
        name="T2",
    )
    return {"T1": t1, "T2": t2}


def task_operator_report(task: TaskData) -> dict[str, Any]:
    if task.train.mask is None:
        train_mask_ids = []
        n_distinct_train_A = 1
    else:
        train_mask_ids = [tensor_id(task.train.mask[i]) for i in range(len(task.train))]
        n_distinct_train_A = len(set(train_mask_ids))
    return {
        "task": task.name,
        "accel": task.accel,
        "modality": task.modality,
        "physics_class": task.physics_class,
        "n_angles": task.n_angles,
        "img_size": task.img_size,
        "mask_family": task.mask_family,
        "mask_generator": task.mask_generator,
        "anatomy": task.anatomy,
        "n_train": len(task.train),
        "n_eval": len(task.eval),
        "train_mask_ids": train_mask_ids,
        "n_distinct_train_A": n_distinct_train_A,
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
