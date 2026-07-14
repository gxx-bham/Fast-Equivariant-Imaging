import argparse
import json
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict

from models.unet import UNet
from functions.experiment_io import (
    create_output_directories,
    load_config_file,
    save_resolved_config,
    save_run_metadata,
)

from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset and Physics classes
# ============================================================================
class test_InpaintingData(Dataset):
    """Inpainting dataset class for loading Urban100 test data."""
    
    def __init__(self, dataset_name='Urban100', num_train_samples=None, num_val_samples=0,
                 num_test_samples=10, crop_size=(512, 512), resize=True, target_size=256):
        """
        Initialize Inpainting test dataset.
        
        Args:
            dataset_name: Name of dataset ('Urban100')
            num_train_samples: Number of training samples (for test split calculation)
            crop_size: Size to crop images
            resize: Whether to resize after cropping
            target_size: Target size after resizing
        """
        # Find dataset path
        if dataset_name == 'Urban100':
            if os.path.exists('../Urban100/'):
                base_path = '../Urban100/'
            else:
                base_path = './Urban100/'
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Set up transforms
        if resize:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])
        
        # Load full dataset
        imgs_path = os.path.join(base_path, 'all/')
        
        if not os.path.exists(imgs_path):
            raise FileNotFoundError(f"Dataset path not found: {imgs_path}")
        
        # Load all images using ImageFolder
        full_dataset = datasets.ImageFolder(imgs_path, transform=self.transform, 
                                            target_transform=None)
        
        # Get all images
        total_samples = len(full_dataset)
        
        # Default to the experiment split if not specified.
        if num_train_samples is None:
            num_train_samples = 90
        
        # Test set: after train and validation samples, up to num_test_samples
        test_start = num_train_samples + num_val_samples
        test_end = min(test_start + num_test_samples, total_samples)
        self.indices = list(range(test_start, test_end))
        
        self.full_dataset = full_dataset
    
    def __getitem__(self, index):
        """Get data sample at specified index."""
        actual_index = self.indices[index]
        img, _ = self.full_dataset[actual_index]  # Ignore class label
        return img
    
    def __len__(self):
        """Return dataset size."""
        return len(self.indices)


class Inpainting:
    """Inpainting physics model with mask-based observation."""
    
    def __init__(self, img_height=256, img_width=256, mask_rate=0.3,
                 resize=False, device='cuda:0'):
        """
        Initialize Inpainting physics model.
        
        Args:
            img_height: Image height
            img_width: Image width
            mask_rate: Fraction of pixels to mask
            resize: Whether to resize images
            device: Computing device
        """
        mask_path = f'./Urban100/mask_random{mask_rate}.pt'
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path).to(device)
        else:
            self.mask = torch.ones(img_height, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            os.makedirs('./Urban100', exist_ok=True)
            torch.save(self.mask, mask_path)

    def A(self, x):
        """Forward operator: apply mask."""
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_dagger(self, x):
        """Pseudoinverse operator: apply mask (same as forward for inpainting)."""
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_adjoint(self, x):
        """Adjoint operator: apply mask (same as forward for inpainting)."""
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)


def cal_psnr(a, b):
    """
    Calculate PSNR between two images.
    Args:
        a: prediction
        b: ground-truth
    """
    alpha = np.sqrt(a.shape[-1] * a.shape[-2])
    return 20*torch.log10(alpha*torch.norm(b, float('inf'))/torch.norm(b-a, 2)).detach().cpu().numpy()


def cal_ssim(a, b, multichannel=True):
    """
    Calculate SSIM between two images.
    Args:
        a: prediction [B, C, H, W] or [C, H, W]
        b: ground-truth [B, C, H, W] or [C, H, W]
    """
    # For RGB images, need to handle channel dimension differently
    if len(a.shape) == 4:  # [B, C, H, W]
        a = a[0]  # Take first batch
        b = b[0]
    
    # Permute from [C, H, W] to [H, W, C] for SSIM calculation
    a = a.detach().permute(1, 2, 0).cpu().numpy()
    b = b.detach().permute(1, 2, 0).cpu().numpy()
    
    b = img_as_float(b)
    a = img_as_float(a)
    
    # For RGB images, multichannel should be True (or channel_axis=2 for newer skimage)
    try:
        # Try new API first (skimage >= 0.19)
        return ssim(b, a, data_range=a.max() - a.min(), channel_axis=2)
    except (TypeError, ValueError):
        # Fall back to old API
        return ssim(b, a, data_range=a.max() - a.min(), multichannel=multichannel)


def _method_to_slug(method):
    if method not in METHOD_TO_SLUG:
        raise ValueError(f"Unsupported method: {method}")
    return METHOD_TO_SLUG[method]


def _build_eval_parser():
    parser = argparse.ArgumentParser(description='FEI Urban100 evaluation entrypoint.')
    parser.add_argument('--config', required=True, help='Path to the experiment config file.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint to evaluate.')
    parser.add_argument('--device', default=None, help='Device override, e.g. cpu or cuda:0.')
    parser.add_argument('--output-dir', default=None, help='Directory to save evaluation artifacts.')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap on evaluated test samples.')
    return parser


def _is_debug_eval(args):
    checkpoint_path = os.path.normpath(args.checkpoint)
    return args.max_samples is not None or 'outputs_smoke' in checkpoint_path.split(os.sep)


def _run_inpainting_evaluation(args):
    config = load_config_file(args.config)
    device = args.device or config.get('device', 'cuda:0')
    method = config['method']
    debug_only = _is_debug_eval(args)

    if args.output_dir is None:
        output_dirs = create_output_directories(
            task=config['task'],
            method=_method_to_slug(method),
            seed=config.get('seed', 0),
            base_dir='outputs_smoke' if debug_only else config.get('output', {}).get('root_dir', 'outputs'),
        )
        output_root = output_dirs['root']
    else:
        output_root = args.output_dir
        os.makedirs(output_root, exist_ok=True)

    save_resolved_config(config, os.path.join(output_root, 'eval_config_resolved.yaml'))
    if debug_only:
        with open(os.path.join(output_root, 'DEBUG_ONLY_SMOKE_EVAL.txt'), 'w', encoding='utf-8') as handle:
            handle.write('Diagnostic execution-check output; not a reported paper result.\n')

    dataset_cfg = config.get('dataset', {})
    physics_cfg = config.get('physics', {})
    model_cfg = config.get('model', {})

    dataloader = DataLoader(
        dataset=test_InpaintingData(
            dataset_name=dataset_cfg.get('name', 'Urban100'),
            num_train_samples=dataset_cfg.get('train_samples', 90),
            num_val_samples=dataset_cfg.get('val_samples', 0),
            num_test_samples=dataset_cfg.get('test_samples', 10),
            crop_size=tuple(dataset_cfg.get('crop_size', (512, 512))),
            resize=dataset_cfg.get('resize', True),
            target_size=dataset_cfg.get('target_size', 256),
        ),
        batch_size=1,
        shuffle=False,
    )

    physics = Inpainting(
        img_height=model_cfg.get('image_height', 256),
        img_width=model_cfg.get('image_width', 256),
        mask_rate=physics_cfg.get('mask_rate', 0.6),
        device=device,
    )
    model = UNet(
        in_channels=model_cfg.get('in_channels', 3),
        out_channels=model_cfg.get('out_channels', 3),
        compact=model_cfg.get('compact', 4),
        residual=model_cfg.get('residual', True),
        circular_padding=model_cfg.get('circular_padding', True),
        cat=model_cfg.get('cat', True),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    metrics_collector = defaultdict(lambda: {'psnr': [], 'ssim': []})
    max_samples = args.max_samples

    for index, x in enumerate(dataloader):
        if max_samples is not None and index >= max_samples:
            break

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.type(torch.float).to(device)

        y = physics.A(x)
        masked = physics.A_dagger(y)
        prediction = model(masked)

        metrics_collector['Masked']['psnr'].append(float(cal_psnr(masked, x)))
        metrics_collector['Masked']['ssim'].append(float(cal_ssim(masked, x)))
        metrics_collector[method]['psnr'].append(float(cal_psnr(prediction, x)))
        metrics_collector[method]['ssim'].append(float(cal_ssim(prediction, x)))

    summary = {
        'task': config['task'],
        'method': method,
        'checkpoint': args.checkpoint,
        'device': device,
        'debug_only': debug_only,
        'num_samples_evaluated': len(metrics_collector['Masked']['psnr']),
        'metrics': {},
    }

    for metric_method in ['Masked', method]:
        metric_summary = {}
        for metric_name, values in metrics_collector[metric_method].items():
            if len(values) == 0:
                continue
            values_array = np.array(values, dtype=np.float64)
            metric_summary[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                'values': [float(v) for v in values_array.tolist()],
            }
        summary['metrics'][metric_method] = metric_summary

    save_run_metadata(os.path.join(output_root, 'eval_summary.json'), summary)

    stats_file = os.path.join(output_root, 'test_metrics_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as handle:
        handle.write("Method\tPSNR_mean\tPSNR_std\tSSIM_mean\tSSIM_std\n")
        for metric_method in ['Masked', method]:
            metric_entry = summary['metrics'][metric_method]
            handle.write(
                f"{metric_method}\t"
                f"{metric_entry['psnr']['mean']:.6f}\t{metric_entry['psnr']['std']:.6f}\t"
                f"{metric_entry['ssim']['mean']:.6f}\t{metric_entry['ssim']['std']:.6f}\n"
            )

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    _run_inpainting_evaluation(_build_eval_parser().parse_args())
