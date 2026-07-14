import argparse
import json
import torch
import os
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from collections import defaultdict

from models.unet import UNet

from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float

from functions.radon import Radon, IRadon
from functions.experiment_io import (
    create_output_directories,
    load_config_file,
    save_resolved_config,
    save_run_metadata,
)
from kornia.geometry.transform import rotate
import random

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset and Physics classes
# ============================================================================
class test_CTData(Dataset):
    def __init__(self, root_dir='./CT/CT100_128x128.mat', num_train_samples=90, num_val_samples=0, num_test_samples=10):
        mat_data = scio.loadmat(root_dir)
        x = torch.from_numpy(mat_data['DATA'])

        test_start = num_train_samples + num_val_samples
        self.x = x[test_start:test_start + num_test_samples, ...]
        self.x = self.x.type(torch.FloatTensor)

    def __getitem__(self, index):
        x = self.x[index]
        return x

    def __len__(self):
        return len(self.x)


PI = 4*torch.ones(1).atan()


class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle = False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)

    # IRadon is a rescaled inverse rather than the exact adjoint.
    def A_adjoint(self, y):
        return self.iradon(y) / PI.item() * (2 * len(self.iradon.theta))


def cal_psnr(a, b):
    """
    Calculate PSNR between two images.
    Args:
        a: prediction
        b: ground-truth
    """
    alpha = np.sqrt(a.shape[-1] * a.shape[-2])
    return 20*torch.log10(alpha*torch.norm(b, float('inf'))/torch.norm(b-a, 2)).detach().cpu().numpy()


def cal_equiv(x_net, x_gt, physics, model, transform=None, n_trans=5, device='cuda:0', db=True):
    """
    Compute the EQUIV statistic used by the CT evaluation entrypoint.

    For each sampled rotation R, the implementation compares R(x_net) with
    model(A_dagger(A(R(x_gt)))). The mean squared discrepancy is reported in
    decibels when ``db`` is true.

    Args:
        x_net: Reconstructed image with shape [1, C, H, W].
        x_gt: Reference image with shape [1, C, H, W].
        physics: CT forward and pseudoinverse operators.
        model: Reconstruction network.
        transform: Reserved for interface compatibility.
        n_trans: Number of sampled rotations.
        device: Evaluation device.
        db: Convert the mean squared discrepancy to decibels.

    Returns:
        EQUIV in decibels when ``db`` is true, otherwise mean squared error.
    """
    return _equiv_metric_fn(
        x_net=x_net,
        x_gt=x_gt,
        physics=physics,
        model=model,
        transform=transform,
        n_trans=n_trans,
        device=device,
        db=db,
    )


def _get_rotation_angles(n_trans, random_rotate=True):
    """Return randomly sampled angles or a deterministic right-angle subset."""
    if random_rotate:
        # Sample angles in degrees from the same range used during training.
        theta_list = random.sample(list(np.arange(1, 359)), n_trans)
    else:
        available_angles = [90, 180, 270]
        theta_list = available_angles[:min(n_trans, len(available_angles))]
    
    return theta_list


def _equiv_metric_fn(x_net, x_gt, physics, model, transform, n_trans, device, db):
    """Core EQUIV computation; see :func:`cal_equiv` for the definition."""
    model.eval()

    angles = _get_rotation_angles(n_trans, random_rotate=True)

    out = None
    n_samples = len(angles)

    with torch.no_grad():
        for angle in angles:
            angle_tensor = torch.tensor([float(angle)], dtype=torch.float32).to(device)

            x_net_rotated = rotate(x_net, angle_tensor)

            gt_rotated = rotate(x_gt, angle_tensor)
            y_rotated = physics.A(gt_rotated)
            fbp_rotated = physics.A_dagger(y_rotated)
            x_net_re_reconstructed = model(fbp_rotated)

            mse = torch.mean((x_net_re_reconstructed - x_net_rotated) ** 2)

            if out is None:
                out = mse
            else:
                out = out + mse

    out = out / n_samples

    if db:
        out = -10 * torch.log10(out + 1e-10)

    return out.item()


def cal_ssim(a, b, multichannel=False):
    """
    Calculate SSIM between two images.
    Args:
        a: prediction
        b: ground-truth
    """
    b = img_as_float(b.squeeze().detach().cpu().numpy())
    a = img_as_float(a.squeeze().detach().cpu().numpy())
    return ssim(b, a, data_range=a.max() - a.min(), multichannel=multichannel)


def load_model(net, ckp, device):
    """Load a checkpoint and return the model in evaluation mode."""
    checkpoint = torch.load(ckp, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device).eval()
    return net


METHOD_TO_SLUG = {
    'EI': 'ei',
    'FEI-O1': 'fei_o1',
    'FEI-O2': 'fei_o2',
    'PnP-FEI-O1': 'pnp_fei_o1',
    'PnP-FEI-O2': 'pnp_fei_o2',
}


def _method_to_slug(method):
    if method not in METHOD_TO_SLUG:
        raise ValueError(f"Unsupported method: {method}")
    return METHOD_TO_SLUG[method]


def _build_eval_parser():
    parser = argparse.ArgumentParser(description='FEI CT evaluation entrypoint.')
    parser.add_argument('--config', required=True, help='Path to the experiment config file.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint to evaluate.')
    parser.add_argument('--device', default=None, help='Device override, e.g. cpu or cuda:0.')
    parser.add_argument('--output-dir', default=None, help='Directory to save evaluation artifacts.')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap on evaluated test samples.')
    return parser


def _is_debug_eval(args):
    checkpoint_path = os.path.normpath(args.checkpoint)
    return args.max_samples is not None or 'outputs_smoke' in checkpoint_path.split(os.sep)


def _run_ct_evaluation(args):
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
    transform_cfg = config.get('transform', {})
    model_cfg = config.get('model', {})

    dataloader = DataLoader(
        dataset=test_CTData(
            root_dir=dataset_cfg.get('path', './CT/CT100_128x128.mat'),
            num_train_samples=dataset_cfg.get('train_samples', 90),
            num_val_samples=dataset_cfg.get('val_samples', 0),
            num_test_samples=dataset_cfg.get('test_samples', 10),
        ),
        batch_size=1,
        shuffle=False,
    )

    physics = CT(
        img_width=model_cfg.get('image_width', 128),
        radon_view=physics_cfg.get('radon_view', 50),
        circle=physics_cfg.get('circle', False),
        device=device,
    )
    model = load_model(
        UNet(
            in_channels=model_cfg.get('in_channels', 1),
            out_channels=model_cfg.get('out_channels', 1),
            compact=model_cfg.get('compact', 4),
            residual=model_cfg.get('residual', True),
            circular_padding=model_cfg.get('circular_padding', True),
            cat=model_cfg.get('cat', True),
        ).to(device),
        args.checkpoint,
        device,
    )

    metrics_collector = defaultdict(lambda: {'psnr': [], 'ssim': [], 'equiv': []})
    max_samples = args.max_samples
    n_trans_equiv = transform_cfg.get('n_trans', 5)

    for index, x in enumerate(dataloader):
        if max_samples is not None and index >= max_samples:
            break

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(torch.float).to(device)

        y = physics.A(x)
        fbp = physics.A_dagger(y)
        prediction = model(fbp)

        metrics_collector['FBP']['psnr'].append(float(cal_psnr(fbp, x)))
        metrics_collector['FBP']['ssim'].append(float(cal_ssim(fbp, x)))
        metrics_collector[method]['psnr'].append(float(cal_psnr(prediction, x)))
        metrics_collector[method]['ssim'].append(float(cal_ssim(prediction, x)))
        metrics_collector[method]['equiv'].append(
            float(cal_equiv(prediction, x, physics, model, n_trans=n_trans_equiv, device=device))
        )

    summary = {
        'task': config['task'],
        'method': method,
        'checkpoint': args.checkpoint,
        'device': device,
        'debug_only': debug_only,
        'num_samples_evaluated': len(metrics_collector['FBP']['psnr']),
        'metrics': {},
    }

    for metric_method in ['FBP', method]:
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
        handle.write("Method\tPSNR_mean\tPSNR_std\tSSIM_mean\tSSIM_std\tEQUIV_mean\tEQUIV_std\n")
        for metric_method in ['FBP', method]:
            metric_entry = summary['metrics'][metric_method]
            equiv_entry = metric_entry.get('equiv', {})
            handle.write(
                f"{metric_method}\t"
                f"{metric_entry['psnr']['mean']:.6f}\t{metric_entry['psnr']['std']:.6f}\t"
                f"{metric_entry['ssim']['mean']:.6f}\t{metric_entry['ssim']['std']:.6f}\t"
                f"{equiv_entry.get('mean', float('nan')):.6f}\t{equiv_entry.get('std', float('nan')):.6f}\n"
            )

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    _run_ct_evaluation(_build_eval_parser().parse_args())
