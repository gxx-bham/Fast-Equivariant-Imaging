"""
Training script for CT reconstruction models based on Equivariant Imaging and FEI.

Supported algorithms:
- EI: Equivariant Imaging (vanilla version without noise model)
- FEI-O1: FEI Option 1 with NAG optimization
- FEI-O2: FEI Option 2 with ADMM optimization
- PnP-FEI-O1: FEI Option 1 with a pretrained plug-and-play denoiser
- PnP-FEI-O2: FEI Option 2 with a pretrained plug-and-play denoiser
"""

# ============================================================================
# Import libraries
# ============================================================================
# Standard library
import argparse
import copy
import itertools
import json
import os
import random
import time
import warnings
from datetime import datetime, timezone

# Third-party libraries - numerical computation
import numpy as np
import scipy.io as scio

# PyTorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset

# PyTorch vision related
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Image processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float
from kornia.geometry.transform import rotate

# DeepInv framework
import deepinv as dinv

# Local modules
from functions.experiment_io import (
    build_run_metadata,
    collect_environment_info,
    collect_git_info,
    create_output_directories,
    load_config_file,
    save_environment_info,
    save_git_info,
    save_resolved_config,
    save_run_metadata,
    set_global_seed,
)
from functions.radon import Radon, IRadon
from functions.nn import adjust_learning_rate
from functions.utils_dinv import get_model, EquivariantDenoiser
from models.unet import UNet

# Ignore warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Constant definitions
# ============================================================================
PI = 4 * torch.ones(1).atan()


# ============================================================================
# Dataset class
# ============================================================================
class CTData(Dataset):
    """CT dataset class for loading data for CT reconstruction tasks."""
    
    def __init__(self, root_dir='./CT/CT100_128x128.mat', mode='train', num_train_samples=None):
        """
        Initialize CT dataset.
        
        Args:
            root_dir: Path to MAT data file
            mode: 'train' or 'test', for dataset split
            num_train_samples: Number of training samples (if None, uses default 90)
        """
        mat_data = scio.loadmat(root_dir)
        x = torch.from_numpy(mat_data['DATA'])
        
        # Default to the experiment split if not specified.
        if num_train_samples is None:
            num_train_samples = 90

        if mode == 'train':
            self.x = x[0: num_train_samples]
        elif mode == 'test':
            # Test set starts from 90 to end (typically 100 total samples)
            self.x = x[num_train_samples: num_train_samples + 10, ...]
        else:
            raise ValueError(f"Unsupported mode: {mode}, should be 'train' or 'test'")

        self.x = self.x.type(torch.FloatTensor)

    def __getitem__(self, index):
        """Get data sample at specified index."""
        x = self.x[index]
        return x

    def __len__(self):
        """Return dataset size."""
        return len(self.x)


# ============================================================================
# Inpainting dataset class
# ============================================================================
class InpaintingData(Dataset):
    """Inpainting dataset class for loading Urban100 or similar image datasets."""
    
    def __init__(self, dataset_name='Urban100', mode='train', num_train_samples=None,
                 crop_size=(512, 512), resize=True, target_size=256):
        """
        Initialize Inpainting dataset.
        
        Args:
            dataset_name: Name of dataset ('Urban100')
            mode: 'train' or 'test', for dataset split
            num_train_samples: Number of training samples (if None, uses all available)
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
        # For Urban100, we use the 'all' folder which contains all images
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
        
        # Split dataset
        if mode == 'train':
            # Training set: first num_train_samples
            self.indices = list(range(0, min(num_train_samples, total_samples)))
        elif mode == 'test':
            # Test set: after training samples, up to 10 samples
            test_start = num_train_samples
            test_end = min(test_start + 10, total_samples)
            self.indices = list(range(test_start, test_end))
        else:
            raise ValueError(f"Unsupported mode: {mode}, should be 'train' or 'test'")
        
        self.full_dataset = full_dataset
        self.mode = mode
        
    def __getitem__(self, index):
        """Get data sample at specified index."""
        actual_index = self.indices[index]
        img, _ = self.full_dataset[actual_index]  # Ignore class label
        return img
    
    def __len__(self):
        """Return dataset size."""
        return len(self.indices)


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


def _task_to_internal(task):
    if task == 'urban100':
        return 'inpainting'
    return task


def _to_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


def _resolve_checkpoint_dir(task, lr_value, method, epochs, output_dirs=None):
    if output_dirs and 'checkpoints' in output_dirs:
        checkpoint_dir = output_dirs['checkpoints']
    else:
        fallback_names = {
            'EI': 'Avg-EI-New-vani',
            'FEI-O1': 'Avg-FEI-Option1',
            'FEI-O2': 'Avg-FEI-Option2',
            'PnP-FEI-O1': 'Avg-PnP-FEI-Option1',
            'PnP-FEI-O2': 'Avg-PnP-FEI-Option2',
        }
        checkpoint_dir = f'./{task}_ckp/{fallback_names[method]}/epochs_{epochs}_lr_{lr_value}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def _resolve_results_dir(task, lr_value, method, output_dirs=None):
    if output_dirs and 'root' in output_dirs:
        root_dir = output_dirs['root']
    else:
        fallback_names = {
            'EI': 'Results-Avg-EI-New-vani',
            'FEI-O1': f'Results-Avg-FEI-Option1_lr_{lr_value}',
            'FEI-O2': f'Results-Avg-FEI-Option2_lr_{lr_value}',
            'PnP-FEI-O1': f'Results-Avg-PnP-FEI-Option1_lr_{lr_value}',
            'PnP-FEI-O2': f'Results-Avg-PnP-FEI-Option2_lr_{lr_value}',
        }
        root_dir = os.path.join(f'{task}-Results', fallback_names[method])
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def _build_training_dataset_from_config(config):
    task = _task_to_internal(config['task'])
    dataset_cfg = config.get('dataset', {})

    if task == 'ct':
        dataset = CTData(
            root_dir=dataset_cfg.get('path', './CT/CT100_128x128.mat'),
            mode='train',
            num_train_samples=dataset_cfg.get('train_samples', 90),
        )
    elif task == 'inpainting':
        dataset = InpaintingData(
            dataset_name=dataset_cfg.get('name', 'Urban100'),
            mode='train',
            num_train_samples=dataset_cfg.get('train_samples', 90),
            crop_size=_to_tuple(dataset_cfg.get('crop_size', (512, 512))),
            resize=dataset_cfg.get('resize', True),
            target_size=dataset_cfg.get('target_size', 256),
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return dataset


def _build_physics_from_config(config, device):
    task = _task_to_internal(config['task'])
    physics_cfg = config.get('physics', {})

    if task == 'ct':
        return CT(
            img_width=config.get('model', {}).get('image_width', 128),
            radon_view=physics_cfg.get('radon_view', 50),
            circle=physics_cfg.get('circle', False),
            device=device,
        )

    if task == 'inpainting':
        return Inpainting(
            img_height=config.get('model', {}).get('image_height', 256),
            img_width=config.get('model', {}).get('image_width', 256),
            mask_rate=physics_cfg.get('mask_rate', 0.6),
            device=device,
        )

    raise ValueError(f"Unsupported task: {task}")


def _build_transform_from_config(config):
    task = _task_to_internal(config['task'])
    transform_cfg = config.get('transform', {})

    if task == 'ct':
        return Rotate(
            n_trans=transform_cfg.get('n_trans', 5),
            random_rotate=transform_cfg.get('random_rotate', True),
        )

    if task == 'inpainting':
        return Shift(
            n_trans=transform_cfg.get('n_trans', 3),
            max_offset=transform_cfg.get('max_offset', 0),
        )

    raise ValueError(f"Unsupported task: {task}")


def _build_model_from_config(config, device):
    model_cfg = config.get('model', {})
    net = UNet(
        in_channels=model_cfg.get('in_channels', 1),
        out_channels=model_cfg.get('out_channels', 1),
        compact=model_cfg.get('compact', 4),
        residual=model_cfg.get('residual', True),
        circular_padding=model_cfg.get('circular_padding', True),
        cat=model_cfg.get('cat', True),
    )
    return net.to(device)


def _apply_train_sample_cap(dataset, max_train_samples):
    if max_train_samples is None:
        return dataset

    capped_length = min(len(dataset), int(max_train_samples))
    return Subset(dataset, list(range(capped_length)))


def _get_split_sizes(dataset_cfg):
    train_samples = int(dataset_cfg.get('train_samples', 90))
    val_samples = int(dataset_cfg.get('val_samples', 0))
    test_samples = int(dataset_cfg.get('test_samples', 10))
    return train_samples, val_samples, test_samples


def _get_split_descriptors(task, dataset_cfg):
    train_samples, val_samples, test_samples = _get_split_sizes(dataset_cfg)
    train_end = train_samples
    val_end = train_samples + val_samples
    test_end = val_end + test_samples
    return {
        'train_split': f'0:{train_end}',
        'val_split': f'{train_end}:{val_end}' if val_samples > 0 else None,
        'test_split': f'{val_end}:{test_end}',
        'split_policy': dataset_cfg.get('split_policy', 'first90_last10'),
    }


def _default_snapshot_epochs(task):
    if task == 'ct':
        return [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    if task == 'urban100':
        return [100, 300, 500, 1000, 1500, 2000]
    return []


def _build_validation_dataloader_for_task(task, batch_size=1, dataset_config=None, max_val_samples=None):
    dataset_config = dataset_config or {}
    train_samples, val_samples, _ = _get_split_sizes(dataset_config)
    train_plus_val = train_samples + val_samples

    if task == 'ct':
        base_dataset = CTData(
            root_dir=dataset_config.get('path', './CT/CT100_128x128.mat'),
            mode='train',
            num_train_samples=train_plus_val,
        )
    elif task == 'inpainting':
        base_dataset = InpaintingData(
            dataset_name=dataset_config.get('name', 'Urban100'),
            mode='train',
            num_train_samples=train_plus_val,
            crop_size=_to_tuple(dataset_config.get('crop_size', (512, 512))),
            resize=dataset_config.get('resize', True),
            target_size=dataset_config.get('target_size', 256),
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    val_indices = list(range(train_samples, train_plus_val))
    if max_val_samples is not None:
        capped_length = min(len(val_indices), int(max_val_samples))
        val_indices = val_indices[:capped_length]

    val_dataset = Subset(base_dataset, val_indices)
    return DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


def _resolve_smoke_settings(args):
    smoke_test = bool(args.smoke_test)
    max_epochs = args.max_epochs
    max_batches = args.max_batches

    if smoke_test:
        if max_epochs is None:
            max_epochs = 1
        if max_batches is None:
            max_batches = 2

    return {
        'smoke_test': smoke_test,
        'max_epochs': max_epochs,
        'max_batches': max_batches,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': 2 if smoke_test else None,
    }


def _resolve_runtime_config(config, args):
    resolved = copy.deepcopy(config)
    output_cfg = resolved.setdefault('output', {})
    smoke_settings = _resolve_smoke_settings(args)

    if args.seed is not None:
        resolved['seed'] = args.seed
    else:
        resolved['seed'] = resolved.get('seed', 0)

    if args.device:
        resolved['device'] = args.device
    else:
        resolved['device'] = resolved.get('device', 'cuda:0')

    if args.output_dir:
        output_cfg['root_dir'] = args.output_dir
    elif smoke_settings['smoke_test']:
        output_cfg['root_dir'] = 'outputs_smoke'

    resolved['runtime'] = {
        'resume': args.resume,
        'dry_run_config': bool(args.dry_run_config),
        'smoke_test': smoke_settings['smoke_test'],
        'max_epochs': smoke_settings['max_epochs'],
        'max_batches': smoke_settings['max_batches'],
        'max_train_samples': smoke_settings['max_train_samples'],
        'max_val_samples': smoke_settings['max_val_samples'],
        'log_interval': int(args.log_interval),
    }

    dataset_cfg = resolved.setdefault('dataset', {})
    train_cfg = resolved.setdefault('train', {})
    task_name = resolved.get('task')

    dataset_cfg.setdefault('train_samples', 90)
    dataset_cfg.setdefault('val_samples', 0)
    dataset_cfg.setdefault('test_samples', 10)
    dataset_cfg.setdefault('split_policy', 'first90_last10')
    train_cfg.setdefault('validation_interval', 0)
    train_cfg.setdefault('snapshot_epochs', _default_snapshot_epochs(task_name))

    selection_cfg = resolved.setdefault('selection', {})
    selection_cfg.setdefault('checkpoint_selection_metric', 'final_epoch')
    selection_cfg.setdefault('evaluation_checkpoint', 'final_model.pth.tar')
    selection_cfg.setdefault('early_stopping_metric', 'none')

    return resolved


# ============================================================================
# Test evaluation function
# ============================================================================
def evaluate_test_set(net, test_dataloader, physics, device, dtype):
    """
    Evaluate model performance on test set.
    
    Args:
        net: Neural network model
        test_dataloader: Test data loader
        physics: CT physics model
        device: Computing device
        dtype: Data type
        
    Returns:
        avg_psnr: Average PSNR across test samples
        avg_mse: Average MSE across test samples
        psnr_list: List of PSNR values for each sample
        mse_list: List of MSE values for each sample
    """
    net.eval()
    psnr_list = []
    mse_list = []
    
    with torch.no_grad():
        for x_batch in test_dataloader:
            x = _prepare_batch_data(x_batch, dtype, device)
            
            # Generate noisy-free measurement for testing
            y0 = physics.A(x)
            x0 = physics.A_dagger(y0)
            
            # Model inference
            x_pred = net(x0)
            
            # Compute metrics for each sample in the batch
            if len(x_pred.shape) == 4 and x_pred.shape[0] > 1:
                # Batch processing: compute metrics for each sample separately
                batch_size = x_pred.shape[0]
                for b in range(batch_size):
                    psnr_b = cal_psnr(x_pred[b:b+1], x[b:b+1])
                    mse_b = cal_mse(x_pred[b:b+1], x[b:b+1])
                    psnr_list.append(psnr_b)
                    mse_list.append(mse_b)
            else:
                # Single sample processing
                psnr = cal_psnr(x_pred, x)
                mse = cal_mse(x_pred, x)
                psnr_list.append(psnr)
                mse_list.append(mse)
    
    net.train()
    return np.mean(psnr_list), np.mean(mse_list), psnr_list, mse_list


def _limited_batch_iterator(dataloader, max_batches=None):
    batch_iter = enumerate(dataloader)
    if max_batches is None:
        return batch_iter
    return itertools.islice(batch_iter, int(max_batches))


def _effective_iterations_per_epoch(dataloader, max_batches=None):
    if max_batches is None:
        return len(dataloader)
    return min(len(dataloader), int(max_batches))


def _should_run_validation(epoch, epochs, validation_interval):
    if validation_interval is None or int(validation_interval) <= 0:
        return False
    epoch_number = epoch + 1
    return (
        epoch_number == 1
        or (validation_interval > 0 and epoch_number % int(validation_interval) == 0)
        or epoch == epochs - 1
    )


def _merge_checkpoint_type(current_value, new_value):
    if not new_value:
        return current_value
    if not current_value:
        return new_value
    existing = current_value.split('+')
    if new_value in existing:
        return current_value
    return f"{current_value}+{new_value}"


def _build_checkpoint_state(net, optimizer, epoch, extra_state=None):
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if extra_state:
        state.update(extra_state)
    return state


def _save_explicit_checkpoint(net, optimizer, epoch, checkpoint_path, extra_state=None):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    state = _build_checkpoint_state(net, optimizer, epoch, extra_state=extra_state)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def _save_snapshot_checkpoint(net, optimizer, epoch, save_path, snapshot_epochs, extra_state=None):
    snapshot_epochs = snapshot_epochs or []
    epoch_number = epoch + 1
    if epoch_number not in set(int(item) for item in snapshot_epochs):
        return None

    snapshot_dir = os.path.join(save_path, 'snapshots')
    snapshot_path = os.path.join(snapshot_dir, f'epoch_{epoch_number}.pth.tar')
    return _save_explicit_checkpoint(net, optimizer, epoch, snapshot_path, extra_state=extra_state)


def _run_validation_pass(net, val_dataloader, physics, device, dtype, epoch, optimizer,
                         save_path, val_psnr_dict, val_mse_dict, best_val_psnr,
                         selected_epoch, no_val_improve_count, last_val_epoch,
                         training_log_context=None, evaluation_checkpoint='validation_selected_model.pth.tar'):
    val_psnr, val_mse, val_psnr_list, val_mse_list = evaluate_test_set(
        net, val_dataloader, physics, device, dtype
    )
    val_psnr_dict[epoch] = (val_psnr, val_psnr_list)
    val_mse_dict[epoch] = (val_mse, val_mse_list)

    epoch_number = epoch + 1
    best_checkpoint_saved = False
    best_checkpoint_path = None
    checkpoint_type = None

    if val_psnr > best_val_psnr:
        best_val_psnr = val_psnr
        selected_epoch = epoch_number
        no_val_improve_count = 0
        best_checkpoint_path = os.path.join(save_path, evaluation_checkpoint)
        _save_explicit_checkpoint(
            net,
            optimizer,
            epoch,
            best_checkpoint_path,
            extra_state={
                'best_val_psnr': _json_safe_number(best_val_psnr),
                'selected_epoch': _json_safe_number(selected_epoch),
                'selection_metric': 'val_psnr',
            },
        )
        best_checkpoint_saved = True
        checkpoint_type = 'validation_selected_model'
    else:
        if last_val_epoch is not None:
            no_val_improve_count += epoch_number - last_val_epoch

    last_val_epoch = epoch_number

    if training_log_context:
        _log_validation_metrics(
            training_log_context,
            epoch=epoch,
            val_psnr=val_psnr,
            val_mse=val_mse,
            best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
            no_val_improve_count=no_val_improve_count,
            selected_epoch=selected_epoch,
            checkpoint_saved=best_checkpoint_saved,
            checkpoint_type=checkpoint_type,
            early_stop_triggered=False,
        )
        if best_checkpoint_saved:
            _emit_checkpoint_notice(
                training_log_context,
                epoch,
                best_checkpoint_path,
                'validation_selected_model',
                best_val_psnr=best_val_psnr,
                selected_epoch=selected_epoch,
            )
    else:
        print(
            f"  [Validation] Epoch {epoch_number}: PSNR = {val_psnr:.4f} dB, "
            f"MSE = {val_mse:.6f}, best_val_psnr = {best_val_psnr:.4f}"
        )
        if best_checkpoint_saved:
            print(f"  [Best Val Model] Updated! Validation PSNR: {best_val_psnr:.4f} dB at epoch {selected_epoch}")

    return {
        'val_psnr': val_psnr,
        'val_mse': val_mse,
        'val_psnr_list': val_psnr_list,
        'val_mse_list': val_mse_list,
        'best_val_psnr': best_val_psnr,
        'selected_epoch': selected_epoch,
        'no_val_improve_count': no_val_improve_count,
        'last_val_epoch': last_val_epoch,
        'best_checkpoint_saved': best_checkpoint_saved,
        'best_checkpoint_path': best_checkpoint_path,
        'checkpoint_type': checkpoint_type,
    }


def _finalize_training_run(net, optimizer, last_completed_epoch, save_path, output_root,
                           training_log_context, epochs, early_stopped, best_val_psnr,
                           selected_epoch, validation_interval, patience,
                           evaluation_checkpoint):
    epochs_completed = max(0, int(last_completed_epoch) + 1)
    summary_best_val_psnr = best_val_psnr if np.isfinite(best_val_psnr) else None
    total_train_time_sec = None
    average_epoch_time_sec = None

    final_checkpoint = None
    latest_checkpoint = None
    if last_completed_epoch >= 0:
        selection_metric = (
            training_log_context['checkpoint_selection_metric']
            if training_log_context is not None
            else 'final_epoch'
        )
        checkpoint_epoch = epochs_completed if selection_metric == 'final_epoch' else selected_epoch
        extra_state = {
            'best_val_psnr': _json_safe_number(summary_best_val_psnr),
            'selected_epoch': _json_safe_number(checkpoint_epoch),
            'selection_metric': selection_metric,
        }
        final_checkpoint = os.path.join(save_path, 'final_model.pth.tar')
        latest_checkpoint = os.path.join(save_path, 'latest.pth.tar')
        _save_explicit_checkpoint(net, optimizer, last_completed_epoch, final_checkpoint, extra_state=extra_state)
        _save_explicit_checkpoint(net, optimizer, last_completed_epoch, latest_checkpoint, extra_state=extra_state)
        if training_log_context:
            _emit_checkpoint_notice(
                training_log_context,
                last_completed_epoch,
                final_checkpoint,
                'final_model',
                best_val_psnr=summary_best_val_psnr,
                selected_epoch=selected_epoch,
            )
            _emit_checkpoint_notice(
                training_log_context,
                last_completed_epoch,
                latest_checkpoint,
                'latest_checkpoint',
                best_val_psnr=summary_best_val_psnr,
                selected_epoch=selected_epoch,
            )

    progress_path = None if training_log_context is None else training_log_context['progress_path']
    if progress_path and os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as handle:
                epoch_records = [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]
            epoch_events = [item for item in epoch_records if item.get('event') == 'epoch']
            if epoch_events:
                elapsed_values = [
                    item.get('elapsed_sec')
                    for item in epoch_events
                    if item.get('elapsed_sec') is not None
                ]
                if elapsed_values:
                    total_train_time_sec = float(elapsed_values[-1])
                epoch_time_values = [
                    item.get('epoch_time_sec')
                    for item in epoch_events
                    if item.get('epoch_time_sec') is not None
                ]
                if epoch_time_values:
                    average_epoch_time_sec = float(sum(epoch_time_values) / len(epoch_time_values))
        except Exception:
            total_train_time_sec = None
            average_epoch_time_sec = None

    summary = {
        'task': None if training_log_context is None else training_log_context['task'],
        'method': None if training_log_context is None else training_log_context['method'],
        'seed': None if training_log_context is None else int(training_log_context['seed']),
        'epochs_requested': int(epochs),
        'epochs_completed': epochs_completed,
        'checkpoint_selection_metric': None if training_log_context is None else training_log_context['checkpoint_selection_metric'],
        'selected_epoch': _json_safe_number(
            epochs_completed
            if training_log_context is not None
            and training_log_context['checkpoint_selection_metric'] == 'final_epoch'
            else selected_epoch
        ),
        'selected_checkpoint': (
            os.path.join(save_path, evaluation_checkpoint)
            if os.path.exists(os.path.join(save_path, evaluation_checkpoint))
            else None
        ),
        'total_train_time_sec': _json_safe_number(total_train_time_sec),
        'average_epoch_time_sec': _json_safe_number(average_epoch_time_sec),
        'train_split': None if training_log_context is None else training_log_context['train_split'],
        'test_split': None if training_log_context is None else training_log_context['test_split'],
        'evaluation_checkpoint': evaluation_checkpoint,
        'final_checkpoint': final_checkpoint,
        'latest_checkpoint': latest_checkpoint,
    }
    if int(validation_interval) > 0:
        summary.update(
            {
                'validation_interval': int(validation_interval),
                'val_split': None if training_log_context is None else training_log_context['val_split'],
                'best_val_psnr': _json_safe_number(summary_best_val_psnr),
                'early_stopped': bool(early_stopped),
                'early_stop_metric': (
                    None if training_log_context is None else training_log_context['early_stopping_metric']
                ),
                'patience': int(patience),
            }
        )
    save_run_metadata(os.path.join(output_root, 'training_summary.json'), summary)
    return summary


# ============================================================================
# CT physics model class
# ============================================================================
class CT:
    """CT imaging physics model, containing Radon transform."""
    
    def __init__(self, img_width, radon_view, uniform=True, circle=False, device='cuda:0'):
        """
        Initialize CT physics model.
        
        Args:
            img_width: Image width
            radon_view: Number of angular views for Radon transform
            uniform: Whether to use uniform angular distribution
            circle: Whether to use circular projection
            device: Computing device
        """
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
            
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)

    def A(self, x):
        """
        Forward operator: perform Radon transform (projection).
        
        Args:
            x: Input image
            
        Returns:
            y: Projection data (sinogram)
        """
        return self.radon(x)

    def A_dagger(self, y):
        """
        Pseudoinverse operator: perform inverse Radon transform (filtered back-projection).
        
        Returns:
            x: Reconstructed image
        """
        return self.iradon(y)

    def A_adjoint(self, y):
        """
        Adjoint operator: adjoint of Radon transform (rescaled version).
        
        Note: IRadon is not an exact adjoint, but a rescaled version.
        """
        return self.iradon(y) / PI.item() * (2 * len(self.iradon.theta))


# ============================================================================
# Inpainting physics model class
# ============================================================================
class Inpainting:
    """Inpainting physics model with mask-based observation."""
    
    def __init__(self, img_height=512, img_width=512, mask_rate=0.3,
                 resize=False, device='cuda:0'):
        """
        Initialize Inpainting physics model.
        
        Args:
            img_height: Image height
            img_width: Image width
            mode: Mask mode ('random')
            mask_rate: Percentage of pixels to mask
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


# ============================================================================
# Data augmentation: rotation transform (for CT)
# ============================================================================
class Rotate:
    """Rotation transform class for data augmentation in equivariant imaging."""
    
    def __init__(self, n_trans, random_rotate=True):
        """
        Initialize rotation transform.
        
        Args:
            n_trans: Number of transforms
            random_rotate: Whether to randomly select rotation angles
        """
        self.n_trans = n_trans
        self.random_rotate = random_rotate
        
    def apply(self, x):
        """Apply rotation transform."""
        return rotate_dgm(x, self.n_trans, self.random_rotate)


def rotate_dgm(data, n_trans=5, random_rotate=False):
    """
    Apply multiple rotation transforms to data for data augmentation.
    
    Args:
        data: Input data tensor
        n_trans: Number of transforms
        random_rotate: Whether to randomly select angles
        
    Returns:
        Rotated data (concatenated along dimension 0)
    """
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 359)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))

    data = torch.cat([
        data if theta == 0 else rotate(data, torch.Tensor([theta]).type_as(data))
        for theta in theta_list
    ], dim=0)
    return data


# ============================================================================
# Data augmentation: shift transform (for Inpainting)
# ============================================================================
class Shift:
    """Shift transform class for data augmentation in inpainting tasks."""
    
    def __init__(self, n_trans, max_offset=0):
        """
        Initialize shift transform.
        
        Args:
            n_trans: Number of transforms
            max_offset: Maximum shift offset (0 means use full range)
        """
        self.n_trans = n_trans
        self.max_offset = max_offset
        
    def apply(self, x):
        """Apply shift transform."""
        return shift_random(x, self.n_trans, self.max_offset)


def shift_random(x, n_trans=5, max_offset=0):
    """
    Apply multiple random shift transforms to data for data augmentation.
    
    Args:
        x: Input data tensor
        n_trans: Number of transforms
        max_offset: Maximum shift offset (0 means use full range)
        
    Returns:
        Shifted data (concatenated along dimension 0)
    """
    H, W = x.shape[-2], x.shape[-1]
    assert n_trans <= H - 1 and n_trans <= W - 1, f'n_shifts should be less than {H-1}'

    if max_offset == 0:
        shifts_row = random.sample(list(np.concatenate([-1 * np.arange(1, H), np.arange(1, H)])), n_trans)
        shifts_col = random.sample(list(np.concatenate([-1 * np.arange(1, W), np.arange(1, W)])), n_trans)
    else:
        assert max_offset <= min(H, W), 'max_offset must be less than min(H,W)'
        shifts_row = random.sample(list(np.concatenate([-1 * np.arange(1, max_offset), np.arange(1, max_offset)])), n_trans)
        shifts_col = random.sample(list(np.concatenate([-1 * np.arange(1, max_offset), np.arange(1, max_offset)])), n_trans)

    x = torch.cat([
        x if n_trans == 0 else torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x) 
        for sx, sy in zip(shifts_row, shifts_col)
    ], dim=0)
    return x


# ============================================================================
# Dataset loader for inpainting (Urban100)
# ============================================================================
def DIP_CVData(dataset_name='Urban100', mode='dip', batch_size=1, shuffle=True, 
               crop_size=(512, 512), resize=True):
    """
    Load computer vision dataset for inpainting tasks.
    
    Args:
        dataset_name: Name of dataset ('Urban100')
        mode: Data mode ('train', 'test', or 'dip')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        crop_size: Size to crop images
        resize: Whether to resize after cropping
        
    Returns:
        DataLoader for the specified dataset
    """
    if dataset_name == 'Urban100':
        if os.path.exists('../Urban100/'):
            imgs_path = '../Urban100/'
        else:
            imgs_path = './Urban100/'

    if resize:
        transform_data = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(256),
            transforms.ToTensor()
        ])
    else:
        transform_data = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

    if mode == 'train':
        imgs_path = imgs_path + 'train/'
    elif mode == 'test':
        imgs_path = imgs_path + 'test/'
    elif mode == 'dip':
        imgs_path = imgs_path + 'dip/'

    dataset = datasets.ImageFolder(imgs_path, transform=transform_data, target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


# ============================================================================
# Evaluation metric functions
# ============================================================================
def cal_psnr(prediction, ground_truth):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        prediction: Predicted image (can be single sample or batch)
        ground_truth: Ground truth image (can be single sample or batch)
        
    Returns:
        PSNR value (scalar)
        - For single sample [C,H,W] or [1,C,H,W]: returns single PSNR value
        - For batch [B,C,H,W] where B>1: returns average PSNR across batch
    """
    # Handle batch dimension
    if len(prediction.shape) == 4 and prediction.shape[0] > 1:  # True batch format [B>1, C, H, W]
        batch_size = prediction.shape[0]
        psnr_values = []
        for b in range(batch_size):
            pred_b = prediction[b:b+1]
            gt_b = ground_truth[b:b+1]
            alpha = np.sqrt(pred_b.shape[-1] * pred_b.shape[-2])
            psnr = 20 * torch.log10(
                alpha * torch.norm(gt_b, float('inf')) / torch.norm(gt_b - pred_b, 2)
            ).detach().cpu().numpy()
            psnr_values.append(psnr)
        return np.mean(psnr_values)  # Return average PSNR for batch
    else:
        # Single sample: [C,H,W] or [1,C,H,W]
        alpha = np.sqrt(prediction.shape[-1] * prediction.shape[-2])
        return 20 * torch.log10(
            alpha * torch.norm(ground_truth, float('inf')) / torch.norm(ground_truth - prediction, 2)
        ).detach().cpu().numpy()


def cal_mse(prediction, ground_truth, mask=None):
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        prediction: Predicted image (can be single sample or batch)
        ground_truth: Ground truth image (can be single sample or batch)
        mask: Optional mask for computing specific regions
        
    Returns:
        MSE value (scalar)
        - For single sample [C,H,W] or [1,C,H,W]: returns single MSE value
        - For batch [B,C,H,W] where B>1: returns average MSE across batch
    """
    # Handle batch dimension (consistent with cal_psnr)
    if len(prediction.shape) == 4 and prediction.shape[0] > 1:  # True batch format [B>1, C, H, W]
        batch_size = prediction.shape[0]
        mse_values = []
        for b in range(batch_size):
            pred_b = prediction[b:b+1]
            gt_b = ground_truth[b:b+1]
            if mask is None:
                mse = nn.MSELoss()(pred_b, gt_b).item()
            else:
                mask_b = mask[b:b+1]
                mse = nn.MSELoss()(pred_b[mask_b > 0], gt_b[mask_b > 0]).item()
            mse_values.append(mse)
        return np.mean(mse_values)  # Return average MSE for batch
    else:
        # Single sample: [C,H,W] or [1,C,H,W]
        if mask is None:
            return nn.MSELoss()(prediction, ground_truth).item()
        else:
            return nn.MSELoss()(prediction[mask > 0], ground_truth[mask > 0]).item()


def cal_ssim(prediction, ground_truth, multichannel=False):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        prediction: Predicted image
        ground_truth: Ground truth image
        multichannel: Whether it is a multi-channel image
        
    Returns:
        SSIM value (scalar)
    """
    ground_truth_float = img_as_float(ground_truth.squeeze().detach().cpu().numpy())
    prediction_float = img_as_float(prediction.squeeze().detach().cpu().numpy())
    return ssim(
        ground_truth_float, 
        prediction_float, 
        data_range=prediction_float.max() - prediction_float.min(),
        multichannel=multichannel
    )


# ============================================================================
# Training helper functions
# ============================================================================
def _prepare_batch_data(x_batch, dtype, device):
    """
    Prepare batch data, ensuring correct shape and type.
    
    Args:
        x_batch: Raw batch data
        dtype: Target data type
        device: Target device
        
    Returns:
        Processed batch data
    """
    x = x_batch[0] if isinstance(x_batch, list) else x_batch
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    return x.type(dtype).to(device)


def _setup_loss_criterion(loss_type, device):
    """
    Setup loss functions.
    
    Args:
        loss_type: 'l2' or 'l1'
        device: Computing device
        
    Returns:
        (criterion_mc, criterion_ei): Loss functions for measurement consistency and equivariance
    """
    if loss_type == 'l2':
        criterion_mc = nn.MSELoss().to(device)
        criterion_ei = nn.MSELoss().to(device)
    elif loss_type == 'l1':
        criterion_mc = nn.L1Loss().to(device)
        criterion_ei = nn.L1Loss().to(device)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}, should be 'l2' or 'l1'")
    return criterion_mc, criterion_ei


def _save_checkpoint(net, optimizer, epoch, save_path, save_latest=True):
    """
    Save model checkpoint.
    
    Args:
        net: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Save path
        save_latest: Whether to also save as latest.pth.tar (for resuming training)
    """
    checkpoint_file = os.path.join(save_path, f'ckp_{epoch}.pth.tar')
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, checkpoint_file)
    
    # Also save as latest checkpoint for easy resuming
    if save_latest:
        latest_file = os.path.join(save_path, 'latest.pth.tar')
        torch.save(state, latest_file)
    
    return checkpoint_file


def _load_checkpoint(checkpoint_path, net, optimizer=None, device=None):
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Checkpoint file path
        net: Neural network model
        optimizer: Optimizer (optional, if provided will restore optimizer state)
        device: Device
        
    Returns:
        epoch: Saved epoch number, returns None if checkpoint doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    return epoch


def _save_training_results(results_folder, psnr_array, mse_array, crit_mc_array, 
                           crit_ei_array, crit_total_array, time_array):
    """
    Save training results to files.
    
    Args:
        results_folder: Results save folder
        psnr_array: PSNR array
        mse_array: MSE array
        crit_mc_array: MC loss array
        crit_ei_array: EI loss array
        crit_total_array: Total loss array
        time_array: Time array
    """
    results_dict = {
        'psnr.npy': psnr_array,
        'mse.npy': mse_array,
        'mc_loss.npy': crit_mc_array,
        'ei_loss.npy': crit_ei_array,
        'total_loss.npy': crit_total_array,
        'time.npy': time_array
    }
    
    for filename, array in results_dict.items():
        filepath = os.path.join(results_folder, filename)
        with open(filepath, 'wb') as f:
            np.save(f, array)


def _log_line(message):
    print(message, flush=True)


def _json_safe_number(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def _format_metric(value, precision=4):
    safe_value = _json_safe_number(value)
    if safe_value is None:
        return 'n/a'
    return f"{safe_value:.{precision}f}"


def _append_jsonl_line(path, payload):
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_training_log_context(config, output_dirs, config_path):
    train_cfg = config.get('train', {})
    runtime_cfg = config.get('runtime', {})
    dataset_cfg = config.get('dataset', {})
    selection_cfg = config.get('selection', {})
    split_descriptors = _get_split_descriptors(config.get('task'), dataset_cfg)
    total_epochs = runtime_cfg.get('max_epochs') or train_cfg.get('epochs', 1)
    return {
        'config_path': config_path,
        'task': config.get('task'),
        'method': config.get('method'),
        'seed': config.get('seed'),
        'device': config.get('device'),
        'output_dir': output_dirs['root'],
        'progress_path': os.path.join(output_dirs['root'], 'train_progress.jsonl'),
        'log_interval': max(1, int(runtime_cfg.get('log_interval', 50))),
        'batch_size': train_cfg.get('batch_size'),
        'total_epochs': total_epochs,
        'smoke_test': bool(runtime_cfg.get('smoke_test')),
        'validation_interval': int(train_cfg.get('validation_interval', 0)),
        'patience': int(train_cfg.get('patience', 500)),
        'train_split': split_descriptors['train_split'],
        'val_split': split_descriptors['val_split'],
        'test_split': split_descriptors['test_split'],
        'split_policy': split_descriptors['split_policy'],
        'checkpoint_selection_metric': selection_cfg.get('checkpoint_selection_metric', 'final_epoch'),
        'evaluation_checkpoint': selection_cfg.get('evaluation_checkpoint', 'final_model.pth.tar'),
        'early_stopping_metric': selection_cfg.get('early_stopping_metric', 'none'),
    }


def _build_progress_payload(context, event, epoch=None, total_epochs=None):
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event': event,
        'epoch': None if epoch is None else int(epoch),
        'total_epochs': int(total_epochs if total_epochs is not None else context['total_epochs']),
        'elapsed_sec': None,
        'epoch_time_sec': None,
        'method': context['method'],
        'task': context['task'],
        'seed': int(context['seed']),
        'device': context['device'],
        'output_dir': context['output_dir'],
        'loss_total': None,
        'loss_mc': None,
        'loss_ei': None,
        'psnr': None,
        'ssim': None,
        'checkpoint_saved': None,
        'checkpoint_type': None,
    }


def _emit_run_start(context):
    _log_line("[Run Start]")
    _log_line(f"  config_path: {context['config_path']}")
    _log_line(f"  task: {context['task']}")
    _log_line(f"  method: {context['method']}")
    _log_line(f"  seed: {context['seed']}")
    _log_line(f"  device: {context['device']}")
    _log_line(f"  output_dir: {context['output_dir']}")
    _log_line(f"  total_epochs: {context['total_epochs']}")
    _log_line(f"  batch_size: {context['batch_size']}")
    _log_line(f"  log_interval: {context['log_interval']}")
    if context['validation_interval'] > 0:
        _log_line(f"  validation_interval: {context['validation_interval']}")
        _log_line(f"  patience: {context['patience']}")
    _log_line(f"  checkpoint_selection_metric: {context['checkpoint_selection_metric']}")
    _log_line(f"  evaluation_checkpoint: {context['evaluation_checkpoint']}")
    _log_line(f"  train_split: {context['train_split']}")
    if context['val_split'] is not None:
        _log_line(f"  val_split: {context['val_split']}")
    _log_line(f"  test_split: {context['test_split']}")

    payload = _build_progress_payload(context, 'run_start')
    payload.update(
        {
            'train_split': context['train_split'],
            'test_split': context['test_split'],
            'split_policy': context['split_policy'],
            'checkpoint_selection_metric': context['checkpoint_selection_metric'],
            'evaluation_checkpoint': context['evaluation_checkpoint'],
        }
    )
    if context['validation_interval'] > 0:
        payload.update(
            {
                'val_split': context['val_split'],
                'validation_interval': int(context['validation_interval']),
                'patience': int(context['patience']),
                'early_stopping_metric': context['early_stopping_metric'],
            }
        )
    _append_jsonl_line(context['progress_path'], payload)


def _emit_init_milestone(context, message, extra=None):
    _log_line(f"[Init] {message}")
    payload = _build_progress_payload(context, 'init')
    payload['message'] = message
    if extra:
        payload.update(extra)
    _append_jsonl_line(context['progress_path'], payload)


def _emit_checkpoint_notice(context, epoch, checkpoint_path, label, best_val_psnr=None, selected_epoch=None):
    _log_line(
        f"[Checkpoint] {label}: epoch {epoch + 1}/{context['total_epochs']} saved to {checkpoint_path}"
        + (f" | best_val_psnr={_format_metric(best_val_psnr)} dB" if best_val_psnr is not None else "")
    )
    payload = _build_progress_payload(context, 'checkpoint', epoch=epoch + 1)
    payload.update(
        {
            'checkpoint_saved': True,
            'checkpoint_type': label,
            'checkpoint_label': label,
            'checkpoint_path': checkpoint_path,
        }
    )
    if context['validation_interval'] > 0:
        payload.update(
            {
                'best_psnr': _json_safe_number(best_val_psnr),
                'best_val_psnr': _json_safe_number(best_val_psnr),
                'selected_epoch': _json_safe_number(selected_epoch),
            }
        )
    _append_jsonl_line(context['progress_path'], payload)


def _log_test_metrics(context, epoch, test_psnr, test_mse):
    _log_line(
        f"[Test] epoch {epoch + 1}/{context['total_epochs']} "
        f"psnr={_format_metric(test_psnr)} dB mse={_format_metric(test_mse, precision=6)}"
    )


def _log_validation_metrics(context, epoch, val_psnr, val_mse, best_val_psnr=None,
                            no_val_improve_count=None, selected_epoch=None,
                            checkpoint_saved=False, checkpoint_type=None,
                            early_stop_triggered=False):
    _log_line(
        f"[Validation] epoch {epoch + 1}/{context['total_epochs']} "
        f"psnr={_format_metric(val_psnr)} dB "
        f"mse={_format_metric(val_mse, precision=6)} "
        f"best_val_psnr={_format_metric(best_val_psnr)} "
        f"no_val_improve_count={_json_safe_number(no_val_improve_count)} "
        f"selected_epoch={_json_safe_number(selected_epoch)}"
    )
    payload = _build_progress_payload(context, 'validation', epoch=epoch + 1)
    payload.update(
        {
            'val_psnr': _json_safe_number(val_psnr),
            'best_psnr': _json_safe_number(best_val_psnr),
            'best_val_psnr': _json_safe_number(best_val_psnr),
            'no_val_improve_count': _json_safe_number(no_val_improve_count),
            'selected_epoch': _json_safe_number(selected_epoch),
            'checkpoint_saved': bool(checkpoint_saved),
            'checkpoint_type': checkpoint_type,
            'early_stop_triggered': bool(early_stop_triggered),
        }
    )
    _append_jsonl_line(context['progress_path'], payload)


def _emit_early_stop(context, epoch, best_val_psnr, no_val_improve_count, selected_epoch):
    _log_line(
        f"[Early Stop] epoch {epoch + 1}/{context['total_epochs']} "
        f"best_val_psnr={_format_metric(best_val_psnr)} "
        f"no_val_improve_count={_json_safe_number(no_val_improve_count)} "
        f"selected_epoch={_json_safe_number(selected_epoch)}"
    )
    payload = _build_progress_payload(context, 'early_stop', epoch=epoch + 1)
    payload.update(
        {
            'best_psnr': _json_safe_number(best_val_psnr),
            'best_val_psnr': _json_safe_number(best_val_psnr),
            'no_val_improve_count': _json_safe_number(no_val_improve_count),
            'selected_epoch': _json_safe_number(selected_epoch),
            'checkpoint_saved': False,
            'checkpoint_type': None,
            'early_stop_triggered': True,
        }
    )
    _append_jsonl_line(context['progress_path'], payload)


def _log_epoch_progress(context, epoch, total_epochs, epoch_time_sec, elapsed_sec,
                        loss_total=None, loss_mc=None, loss_ei=None, psnr=None,
                        ssim=None, best_psnr=None, val_psnr=None, best_val_psnr=None,
                        no_val_improve_count=None, selected_epoch=None,
                        checkpoint_saved=False, checkpoint_type=None,
                        early_stop_triggered=False):
    payload = _build_progress_payload(context, 'epoch', epoch=epoch + 1, total_epochs=total_epochs)
    validation_enabled = context['validation_interval'] > 0
    payload.update(
        {
            'elapsed_sec': _json_safe_number(elapsed_sec),
            'epoch_time_sec': _json_safe_number(epoch_time_sec),
            'loss_total': _json_safe_number(loss_total),
            'loss_mc': _json_safe_number(loss_mc),
            'loss_ei': _json_safe_number(loss_ei),
            'psnr': _json_safe_number(psnr),
            'ssim': _json_safe_number(ssim),
            'checkpoint_saved': bool(checkpoint_saved),
            'checkpoint_type': checkpoint_type,
        }
    )
    if validation_enabled:
        payload.update(
            {
                'best_psnr': _json_safe_number(best_psnr),
                'val_psnr': _json_safe_number(val_psnr),
                'best_val_psnr': _json_safe_number(best_val_psnr),
                'no_val_improve_count': _json_safe_number(no_val_improve_count),
                'selected_epoch': _json_safe_number(selected_epoch),
                'early_stop_triggered': bool(early_stop_triggered),
            }
        )
    _append_jsonl_line(context['progress_path'], payload)

    should_print = (
        epoch == 0
        or epoch == total_epochs - 1
        or ((epoch + 1) % context['log_interval'] == 0)
    )
    if not should_print:
        return

    avg_epoch_time_sec = elapsed_sec / float(epoch + 1)
    fields = [
        f"[Epoch {epoch + 1}/{total_epochs}]",
        f"elapsed={_format_metric(elapsed_sec, precision=1)}s",
        f"epoch_time={_format_metric(epoch_time_sec, precision=1)}s",
        f"avg_epoch_time={_format_metric(avg_epoch_time_sec, precision=1)}s",
        f"loss_total={_format_metric(loss_total, precision=6)}",
        f"loss_mc={_format_metric(loss_mc, precision=6)}",
        f"loss_ei={_format_metric(loss_ei, precision=6)}",
        f"psnr={_format_metric(psnr)}",
        f"ssim={_format_metric(ssim)}",
    ]
    if validation_enabled:
        fields.extend(
            [
                f"val_psnr={_format_metric(val_psnr)}",
                f"best_val_psnr={_format_metric(best_val_psnr)}",
                f"no_val_improve_count={_format_metric(no_val_improve_count, precision=0)}",
                f"selected_epoch={_format_metric(selected_epoch, precision=0)}",
                f"early_stop_triggered={bool(early_stop_triggered)}",
            ]
        )
    fields.extend(
        [
            f"checkpoint_saved={bool(checkpoint_saved)}",
            f"checkpoint_type={checkpoint_type or 'n/a'}",
        ]
    )
    _log_line(" ".join(fields))


def _plot_train_validation_comparison(results_folder, train_psnr, test_psnr_dict, test_mse_dict,
                                 algorithm_name='', task='ct', figure_dir=None,
                                 split_name='Validation', file_prefix='validation'):
    """
    Plot training and optional validation PSNR curves.
    
    Args:
        results_folder: Folder to save plots
        train_psnr: Training PSNR array of shape (num_iterations, epochs)
        test_psnr_dict: Dict with keys as epoch numbers and values as (avg_psnr, psnr_list)
        test_mse_dict: Dict with keys as epoch numbers and values as (avg_mse, mse_list)
        algorithm_name: Name of algorithm for title
        task: Task type ('ct' or 'inpainting')
    """
    import matplotlib.pyplot as plt
    
    # Extract test epochs and values
    test_epochs = sorted(test_psnr_dict.keys())
    test_psnr_mean = [test_psnr_dict[e][0] for e in test_epochs]
    test_psnr_std = [np.std(test_psnr_dict[e][1]) for e in test_epochs]
    
    # Compute training PSNR statistics
    train_epochs = np.arange(train_psnr.shape[1])
    train_psnr_mean = train_psnr.mean(axis=0)
    train_psnr_std = train_psnr.std(axis=0, ddof=1)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training curve
    plt.plot(train_epochs, train_psnr_mean, 
             label='Training PSNR', color='blue', linestyle='-', linewidth=2)
    plt.fill_between(train_epochs, 
                     train_psnr_mean - train_psnr_std,
                     train_psnr_mean + train_psnr_std,
                     color='blue', alpha=0.2)
    
    # Plot holdout curve
    plt.plot(test_epochs, test_psnr_mean,
             label=f'{split_name} PSNR', color='red', linestyle='--', linewidth=2, marker='o', markersize=6)
    plt.fill_between(test_epochs,
                     np.array(test_psnr_mean) - np.array(test_psnr_std),
                     np.array(test_psnr_mean) + np.array(test_psnr_std),
                     color='red', alpha=0.2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'{algorithm_name} - Training vs {split_name} PSNR', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    figure_dir = figure_dir or results_folder
    os.makedirs(figure_dir, exist_ok=True)
    save_path = os.path.join(figure_dir, f'train_{file_prefix}_psnr_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved train-{split_name.lower()} comparison plot to: {save_path}")
    
    # Save test results to npy files
    test_psnr_array = np.array([test_psnr_dict[e][0] for e in test_epochs])
    test_mse_array = np.array([test_mse_dict[e][0] for e in test_epochs])
    test_epochs_array = np.array(test_epochs)
    
    # Save all test sample PSNR values for confidence interval calculation
    test_psnr_all_samples = []
    for e in test_epochs:
        psnr_list = test_psnr_dict[e][1]
        test_psnr_all_samples.append(psnr_list)
    
    # Pad lists to same length if needed
    max_len = max(len(psnr_list) for psnr_list in test_psnr_all_samples) if test_psnr_all_samples else 0
    if max_len > 0:
        test_psnr_all_samples_padded = []
        for psnr_list in test_psnr_all_samples:
            if len(psnr_list) < max_len:
                psnr_list = list(psnr_list) + [psnr_list[-1]] * (max_len - len(psnr_list))
            test_psnr_all_samples_padded.append(psnr_list)
        test_psnr_all_samples_array = np.array(test_psnr_all_samples_padded)
    else:
        test_psnr_all_samples_array = np.array(test_psnr_all_samples)
    
    np.save(os.path.join(results_folder, f'{file_prefix}_psnr.npy'), test_psnr_array)
    np.save(os.path.join(results_folder, f'{file_prefix}_psnr_all_samples.npy'), test_psnr_all_samples_array)
    np.save(os.path.join(results_folder, f'{file_prefix}_mse.npy'), test_mse_array)
    np.save(os.path.join(results_folder, f'{file_prefix}_epochs.npy'), test_epochs_array)


def _save_intermediate_images(epoch, task, results_folder, x0, x, results, i, save_dir=None):
    """
    Save intermediate result images (for visualization).
    
    Args:
        epoch: Current epoch
        task: Task type ('ct' or 'inpainting')
        results_folder: Results save folder
        x0: Pseudoinverse reconstructed image
        x: Ground truth image
        results: Current result image
        i: Sample index
    """
    save_dir = save_dir or results_folder
    os.makedirs(save_dir, exist_ok=True)

    if epoch == 0:
        if task == 'ct':
            plt.imsave(
                os.path.join(save_dir, f'pseudo_{i}.png'),
                x0.squeeze().detach().cpu().numpy(),
                cmap='gray'
            )
            plt.imsave(
                os.path.join(save_dir, f'target_{i}.png'),
                x.squeeze().detach().cpu().numpy(),
                cmap='gray'
            )
        elif task == 'inpainting':
            plt.imsave(
                os.path.join(save_dir, f'pseudo_{i}.png'),
                x0.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            )
            plt.imsave(
                os.path.join(save_dir, f'target_{i}.png'),
                x.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            )

    # Periodically save result images
    if (epoch % 500 == 0 or epoch == 100):
        if task == 'ct':
            plt.imsave(
                os.path.join(save_dir, f'epoch_{epoch}.png'),
                results.squeeze().detach().cpu().numpy(),
                cmap='gray'
            )
        elif task == 'inpainting':
            plt.imsave(
                os.path.join(save_dir, f'epoch_{epoch}.png'),
                np.clip(results.squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            )


def _print_training_progress(epoch, epochs, crit_mc_array, crit_ei_array, 
                             psnr_array, mse_array, verbose=True):
    """
    Print training progress information.
    
    Args:
        epoch: Current epoch
        epochs: Total number of epochs
        crit_mc_array: MC loss array
        crit_ei_array: EI loss array
        psnr_array: PSNR array
        mse_array: MSE array
        verbose: Whether to output verbosely
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Output every 100 epochs (or at epoch 0 and last epoch)
    if epoch == 0 or epoch % 100 == 0 or epoch == epochs - 1:
        # Calculate metrics
        mc_val = crit_mc_array[:, epoch].mean() if epoch < crit_mc_array.shape[1] else 0
        ei_val = crit_ei_array[:, epoch].mean() if epoch < crit_ei_array.shape[1] else 0
        psnr_val = psnr_array[:, epoch].mean() if epoch < psnr_array.shape[1] else 0
        mse_val = mse_array[:, epoch].mean() if epoch < mse_array.shape[1] else 0
        
        progress_msg = (
            f'Epoch[{epoch:4d}/{epochs}]\t'
            f'mc={mc_val:.4e}\t'
            f'ei={ei_val:.4e}\t'
            f'psnr={psnr_val:.4f}\t'
            f'mse={mse_val:.4e}'
        )
        
        print(progress_msg)


# ============================================================================
# NAG algorithm (for FEI Option 1)
# ============================================================================
def NAG(physics, y0, x0, u0, lamb, iters, beta=1e-1, eta=1e-2):
    """
    Nesterov Accelerated Gradient (NAG) algorithm for FEI Option 1.
    
    Args:
        physics: CT physics model
        y0: Measurement data
        x0: Initial reconstruction
        u0: Initial variable
        lamb: Regularization parameter
        iters: Number of iterations
        beta: NAG parameter
        eta: Step size
        
    Returns:
        u0: Optimized reconstruction result
    """
    v_t = torch.zeros_like(x0)
    for it in range(iters):
        u_ahead = u0 - beta * v_t
        gradient = physics.A_adjoint(physics.A(u_ahead) - y0) + lamb * (u_ahead - x0)
        v_t = beta * v_t + eta * gradient
        u0 = u0 - v_t
    return u0


def NAG_PnP(physics, y0, x0, u0, lamb, denoiser, sigma, iters, beta=1e-1, eta=1e-2):
    """
    Nesterov Accelerated Gradient (NAG) with Plug-and-Play prior for FEI Option 1.
    
    Args:
        physics: CT physics model
        y0: Measurement data
        x0: Initial reconstruction
        u0: Initial variable
        lamb: Regularization parameter
        denoiser: Plug-and-Play denoiser
        sigma: Noise level for denoiser
        iters: Number of iterations
        beta: NAG parameter
        eta: Step size
        
    Returns:
        u0: Optimized reconstruction result
    """
    v_t = torch.zeros_like(x0)
    for it in range(iters):
        u_ahead = u0 - beta * v_t
        gradient = physics.A_adjoint(physics.A(u_ahead) - y0) + lamb * (u_ahead - x0)
        # Apply PnP denoiser
        with torch.no_grad():
            u_denoised = denoiser(u_ahead - eta * gradient, sigma)
        v_t = beta * v_t + (u_ahead - u_denoised) / eta
        u0 = u0 - v_t
    return u0


def _freeze_latent_targets(physics, transform, x1):
    """
    Build fixed latent/equivariance targets for the pseudo-supervision step.

    The manuscript FEI updates alternate between:
    1. latent reconstruction with theta fixed, and
    2. pseudo-supervision with the latent treated as a fixed target.
    """
    with torch.no_grad():
        x1_target = x1.detach()
        x2_target = transform.apply(x1_target).detach()
        x2_T = physics.A_dagger(physics.A(x2_target))
    return x1_target, x2_target, x2_T


def _linearized_admm_latent_step(physics, y0, x0_fixed, dual_fixed, lamb, eta, iters,
                                 denoiser=None, sigma=None):
    """
    Linearized ADMM-inspired latent step using the manuscript scaled-dual convention.

    We store the scaled dual variable as the same L that appears in
    ||u - F_theta(y) + L||^2. With that convention the latent update is
    u <- u - eta * [grad f_mc + lambda * (u - F_theta(y) + L)].
    """
    with torch.no_grad():
        x1 = x0_fixed.clone()
        for _ in range(iters):
            x_grad = physics.A_adjoint(physics.A(x1) - y0) + lamb * (x1 - x0_fixed + dual_fixed)
            x1 = x1 - eta * x_grad
            if denoiser is not None:
                x1 = denoiser(x1, sigma)
        return x1.detach()


# ============================================================================
# Training function: EI (vanilla Equivariant Imaging)
# ============================================================================
def train_ei(net, ckp_interval, dataloader, epochs, physics, transform, alpha, 
             dtype, task, pretrained=None, loss_type='l2', lr_cos=True, 
             device=None, lr=None, schedule=None, num_train_samples=None, verbose=True,
             resume_from=None, patience=500, dataset_config=None, output_dirs=None,
             training_log_context=None, validation_interval=0, snapshot_epochs=None,
             selection_config=None, max_batches=None, max_val_samples=None):
    """
    Train network using vanilla Equivariant Imaging method (without noise model).
    
    Args:
        net: Neural network model
        ckp_interval: Checkpoint save interval
        dataloader: Data loader
        epochs: Number of training epochs
        physics: CT physics model
        transform: Data transform (rotation transform)
        alpha: EI loss weight
        dtype: Data type
        task: Task type ('ct' or 'inpainting')
        pretrained: Pretrained model path
        loss_type: Loss type ('l2' or 'l1')
        lr_cos: Whether to use cosine learning rate schedule
        device: Computing device
        lr: Learning rate dictionary {'G', 'WD'}
        schedule: Learning rate schedule list
        num_train_samples: Number of training samples
        verbose: Whether to output verbosely
        resume_from: Checkpoint path for resuming training
        patience: Optional validation patience (inactive when validation is disabled)
    """
    # Path setup
    save_path = _resolve_checkpoint_dir(task, lr['G'], 'EI', epochs, output_dirs=output_dirs)
    results_folder = _resolve_results_dir(task, lr['G'], 'EI', output_dirs=output_dirs)
    figure_dir = output_dirs.get('figures', results_folder) if output_dirs else results_folder
    recon_dir = output_dirs.get('reconstructions', results_folder) if output_dirs else results_folder

    selection_config = selection_config or {}
    evaluation_checkpoint = selection_config.get('evaluation_checkpoint', 'final_model.pth.tar')

    # The supplied configurations disable training-time validation.
    val_dataloader = None
    if int(validation_interval) > 0:
        val_dataloader = _build_validation_dataloader_for_task(
            task=task,
            batch_size=1,
            dataset_config=dataset_config,
            max_val_samples=max_val_samples,
        )
    
    val_psnr_dict = {}  # {epoch: (avg_psnr, psnr_list)}
    val_mse_dict = {}   # {epoch: (avg_mse, mse_list)}

    # Setup loss functions and optimizer
    criterion_mc, criterion_ei = _setup_loss_criterion(loss_type, device)
    optimizer = Adam(net.parameters(), lr=lr['G'], weight_decay=lr['WD'])
    if training_log_context:
        _emit_init_milestone(training_log_context, 'Optimizer initialized')

    # Resume training
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch = _load_checkpoint(resume_from, net, optimizer, device)
            if start_epoch is not None:
                start_epoch += 1
                print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print("Failed to load checkpoint, starting from scratch.")
        else:
            latest_checkpoint = os.path.join(save_path, 'latest.pth.tar')
            if os.path.exists(latest_checkpoint):
                print(f"Found latest checkpoint: {latest_checkpoint}")
                start_epoch = _load_checkpoint(latest_checkpoint, net, optimizer, device)
                if start_epoch is not None:
                    start_epoch += 1
                    print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print(f"Resume checkpoint not found: {resume_from}")
                print("Starting training from scratch.")
    
    # Load pretrained model (if not resuming and pretrained is provided)
    if start_epoch == 0 and pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained model from: {pretrained}")

    # Calculate actual number of iterations per epoch
    num_iterations_per_epoch = _effective_iterations_per_epoch(dataloader, max_batches)
    
    # Initialize recording arrays
    psnr_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_mc_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_ei_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_total_array = np.zeros((num_iterations_per_epoch, epochs))
    mse_array = np.zeros((num_iterations_per_epoch, epochs))
    time_array = np.zeros((1, epochs))

    results_list = [None] * num_iterations_per_epoch

    # Optional validation-monitoring state
    best_val_psnr = -np.inf
    selected_epoch = None
    no_improve_count = 0
    last_val_epoch = None
    last_completed_epoch = start_epoch - 1
    latest_val_psnr = None
    early_stopped = False
    run_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
        
        for i, x_batch in _limited_batch_iterator(dataloader, max_batches):
            # Prepare data
            x = _prepare_batch_data(x_batch, dtype, device)

            # Forward pass
            y0 = physics.A(x)  # Noiseless measurement
            x0 = physics.A_dagger(y0)
            x1 = net(x0)
            y1 = physics.A(x1)
            
            # Measurement consistency loss
            loss_mc = criterion_mc(y1, y0)
            
            # Equivariant imaging loss: apply transform
            x2 = transform.apply(x1)
            x2_T = physics.A_dagger(physics.A(x2))
            x3 = net(x2_T)
            loss_ei = criterion_ei(x3, x2)
            
            loss = loss_mc + alpha * loss_ei

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            crit_mc_array[i, epoch] = loss_mc.item()
            crit_ei_array[i, epoch] = loss_ei.item()
            crit_total_array[i, epoch] = loss.item()

            # Update results (exponential moving average)
            if epoch == 0:
                results = x1.clone() if len(x1.shape) == 4 and x1.shape[0] > 1 else x1
            else:
                prev_results = results_list[i]
                if prev_results is not None:
                    results = 0.01 * x1 + 0.99 * prev_results
                else:
                    results = x1.clone() if len(x1.shape) == 4 and x1.shape[0] > 1 else x1
            results_list[i] = results
            
            # Compute metrics on EMA results
            psnr = cal_psnr(results, x)
            mse = cal_mse(results, x)
            
            psnr_array[i, epoch] = psnr
            mse_array[i, epoch] = mse

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_array[0, epoch] = elapsed_time
        last_completed_epoch = epoch
        checkpoint_saved = False
        checkpoint_type = None
        early_stop_triggered = False
        current_val_psnr = latest_val_psnr

        # Save images
        if epoch == 0 or (epoch % 500 == 0 or epoch == 100):
            # For batch data, use first sample for visualization
            if len(x0.shape) == 4 and x0.shape[0] > 1:
                x0_viz = x0[0:1]
                x_viz = x[0:1] if len(x.shape) == 4 and x.shape[0] > 1 else x
            else:
                x0_viz = x0
                x_viz = x
            
            if len(results.shape) == 4 and results.shape[0] > 1:
                results_viz = results[0:1]
            else:
                results_viz = results
                
            _save_intermediate_images(
                epoch,
                task,
                results_folder,
                x0_viz,
                x_viz,
                results_viz,
                i,
                save_dir=recon_dir,
            )
        
        if _should_run_validation(epoch, epochs, validation_interval):
            validation_state = _run_validation_pass(
                net=net,
                val_dataloader=val_dataloader,
                physics=physics,
                device=device,
                dtype=dtype,
                epoch=epoch,
                optimizer=optimizer,
                save_path=save_path,
                val_psnr_dict=val_psnr_dict,
                val_mse_dict=val_mse_dict,
                best_val_psnr=best_val_psnr,
                selected_epoch=selected_epoch,
                no_val_improve_count=no_improve_count,
                last_val_epoch=last_val_epoch,
                training_log_context=training_log_context,
                evaluation_checkpoint=evaluation_checkpoint,
            )
            current_val_psnr = validation_state['val_psnr']
            latest_val_psnr = current_val_psnr
            best_val_psnr = validation_state['best_val_psnr']
            selected_epoch = validation_state['selected_epoch']
            no_improve_count = validation_state['no_val_improve_count']
            last_val_epoch = validation_state['last_val_epoch']
            if validation_state['best_checkpoint_saved']:
                checkpoint_saved = True
                checkpoint_type = _merge_checkpoint_type(checkpoint_type, validation_state['checkpoint_type'])

        snapshot_path = _save_snapshot_checkpoint(
            net,
            optimizer,
            epoch,
            save_path,
            snapshot_epochs,
            extra_state={
                'best_val_psnr': _json_safe_number(best_val_psnr if np.isfinite(best_val_psnr) else None),
                'selected_epoch': _json_safe_number(selected_epoch),
            },
        )
        if snapshot_path is not None:
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'snapshot')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    snapshot_path,
                    'snapshot',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        # Save checkpoint
        if (epoch % ckp_interval == 0 and epoch > 0) or epoch + 1 == epochs:
            checkpoint_path = _save_checkpoint(net, optimizer, epoch, save_path, save_latest=True)
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'latest_checkpoint')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    checkpoint_path,
                    'latest_checkpoint',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        if int(validation_interval) > 0 and no_improve_count >= patience and (epoch + 1) >= validation_interval:
            early_stop_triggered = True
            early_stopped = True
            if training_log_context:
                _emit_early_stop(
                    training_log_context,
                    epoch,
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    no_val_improve_count=no_improve_count,
                    selected_epoch=selected_epoch,
                )
            else:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"best_val_psnr={best_val_psnr:.4f}, "
                    f"no_val_improve_count={no_improve_count}, selected_epoch={selected_epoch}"
                )

        if training_log_context:
            _log_epoch_progress(
                training_log_context,
                epoch=epoch,
                total_epochs=epochs,
                epoch_time_sec=elapsed_time,
                elapsed_sec=time.time() - run_start_time,
                loss_total=crit_total_array[:, epoch].mean(),
                loss_mc=crit_mc_array[:, epoch].mean(),
                loss_ei=crit_ei_array[:, epoch].mean(),
                psnr=psnr_array[:, epoch].mean(),
                ssim=None,
                best_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                val_psnr=current_val_psnr,
                best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                no_val_improve_count=no_improve_count,
                selected_epoch=selected_epoch,
                checkpoint_saved=checkpoint_saved,
                checkpoint_type=checkpoint_type,
                early_stop_triggered=early_stop_triggered,
            )
        if early_stop_triggered:
            break

    # Save all results
    _save_training_results(results_folder, psnr_array, mse_array, crit_mc_array,
                          crit_ei_array, crit_total_array, time_array)
    
    # Plot train vs validation comparison
    if len(val_psnr_dict) > 0:
        _plot_train_validation_comparison(
            results_folder,
            psnr_array,
            val_psnr_dict,
            val_mse_dict,
            algorithm_name='EI',
            task=task,
            figure_dir=figure_dir,
            split_name='Validation',
            file_prefix='validation',
        )

    _finalize_training_run(
        net=net,
        optimizer=optimizer,
        last_completed_epoch=last_completed_epoch,
        save_path=save_path,
        output_root=output_dirs['root'] if output_dirs else results_folder,
        training_log_context=training_log_context,
        epochs=epochs,
        early_stopped=early_stopped,
        best_val_psnr=best_val_psnr,
        selected_epoch=selected_epoch,
        validation_interval=validation_interval,
        patience=patience,
        evaluation_checkpoint=evaluation_checkpoint,
    )


# ============================================================================
# Training function: FEI Option 1 (NAG)
# ============================================================================
def train_fei_option1(net, ckp_interval, dataloader, epochs, physics, transform, alpha, lamb,
                      dtype, task, pretrained=None, loss_type='l2', lr_cos=True,
                      device=None, lr=None, schedule=None, num_train_samples=None, 
                      verbose=True, resume_from=None, patience=500, dataset_config=None,
                      output_dirs=None, training_log_context=None, inner_iters=10, beta=1e-1, eta=1e-2,
                      validation_interval=0, snapshot_epochs=None, selection_config=None,
                      max_batches=None, max_val_samples=None):
    """
    Train network using FEI Option 1 (NAG) method.
    
    Args:
        lamb: FEI regularization parameter (multiplier)
        Other parameters same as train_ei
    """
    # Path setup
    save_path = _resolve_checkpoint_dir(task, lr['G'], 'FEI-O1', epochs, output_dirs=output_dirs)
    results_folder = _resolve_results_dir(task, lr['G'], 'FEI-O1', output_dirs=output_dirs)
    figure_dir = output_dirs.get('figures', results_folder) if output_dirs else results_folder
    recon_dir = output_dirs.get('reconstructions', results_folder) if output_dirs else results_folder

    selection_config = selection_config or {}
    evaluation_checkpoint = selection_config.get('evaluation_checkpoint', 'final_model.pth.tar')

    # The supplied configurations disable training-time validation.
    val_dataloader = None
    if int(validation_interval) > 0:
        val_dataloader = _build_validation_dataloader_for_task(
            task=task,
            batch_size=1,
            dataset_config=dataset_config,
            max_val_samples=max_val_samples,
        )
    
    val_psnr_dict = {}
    val_mse_dict = {}

    # Setup loss functions and optimizer
    criterion_mc, criterion_ei = _setup_loss_criterion(loss_type, device)
    optimizer = Adam(net.parameters(), lr=lr['G'], weight_decay=lr['WD'])
    if training_log_context:
        _emit_init_milestone(training_log_context, 'Optimizer initialized')

    # Resume training
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch = _load_checkpoint(resume_from, net, optimizer, device)
            if start_epoch is not None:
                start_epoch += 1
                print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print("Failed to load checkpoint, starting from scratch.")
        else:
            latest_checkpoint = os.path.join(save_path, 'latest.pth.tar')
            if os.path.exists(latest_checkpoint):
                print(f"Found latest checkpoint: {latest_checkpoint}")
                start_epoch = _load_checkpoint(latest_checkpoint, net, optimizer, device)
                if start_epoch is not None:
                    start_epoch += 1
                    print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print(f"Resume checkpoint not found: {resume_from}")
                print("Starting training from scratch.")
    
    # Load pretrained model
    if start_epoch == 0 and pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained model from: {pretrained}")

    # Calculate actual number of iterations per epoch
    num_iterations_per_epoch = _effective_iterations_per_epoch(dataloader, max_batches)
    
    # Initialize recording arrays
    psnr_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_mc_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_ei_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_total_array = np.zeros((num_iterations_per_epoch, epochs))
    mse_array = np.zeros((num_iterations_per_epoch, epochs))
    time_array = np.zeros((1, epochs))

    results_list = [None] * num_iterations_per_epoch

    # Optional validation-monitoring state
    best_val_psnr = -np.inf
    selected_epoch = None
    no_improve_count = 0
    last_val_epoch = None
    last_completed_epoch = start_epoch - 1
    latest_val_psnr = None
    early_stopped = False
    run_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
        
        for i, x_batch in _limited_batch_iterator(dataloader, max_batches):
            # Prepare data
            x = _prepare_batch_data(x_batch, dtype, device)

            # Forward pass
            y0 = physics.A(x)
            z = physics.A_dagger(y0)
            x0 = net(z)

            # Manuscript FEI-O1 alternates between a latent step with theta fixed
            # and a pseudo-supervision step against fixed latent/equivariance targets.
            with torch.no_grad():
                x0_fixed = x0.detach()
                x1 = NAG(physics, y0, x0_fixed, u0=x0_fixed, lamb=lamb, iters=inner_iters,
                         beta=beta, eta=eta)
            x1_target, x2_target, x2_T = _freeze_latent_targets(physics, transform, x1)
            x3 = net(x2_T)

            loss_mc = criterion_mc(x0, x1_target)
            loss_ei = criterion_ei(x3, x2_target)
            loss = loss_mc + alpha * loss_ei

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            crit_mc_array[i, epoch] = loss_mc.item()
            crit_ei_array[i, epoch] = loss_ei.item()
            crit_total_array[i, epoch] = loss.item()

            # Update results (exponential moving average on x0, not x1)
            if epoch == 0:
                results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            else:
                prev_results = results_list[i]
                if prev_results is not None:
                    results = 0.01 * x0 + 0.99 * prev_results
                else:
                    results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            results_list[i] = results
            
            # Compute metrics
            psnr = cal_psnr(results, x)
            mse = cal_mse(results, x)
            
            psnr_array[i, epoch] = psnr
            mse_array[i, epoch] = mse

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_array[0, epoch] = elapsed_time
        last_completed_epoch = epoch
        checkpoint_saved = False
        checkpoint_type = None
        early_stop_triggered = False
        current_val_psnr = latest_val_psnr

        # Save images
        if epoch == 0 or (epoch % 500 == 0 or epoch == 100):
            if len(z.shape) == 4 and z.shape[0] > 1:
                z_viz = z[0:1]
                x_viz = x[0:1] if len(x.shape) == 4 and x.shape[0] > 1 else x
            else:
                z_viz = z
                x_viz = x
            
            if len(results.shape) == 4 and results.shape[0] > 1:
                results_viz = results[0:1]
            else:
                results_viz = results
                
            _save_intermediate_images(
                epoch,
                task,
                results_folder,
                z_viz,
                x_viz,
                results_viz,
                i,
                save_dir=recon_dir,
            )
        
        if _should_run_validation(epoch, epochs, validation_interval):
            validation_state = _run_validation_pass(
                net=net,
                val_dataloader=val_dataloader,
                physics=physics,
                device=device,
                dtype=dtype,
                epoch=epoch,
                optimizer=optimizer,
                save_path=save_path,
                val_psnr_dict=val_psnr_dict,
                val_mse_dict=val_mse_dict,
                best_val_psnr=best_val_psnr,
                selected_epoch=selected_epoch,
                no_val_improve_count=no_improve_count,
                last_val_epoch=last_val_epoch,
                training_log_context=training_log_context,
                evaluation_checkpoint=evaluation_checkpoint,
            )
            current_val_psnr = validation_state['val_psnr']
            latest_val_psnr = current_val_psnr
            best_val_psnr = validation_state['best_val_psnr']
            selected_epoch = validation_state['selected_epoch']
            no_improve_count = validation_state['no_val_improve_count']
            last_val_epoch = validation_state['last_val_epoch']
            if validation_state['best_checkpoint_saved']:
                checkpoint_saved = True
                checkpoint_type = _merge_checkpoint_type(checkpoint_type, validation_state['checkpoint_type'])

        snapshot_path = _save_snapshot_checkpoint(
            net,
            optimizer,
            epoch,
            save_path,
            snapshot_epochs,
            extra_state={
                'best_val_psnr': _json_safe_number(best_val_psnr if np.isfinite(best_val_psnr) else None),
                'selected_epoch': _json_safe_number(selected_epoch),
            },
        )
        if snapshot_path is not None:
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'snapshot')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    snapshot_path,
                    'snapshot',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        # Save checkpoint
        if (epoch % ckp_interval == 0 and epoch > 0) or epoch + 1 == epochs:
            checkpoint_path = _save_checkpoint(net, optimizer, epoch, save_path, save_latest=True)
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'latest_checkpoint')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    checkpoint_path,
                    'latest_checkpoint',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        if int(validation_interval) > 0 and no_improve_count >= patience and (epoch + 1) >= validation_interval:
            early_stop_triggered = True
            early_stopped = True
            if training_log_context:
                _emit_early_stop(
                    training_log_context,
                    epoch,
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    no_val_improve_count=no_improve_count,
                    selected_epoch=selected_epoch,
                )
            else:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"best_val_psnr={best_val_psnr:.4f}, "
                    f"no_val_improve_count={no_improve_count}, selected_epoch={selected_epoch}"
                )

        if training_log_context:
            _log_epoch_progress(
                training_log_context,
                epoch=epoch,
                total_epochs=epochs,
                epoch_time_sec=elapsed_time,
                elapsed_sec=time.time() - run_start_time,
                loss_total=crit_total_array[:, epoch].mean(),
                loss_mc=crit_mc_array[:, epoch].mean(),
                loss_ei=crit_ei_array[:, epoch].mean(),
                psnr=psnr_array[:, epoch].mean(),
                ssim=None,
                best_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                val_psnr=current_val_psnr,
                best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                no_val_improve_count=no_improve_count,
                selected_epoch=selected_epoch,
                checkpoint_saved=checkpoint_saved,
                checkpoint_type=checkpoint_type,
                early_stop_triggered=early_stop_triggered,
            )
        if early_stop_triggered:
            break

    # Save all results
    _save_training_results(results_folder, psnr_array, mse_array, crit_mc_array,
                          crit_ei_array, crit_total_array, time_array)
    
    # Plot train vs validation comparison
    if len(val_psnr_dict) > 0:
        _plot_train_validation_comparison(
            results_folder,
            psnr_array,
            val_psnr_dict,
            val_mse_dict,
            algorithm_name='FEI-Option1',
            task=task,
            figure_dir=figure_dir,
            split_name='Validation',
            file_prefix='validation',
        )

    _finalize_training_run(
        net=net,
        optimizer=optimizer,
        last_completed_epoch=last_completed_epoch,
        save_path=save_path,
        output_root=output_dirs['root'] if output_dirs else results_folder,
        training_log_context=training_log_context,
        epochs=epochs,
        early_stopped=early_stopped,
        best_val_psnr=best_val_psnr,
        selected_epoch=selected_epoch,
        validation_interval=validation_interval,
        patience=patience,
        evaluation_checkpoint=evaluation_checkpoint,
    )


# ============================================================================
# Training function: FEI Option 2 (ADMM)
# ============================================================================
def train_fei_option2(net, ckp_interval, dataloader, epochs, physics, transform, alpha, lamb, eta,
                      dtype, task, pretrained=None, loss_type='l2', lr_cos=True,
                      device=None, lr=None, schedule=None, num_train_samples=None,
                      verbose=True, resume_from=None, patience=500, dataset_config=None,
                      output_dirs=None, training_log_context=None, inner_iters=1,
                      validation_interval=0, snapshot_epochs=None, selection_config=None,
                      max_batches=None, max_val_samples=None):
    """
    Train network using FEI Option 2 (ADMM) method.
    
    Args:
        lamb: FEI regularization parameter
        eta: Gradient descent step size (for ADMM)
        Other parameters same as train_ei
    """
    # Path setup
    save_path = _resolve_checkpoint_dir(task, lr['G'], 'FEI-O2', epochs, output_dirs=output_dirs)
    results_folder = _resolve_results_dir(task, lr['G'], 'FEI-O2', output_dirs=output_dirs)
    figure_dir = output_dirs.get('figures', results_folder) if output_dirs else results_folder
    recon_dir = output_dirs.get('reconstructions', results_folder) if output_dirs else results_folder

    selection_config = selection_config or {}
    evaluation_checkpoint = selection_config.get('evaluation_checkpoint', 'final_model.pth.tar')

    # The supplied configurations disable training-time validation.
    val_dataloader = None
    if int(validation_interval) > 0:
        val_dataloader = _build_validation_dataloader_for_task(
            task=task,
            batch_size=1,
            dataset_config=dataset_config,
            max_val_samples=max_val_samples,
        )
    
    val_psnr_dict = {}
    val_mse_dict = {}

    # Setup loss functions and optimizer
    criterion_mc, criterion_ei = _setup_loss_criterion(loss_type, device)
    optimizer = Adam(net.parameters(), lr=lr['G'], weight_decay=lr['WD'])
    if training_log_context:
        _emit_init_milestone(training_log_context, 'Optimizer initialized')

    # Resume training
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch = _load_checkpoint(resume_from, net, optimizer, device)
            if start_epoch is not None:
                start_epoch += 1
                print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print("Failed to load checkpoint, starting from scratch.")
        else:
            latest_checkpoint = os.path.join(save_path, 'latest.pth.tar')
            if os.path.exists(latest_checkpoint):
                print(f"Found latest checkpoint: {latest_checkpoint}")
                start_epoch = _load_checkpoint(latest_checkpoint, net, optimizer, device)
                if start_epoch is not None:
                    start_epoch += 1
                    print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print(f"Resume checkpoint not found: {resume_from}")
                print("Starting training from scratch.")
    
    # Load pretrained model
    if start_epoch == 0 and pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained model from: {pretrained}")

    # Calculate actual number of iterations per epoch
    num_iterations_per_epoch = _effective_iterations_per_epoch(dataloader, max_batches)
    
    # Initialize recording arrays
    psnr_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_mc_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_ei_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_total_array = np.zeros((num_iterations_per_epoch, epochs))
    mse_array = np.zeros((num_iterations_per_epoch, epochs))
    time_array = np.zeros((1, epochs))

    # Per-sample scaled dual variables for the manuscript convention
    # ||u - F_theta(y) + L||^2 and L <- L + u - F_theta(y).
    L0_list = [None] * num_iterations_per_epoch
    results_list = [None] * num_iterations_per_epoch

    # Optional validation-monitoring state
    best_val_psnr = -np.inf
    selected_epoch = None
    no_improve_count = 0
    last_val_epoch = None
    last_completed_epoch = start_epoch - 1
    latest_val_psnr = None
    early_stopped = False
    run_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
        
        for i, x_batch in _limited_batch_iterator(dataloader, max_batches):
            # Prepare data
            x = _prepare_batch_data(x_batch, dtype, device)

            # Forward pass
            y0 = physics.A(x)
            z = physics.A_dagger(y0)
            x0 = net(z)

            x0_fixed = x0.detach()
            if L0_list[i] is None:
                L0_list[i] = torch.zeros_like(x0_fixed)
            L0 = L0_list[i].detach()

            x1 = _linearized_admm_latent_step(
                physics=physics,
                y0=y0,
                x0_fixed=x0_fixed,
                dual_fixed=L0,
                lamb=lamb,
                eta=eta,
                iters=inner_iters,
            )
            x1_target, x2_target, x2_T = _freeze_latent_targets(physics, transform, x1)
            x3 = net(x2_T)

            # Equivalent to ||F_theta(y) - L - x1||^2 from Algorithm 4.2.
            loss_mc = criterion_mc(x0 - L0, x1_target)
            loss_ei = criterion_ei(x3, x2_target)
            loss = loss_mc + alpha * loss_ei

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x0_updated = net(z).detach()
                L0_list[i] = (L0 + x1_target - x0_updated).detach()

            # Record losses
            crit_mc_array[i, epoch] = loss_mc.item()
            crit_ei_array[i, epoch] = loss_ei.item()
            crit_total_array[i, epoch] = loss.item()

            # Update results (exponential moving average on x0, not x1)
            if epoch == 0:
                results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            else:
                prev_results = results_list[i]
                if prev_results is not None:
                    results = 0.01 * x0 + 0.99 * prev_results
                else:
                    results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            results_list[i] = results
            
            # Compute metrics
            psnr = cal_psnr(results, x)
            mse = cal_mse(results, x)
            
            psnr_array[i, epoch] = psnr
            mse_array[i, epoch] = mse

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_array[0, epoch] = elapsed_time
        last_completed_epoch = epoch
        checkpoint_saved = False
        checkpoint_type = None
        early_stop_triggered = False
        current_val_psnr = latest_val_psnr

        # Save images
        if epoch == 0 or (epoch % 500 == 0 or epoch == 100):
            if len(z.shape) == 4 and z.shape[0] > 1:
                z_viz = z[0:1]
                x_viz = x[0:1] if len(x.shape) == 4 and x.shape[0] > 1 else x
            else:
                z_viz = z
                x_viz = x
            
            if len(results.shape) == 4 and results.shape[0] > 1:
                results_viz = results[0:1]
            else:
                results_viz = results
                
            _save_intermediate_images(
                epoch,
                task,
                results_folder,
                z_viz,
                x_viz,
                results_viz,
                i,
                save_dir=recon_dir,
            )
        
        if _should_run_validation(epoch, epochs, validation_interval):
            validation_state = _run_validation_pass(
                net=net,
                val_dataloader=val_dataloader,
                physics=physics,
                device=device,
                dtype=dtype,
                epoch=epoch,
                optimizer=optimizer,
                save_path=save_path,
                val_psnr_dict=val_psnr_dict,
                val_mse_dict=val_mse_dict,
                best_val_psnr=best_val_psnr,
                selected_epoch=selected_epoch,
                no_val_improve_count=no_improve_count,
                last_val_epoch=last_val_epoch,
                training_log_context=training_log_context,
                evaluation_checkpoint=evaluation_checkpoint,
            )
            current_val_psnr = validation_state['val_psnr']
            latest_val_psnr = current_val_psnr
            best_val_psnr = validation_state['best_val_psnr']
            selected_epoch = validation_state['selected_epoch']
            no_improve_count = validation_state['no_val_improve_count']
            last_val_epoch = validation_state['last_val_epoch']
            if validation_state['best_checkpoint_saved']:
                checkpoint_saved = True
                checkpoint_type = _merge_checkpoint_type(checkpoint_type, validation_state['checkpoint_type'])

        snapshot_path = _save_snapshot_checkpoint(
            net,
            optimizer,
            epoch,
            save_path,
            snapshot_epochs,
            extra_state={
                'best_val_psnr': _json_safe_number(best_val_psnr if np.isfinite(best_val_psnr) else None),
                'selected_epoch': _json_safe_number(selected_epoch),
            },
        )
        if snapshot_path is not None:
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'snapshot')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    snapshot_path,
                    'snapshot',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        # Save checkpoint
        if (epoch % ckp_interval == 0 and epoch > 0) or epoch + 1 == epochs:
            checkpoint_path = _save_checkpoint(net, optimizer, epoch, save_path, save_latest=True)
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'latest_checkpoint')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    checkpoint_path,
                    'latest_checkpoint',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        if int(validation_interval) > 0 and no_improve_count >= patience and (epoch + 1) >= validation_interval:
            early_stop_triggered = True
            early_stopped = True
            if training_log_context:
                _emit_early_stop(
                    training_log_context,
                    epoch,
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    no_val_improve_count=no_improve_count,
                    selected_epoch=selected_epoch,
                )
            else:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"best_val_psnr={best_val_psnr:.4f}, "
                    f"no_val_improve_count={no_improve_count}, selected_epoch={selected_epoch}"
                )

        if training_log_context:
            _log_epoch_progress(
                training_log_context,
                epoch=epoch,
                total_epochs=epochs,
                epoch_time_sec=elapsed_time,
                elapsed_sec=time.time() - run_start_time,
                loss_total=crit_total_array[:, epoch].mean(),
                loss_mc=crit_mc_array[:, epoch].mean(),
                loss_ei=crit_ei_array[:, epoch].mean(),
                psnr=psnr_array[:, epoch].mean(),
                ssim=None,
                best_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                val_psnr=current_val_psnr,
                best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                no_val_improve_count=no_improve_count,
                selected_epoch=selected_epoch,
                checkpoint_saved=checkpoint_saved,
                checkpoint_type=checkpoint_type,
                early_stop_triggered=early_stop_triggered,
            )
        if early_stop_triggered:
            break

    # Save all results
    _save_training_results(results_folder, psnr_array, mse_array, crit_mc_array,
                          crit_ei_array, crit_total_array, time_array)
    
    # Plot train vs validation comparison
    if len(val_psnr_dict) > 0:
        _plot_train_validation_comparison(
            results_folder,
            psnr_array,
            val_psnr_dict,
            val_mse_dict,
            algorithm_name='FEI-Option2',
            task=task,
            figure_dir=figure_dir,
            split_name='Validation',
            file_prefix='validation',
        )

    _finalize_training_run(
        net=net,
        optimizer=optimizer,
        last_completed_epoch=last_completed_epoch,
        save_path=save_path,
        output_root=output_dirs['root'] if output_dirs else results_folder,
        training_log_context=training_log_context,
        epochs=epochs,
        early_stopped=early_stopped,
        best_val_psnr=best_val_psnr,
        selected_epoch=selected_epoch,
        validation_interval=validation_interval,
        patience=patience,
        evaluation_checkpoint=evaluation_checkpoint,
    )


# ============================================================================
# Training function: PnP-FEI Option 1 (NAG with PnP)
# ============================================================================
def train_pnp_fei_option1(net, ckp_interval, dataloader, epochs, physics, transform, alpha, lamb, eta, sigma,
                          dtype, task, pretrained=None, pnp='BM3D', EQ_PnP=False, loss_type='l2', 
                          lr_cos=True, device=None, lr=None, schedule=None, num_train_samples=None,
                          verbose=True, resume_from=None, patience=500, dataset_config=None,
                          output_dirs=None, training_log_context=None, inner_iters=1, beta=1e-1,
                          validation_interval=0, snapshot_epochs=None, selection_config=None,
                          max_batches=None, max_val_samples=None):
    """
    Train network using PnP-FEI Option 1 (direct PnP with gradient descent) method.
    
    Args:
        lamb: FEI regularization parameter
        eta: Step size for gradient descent
        sigma: Noise level for PnP denoiser
        pnp: PnP denoiser type ('BM3D', 'dncnn', 'drunet', etc.)
        EQ_PnP: Whether to use equivariant PnP
        Other parameters same as train_fei_option1
    """
    # Path setup
    save_path = _resolve_checkpoint_dir(task, lr['G'], 'PnP-FEI-O1', epochs, output_dirs=output_dirs)
    results_folder = _resolve_results_dir(task, lr['G'], 'PnP-FEI-O1', output_dirs=output_dirs)
    figure_dir = output_dirs.get('figures', results_folder) if output_dirs else results_folder
    recon_dir = output_dirs.get('reconstructions', results_folder) if output_dirs else results_folder

    selection_config = selection_config or {}
    evaluation_checkpoint = selection_config.get('evaluation_checkpoint', 'final_model.pth.tar')

    # The supplied configurations disable training-time validation.
    val_dataloader = None
    if int(validation_interval) > 0:
        val_dataloader = _build_validation_dataloader_for_task(
            task=task,
            batch_size=1,
            dataset_config=dataset_config,
            max_val_samples=max_val_samples,
        )
    
    val_psnr_dict = {}
    val_mse_dict = {}

    # Setup PnP denoiser
    if pnp == 'BM3D':
        denoiser = dinv.models.BM3D()
    else:
        # Determine number of channels based on task
        channels = 1 if task == 'ct' else 3
        denoiser = get_model(pnp, device=device, channels=channels)

    # Use equivariant PnP if specified
    if EQ_PnP:
        if task == 'ct':
            # For CT: use random rotation
            denoiser = EquivariantDenoiser(denoiser, rand_rot=True, mean_rot=False,
                                          rand_translations=False, clamp=True)
        elif task == 'inpainting':
            # For inpainting: use random translation
            denoiser = EquivariantDenoiser(denoiser, rand_rot=False, mean_rot=False,
                                          rand_translations=True, clamp=True)

    # Setup loss functions and optimizer
    criterion_mc, criterion_ei = _setup_loss_criterion(loss_type, device)
    optimizer = Adam(net.parameters(), lr=lr['G'], weight_decay=lr['WD'])
    if training_log_context:
        _emit_init_milestone(training_log_context, 'Optimizer initialized')

    # Resume training
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch = _load_checkpoint(resume_from, net, optimizer, device)
            if start_epoch is not None:
                start_epoch += 1
                print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print("Failed to load checkpoint, starting from scratch.")
        else:
            latest_checkpoint = os.path.join(save_path, 'latest.pth.tar')
            if os.path.exists(latest_checkpoint):
                print(f"Found latest checkpoint: {latest_checkpoint}")
                start_epoch = _load_checkpoint(latest_checkpoint, net, optimizer, device)
                if start_epoch is not None:
                    start_epoch += 1
                    print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print(f"Resume checkpoint not found: {resume_from}")
                print("Starting training from scratch.")
    
    # Load pretrained model
    if start_epoch == 0 and pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained model from: {pretrained}")

    # Calculate actual number of iterations per epoch
    num_iterations_per_epoch = _effective_iterations_per_epoch(dataloader, max_batches)
    
    # Initialize recording arrays
    psnr_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_mc_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_ei_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_total_array = np.zeros((num_iterations_per_epoch, epochs))
    mse_array = np.zeros((num_iterations_per_epoch, epochs))
    time_array = np.zeros((1, epochs))

    results_list = [None] * num_iterations_per_epoch

    # Optional validation-monitoring state
    best_val_psnr = -np.inf
    selected_epoch = None
    no_improve_count = 0
    last_val_epoch = None
    last_completed_epoch = start_epoch - 1
    latest_val_psnr = None
    early_stopped = False
    run_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
        
        for i, x_batch in _limited_batch_iterator(dataloader, max_batches):
            # Prepare data
            x = _prepare_batch_data(x_batch, dtype, device)

            # Forward pass
            y0 = physics.A(x)
            z = physics.A_dagger(y0)
            x0 = net(z)

            # PnP-FEI-O1 follows the Option-1 latent path, with denoising after
            # the NAG-style latent update and fixed targets in pseudo-supervision.
            with torch.no_grad():
                x0_fixed = x0.detach()
                x1 = NAG_PnP(
                    physics=physics,
                    y0=y0,
                    x0=x0_fixed,
                    u0=x0_fixed,
                    lamb=lamb,
                    denoiser=denoiser,
                    sigma=sigma,
                    iters=inner_iters,
                    beta=beta,
                    eta=eta,
                )
            x1_target, x2_target, x2_T = _freeze_latent_targets(physics, transform, x1)
            x3 = net(x2_T)

            loss_mc = criterion_mc(x0, x1_target)
            loss_ei = criterion_ei(x3, x2_target)
            loss = loss_mc + alpha * loss_ei

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            crit_mc_array[i, epoch] = loss_mc.item()
            crit_ei_array[i, epoch] = loss_ei.item()
            crit_total_array[i, epoch] = loss.item()

            # Update results (exponential moving average on x0)
            if epoch == 0:
                results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            else:
                prev_results = results_list[i]
                if prev_results is not None:
                    results = 0.01 * x0 + 0.99 * prev_results
                else:
                    results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            results_list[i] = results
            
            # Compute metrics
            psnr = cal_psnr(results, x)
            mse = cal_mse(results, x)
            
            psnr_array[i, epoch] = psnr
            mse_array[i, epoch] = mse

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_array[0, epoch] = elapsed_time
        last_completed_epoch = epoch
        checkpoint_saved = False
        checkpoint_type = None
        early_stop_triggered = False
        current_val_psnr = latest_val_psnr

        # Save images
        if epoch == 0 or (epoch % 500 == 0 or epoch == 100):
            if len(z.shape) == 4 and z.shape[0] > 1:
                z_viz = z[0:1]
                x_viz = x[0:1] if len(x.shape) == 4 and x.shape[0] > 1 else x
            else:
                z_viz = z
                x_viz = x
            
            if len(results.shape) == 4 and results.shape[0] > 1:
                results_viz = results[0:1]
            else:
                results_viz = results
                
            _save_intermediate_images(
                epoch,
                task,
                results_folder,
                z_viz,
                x_viz,
                results_viz,
                i,
                save_dir=recon_dir,
            )
        
        if _should_run_validation(epoch, epochs, validation_interval):
            validation_state = _run_validation_pass(
                net=net,
                val_dataloader=val_dataloader,
                physics=physics,
                device=device,
                dtype=dtype,
                epoch=epoch,
                optimizer=optimizer,
                save_path=save_path,
                val_psnr_dict=val_psnr_dict,
                val_mse_dict=val_mse_dict,
                best_val_psnr=best_val_psnr,
                selected_epoch=selected_epoch,
                no_val_improve_count=no_improve_count,
                last_val_epoch=last_val_epoch,
                training_log_context=training_log_context,
                evaluation_checkpoint=evaluation_checkpoint,
            )
            current_val_psnr = validation_state['val_psnr']
            latest_val_psnr = current_val_psnr
            best_val_psnr = validation_state['best_val_psnr']
            selected_epoch = validation_state['selected_epoch']
            no_improve_count = validation_state['no_val_improve_count']
            last_val_epoch = validation_state['last_val_epoch']
            if validation_state['best_checkpoint_saved']:
                checkpoint_saved = True
                checkpoint_type = _merge_checkpoint_type(checkpoint_type, validation_state['checkpoint_type'])

        snapshot_path = _save_snapshot_checkpoint(
            net,
            optimizer,
            epoch,
            save_path,
            snapshot_epochs,
            extra_state={
                'best_val_psnr': _json_safe_number(best_val_psnr if np.isfinite(best_val_psnr) else None),
                'selected_epoch': _json_safe_number(selected_epoch),
            },
        )
        if snapshot_path is not None:
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'snapshot')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    snapshot_path,
                    'snapshot',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        # Save checkpoint
        if (epoch % ckp_interval == 0 and epoch > 0) or epoch + 1 == epochs:
            checkpoint_path = _save_checkpoint(net, optimizer, epoch, save_path, save_latest=True)
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'latest_checkpoint')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    checkpoint_path,
                    'latest_checkpoint',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        if int(validation_interval) > 0 and no_improve_count >= patience and (epoch + 1) >= validation_interval:
            early_stop_triggered = True
            early_stopped = True
            if training_log_context:
                _emit_early_stop(
                    training_log_context,
                    epoch,
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    no_val_improve_count=no_improve_count,
                    selected_epoch=selected_epoch,
                )
            else:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"best_val_psnr={best_val_psnr:.4f}, "
                    f"no_val_improve_count={no_improve_count}, selected_epoch={selected_epoch}"
                )

        if training_log_context:
            _log_epoch_progress(
                training_log_context,
                epoch=epoch,
                total_epochs=epochs,
                epoch_time_sec=elapsed_time,
                elapsed_sec=time.time() - run_start_time,
                loss_total=crit_total_array[:, epoch].mean(),
                loss_mc=crit_mc_array[:, epoch].mean(),
                loss_ei=crit_ei_array[:, epoch].mean(),
                psnr=psnr_array[:, epoch].mean(),
                ssim=None,
                best_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                val_psnr=current_val_psnr,
                best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                no_val_improve_count=no_improve_count,
                selected_epoch=selected_epoch,
                checkpoint_saved=checkpoint_saved,
                checkpoint_type=checkpoint_type,
                early_stop_triggered=early_stop_triggered,
            )
        if early_stop_triggered:
            break

    # Save all results
    _save_training_results(results_folder, psnr_array, mse_array, crit_mc_array,
                          crit_ei_array, crit_total_array, time_array)
    
    # Plot train vs validation comparison
    if len(val_psnr_dict) > 0:
        _plot_train_validation_comparison(
            results_folder,
            psnr_array,
            val_psnr_dict,
            val_mse_dict,
            algorithm_name='PnP-FEI-Option1',
            task=task,
            figure_dir=figure_dir,
            split_name='Validation',
            file_prefix='validation',
        )

    _finalize_training_run(
        net=net,
        optimizer=optimizer,
        last_completed_epoch=last_completed_epoch,
        save_path=save_path,
        output_root=output_dirs['root'] if output_dirs else results_folder,
        training_log_context=training_log_context,
        epochs=epochs,
        early_stopped=early_stopped,
        best_val_psnr=best_val_psnr,
        selected_epoch=selected_epoch,
        validation_interval=validation_interval,
        patience=patience,
        evaluation_checkpoint=evaluation_checkpoint,
    )


# ============================================================================
# Training function: PnP-FEI Option 2 (ADMM with PnP)
# ============================================================================
def train_pnp_fei_option2(net, ckp_interval, dataloader, epochs, physics, transform, alpha, lamb, eta, sigma,
                          dtype, task, pretrained=None, pnp='BM3D', EQ_PnP=False, loss_type='l2',
                          lr_cos=True, device=None, lr=None, schedule=None, num_train_samples=None,
                          verbose=True, resume_from=None, patience=500, dataset_config=None,
                          output_dirs=None, training_log_context=None, inner_iters=1,
                          validation_interval=0, snapshot_epochs=None, selection_config=None,
                          max_batches=None, max_val_samples=None):
    """
    Train network using PnP-FEI Option 2 (ADMM with Plug-and-Play prior) method.
    
    Args:
        lamb: FEI regularization parameter
        eta: Gradient descent step size
        sigma: Noise level for PnP denoiser
        pnp: PnP denoiser type ('BM3D', 'dncnn', 'drunet', etc.)
        EQ_PnP: Whether to use equivariant PnP
        Other parameters same as train_fei_option2
    """
    # Path setup
    save_path = _resolve_checkpoint_dir(task, lr['G'], 'PnP-FEI-O2', epochs, output_dirs=output_dirs)
    results_folder = _resolve_results_dir(task, lr['G'], 'PnP-FEI-O2', output_dirs=output_dirs)
    figure_dir = output_dirs.get('figures', results_folder) if output_dirs else results_folder
    recon_dir = output_dirs.get('reconstructions', results_folder) if output_dirs else results_folder

    selection_config = selection_config or {}
    evaluation_checkpoint = selection_config.get('evaluation_checkpoint', 'final_model.pth.tar')

    # The supplied configurations disable training-time validation.
    val_dataloader = None
    if int(validation_interval) > 0:
        val_dataloader = _build_validation_dataloader_for_task(
            task=task,
            batch_size=1,
            dataset_config=dataset_config,
            max_val_samples=max_val_samples,
        )
    
    val_psnr_dict = {}
    val_mse_dict = {}

    # Setup PnP denoiser
    if pnp == 'BM3D':
        denoiser = dinv.models.BM3D()
    else:
        # Determine number of channels based on task
        channels = 1 if task == 'ct' else 3
        denoiser = get_model(pnp, device=device, channels=channels)

    # Use equivariant PnP if specified
    if EQ_PnP:
        if task == 'ct':
            # For CT: use random rotation
            denoiser = EquivariantDenoiser(denoiser, rand_rot=True, mean_rot=False,
                                          rand_translations=False, clamp=True)
        elif task == 'inpainting':
            # For inpainting: use random translation
            denoiser = EquivariantDenoiser(denoiser, rand_rot=False, mean_rot=False,
                                          rand_translations=True, clamp=True)

    # Setup loss functions and optimizer
    criterion_mc, criterion_ei = _setup_loss_criterion(loss_type, device)
    optimizer = Adam(net.parameters(), lr=lr['G'], weight_decay=lr['WD'])
    if training_log_context:
        _emit_init_milestone(training_log_context, 'Optimizer initialized')

    # Resume training
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch = _load_checkpoint(resume_from, net, optimizer, device)
            if start_epoch is not None:
                start_epoch += 1
                print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print("Failed to load checkpoint, starting from scratch.")
        else:
            latest_checkpoint = os.path.join(save_path, 'latest.pth.tar')
            if os.path.exists(latest_checkpoint):
                print(f"Found latest checkpoint: {latest_checkpoint}")
                start_epoch = _load_checkpoint(latest_checkpoint, net, optimizer, device)
                if start_epoch is not None:
                    start_epoch += 1
                    print(f"Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
            else:
                print(f"Resume checkpoint not found: {resume_from}")
                print("Starting training from scratch.")
    
    # Load pretrained model
    if start_epoch == 0 and pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained model from: {pretrained}")

    # Calculate actual number of iterations per epoch
    num_iterations_per_epoch = _effective_iterations_per_epoch(dataloader, max_batches)
    
    # Initialize recording arrays
    psnr_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_mc_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_ei_array = np.zeros((num_iterations_per_epoch, epochs))
    crit_total_array = np.zeros((num_iterations_per_epoch, epochs))
    mse_array = np.zeros((num_iterations_per_epoch, epochs))
    time_array = np.zeros((1, epochs))

    # Per-sample scaled dual variables for the manuscript convention
    # ||u - F_theta(y) + L||^2 and L <- L + u - F_theta(y).
    L0_list = [None] * num_iterations_per_epoch
    results_list = [None] * num_iterations_per_epoch

    # Optional validation-monitoring state
    best_val_psnr = -np.inf
    selected_epoch = None
    no_improve_count = 0
    last_val_epoch = None
    last_completed_epoch = start_epoch - 1
    latest_val_psnr = None
    early_stopped = False
    run_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
        
        for i, x_batch in _limited_batch_iterator(dataloader, max_batches):
            # Prepare data
            x = _prepare_batch_data(x_batch, dtype, device)

            # Forward pass
            y0 = physics.A(x)
            z = physics.A_dagger(y0)
            x0 = net(z)

            x0_fixed = x0.detach()
            if L0_list[i] is None:
                L0_list[i] = torch.zeros_like(x0_fixed)
            L0 = L0_list[i].detach()

            x1 = _linearized_admm_latent_step(
                physics=physics,
                y0=y0,
                x0_fixed=x0_fixed,
                dual_fixed=L0,
                lamb=lamb,
                eta=eta,
                iters=inner_iters,
                denoiser=denoiser,
                sigma=sigma,
            )
            x1_target, x2_target, x2_T = _freeze_latent_targets(physics, transform, x1)
            x3 = net(x2_T)

            # Equivalent to ||F_theta(y) - L - x1||^2 from Algorithm 4.3.
            loss_mc = criterion_mc(x0 - L0, x1_target)
            loss_ei = criterion_ei(x3, x2_target)
            loss = loss_mc + alpha * loss_ei

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x0_updated = net(z).detach()
                L0_list[i] = (L0 + x1_target - x0_updated).detach()

            # Record losses
            crit_mc_array[i, epoch] = loss_mc.item()
            crit_ei_array[i, epoch] = loss_ei.item()
            crit_total_array[i, epoch] = loss.item()

            # Update results (exponential moving average on x0)
            if epoch == 0:
                results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            else:
                prev_results = results_list[i]
                if prev_results is not None:
                    results = 0.01 * x0 + 0.99 * prev_results
                else:
                    results = x0.clone() if len(x0.shape) == 4 and x0.shape[0] > 1 else x0
            results_list[i] = results
            
            # Compute metrics
            psnr = cal_psnr(results, x)
            mse = cal_mse(results, x)
            
            psnr_array[i, epoch] = psnr
            mse_array[i, epoch] = mse

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_array[0, epoch] = elapsed_time
        last_completed_epoch = epoch
        checkpoint_saved = False
        checkpoint_type = None
        early_stop_triggered = False
        current_val_psnr = latest_val_psnr

        # Save images
        if epoch == 0 or (epoch % 500 == 0 or epoch == 100):
            if len(z.shape) == 4 and z.shape[0] > 1:
                z_viz = z[0:1]
                x_viz = x[0:1] if len(x.shape) == 4 and x.shape[0] > 1 else x
            else:
                z_viz = z
                x_viz = x
            
            if len(results.shape) == 4 and results.shape[0] > 1:
                results_viz = results[0:1]
            else:
                results_viz = results
                
            _save_intermediate_images(
                epoch,
                task,
                results_folder,
                z_viz,
                x_viz,
                results_viz,
                i,
                save_dir=recon_dir,
            )
        
        if _should_run_validation(epoch, epochs, validation_interval):
            validation_state = _run_validation_pass(
                net=net,
                val_dataloader=val_dataloader,
                physics=physics,
                device=device,
                dtype=dtype,
                epoch=epoch,
                optimizer=optimizer,
                save_path=save_path,
                val_psnr_dict=val_psnr_dict,
                val_mse_dict=val_mse_dict,
                best_val_psnr=best_val_psnr,
                selected_epoch=selected_epoch,
                no_val_improve_count=no_improve_count,
                last_val_epoch=last_val_epoch,
                training_log_context=training_log_context,
                evaluation_checkpoint=evaluation_checkpoint,
            )
            current_val_psnr = validation_state['val_psnr']
            latest_val_psnr = current_val_psnr
            best_val_psnr = validation_state['best_val_psnr']
            selected_epoch = validation_state['selected_epoch']
            no_improve_count = validation_state['no_val_improve_count']
            last_val_epoch = validation_state['last_val_epoch']
            if validation_state['best_checkpoint_saved']:
                checkpoint_saved = True
                checkpoint_type = _merge_checkpoint_type(checkpoint_type, validation_state['checkpoint_type'])

        snapshot_path = _save_snapshot_checkpoint(
            net,
            optimizer,
            epoch,
            save_path,
            snapshot_epochs,
            extra_state={
                'best_val_psnr': _json_safe_number(best_val_psnr if np.isfinite(best_val_psnr) else None),
                'selected_epoch': _json_safe_number(selected_epoch),
            },
        )
        if snapshot_path is not None:
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'snapshot')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    snapshot_path,
                    'snapshot',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        # Save checkpoint
        if (epoch % ckp_interval == 0 and epoch > 0) or epoch + 1 == epochs:
            checkpoint_path = _save_checkpoint(net, optimizer, epoch, save_path, save_latest=True)
            checkpoint_saved = True
            checkpoint_type = _merge_checkpoint_type(checkpoint_type, 'latest_checkpoint')
            if training_log_context:
                _emit_checkpoint_notice(
                    training_log_context,
                    epoch,
                    checkpoint_path,
                    'latest_checkpoint',
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    selected_epoch=selected_epoch,
                )

        if int(validation_interval) > 0 and no_improve_count >= patience and (epoch + 1) >= validation_interval:
            early_stop_triggered = True
            early_stopped = True
            if training_log_context:
                _emit_early_stop(
                    training_log_context,
                    epoch,
                    best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                    no_val_improve_count=no_improve_count,
                    selected_epoch=selected_epoch,
                )
            else:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"best_val_psnr={best_val_psnr:.4f}, "
                    f"no_val_improve_count={no_improve_count}, selected_epoch={selected_epoch}"
                )

        if training_log_context:
            _log_epoch_progress(
                training_log_context,
                epoch=epoch,
                total_epochs=epochs,
                epoch_time_sec=elapsed_time,
                elapsed_sec=time.time() - run_start_time,
                loss_total=crit_total_array[:, epoch].mean(),
                loss_mc=crit_mc_array[:, epoch].mean(),
                loss_ei=crit_ei_array[:, epoch].mean(),
                psnr=psnr_array[:, epoch].mean(),
                ssim=None,
                best_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                val_psnr=current_val_psnr,
                best_val_psnr=best_val_psnr if np.isfinite(best_val_psnr) else None,
                no_val_improve_count=no_improve_count,
                selected_epoch=selected_epoch,
                checkpoint_saved=checkpoint_saved,
                checkpoint_type=checkpoint_type,
                early_stop_triggered=early_stop_triggered,
            )
        if early_stop_triggered:
            break

    # Save all results
    _save_training_results(results_folder, psnr_array, mse_array, crit_mc_array,
                          crit_ei_array, crit_total_array, time_array)
    
    # Plot train vs validation comparison
    if len(val_psnr_dict) > 0:
        _plot_train_validation_comparison(
            results_folder,
            psnr_array,
            val_psnr_dict,
            val_mse_dict,
            algorithm_name='PnP-FEI-Option2',
            task=task,
            figure_dir=figure_dir,
            split_name='Validation',
            file_prefix='validation',
        )

    _finalize_training_run(
        net=net,
        optimizer=optimizer,
        last_completed_epoch=last_completed_epoch,
        save_path=save_path,
        output_root=output_dirs['root'] if output_dirs else results_folder,
        training_log_context=training_log_context,
        epochs=epochs,
        early_stopped=early_stopped,
        best_val_psnr=best_val_psnr,
        selected_epoch=selected_epoch,
        validation_interval=validation_interval,
        patience=patience,
        evaluation_checkpoint=evaluation_checkpoint,
    )


# ============================================================================
# Main program entry
# ============================================================================
def _build_train_parser():
    parser = argparse.ArgumentParser(description='FEI experiment training entrypoint.')
    parser.add_argument('--config', required=True, help='Path to an experiment YAML/JSON config.')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config.')
    parser.add_argument('--device', default=None, help='Override device from config, e.g. cpu or cuda:0.')
    parser.add_argument('--output-dir', default=None, help='Base output directory. Defaults to config output.root_dir.')
    parser.add_argument('--resume', default=None, help='Checkpoint path to resume from.')
    parser.add_argument('--log-interval', type=int, default=50, help='Print training progress every N epochs.')
    parser.add_argument('--smoke-test', action='store_true', help='Run a reduced-scale diagnostic execution check.')
    parser.add_argument('--max-epochs', type=int, default=None, help='Optional epoch cap for diagnostic runs.')
    parser.add_argument('--max-batches', type=int, default=None, help='Optional per-epoch batch cap for diagnostic runs.')
    parser.add_argument('--max-train-samples', type=int, default=None, help='Optional cap on the number of training samples used.')
    parser.add_argument(
        '--dry-run-config',
        action='store_true',
        help='Resolve config, create output folders, save metadata, print config, then exit.',
    )
    return parser


def _prepare_run_artifacts(config):
    output_cfg = config.get('output', {})
    output_dirs = create_output_directories(
        task=config['task'],
        method=_method_to_slug(config['method']),
        seed=config['seed'],
        base_dir=output_cfg.get('root_dir', 'outputs'),
    )

    if output_cfg.get('save_resolved_config', True):
        save_resolved_config(config, os.path.join(output_dirs['root'], 'config_resolved.yaml'))

    environment_info = collect_environment_info()
    git_info = collect_git_info()

    if output_cfg.get('save_environment_info', True):
        save_environment_info(os.path.join(output_dirs['root'], 'environment.json'))

    if output_cfg.get('save_git_info', True):
        save_git_info(os.path.join(output_dirs['root'], 'git_info.json'))

    metadata = build_run_metadata(
        config=config,
        output_dirs=output_dirs,
        git_info=git_info,
        environment_info=environment_info,
        extra={
            'device': config['device'],
            'method_slug': _method_to_slug(config['method']),
            'task_internal': _task_to_internal(config['task']),
        },
    )

    if output_cfg.get('save_run_metadata', True):
        save_run_metadata(os.path.join(output_dirs['root'], 'run_metadata.json'), metadata)

    if config.get('runtime', {}).get('smoke_test'):
        debug_marker = os.path.join(output_dirs['root'], 'DEBUG_ONLY_SMOKE_TEST.txt')
        with open(debug_marker, 'w', encoding='utf-8') as handle:
            handle.write('Diagnostic execution-check output; not a reported paper result.\n')

    return output_dirs


def _verify_smoke_outputs(config, output_dirs, device):
    checkpoints_dir = output_dirs['checkpoints']
    checkpoint_candidates = [
        os.path.join(checkpoints_dir, 'final_model.pth.tar'),
        os.path.join(checkpoints_dir, 'latest.pth.tar'),
    ]
    checkpoint_candidates.extend(
        [
            os.path.join(checkpoints_dir, filename)
            for filename in os.listdir(checkpoints_dir)
            if filename.endswith('.pth.tar')
        ]
    )

    checkpoint_path = next((path for path in checkpoint_candidates if os.path.exists(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f'No smoke-test checkpoint found under: {checkpoints_dir}')

    verification_model = _build_model_from_config(config, device=device)
    loaded_epoch = _load_checkpoint(checkpoint_path, verification_model, optimizer=None, device=device)

    verification_summary = {
        'debug_only': True,
        'verified_checkpoint': checkpoint_path,
        'loaded_epoch': loaded_epoch,
        'config_resolved_exists': os.path.exists(os.path.join(output_dirs['root'], 'config_resolved.yaml')),
        'environment_exists': os.path.exists(os.path.join(output_dirs['root'], 'environment.json')),
        'run_metadata_exists': os.path.exists(os.path.join(output_dirs['root'], 'run_metadata.json')),
        'training_summary_exists': os.path.exists(os.path.join(output_dirs['root'], 'training_summary.json')),
        'final_model_exists': os.path.exists(os.path.join(checkpoints_dir, 'final_model.pth.tar')),
    }
    save_run_metadata(os.path.join(output_dirs['root'], 'smoke_verification.json'), verification_summary)


def _run_training_from_config(args):
    config = load_config_file(args.config)
    resolved_config = _resolve_runtime_config(config, args)
    method = resolved_config.get('method')

    _log_line(f"[Init] Config loaded: {args.config}")

    if method not in METHOD_TO_SLUG:
        raise ValueError(f"Unsupported method in config: {method}")

    device = resolved_config['device']
    task = _task_to_internal(resolved_config['task'])
    dataset_cfg = resolved_config.get('dataset', {})
    train_cfg = resolved_config.get('train', {})
    method_cfg = resolved_config.get('method_settings', {})
    runtime_cfg = resolved_config.get('runtime', {})

    set_global_seed(resolved_config['seed'])
    output_dirs = _prepare_run_artifacts(resolved_config)
    training_log_context = _build_training_log_context(resolved_config, output_dirs, args.config)
    if not args.resume and os.path.exists(training_log_context['progress_path']):
        os.remove(training_log_context['progress_path'])
    _emit_run_start(training_log_context)

    if args.dry_run_config:
        print(json.dumps(resolved_config, indent=2, sort_keys=True))
        print(f"Resolved output root: {output_dirs['root']}")
        return

    dataset = _build_training_dataset_from_config(resolved_config)
    dataset = _apply_train_sample_cap(dataset, runtime_cfg.get('max_train_samples'))
    _emit_init_milestone(
        training_log_context,
        'Dataset loaded',
        extra={'dataset_size': len(dataset)},
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_cfg.get('batch_size', 1),
        shuffle=False,
    )
    physics = _build_physics_from_config(resolved_config, device=device)
    transform = _build_transform_from_config(resolved_config)
    net = _build_model_from_config(resolved_config, device=device)
    _emit_init_milestone(training_log_context, 'Model initialized')

    lr = {
        'G': train_cfg.get('lr_G', 1e-3),
        'WD': train_cfg.get('weight_decay', 1e-8),
    }
    common_kwargs = {
        'net': net,
        'ckp_interval': train_cfg.get('ckp_interval', train_cfg.get('epochs', 1)),
        'dataloader': dataloader,
        'epochs': runtime_cfg.get('max_epochs') or train_cfg.get('epochs', 1),
        'physics': physics,
        'dtype': torch.float,
        'task': task,
        'pretrained': None,
        'loss_type': train_cfg.get('loss_type', 'l2'),
        'device': device,
        'lr': lr,
        'schedule': train_cfg.get('schedule'),
        'num_train_samples': dataset_cfg.get('train_samples'),
        'patience': train_cfg.get('patience', 500),
        'resume_from': args.resume,
        'dataset_config': dataset_cfg,
        'output_dirs': output_dirs,
        'training_log_context': training_log_context,
        'validation_interval': train_cfg.get('validation_interval', 0),
        'snapshot_epochs': train_cfg.get('snapshot_epochs', _default_snapshot_epochs(resolved_config.get('task'))),
        'selection_config': resolved_config.get('selection', {}),
        'max_batches': runtime_cfg.get('max_batches'),
        'max_val_samples': runtime_cfg.get('max_val_samples'),
    }

    alpha = method_cfg.get('alpha', 1.0)
    lamb = method_cfg.get('lambda', 0.1)
    beta = method_cfg.get('beta', 0.1)
    eta_default = 1e-2 if task == 'ct' else 1.0
    eta = method_cfg.get('eta', eta_default)

    if method == 'EI':
        _emit_init_milestone(training_log_context, 'Training loop started')
        train_ei(transform=transform, alpha=alpha, **common_kwargs)
        if runtime_cfg.get('smoke_test'):
            _verify_smoke_outputs(resolved_config, output_dirs, device)
        return

    if method == 'FEI-O1':
        _emit_init_milestone(training_log_context, 'Training loop started')
        train_fei_option1(
            transform=transform,
            alpha=alpha,
            lamb=lamb,
            beta=beta,
            eta=eta,
            inner_iters=method_cfg.get('J', 10),
            **common_kwargs,
        )
        if runtime_cfg.get('smoke_test'):
            _verify_smoke_outputs(resolved_config, output_dirs, device)
        return

    if method == 'FEI-O2':
        _emit_init_milestone(training_log_context, 'Training loop started')
        train_fei_option2(
            transform=transform,
            alpha=alpha,
            lamb=lamb,
            eta=eta,
            inner_iters=method_cfg.get('J', 1),
            **common_kwargs,
        )
        if runtime_cfg.get('smoke_test'):
            _verify_smoke_outputs(resolved_config, output_dirs, device)
        return

    if method == 'PnP-FEI-O1':
        _emit_init_milestone(training_log_context, 'Training loop started')
        train_pnp_fei_option1(
            transform=transform,
            alpha=alpha,
            lamb=lamb,
            eta=eta,
            sigma=method_cfg.get('pnp_sigma', 0.01),
            pnp=method_cfg.get('pnp_denoiser', 'dncnn'),
            EQ_PnP=method_cfg.get('eq_pnp', False),
            inner_iters=method_cfg.get('J', 1),
            beta=beta,
            **common_kwargs,
        )
        if runtime_cfg.get('smoke_test'):
            _verify_smoke_outputs(resolved_config, output_dirs, device)
        return

    if method == 'PnP-FEI-O2':
        _emit_init_milestone(training_log_context, 'Training loop started')
        train_pnp_fei_option2(
            transform=transform,
            alpha=alpha,
            lamb=lamb,
            eta=eta,
            sigma=method_cfg.get('pnp_sigma', 0.01),
            pnp=method_cfg.get('pnp_denoiser', 'dncnn'),
            EQ_PnP=method_cfg.get('eq_pnp', False),
            inner_iters=method_cfg.get('J', 1),
            **common_kwargs,
        )
        if runtime_cfg.get('smoke_test'):
            _verify_smoke_outputs(resolved_config, output_dirs, device)
        return

    raise ValueError(f"Unsupported method: {method}")


if __name__ == '__main__':
    parser = _build_train_parser()
    _run_training_from_config(parser.parse_args())
