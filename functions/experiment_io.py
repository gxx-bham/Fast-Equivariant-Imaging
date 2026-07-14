import json
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed, deterministic=False):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def create_output_directories(task, method, seed, base_dir="outputs", extra_subdirs=None):
    root = Path(base_dir) / str(task) / str(method) / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)

    directories = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "figures": root / "figures",
        "reconstructions": root / "reconstructions",
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    if extra_subdirs:
        for name in extra_subdirs:
            path = root / str(name)
            path.mkdir(parents=True, exist_ok=True)
            directories[str(name)] = path

    return {key: str(path) for key, path in directories.items()}


def load_config_file(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Loading YAML configs requires PyYAML to be installed."
            ) from exc

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        return {} if data is None else data

    raise ValueError(f"Unsupported config format: {path}")


def save_resolved_config(config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return str(output_path)

    if suffix in {".yaml", ".yml"}:
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(_dict_to_yaml(config))
            if not str(_dict_to_yaml(config)).endswith("\n"):
                handle.write("\n")
        return str(output_path)

    raise ValueError(f"Unsupported resolved config format: {output_path}")


def collect_environment_info():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "numpy_version": np.__version__,
    }

    if torch.cuda.is_available():
        try:
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["torch_cuda_device_name"] = "unavailable"

    return info


def save_environment_info(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = collect_environment_info()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return str(output_path)


def collect_git_info(repo_root=None):
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    info = {
        "repo_root": str(repo_root.resolve()),
        "git_available": False,
        "commit": None,
        "branch": None,
        "is_dirty": None,
        "error": None,
    }

    try:
        commit = _run_git_command(["git", "rev-parse", "HEAD"], repo_root)
        branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        status = _run_git_command(["git", "status", "--porcelain"], repo_root)
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["git_available"] = True
    info["commit"] = commit
    info["branch"] = branch
    info["is_dirty"] = bool(status)
    return info


def save_git_info(output_path, repo_root=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = collect_git_info(repo_root=repo_root)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return str(output_path)


def save_run_metadata(output_path, metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return str(output_path)


def build_run_metadata(config, output_dirs=None, git_info=None, environment_info=None, extra=None):
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config.get("config_name"),
        "stage": config.get("stage"),
        "task": config.get("task"),
        "method": config.get("method"),
        "seed": config.get("seed"),
        "output_dirs": output_dirs or {},
        "git": git_info,
        "environment": environment_info,
    }

    if extra:
        metadata.update(extra)

    return metadata


def _run_git_command(command, cwd):
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _dict_to_yaml(data):
    lines = _yaml_lines(data, indent=0)
    return "\n".join(lines) + "\n"


def _yaml_lines(value, indent):
    prefix = " " * indent

    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines

    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines

    return [f"{prefix}{_yaml_scalar(value)}"]


def _yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "!", "%", "@", "`"]) or text.strip() != text:
        return json.dumps(text)
    lowered = text.lower()
    if lowered in {"true", "false", "null", "~"}:
        return json.dumps(text)
    return text
