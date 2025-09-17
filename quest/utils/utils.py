import copy
import json
import os
import random
from pathlib import Path
import quest.utils.tensor_utils as TensorUtils
import numpy as np
import torch
import torch.nn as nn
import warnings
from natsort import natsorted
from typing import Any, Optional, Union

def get_experiment_dir(
    cfg: Any,
    evaluate: bool = False,
    allow_overlap: bool = False,
) -> tuple[str, str]:
    """
    Build the experiment directory path and a short experiment name.

    Args:
        cfg (Any):
            A config-like object with the following expected attributes:
            - output_prefix (str): Root output directory.
            - task.suite_name (str): Task suite identifier.
            - task.benchmark_name (str): Benchmark identifier.
            - algo.name (str): Algorithm name.
            - exp_name (str): Experiment name.
            - variant_name (str | None): Optional variant subfolder.
            - seed (int): Random seed; appended if not equal to 10000.
            - make_unique_experiment_dir (bool): Whether to create run_XXX subfolders.
            - stage (int | str): Stage index, used when not making unique dirs.
            - training.resume (bool): If True, allows reusing an existing folder.
        evaluate (bool, optional):
            If True, put results under an "evaluate" subdirectory of ``output_prefix``.
            Defaults to False.
        allow_overlap (bool, optional):
            If False and ``cfg.make_unique_experiment_dir`` is False and training is not
            resuming, assert that the computed directory does not already exist.
            Defaults to False.

    Returns:
        tuple[str, str]:
            - experiment_dir: Full directory path to write outputs.
            - experiment_name: A concise name derived from the path (relative to
              ``output_prefix``) with path separators replaced by underscores.
    """
    # Base prefix, optionally nested under "evaluate" when evaluating.
    prefix = cfg.output_prefix
    if evaluate:
        prefix = os.path.join(prefix, 'evaluate')

    # Sanitize user-provided components to avoid accidental leading/trailing slashes
    exp_name = str(cfg.exp_name).strip().strip("'\"").strip('/\\')
    variant_name = None
    if cfg.variant_name is not None:
        variant_name = str(cfg.variant_name).strip().strip("'\"").strip('/\\')

    # Core hierarchy: <prefix>/<suite>/<benchmark>/<algo>/<exp_name>
    base_parts = [
        prefix,
        cfg.task.suite_name,
        cfg.task.benchmark_name,
        cfg.algo.name,
        exp_name,
    ]
    if variant_name is not None:
        base_parts.append(variant_name)
    if cfg.seed != 10000:
        base_parts.append(str(cfg.seed))
    base_dir = os.path.join(*base_parts)

    if cfg.make_unique_experiment_dir:
        # Create an incrementing run_XXX folder to avoid collisions.
        experiment_id = 0
        experiment_dir = base_dir
        if os.path.exists(experiment_dir):
            for path in Path(experiment_dir).glob("run_*"):
                if not path.is_dir():
                    continue
                try:
                    folder_id = int(str(path).split("run_")[-1])
                    if folder_id > experiment_id:
                        experiment_id = folder_id
                except BaseException:
                    # Ignore folders that don't match the pattern strictly
                    pass
            experiment_id += 1

        experiment_dir = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    else:
        # Otherwise, organize by training stage.
        experiment_dir = os.path.join(base_dir, f'stage_{cfg.stage}')

        # Prevent accidental overwrites unless explicitly allowed or resuming.
        if not allow_overlap and not cfg.training.resume:
            assert not os.path.exists(experiment_dir), (
                'cfg.make_unique_experiment_dir=false but '
                f"{experiment_dir} is already occupied"
            )

    # Derive a short display name from the path relative to output_prefix.
    rel_path = os.path.relpath(experiment_dir, cfg.output_prefix)
    experiment_name = rel_path.replace(os.sep, '_')
    return experiment_dir, experiment_name

def get_experiment_dir_for_mixed_dataset(
    cfg: Any,
    evaluate: bool = False,
    allow_overlap: bool = False,
) -> tuple[str, str]:
    """
    Build the experiment directory path and a short experiment name for mixed datasets.

    Args:
        cfg (Any):
            A config-like object with the same expected attributes as in
            `get_experiment_dir`, but tailored for mixed dataset scenarios.

    Returns:
        tuple[str, str]:
            - experiment_dir: Full directory path to write outputs.
            - experiment_name: A concise name derived from the path (relative to
              ``output_prefix``) with path separators replaced by underscores.
    """
    # Base prefix, optionally nested under "evaluate" when evaluating.
    prefix = cfg.output_prefix
    if evaluate:
        prefix = os.path.join(prefix, 'evaluate')

    # Sanitize user-provided components to avoid accidental leading/trailing slashes
    exp_name = str(cfg.exp_name).strip().strip("'\"").strip('/\\')
    variant_name = None
    if cfg.variant_name is not None:
        variant_name = str(cfg.variant_name).strip().strip("'\"").strip('/\\')

    # Core hierarchy: <prefix>/<suite>/mixed_datasets/<algo>/<exp_name>
    base_parts = [
        prefix,
        cfg.task.suite_name,
        'mixed_datasets',
        cfg.algo.name,
        exp_name,
    ]
    if variant_name is not None:
        base_parts.append(variant_name)
    if cfg.seed != 10000:
        base_parts.append(str(cfg.seed))
    base_dir = os.path.join(*base_parts)

    if cfg.make_unique_experiment_dir:
        # Create an incrementing run_XXX folder to avoid collisions.
        experiment_id = 0
        experiment_dir = base_dir
        if os.path.exists(experiment_dir):
            for path in Path(experiment_dir).glob("run_*"):
                if not path.is_dir():
                    continue
                try:
                    folder_id = int(str(path).split("run_")[-1])
                    if folder_id > experiment_id:
                        experiment_id = folder_id
                except BaseException:
                    # Ignore folders that don't match the pattern strictly
                    pass
            experiment_id += 1

        experiment_dir = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    else:
        # Otherwise, organize by training stage.
        experiment_dir = os.path.join(base_dir, f'stage_{cfg.stage}')

        # Prevent accidental overwrites unless explicitly allowed or resuming.
        if not allow_overlap and not cfg.training.resume:
            assert not os.path.exists(experiment_dir), (
                'cfg.make_unique_experiment_dir=false but '
                f"{experiment_dir} is already occupied"
            )

    # Derive a short display name from the path relative to output_prefix.
    rel_path = os.path.relpath(experiment_dir, cfg.output_prefix)
    experiment_name = rel_path.replace(os.sep, '_')
    return experiment_dir, experiment_name

def get_checkpoint_with_selection(checkpoint_dir: Union[str, os.PathLike[str]]) -> Optional[str]:
    """Select a checkpoint file interactively from a directory.

    If ``checkpoint_dir`` points to a file, it is returned immediately. If it
    points to a directory, the files inside are listed (natural sort), printed,
    and the user is prompted to select one.

    Args:
        checkpoint_dir (str | os.PathLike):
            Path to a checkpoint file, or to a directory containing checkpoint files.

    Returns:
        Optional[str]:
            Full path to the selected checkpoint file (as provided, not necessarily
            absolute), or ``None`` if the selection is cancelled via keyboard
            interrupt (Ctrl-C).

    Raises:
        FileNotFoundError: If ``checkpoint_dir`` does not exist.
        ValueError: If ``checkpoint_dir`` is a directory but contains no files.
    """
    # Normalize PathLike to string for consistent operations
    checkpoint_dir = os.fspath(checkpoint_dir)

    # If a file path is provided, return it as-is
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    # Provide an explicit error if the path does not exist
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Path does not exist: {checkpoint_dir}")

    # Gather files directly under the directory and sort naturally for readability
    onlyfiles = [
        f for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))
    ]
    onlyfiles = natsorted(onlyfiles)
    
    if not onlyfiles:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    print(f"Available checkpoints in {checkpoint_dir}:")
    for i, filename in enumerate(onlyfiles):
        print(f"  {i + 1}: {filename}")
    
    while True:
        try:
            choice = input(f"Please select a checkpoint (1-{len(onlyfiles)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(onlyfiles):
                selected_file = onlyfiles[choice_idx]
                print(f"Selected: {selected_file}")
                return os.path.join(checkpoint_dir, selected_file)
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(onlyfiles)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def get_latest_checkpoint(checkpoint_dir):
    """Keep the original function for backward compatibility"""
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    onlyfiles = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
    onlyfiles = natsorted(onlyfiles)
    best_file = onlyfiles[-1]
    return os.path.join(checkpoint_dir, best_file)

def soft_load_state_dict(model, loaded_state_dict):
    # loaded_state_dict['task_encoder.weight'] = loaded_state_dict['task_encodings.weight']
    
    current_model_dict = model.state_dict()
    new_state_dict = {}

    for k in current_model_dict.keys():
        if k in loaded_state_dict:
            v = loaded_state_dict[k]
            if not hasattr(v, 'size') or v.size() == current_model_dict[k].size():
                new_state_dict[k] = v
            else:
                warnings.warn(f'Cannot load checkpoint parameter {k} with shape {loaded_state_dict[k].shape}'
                            f'into model with corresponding parameter shape {current_model_dict[k].shape}. Skipping')
                new_state_dict[k] = current_model_dict[k]
        else:
            new_state_dict[k] = current_model_dict[k]
            warnings.warn(f'Model parameter {k} does not exist in checkpoint. Skipping')
    for k in loaded_state_dict.keys():
        if k not in current_model_dict:
            warnings.warn(f'Loaded checkpoint parameter {k} does not exist in model. Skipping')
    
    model.load_state_dict(new_state_dict)

def map_tensor_to_device(data, device):
    """Move data to the device specified by device."""
    return TensorUtils.map_tensor(
        data, lambda x: safe_device(x, device=device)
    )

def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()

def extract_state_dicts(inp):

    if not (isinstance(inp, dict) or isinstance(inp, list)):
        if hasattr(inp, 'state_dict'):
            return inp.state_dict()
        else:
            return inp
    elif isinstance(inp, list):
        out_list = []
        for value in inp:
            out_list.append(extract_state_dicts(value))
        return out_list
    else:
        out_dict = {}
        for key, value in inp.items():
            out_dict[key] = extract_state_dicts(value)
        return out_dict
        
def save_state(state_dict, path):
    save_dict = extract_state_dicts(state_dict)
    torch.save(save_dict, path)

def load_state(path: Union[str, os.PathLike], map_location: Optional[Union[str, torch.device]] = None):
    """Load a checkpoint with optional device remapping.

    Args:
        path: Path to the checkpoint file.
        map_location: Optional map location to remap storages. Examples: 'cpu',
            torch.device('cuda:0'). If None, will try default loading first and
            fall back to CPU on CUDA device mismatch.

    Returns:
        The object loaded by torch.load.
    """
    try:
        if map_location is not None:
            return torch.load(path, map_location=map_location)
        return torch.load(path)
    except RuntimeError as e:
        # Common when a checkpoint was saved on cuda:1 but only cuda:0 exists now
        msg = str(e)
        if (
            'Attempting to deserialize object on CUDA device' in msg
            or 'Invalid device id' in msg
            or 'CUDA error' in msg
        ):
            warnings.warn(
                f"Falling back to CPU map_location for loading '{path}' due to: {e}"
            )
            return torch.load(path, map_location='cpu')
        raise

def torch_save_model(model, optimizer, scheduler, model_path, cfg=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": cfg,
        },
        model_path,
    )

def torch_load_model(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"], checkpoint["scheduler_state_dict"], checkpoint["cfg"]
