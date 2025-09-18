#!/usr/bin/env python3

import functools
import json
import os
import time
from typing import List, Optional
import fnmatch

import hydra
import torch
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig, OmegaConf

import quest.utils.libero_utils as lu
import quest.utils.utils as utils
from quest.env_runner.libero_runner import LiberoSelectedTaskRunner


def load_model_from_checkpoint(config: DictConfig, checkpoint_path: Optional[str] = None):
    """
    Load model from checkpoint

    Args:
        config (DictConfig): Hydra configuration object.
        checkpoint_path (Optional[str]): Explicit checkpoint file path. When None, fall back to interactive selection.
    """
    if checkpoint_path is None:
        print(f"Loading checkpoint from: {config.checkpoint_dir}")
        checkpoint_path = utils.get_checkpoint_with_selection(config.checkpoint_dir)

    # Load model state, remapping storages to the target device
    device_str = str(config.device)
    map_location = 'cpu' if device_str.lower().startswith('cpu') else device_str
    state_dict = utils.load_state(checkpoint_path, map_location=map_location)
    
    # Create model based on saved config or provided config
    if 'config' in state_dict:
        print('Auto-loading model based on saved parameters')
        policy_config_from_model = state_dict['config']['algo']['policy']
        # Ensure device from current runtime overrides any saved device (e.g., cuda:1)
        try:
            if isinstance(policy_config_from_model, dict):
                # Top-level policy device
                if 'device' in policy_config_from_model:
                    policy_config_from_model['device'] = str(config.device)
                # Common nested module devices
                for subkey in ['diffusion_model', 'encoder', 'vision_encoder', 'policy', 'actor', 'critic']:
                    if subkey in policy_config_from_model and isinstance(policy_config_from_model[subkey], dict):
                        if 'device' in policy_config_from_model[subkey]:
                            policy_config_from_model[subkey]['device'] = str(config.device)
        except Exception:
            # Non-fatal: if structure is unexpected, we'll still set after instantiation via model.to()
            pass
        model = hydra.utils.instantiate(
            policy_config_from_model, 
            shape_meta=config.task.shape_meta
        )
    else:
        print("No saved config found in checkpoint. Cannot infer model architecture.")
        print("Using a default policy configuration.")
        model = hydra.utils.instantiate(
            config.algo.policy,
            shape_meta=config.task.shape_meta
        )

    model.to(config.device)
    model.eval()
    model.load_state_dict(state_dict['model'])
    print(f"Model loaded successfully on {config.device}")

    return model


def list_checkpoints(checkpoint_dir: str, pattern: Optional[str] = None) -> List[str]:
    """List checkpoint files in a directory.

    - Includes files ending with .pth or .pt by default.
    - Optional simple glob-style pattern (e.g., "*epoch_*.pth").
    - Returns sorted list for reproducible order.
    """
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Expected a directory for checkpoint_dir, got: {checkpoint_dir}")

    files = []
    for f in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, f)
        if not os.path.isfile(full):
            continue
        if f.endswith((".pth", ".pt")):
            files.append(f)

    if pattern:
        files = [f for f in files if fnmatch.fnmatch(f, pattern)]

    # Sort alphabetically; names like epoch_0010 will be in numeric order
    files.sort()
    return [os.path.join(checkpoint_dir, f) for f in files]


def create_env_runner(config):
    """
    Build and return a LiberoSelectedTaskRunner configured from the Hydra config.

    Args:
        config (DictConfig): Resolved config providing:
            - task.shape_meta, task.obs_key_mapping: Observation/action spec for LiberoWrapper.
            - task.img_height, task.img_width (int): Camera image resolution for env frames.
            - device (str|torch.device): Device to place env tensors/policy.
            - task.benchmark_name (str): LIBERO benchmark identifier.
            - task.task_id (int): Index of the single task to evaluate within the benchmark.
            - task.task_embedding_format (str|None): How the task embedding is provided to the policy.
            - rollout.rollouts_per_env (int): Number of rollouts to run per parallel env.
            - rollout.num_parallel_envs (int): Number of environments to run in parallel.
            - rollout.max_episode_length (int): Max steps per episode.
            - rollout.fps (int): Frames-per-second used for rendering/saving videos.

    Returns:
        LiberoSelectedTaskRunner: Runner wired with an env_factory that creates
        quest.utils.libero_utils.LiberoWrapper. Intended to be
        reused across one or multiple checkpoint evaluations.
    """
    env_factory = functools.partial(
        lu.LiberoWrapper,
        shape_meta=config.task.shape_meta,
        obs_key_mapping=config.task.obs_key_mapping,
        img_height=config.task.img_height,
        img_width=config.task.img_width,
        device=config.device,
    )

    # Ensure task_ids is a list[int] (supports single int or iterable like OmegaConf ListConfig)
    task_ids_cfg = config.task.task_id
    if isinstance(task_ids_cfg, int):
        task_ids_list = [task_ids_cfg]
    else:
        # Accept any non-string iterable (e.g., OmegaConf ListConfig) and coerce to list
        if isinstance(task_ids_cfg, (str, bytes)):
            raise TypeError(f"config.task.task_id must be an int or a sequence of ints, got string: {task_ids_cfg}")
        try:
            task_ids_list = list(task_ids_cfg)
        except TypeError:
            raise TypeError(f"config.task.task_id must be int or sequence of ints, got: {type(task_ids_cfg)}")
    # Validate and coerce elements to int
    try:
        task_ids_list = [int(t) for t in task_ids_list]
    except (TypeError, ValueError):
        raise TypeError("All elements of config.task.task_id must be integers or coercible to int")

    runner = LiberoSelectedTaskRunner(
        env_factory=env_factory,
        benchmark_name=config.task.benchmark_name,
        task_ids=task_ids_list,
        rollouts_per_env=config.rollout.rollouts_per_env,
        num_parallel_envs=config.rollout.num_parallel_envs,
        max_episode_length=config.rollout.max_episode_length,
        frame_stack=config.algo.frame_stack,
        fps=config.rollout.fps,
        task_embedding_format=config.task.task_embedding_format
    )
    
    return runner


def save_video_fn(video_chw, env_name, idx, save_dir, fps):
    """Save video function"""
    video_dir = os.path.join(save_dir, 'videos', env_name)
    os.makedirs(video_dir, exist_ok=True)
    save_path = os.path.join(video_dir, f'{idx}.mp4')
    clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=fps)
    clip.write_videofile(save_path, fps=fps, verbose=False, logger=None)
    print(f"Video saved: {save_path}")


OmegaConf.register_new_resolver("eval", eval, replace=True)
@hydra.main(config_path="config", config_name="evaluate_single_task", version_base=None)
def main(config: DictConfig):
    device = config.device
    seed = config.seed

    # Set random seed
    torch.manual_seed(seed)

    # Resolve config to avoid interpolation issues
    OmegaConf.resolve(config)

    # Create save directory
    save_dir, _ = utils.get_experiment_dir_for_mixed_dataset(config, evaluate=True)
    os.makedirs(save_dir, exist_ok=True)
    print('Saving to:', save_dir)

    # # Have a look at config
    # print("The config loaded from yaml is:")
    # print(OmegaConf.to_yaml(config))
    # pdb.set_trace()

    # Determine single vs. multi-checkpoint evaluation
    eval_all = getattr(config, 'eval_all_checkpoints', False)
    pattern = getattr(config, 'checkpoint_pattern', None)

    # Create environment runner once; reuse across checkpoints
    env_runner = create_env_runner(config)
    
    # Selected task names (supports multiple task ids)
    # Will be available after runner is created
    
    # print(f"\n=== Single Task Evaluation (Hydra) ===")
    # print(f"Benchmark: {config.task.benchmark_name}")
    # print(f"Task ID: {config.task.task_id}")
    # print(f"Task Name: {task_name}")
    # print(f"Rollouts per env: {config.rollout.rollouts_per_env}")
    # print(f"Number of videos: {config.rollout.n_video}")
    # print(f"Device: {config.device}")
    # print("=" * 40)
    
    if eval_all and os.path.isdir(config.checkpoint_dir):
        print("Evaluating ALL checkpoints in directory...")
        checkpoints = list_checkpoints(config.checkpoint_dir, pattern)
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {config.checkpoint_dir} with pattern: {pattern or '*.pth|*.pt'}")

        summary = []
        selected_task_names = env_runner.env_names
        for i, ckpt_path in enumerate(checkpoints, 1):
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            ckpt_save_dir = os.path.join(save_dir, ckpt_name)
            os.makedirs(ckpt_save_dir, exist_ok=True)

            print(f"\n[{i}/{len(checkpoints)}] Running evaluation for checkpoint: {ckpt_name}")
            model = load_model_from_checkpoint(config, checkpoint_path=ckpt_path)

            # Per-checkpoint video callback
            def save_video_callback_fn(video_chw, env_name, idx, _dir=ckpt_save_dir):
                save_video_fn(video_chw, env_name, idx, _dir, config.rollout.fps)

            start_time = time.time()
            rollout_results = env_runner.run(
                model,
                n_video=config.rollout.n_video,
                do_tqdm=True,
                save_video_fn=save_video_callback_fn if config.rollout.n_video > 0 else None,
            )
            evaluation_time = time.time() - start_time

            # Persist per-ckpt results
            results_data = {
                'args': OmegaConf.to_container(config),
                'checkpoint': ckpt_path,
                'results': rollout_results,
                'evaluation_time': evaluation_time,
                'task_names': selected_task_names,
            }
            results_file = os.path.join(ckpt_save_dir, 'results.json')
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)

            # Print short metrics
            print(f"Success Rate: {rollout_results['rollout']['overall_success_rate']:.3f} | Avg Reward: {rollout_results['rollout']['overall_average_reward']:.3f} | Time: {evaluation_time:.1f}s")
            if config.rollout.n_video > 0:
                print(f"Videos saved to: {os.path.join(ckpt_save_dir, 'videos')}")

            summary.append({
                'checkpoint': ckpt_path,
                'checkpoint_name': ckpt_name,
                'overall_success_rate': rollout_results['rollout']['overall_success_rate'],
                'overall_average_reward': rollout_results['rollout']['overall_average_reward'],
                'environments_solved': rollout_results['rollout']['environments_solved'],
                'task_success_rates': {name: rollout_results['rollout_success_rate'].get(name, None) for name in selected_task_names},
                'evaluation_time_sec': evaluation_time,
            })

            # Free GPU memory before next checkpoint
            try:
                del model
                if str(config.device).startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Save summary at top-level save_dir
        summary_path = os.path.join(save_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'task_names': selected_task_names,
                'num_checkpoints': len(summary),
                'checkpoints': summary,
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    else:
        # Single-checkpoint flow (interactive selection if a directory is given)
        model = load_model_from_checkpoint(config)

        # Define video save callback function
        def save_video_callback_fn(video_chw, env_name, idx):
            save_video_fn(video_chw, env_name, idx, save_dir, config.rollout.fps)

    # Run evaluation
        print("Running evaluation...")
        start_time = time.time()

        rollout_results = env_runner.run(
            model,
            n_video=config.rollout.n_video,
            do_tqdm=True,
            save_video_fn=save_video_callback_fn if config.rollout.n_video > 0 else None
        )

        evaluation_time = time.time() - start_time

        # Print results
        print(f"\n=== Results ===")
        print(f"Success Rate: {rollout_results['rollout']['overall_success_rate']:.3f}")
        print(f"Average Reward: {rollout_results['rollout']['overall_average_reward']:.3f}")
        print(f"Environments Solved: {rollout_results['rollout']['environments_solved']}")
        selected_task_names = env_runner.env_names
        if 'rollout_success_rate' in rollout_results and isinstance(rollout_results['rollout_success_rate'], dict):
            print("Per-task Success Rates:")
            for name in selected_task_names:
                if name in rollout_results['rollout_success_rate']:
                    print(f" - {name}: {rollout_results['rollout_success_rate'][name]:.3f}")
        print(f"Evaluation Time: {evaluation_time:.2f} seconds")

        # Save results
        results_data = {
            'args': OmegaConf.to_container(config),
            'results': rollout_results,
            'evaluation_time': evaluation_time,
            'task_names': selected_task_names
        }

        results_file = os.path.join(save_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to: {results_file}")
        if config.rollout.n_video > 0:
            print(f"Videos saved to: {os.path.join(save_dir, 'videos')}")


if __name__ == "__main__":
    main()
