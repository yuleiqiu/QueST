#!/usr/bin/env python3
"""
Single Task Evaluation Script for LiberoSingleTaskRunner using Hydra.

This script evaluates a trained checkpoint on a single specific task from a Libero benchmark,
leveraging the existing hydra configuration system.

To use this script, you should create a YAML configuration file for your single-task evaluation.
You can create a file `config/evaluate_single_task_custom.yaml` with the following content:

```yaml
# config/evaluate_single_task_custom.yaml
defaults:
  - _self_
  - override hydra/launcher: submitit_local

hydra:
  run:
    dir: ./experiments/evaluate_single_task/${task.benchmark_name}_task${task.task_id}/${now:%Y.%m.%d}/${now:%H.%M.%S}
  sweep:
    dir: ./experiments/evaluate_single_task/${task.benchmark_name}_task${task.task_id}
    subdir: ${hydra.job.num}

# General settings
device: cuda:0
seed: 42
checkpoint_path: /path/to/your/checkpoint.pth # IMPORTANT: Change this path

# Task specific configuration
task:
  suite_name: libero
  benchmark_name: LIBERO_10 # e.g., LIBERO_10, LIBERO_OBJECT
  task_id: 0 # The ID of the task you want to evaluate
  img_height: 128
  img_width: 128
  horizon: 500
  task_embedding_format: clip
  shape_meta:
    action_dim: 7
    observation:
      rgb:
        agentview_rgb: [3, 128, 128]
        eye_in_hand_rgb: [3, 128, 128]
      lowdim:
        joint_states: 7
        ee_pos: 3
        gripper_states: 2
    task:
      type: vector
      dim: 512
  obs_key_mapping:
    agentview_rgb: 'agentview_image'
    eye_in_hand_rgb: 'robot0_eye_in_hand_image'
    gripper_states: 'robot0_gripper_qpos'
    joint_states: 'robot0_joint_pos'
    ee_pos: 'robot0_eef_pos'

# Rollout settings
rollout:
  rollouts_per_env: 20
  num_parallel_envs: 1
  max_episode_length: 500
  n_video: 3
  fps: 24
  debug: false

```

Then, you can run the evaluation with:
    python evaluate_single_task_hydra.py --config-name=evaluate_single_task_custom.yaml

"""

import functools
import json
import os
import pdb
import time
import types
from pprint import pprint

import hydra
import torch
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig, OmegaConf

import quest.utils.libero_utils as lu
import quest.utils.utils as utils
from quest.env_runner.libero_runner import LiberoSingleTaskRunner


def load_model_from_checkpoint(config: DictConfig):
    """
    Load model from checkpoint

    Args:
        config (DictConfig): Hydra configuration object.
    """
    print(f"Loading checkpoint from: {config.checkpoint_dir}")
    checkpoint_path = utils.get_checkpoint_with_selection(config.checkpoint_dir)

    # Load model state
    state_dict = utils.load_state(checkpoint_path)
    
    # Create model based on saved config or provided config
    if 'config' in state_dict:
        print('Auto-loading model based on saved parameters')
        policy_config_from_model = state_dict['config']['algo']['policy']
        # pprint(policy_config_from_model)
        # pdb.set_trace()
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


def create_env_runner(config):
    """
    Create LiberoSingleTaskRunner
    """
    env_factory = functools.partial(
        lu.LiberoWrapper,
        shape_meta=config.task.shape_meta,
        obs_key_mapping=config.task.obs_key_mapping,
        img_height=config.task.img_height,
        img_width=config.task.img_width,
        device=config.device,
    )
    
    runner = LiberoSingleTaskRunner(
        env_factory=env_factory,
        benchmark_name=config.task.benchmark_name,
        task_id=config.task.task_id,
        rollouts_per_env=config.rollout.rollouts_per_env,
        num_parallel_envs=config.rollout.num_parallel_envs,
        max_episode_length=config.rollout.max_episode_length,
        frame_stack=1,
        fps=config.rollout.fps,
        debug=config.rollout.debug,
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
    OmegaConf.resolve(config)
    save_dir, _ = utils.get_experiment_dir(config, evaluate=True)
    os.makedirs(save_dir, exist_ok=True)
    print('Saving to:', save_dir)

    # # Have a look at config
    # print("The config loaded from yaml is:")
    # print(OmegaConf.to_yaml(config))
    # pdb.set_trace()

    # Load model
    model = load_model_from_checkpoint(config)
    # pdb.set_trace()

    # Create environment runner
    env_runner = create_env_runner(config)
    
    # Get task name for display
    benchmark = lu.get_benchmark(config.task.benchmark_name)()
    task_name = benchmark.get_task_names()[config.task.task_id]
    
    print(f"\n=== Single Task Evaluation (Hydra) ===")
    print(f"Benchmark: {config.task.benchmark_name}")
    print(f"Task ID: {config.task.task_id}")
    print(f"Task Name: {task_name}")
    print(f"Rollouts per env: {config.rollout.rollouts_per_env}")
    print(f"Number of videos: {config.rollout.n_video}")
    print(f"Device: {config.device}")
    print("=" * 40)
    
    # Define video save function
    def video_save_fn(video_chw, env_name, idx):
        save_video_fn(video_chw, env_name, idx, save_dir, config.rollout.fps)
    
    # Run evaluation
    print("Running evaluation...")
    start_time = time.time()
    
    rollout_results = env_runner.run(
        model, 
        n_video=config.rollout.n_video,
        do_tqdm=True,
        save_video_fn=video_save_fn if config.rollout.n_video > 0 else None
    )
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Success Rate: {rollout_results['rollout']['overall_success_rate']:.3f}")
    print(f"Average Reward: {rollout_results['rollout']['overall_average_reward']:.3f}")
    print(f"Environments Solved: {rollout_results['rollout']['environments_solved']}")
    if task_name in rollout_results['rollout_success_rate']:
        print(f"Task-specific Success Rate: {rollout_results['rollout_success_rate'][task_name]:.3f}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    
    # Save results
    results_data = {
        'args': OmegaConf.to_container(config),
        'results': rollout_results,
        'evaluation_time': evaluation_time,
        'task_name': task_name
    }
    
    results_file = os.path.join(save_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    if config.rollout.n_video > 0:
        print(f"Videos saved to: {os.path.join(save_dir, 'videos')}")


if __name__ == "__main__":
    main()
