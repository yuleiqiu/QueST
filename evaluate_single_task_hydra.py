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

import os
import time
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from moviepy.editor import ImageSequenceClip
import torch
import functools

import quest.utils.utils as utils
import quest.utils.libero_utils as lu
from quest.env_runner.libero_runner import LiberoSingleTaskRunner


def load_model_from_checkpoint(checkpoint_dir, config):
    """
    Load model from checkpoint

    Args:
        checkpoint_dir: Directory containing (multiple) model checkpoint files.
        config: Hydra configuration object.
    """
    print(f"Loading checkpoint from: {checkpoint_dir}")
    checkpoint_path = utils.get_checkpoint_with_selection(checkpoint_dir)

    # Load model state
    state_dict = utils.load_state(checkpoint_path)
    
    # Create model based on saved config or provided config
    if 'config' in state_dict:
        print('Auto-loading model based on saved parameters')
        policy_config_from_model = state_dict['config']['algo']['policy']        
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


@hydra.main(config_path="config", config_name="evaluate_single_task", version_base=None)
def main(config: DictConfig):
    # Set random seed
    torch.manual_seed(config.seed)
    
    output_dir = os.getcwd() # hydra automatically changes the working directory
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model_from_checkpoint(config.checkpoint_path, config)
    exit(0)  # Temporary exit to avoid running evaluation during testing
    
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
        save_video_fn(video_chw, env_name, idx, output_dir, config.rollout.fps)
    
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
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    if config.rollout.n_video > 0:
        print(f"Videos saved to: {os.path.join(output_dir, 'videos')}")


if __name__ == "__main__":
    main()
