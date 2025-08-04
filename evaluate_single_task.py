#!/usr/bin/env python3
"""
Single Task Evaluation Script for LiberoSingleTaskRunner

This script evaluates a trained checkpoint on a single specific task from a Libero benchmark.
Usage:
    python evaluate_single_task.py \
        --checkpoint_path /path/to/checkpoint \
        --benchmark_name LIBERO_10 \
        --task_id 0 \
        --rollouts_per_env 50 \
        --n_video 3

Example:
    python evaluate_single_task.py \
        --checkpoint_path experiments/libero/LIBERO_10/quest/stage_1/checkpoints/latest.pth \
        --benchmark_name LIBERO_10 \
        --task_id 2 \
        --rollouts_per_env 20 \
        --n_video 1
"""

import os
import argparse
import time
import json
from moviepy.editor import ImageSequenceClip
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import functools

import quest.utils.utils as utils
import quest.utils.libero_utils as lu
from quest.env_runner.libero_runner import LiberoSingleTaskRunner


def create_config_for_single_task(benchmark_name, task_id, args):
    """Create a configuration for single task evaluation"""
    
    # Basic task configuration
    config = DictConfig({
        'device': args.device,
        'seed': args.seed,
        'checkpoint_path': args.checkpoint_path,
        'task': {
            'suite_name': 'libero',
            'benchmark_name': benchmark_name,
            'task_id': task_id,
            'img_height': 128,
            'img_width': 128,
            'horizon': 500, # Some tasks may have different horizons
            'task_embedding_format': 'clip',
            'shape_meta': {
                'action_dim': 7,
                'observation': {
                    'rgb': {
                        'agentview_rgb': [3, 128, 128],
                        'eye_in_hand_rgb': [3, 128, 128]
                        },
                    'lowdim': {
                        'joint_states': 7,
                        'ee_pos': 3,
                        'gripper_states': 2
                        }
                },
                'task': {
                    'type': 'vector',
                    'dim': 512
                }
            },
            'obs_key_mapping': {
                'agentview_rgb': 'agentview_image',
                'eye_in_hand_rgb': 'robot0_eye_in_hand_image',
                'gripper_states': 'robot0_gripper_qpos',
                'joint_states': 'robot0_joint_pos',
                'ee_pos': 'robot0_eef_pos'
            }
        },
        'rollout': {
            'rollouts_per_env': args.rollouts_per_env,
            'num_parallel_envs': args.num_parallel_envs,
            'max_episode_length': 500,
            'n_video': args.n_video
        }
    })
    
    return config


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = utils.load_state(checkpoint_path)
    
    # Create model based on saved config or provided config
    if 'config' in state_dict:
        print('Auto-loading model based on saved parameters')
        model = instantiate(
            state_dict['config']['algo']['policy'], 
            shape_meta=config.task.shape_meta
        )
    else:
        raise ValueError("No saved config found in checkpoint. Cannot infer model architecture.")
    
    model.to(config.device)
    model.eval()
    model.load_state_dict(state_dict['model'])
    print(f"Model loaded successfully on {config.device}")
    
    return model


def create_env_runner(config):
    """Create LiberoSingleTaskRunner"""
    # Create environment factory (partial binds shape/meta and defers task_id, benchmark)
    env_factory = functools.partial(
        lu.LiberoWrapper,
        shape_meta=config.task.shape_meta,
        obs_key_mapping=config.task.obs_key_mapping,
        img_height=config.task.img_height,
        img_width=config.task.img_width,
        device=config.device,
    )
    
    # Create single task runner using the partial env_factory
    runner = LiberoSingleTaskRunner(
        env_factory=env_factory,
        benchmark_name=config.task.benchmark_name,
        task_id=config.task.task_id,
        rollouts_per_env=config.rollout.rollouts_per_env,
        num_parallel_envs=config.rollout.num_parallel_envs,
        max_episode_length=config.rollout.max_episode_length,
        frame_stack=1,  # Default frame stack
        fps=24,
        debug=False,
        task_embedding_format=config.task.task_embedding_format
    )
    
    return runner


def save_video_fn(video_chw, env_name, idx, save_dir):
    """Save video function"""
    video_dir = os.path.join(save_dir, 'videos', env_name)
    os.makedirs(video_dir, exist_ok=True)
    save_path = os.path.join(video_dir, f'{idx}.mp4')
    clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
    clip.write_videofile(save_path, fps=24, verbose=False, logger=None)
    print(f"Video saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate single task with LiberoSingleTaskRunner')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint file')
    parser.add_argument('--benchmark_name', type=str, required=True,
                       help='Libero benchmark name (e.g., LIBERO_10, LIBERO_OBJECT)')
    parser.add_argument('--task_id', type=int, required=True,
                       help='Task ID within the benchmark (0-indexed)')
    parser.add_argument('--rollouts_per_env', type=int, default=50,
                       help='Number of rollouts per environment')
    parser.add_argument('--num_parallel_envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--n_video', type=int, default=3,
                       help='Number of videos to save')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on')
    parser.add_argument('--seed', type=int, default=5000,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create configuration
    config = create_config_for_single_task(args.benchmark_name, args.task_id, args)
    
    # Create output directory
    if args.output_dir is None:
        if args.checkpoint_path:
            checkpoint_name = os.path.basename(args.checkpoint_path)
            checkpoint_name_without_ext = os.path.splitext(checkpoint_name)[0]
            parts = checkpoint_name_without_ext.split('_')
            epoch_tag = '_'.join(parts[2:])
            output_dir = f"./experiments/evaluate_single_task/{args.benchmark_name}_task{args.task_id}/{epoch_tag}"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"./experiments/evaluate_single_task/{args.benchmark_name}_task{args.task_id}_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint_path, config)
    
    # Create environment runner
    env_runner = create_env_runner(config)
    
    # Get task name for display
    benchmark = lu.get_benchmark(args.benchmark_name)()
    task_name = benchmark.get_task_names()[args.task_id]
    
    print(f"\n=== Single Task Evaluation ===")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Task ID: {args.task_id}")
    print(f"Task Name: {task_name}")
    print(f"Rollouts per env: {args.rollouts_per_env}")
    print(f"Number of videos: {args.n_video}")
    print(f"Device: {args.device}")
    print("=" * 40)
    
    # Define video save function
    def video_save_fn(video_chw, env_name, idx):
        save_video_fn(video_chw, env_name, idx, output_dir)
    
    # Run evaluation
    print("Running evaluation...")
    start_time = time.time()
    
    rollout_results = env_runner.run(
        model, 
        n_video=args.n_video,
        do_tqdm=True,
        save_video_fn=video_save_fn if args.n_video > 0 else None
    )
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Success Rate: {rollout_results['rollout']['overall_success_rate']:.3f}")
    print(f"Average Reward: {rollout_results['rollout']['overall_average_reward']:.3f}")
    print(f"Environments Solved: {rollout_results['rollout']['environments_solved']}")
    print(f"Task-specific Success Rate: {rollout_results['rollout_success_rate'][task_name]:.3f}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    
    # Save results
    results_data = {
        'args': vars(args),
        'config': OmegaConf.to_container(config),
        'results': rollout_results,
        'evaluation_time': evaluation_time,
        'task_name': task_name
    }
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    if args.n_video > 0:
        print(f"Videos saved to: {os.path.join(output_dir, 'videos')}")


if __name__ == "__main__":
    main()
