#!/usr/bin/env bash
set -Eeuo pipefail

# This script is used to evaluate the DP model trained on mixed datasets
python evaluate_single_task_hydra.py --config-name=evaluate_single_task_custom -m \
    seed=6000,6001,6002 \
    exp_name=\'2000_0\' \
    checkpoint_dir=experiments/libero/LIBERO_OBJECT_GRID/act/debug/block_16/5000/run_002 \
    task.task_id=[0] \
    algo=act \
    eval_all_checkpoints=false \
    rollout.num_parallel_envs=50 \
    rollout.n_video=50 \
    rollout.rollouts_per_env=50
