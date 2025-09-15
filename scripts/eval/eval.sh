
#!/usr/bin/env bash
set -Eeuo pipefail

# This script is used to evaluate the DP model trained on mixed datasets
python evaluate_single_task_hydra.py --config-name=evaluate_single_task_custom -m \
    seed="range(6000,6003)" \
    checkpoint_dir=./experiments/libero/mixed_datasets/1000_1000/diffusion_policy/5000/run_000 \
    task.task_id=0 \
    algo=diffusion_policy \
    eval_all_checkpoints=false \
    rollout.num_parallel_envs=5 \
    rollout.n_video=50 \
    rollout.rollouts_per_env=50