# This script is used to train a model using mixed datasets.
python train_concat_datasets.py --config-name=train_mixed.yaml \
    task=libero_mixed_base \
    +task.dataset_grid.n_demos=0 \
    +task.dataset_grid.task_ids='[0, 2, 4, 6, 8, 20, 22, 24, 26, 28, 40, 42, 44, 46, 48, 60, 62, 64, 66, 68, 80, 82, 84, 86, 88]' \
    +task.dataset_random.n_demos=100 \
    +task.dataset_random.task_ids='[0]' \
    algo=diffusion_policy \
    exp_name=\'0_100\' \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=true \
    training.n_epochs=100 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=16 \
    logging.group=dp \
    rollout.enabled=false \
    seed=10000 \
    device=cuda:0

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
