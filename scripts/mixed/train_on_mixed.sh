# This script is used to train a model using mixed datasets.
python train_concat_datasets.py --config-name=train_mixed.yaml \
    task=libero_mixed_base \
    +task.dataset_grid.n_demos=0 \
    +task.dataset_random.n_demos=2000 \
    +task.dataset_random.task_ids='[0]' \
    algo=act \
    exp_name=\'0_2000\' \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=true \
    training.n_epochs=100 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=16 \
    logging.group=act \
    rollout.enabled=true \
    seed=10000

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
