# This script is used to train the ACT model
python train_concat_datasets.py --config-name=train_grid.yaml \
    task=libero_object_grid \
    algo=act \
    exp_name=debug \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=true \
    training.n_epochs=100 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=16 \
    logging.project=mixed_datasets \
    seed=5000

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
