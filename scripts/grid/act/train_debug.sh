# This script is used to train the ACT model
python train.py --config-name=train_grid.yaml \
    task=libero_object_grid \
    algo=act \
    exp_name=debug \
    variant_name=block_16 \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    seed=5000

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
