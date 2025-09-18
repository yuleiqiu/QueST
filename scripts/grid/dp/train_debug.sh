# This script is used to train the Diffusion Policy model
python train.py --config-name=train_grid.yaml \
    task=libero_object_grid \
    algo=diffusion_policy \
    exp_name=debug \
    variant_name=block_32 \
    training.use_tqdm=true \
    training.save_all_checkpoints=true \
    training.use_amp=true \
    training.n_epochs=100 \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=true \
    algo.skill_block_size=32 \
    seed=5000 \
    device=cuda:1

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
