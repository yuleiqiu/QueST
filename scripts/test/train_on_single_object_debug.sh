
# This script is used to test compatibility of new dataset

python test_compatibility.py --config-name=train_prior.yaml \
    task=libero_object_single \
    algo=act \
    exp_name=final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    rollout.num_parallel_envs=5 \
    rollout.rollouts_per_env=5 \
    seed=10000

# Note2: change rollout.num_parallel_envs to 1 if libero vectorized env is not working as expected.
