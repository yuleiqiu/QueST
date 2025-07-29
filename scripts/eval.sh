
python evaluate.py \
    task=libero_spatial_single \
    checkpoint_path=./experiments/libero/LIBERO_SPATIAL/diffusion_policy/debug/block_32/5000/stage_1/ \
    algo=diffusion_policy \
    exp_name=debug \
    variant_name=block_32 \
    stage=1 \
    training.use_tqdm=true \
    rollout.n_video=10 \
    rollout.num_parallel_envs=5 \
    seed=5000

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.