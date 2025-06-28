
python evaluate.py \
    task=libero_object \
    checkpoint_task=libero_object_single \
    algo=diffusion_policy \
    exp_name=debug \
    variant_name=block_32 \
    stage=1 \
    training.use_tqdm=true \
    seed=5000 \
    rollout.n_video=10 \
    rollout.num_parallel_envs=5

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.