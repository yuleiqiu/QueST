
python evaluate.py \
    task=libero_object_random \
    checkpoint_path=./experiments/libero/LIBERO_OBJECT_GRID/act/debug/block_16/5000/stage_1/ \
    algo=act \
    exp_name=debug \
    variant_name=block_16 \
    stage=1 \
    training.use_tqdm=true \
    rollout.n_video=10 \
    rollout.rollouts_per_env=100 \
    rollout.num_parallel_envs=5 \
    seed=5000

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.