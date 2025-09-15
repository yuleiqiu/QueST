
python evaluate.py \
    task=libero_object_random \
    checkpoint_path=./experiments/libero/mixed_datasets/500_1500/act/5000/run_000 \
    algo=act \
    exp_name=\'500_1500\' \
    training.use_tqdm=true \
    rollout.n_video=10 \
    rollout.num_parallel_envs=5 \
    seed=6000

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.