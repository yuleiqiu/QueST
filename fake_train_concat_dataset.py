import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from termcolor import colored

import torch
import quest.utils.utils as utils
from torch.utils.data import ConcatDataset


OmegaConf.register_new_resolver("eval", eval, replace=True)


def print_data_structure(data, indent=0):
    """
    Recursively prints the structure of a data object (e.g., a batch from a DataLoader).
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"{prefix}{k}: shape {v.shape}, dtype {v.dtype}")
            elif isinstance(v, dict):
                print(f"{prefix}{k}: dict with keys {list(v.keys())}")
                print_data_structure(v, indent + 1)
            else:
                print(f"{prefix}{k}: type {type(v)}")
    elif isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor: shape {data.shape}, dtype {data.dtype}")
    else:
        print(f"{prefix}type {type(data)}")


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    # print the entire config
    print(OmegaConf.to_yaml(cfg))
    # breakpoint()

    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # start training
    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

    experiment_dir, experiment_name = utils.get_experiment_dir_for_mixed_dataset(cfg)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Experiment name: {experiment_name}")
    breakpoint()
    
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is not None:
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        print(f'Loading from checkpoint {checkpoint_path}')
        state_dict = utils.load_state(checkpoint_path)
        loaded_state_dict = state_dict['model']
        
        # Below line allows loading state dicts with some mismatched parameters
        utils.soft_load_state_dict(model, loaded_state_dict)

        # resuming training since we are loading a checkpoint training the same stage
        if cfg.stage == state_dict['stage']:
            print('Loading from checkpoint')
            for optimizer, opt_state_dict in zip(optimizers, state_dict['optimizers']):
                optimizer.load_state_dict(opt_state_dict)
            for scheduler, sch_state_dict in zip(schedulers, state_dict['schedulers']):
                scheduler.load_state_dict(sch_state_dict)
            scaler.load_state_dict(state_dict['scaler'])
            start_epoch = state_dict['epoch']
            steps = state_dict['steps']
            wandb_id = state_dict['wandb_id']
    else:
        print(colored('\nStarting from scratch', 'yellow'))

    # Prepare dataset
    print("\nBuilding dataset 1...")
    try:
        dataset_grid = hydra.utils.instantiate(cfg.task.dataset_grid)
    except Exception as e:
        print(f"\nError occurred while building Dataset 1: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50 + "\n")

    print("Building dataset 2...")
    try:
        dataset_random = hydra.utils.instantiate(cfg.task.dataset_random)
    except Exception as e:
        print(f"\nError occurred while building Dataset 2: {e}")
        import traceback
        traceback.print_exc()

    # 拼接数据集
    dataset = ConcatDataset([dataset_grid, dataset_random])
    print(f"Concat dataset length: {len(dataset)}")

    # 验证拼接是否成功
    assert len(dataset) == len(dataset_grid) + len(dataset_random), "Concat dataset length is incorrect"
    print("Concat dataset validation successful!\n")

    model.preprocess_dataset(dataset, use_tqdm=train_cfg.use_tqdm)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)

    print(f"Length of dataset: {len(dataset)}")
    for data in train_dataloader:
        print("Structure of a batch:")
        print_data_structure(data)
        break
    # breakpoint()


    """
    Dataset batch structure example
    Assume:
    algo.skill_block_size = 16
    algo.frame_stack = 1
    {
        'actions': Tensor of shape (B, skill_block_size, action_dim), e.g. (128, 16, 7)
        'obs': {
            'agentview_rgb': Tensor of shape (B, frame_stack, C, H, W), e.g. (128, 1, 3, 128, 128)
            'eye_in_hand_rgb': Tensor of shape (B, frame_stack, C, H, W), e.g. (128, 1, 3, 128, 128)
            'joint_states': Tensor of shape (B, frame_stack, joint_dim), e.g. (128, 1, 7)
            'ee_pos': Tensor of shape (B, frame_stack, 3), e.g. (128, 1, 3)
            'gripper_states': Tensor of shape (B, frame_stack, 2), e.g. (128, 1, 2)
        },
        'task_emb': Tensor of shape (B, emb_dim), e.g. (128, 512)
        'task_id': Tensor of shape (B,), e.g. (128,)
    }
    """


    # Now I want to check the model's structure
    print("\nModel structure:")
    print(model)
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
    # breakpoint()

    if cfg.rollout.enabled:
        env_runner = instantiate(cfg.task.env_runner)
        # Comment out the following two lines to debug env runner before starting training
        # rollout_results = env_runner.run(model, n_video=0, do_tqdm=train_cfg.use_tqdm)
        # print(rollout_results)

if __name__ == "__main__":
    main()