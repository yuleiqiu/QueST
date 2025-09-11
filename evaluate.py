import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from hydra import compose
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import quest.utils.utils as utils
from pyinstrument import Profiler
from moviepy.editor import ImageSequenceClip
import json

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="config", config_name='evaluate', version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    OmegaConf.resolve(cfg)
    # create evaluation output directory
    save_dir, _ = utils.get_experiment_dir(cfg, evaluate=True)
    os.makedirs(save_dir, exist_ok=True)
    print('Saving to:', save_dir)

    # determine checkpoint path (allow loading from a different task)
    if cfg.checkpoint_path is None:
        if cfg.checkpoint_task:
            # compose a config for checkpoint lookup with a different task
            ckpt_cfg = compose(
                config_name='evaluate',
                overrides=[
                    f"task={cfg.checkpoint_task}",
                    f"algo={cfg.algo.name}",
                    f"exp_name={cfg.exp_name}",
                    f"variant_name={cfg.variant_name}",
                    f"stage={cfg.stage}",
                    f"seed={cfg.seed}"],
            )
            checkpoint_dir, _ = utils.get_experiment_dir(ckpt_cfg, evaluate=False, allow_overlap=True)
        else:
            checkpoint_dir, _ = utils.get_experiment_dir(cfg, evaluate=False, allow_overlap=True)
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_dir)
    else:
        # checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)
        checkpoint_path = utils.get_checkpoint_with_selection(cfg.checkpoint_path)
    state_dict = utils.load_state(checkpoint_path)
    
    if 'config' in state_dict:
        print('autoloading based on saved parameters')
        model = instantiate(state_dict['config']['algo']['policy'], 
                            shape_meta=cfg.task.shape_meta)
    else:
        model = instantiate(cfg.algo.policy,
                            shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict['model'])

    env_runner = instantiate(cfg.task.env_runner)
    
    print('Running evaluation...')

    def save_video_fn(video_chw, env_name, idx):
        video_dir = os.path.join(save_dir, 'videos', env_name)
        os.makedirs(video_dir, exist_ok=True)
        save_path = os.path.join(video_dir, f'{idx}.mp4')
        clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
        clip.write_videofile(save_path, fps=24, verbose=False, logger=None)

    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()
    rollout_results = env_runner.run(model, n_video=cfg.rollout.n_video, do_tqdm=train_cfg.use_tqdm, save_video_fn=save_video_fn)
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()
    print(
        f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
            | environments solved: {rollout_results['rollout']['environments_solved']}")

    with open(os.path.join(save_dir, 'data.json'), 'w') as f:
        json.dump(rollout_results, f)


if __name__ == "__main__":
    main()
