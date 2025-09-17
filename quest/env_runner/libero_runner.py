"""Libero environment runners for multi-task and single-task evaluation.

This module provides runners that vectorize LIBERO tasks and rollout a policy
with optional video capture. It adds type hints and light runtime validation to
make misuse easier to catch.
"""

from __future__ import annotations

import gc
import multiprocessing
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import wandb

import quest.utils.libero_utils as lu

class LiberoRunner:
    """Run a policy across all tasks in a LIBERO benchmark.

    Args:
        env_factory: Callable that constructs a single environment given
            (task_id, benchmark) and returns a gym-like env.
        benchmark_name: Name of the LIBERO benchmark to load.
        rollouts_per_env: Number of rollout episodes per task.
        num_parallel_envs: Size of the vectorized env (processes/workers).
        max_episode_length: Hard step cap per episode.
        frame_stack: Number of frames to stack in observations.
        fps: FPS to encode saved videos.
        task_embedding_format: Embedding backend for task text (e.g., 'clip').
    """

    def __init__(
        self,
        env_factory: Callable[[int, Any], Any],
        benchmark_name: str,
        rollouts_per_env: int,
        num_parallel_envs: int,
        max_episode_length: int,
        frame_stack: int = 1,
        fps: int = 10,
        task_embedding_format: str = 'clip',
    ) -> None:
        # Basic runtime validation (fast failures for common misuse)
        if not callable(env_factory):
            raise TypeError("env_factory must be callable: (task_id, benchmark) -> env")
        if not isinstance(benchmark_name, str) or not benchmark_name:
            raise TypeError("benchmark_name must be a non-empty string")
        if not isinstance(rollouts_per_env, int) or rollouts_per_env <= 0:
            raise ValueError("rollouts_per_env must be a positive int")
        if not isinstance(num_parallel_envs, int) or num_parallel_envs <= 0:
            raise ValueError("num_parallel_envs must be a positive int")
        if not isinstance(max_episode_length, int) or max_episode_length <= 0:
            raise ValueError("max_episode_length must be a positive int")
        if not isinstance(frame_stack, int) or frame_stack <= 0:
            raise ValueError("frame_stack must be a positive int")
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive int")
        if not isinstance(task_embedding_format, str) or not task_embedding_format:
            raise TypeError("task_embedding_format must be a non-empty string")

        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.benchmark = lu.get_benchmark(benchmark_name)()
        descriptions = [self.benchmark.get_task(i).language for i in range(self.benchmark.n_tasks)]
        task_embs = lu.get_task_embs(task_embedding_format, descriptions)
        self.benchmark.set_task_embs(task_embs)
        self.env_names = self.benchmark.get_task_names()

        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.frame_stack = frame_stack
        if num_parallel_envs > 1:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":  
                multiprocessing.set_start_method("spawn", force=True)
        self.max_episode_length = max_episode_length
        self.fps = fps
        
    def run(
        self,
        policy: Any,
        n_video: int = 0,
        do_tqdm: bool = False,
        save_video_fn: Optional[Callable[[np.ndarray, str, int], None]] = None,
    ) -> Dict[str, Any]:
        """Roll out a policy across all benchmark tasks.

        policy: Either a callable (obs, task_id, task_emb) -> action, or an
            object exposing get_action(obs, task_id, task_emb) and reset().
        n_video: Save the first N episodes per env to video.
        do_tqdm: Show progress bars.
        save_video_fn: Optional callback to persist videos. If not provided,
            videos are returned via Weights & Biases Video objects.
        Returns: Metrics dict with success rates, rewards, and optional videos.
        """
        if not (callable(policy) or hasattr(policy, 'get_action')):
            raise TypeError("policy must be callable or implement get_action()")
        if not isinstance(n_video, int) or n_video < 0:
            raise ValueError("n_video must be a non-negative int")
        if save_video_fn is not None and not callable(save_video_fn):
            raise TypeError("save_video_fn must be callable if provided")

        env_names = self.env_names
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}

        # Show outer progress bar only if there are multiple envs
        outer_disable = not do_tqdm or len(env_names) <= 1
        for env_name in tqdm(env_names, disable=outer_disable):
            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(env_name, policy, render=n_video > 0)
            # progress bar for individual rollouts per environment
            pb = tqdm(total=self.rollouts_per_env, disable=not do_tqdm,
                      desc=f"rollouts/{env_name}")
            for i, (success, total_reward, episode) in enumerate(rollouts):
                pb.update(1)
                any_success = any_success or success
                successes.append(success)
                env_succs.append(success)
                env_rews.append(total_reward)
                rewards.append(total_reward)

                # save only the first n_video episodes
                if i < n_video:
                    if save_video_fn is not None:
                        video_hwc = np.array(episode['render'])
                        video_chw = video_hwc.transpose((0, 3, 1, 2))
                        save_video_fn(video_chw, env_name, i)
                    else:
                        # if no save function provided, just store the video in a list
                        env_video.extend(episode['render'])
            # close rollout progress bar
            pb.close()
             
            per_env_success_rates[env_name] = np.mean(env_succs)
            per_env_rewards[env_name] = np.mean(env_rews)
            per_env_any_success.append(any_success)

            if len(env_video) > 0:
                video_hwc = np.array(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)

        output: Dict[str, Any] = {}
        output['rollout'] = {
            'overall_success_rate': np.mean(successes),
            'overall_average_reward': np.mean(rewards),
            'environments_solved': int(np.sum(per_env_any_success)),
        }
        output['rollout_success_rate'] = {}
        for env_name in env_names:
            output['rollout_success_rate'][env_name] = per_env_success_rates[env_name]
        if len(videos) > 0:
            output['rollout_videos'] = {}
            for env_name in videos:
                output['rollout_videos'][env_name] = videos[env_name]

        return output

    def run_policy_in_env(
        self,
        env_name: str,
        policy: Any,
        render: bool = False,
    ) -> Iterator[Tuple[bool, float, Dict[str, np.ndarray]]]:
        """Run multiple rollouts of a policy in a single task environment.

        Yields per-episode tuples: (success, total_reward, episode_dict)
        where episode_dict contains stacked observations, actions, and optionally
        rendered frames. total_reward is a float for that episode.
        """
        if env_name not in self.env_names:
            raise ValueError(f"Unknown env_name '{env_name}'. Must be one of {self.env_names}.")
        if not (callable(policy) or hasattr(policy, 'get_action')):
            raise TypeError("policy must be callable or implement get_action()")

        env_id = self.env_names.index(env_name)
        env_num = min(self.num_parallel_envs, self.rollouts_per_env)
        env_fn = lambda: lu.LiberoFrameStack(self.env_factory(env_id, self.benchmark), self.frame_stack)
        env = lu.LiberoVectorWrapper(env_fn, self.num_parallel_envs)

        all_init_states = self.benchmark.get_task_init_states(env_id)
        count = 0
        eval_loop_num = (self.rollouts_per_env+self.num_parallel_envs-1)//self.num_parallel_envs

        while count < eval_loop_num:
            indices = np.arange(count * env_num, (count + 1) * env_num) % all_init_states.shape[0]
            init_states_ = all_init_states[indices]
            success, total_reward, episode = self.run_episode(env, 
                                                              env_name, 
                                                              policy,
                                                              init_states_,
                                                              env_num,
                                                              render)
            count += 1
            for k in range(env_num):
                episode_k = {key: value[:,k] for key, value in episode.items()}
                yield success[k], total_reward[k], episode_k
        env._env.close()
        gc.collect()
        del env
    
    def run_episode(
        self,
        env: Any,
        env_name: str,
        policy: Any,
        init_states_: np.ndarray,
        env_num: int,
        render: bool = False,
    ) -> Tuple[List[bool], np.ndarray, Dict[str, np.ndarray]]:
        """Run a single vectorized episode and collect trajectory data."""
        obs, info = env.reset(init_states=init_states_)

        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs, task_id, task_emb: policy_object.get_action(obs, task_id, task_emb)
        
        # Track success flags per vectorized env and cumulative rewards
        success: List[bool] = [False] * env_num
        total_reward: np.ndarray = np.zeros(env_num, dtype=float)

        episode = {key: [value[:,-1]] for key, value in obs.items()}
        episode['actions'] = []
        if render:
            episode['render'] = [env.render()]

        task_id = self.env_names.index(env_name)
        task_emb = self.benchmark.get_task_emb(task_id).repeat(env_num, 1)
        steps = 0
        while steps < self.max_episode_length:
            action = policy(obs, task_id, task_emb)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            """
            Note on "executing action in terminated episode" error:
            This error can occur because `robosuite`'s `MujocoEnv` has an internal `horizon` (default 1000) that acts as a step counter.
            This counter can get out of sync with our external loop counter, causing the environment to terminate prematurely without the wrapper environment being aware.
            Specifically, the internal `done` signal is overridden (see BDDLBaseDomain.step). To prevent this, `task.horizon` should be less than the environment's internal horizon.
            We have modified the `horizon` in `env_wrapper.ControlEnv` to 2000, because our `max_episode_length` is intended to be twice the `task.horizon` of 500.
            """
            next_obs, reward, terminated, truncated, info = env.step(action)
            # reward is expected to be shape (env_num,); accumulate elementwise
            total_reward = total_reward + reward
            obs = next_obs
            for key, value in obs.items():
                episode[key].append(value[:,-1])
            episode['actions'].append(action)
            if render:
                episode['render'].append(env.render())
        
            for k in range(env_num):
                success[k] = success[k] or terminated[k]
            
            if all(success):
                break
            steps += 1

        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode
    
class LiberoSelectedTaskRunner(LiberoRunner):
    """
    Allows evaluating on a specific task by task_id from a benchmark.
    """
    def __init__(
        self,
        env_factory: Callable[[int, Any], Any],
        benchmark_name: str,
        task_ids: List[int],  # specific task IDs to rollout
        rollouts_per_env: int,
        num_parallel_envs: int,
        max_episode_length: int,
        frame_stack: int = 1,
        fps: int = 24,
        task_embedding_format: str = 'clip',
    ) -> None:
        if not task_ids:
            raise ValueError("task_ids list should not be empty")
        if any(not isinstance(tid, int) or tid < 0 for tid in task_ids):
            raise TypeError("task_ids must be a list of non-negative integers")
        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.task_ids = task_ids

        # Initialize benchmark
        self.benchmark = lu.get_benchmark(benchmark_name)()
        
        # Get all task descriptions for embedding computation
        descriptions = [self.benchmark.get_task(i).language for i in range(self.benchmark.n_tasks)]
        task_embs = lu.get_task_embs(task_embedding_format, descriptions)
        self.benchmark.set_task_embs(task_embs)

        # Set env_names to only the target tasks
        all_task_names = self.benchmark.get_task_names()
        for task_id in task_ids:
            if task_id >= len(all_task_names):
                raise ValueError(f"task_id {task_id} is out of range. Benchmark has {len(all_task_names)} tasks")

        # Evaluate the target tasks only
        self.env_names = [all_task_names[task_id] for task_id in self.task_ids]

        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.frame_stack = frame_stack
        if num_parallel_envs > 1:
            if multiprocessing.get_start_method(allow_none=True) != "spawn":  
                multiprocessing.set_start_method("spawn", force=True)
        self.max_episode_length = max_episode_length
        self.fps = fps

        print(f"[LiberoSingleTaskRunner] Initialize following tasks in benchmark {benchmark_name}:")
        print(f"[LiberoSingleTaskRunner] Task_ids: {self.task_ids}")