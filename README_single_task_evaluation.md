# Single Task Evaluation with LiberoSingleTaskRunner

这个目录包含了使用 `LiberoSingleTaskRunner` 来在单个任务上验证训练好的 checkpoint 的代码示例。

## 文件说明

### 1. `evaluate_single_task.py` - 完整的单任务评估脚本

这是一个功能完整的命令行工具，可以直接运行来评估单个任务。

**使用方法：**
```bash
python evaluate_single_task.py \
    --checkpoint_path experiments/libero/LIBERO_10/quest/stage_1/checkpoints/latest.pth \
    --benchmark_name LIBERO_10 \
    --task_id 0 \
    --rollouts_per_env 50 \
    --n_video 3
```

**参数说明：**
- `--checkpoint_path`: 训练好的模型 checkpoint 路径
- `--benchmark_name`: Libero benchmark 名称 (如 LIBERO_10, LIBERO_OBJECT)
- `--task_id`: 要评估的任务 ID (从0开始)
- `--rollouts_per_env`: 每个任务的评估回合数 (默认: 50)
- `--num_parallel_envs`: 并行环境数 (默认: 1)
- `--n_video`: 要保存的视频数量 (默认: 3)
- `--device`: 运行设备 (默认: cuda:0)
- `--output_dir`: 输出目录 (默认: 自动生成)

**功能特点：**
- 自动创建输出目录
- 保存评估结果到 JSON 文件
- 可选择保存评估视频
- 详细的进度显示和结果报告

### 2. `examples/single_task_evaluation_example.py` - 代码示例

这个文件包含了两个示例函数，展示如何在代码中直接使用 `LiberoSingleTaskRunner`。

**示例1: 简单单任务评估**
```python
def simple_single_task_evaluation():
    # 配置参数
    device = 'cuda:0'
    benchmark_name = 'LIBERO_10'
    task_id = 0
    checkpoint_path = 'path/to/your/checkpoint.pth'
    
    # 加载模型、创建环境、运行评估
    # ... (详见代码)
```

**示例2: 批量单任务评估**
```python
def batch_single_task_evaluation(checkpoint_path, benchmark_name, task_ids):
    # 对多个指定任务进行批量评估
    # 例如：评估任务 0, 2, 4
    task_ids = [0, 2, 4]
    results = batch_single_task_evaluation(checkpoint_path, 'LIBERO_10', task_ids)
```

### 3. `config/evaluate_single_task.yaml` - 配置文件模板

这是一个 Hydra 配置文件模板，可以与现有的 `evaluate.py` 脚本一起使用。

**使用方法：**
1. 复制配置文件：`cp config/evaluate_single_task.yaml config/my_single_task_eval.yaml`
2. 修改配置文件中的参数：
   - `task.benchmark_name`: 设置 benchmark 名称
   - `task.task_id`: 设置要评估的任务 ID
   - `checkpoint_path`: 设置 checkpoint 路径
3. 运行评估：`python evaluate.py --config-name=my_single_task_eval`

## 快速开始

### 方法1: 使用独立脚本 (推荐)

```bash
# 评估 LIBERO_10 benchmark 中的任务 0
python evaluate_single_task.py \
    --checkpoint_path experiments/libero/LIBERO_10/quest/stage_1/checkpoints/latest.pth \
    --benchmark_name LIBERO_10 \
    --task_id 0 \
    --rollouts_per_env 20 \
    --n_video 1
```

### 方法2: 使用配置文件

```bash
# 1. 复制并修改配置文件
cp config/evaluate_single_task.yaml config/my_eval.yaml
# 编辑 my_eval.yaml，设置 benchmark_name, task_id, checkpoint_path

# 2. 运行评估
python evaluate.py --config-name=my_eval
```

### 方法3: 在代码中直接使用

```python
from quest.env_runner.libero_runner import LiberoSingleTaskRunner

# 创建单任务运行器
runner = LiberoSingleTaskRunner(
    env_factory=env_factory,
    benchmark_name='LIBERO_10',
    task_id=0,
    rollouts_per_env=50,
    num_parallel_envs=1,
    max_episode_length=600
)

# 运行评估
results = runner.run(policy=model, n_video=3, do_tqdm=True)
```

## 任务 ID 参考

不同 benchmark 的任务数量：
- **LIBERO_10**: 10个任务 (task_id: 0-9)
- **LIBERO_OBJECT**: 10个任务 (task_id: 0-9)  
- **LIBERO_SPATIAL**: 10个任务 (task_id: 0-9)
- **LIBERO_GOAL**: 10个任务 (task_id: 0-9)
- **LIBERO_90**: 90个任务 (task_id: 0-89)

可以通过以下代码查看具体任务名称：
```python
import quest.utils.libero_utils as lu
benchmark = lu.get_benchmark('LIBERO_10')()
task_names = benchmark.get_task_names()
for i, name in enumerate(task_names):
    print(f"Task {i}: {name}")
```

## 输出结果

评估完成后，会输出以下信息：
- **Success Rate**: 成功率 (0-1之间)
- **Average Reward**: 平均奖励
- **Environments Solved**: 解决的环境数量
- **Evaluation Time**: 评估耗时

如果保存了视频，会在输出目录的 `videos/` 子目录中找到 MP4 文件。

## 注意事项

1. **Checkpoint 路径**: 确保 checkpoint 路径正确，文件存在
2. **设备配置**: 根据你的硬件配置选择合适的设备 (cuda:0 或 cpu)
3. **内存使用**: 如果遇到内存问题，可以减少 `rollouts_per_env` 或设置 `num_parallel_envs=1`
4. **任务 ID 范围**: 确保 task_id 在有效范围内 (0 到 benchmark 任务数-1)

## 故障排除

**问题1: "Checkpoint not found"**
- 检查 checkpoint 路径是否正确
- 确保文件确实存在

**问题2: "task_id out of range"**
- 检查指定的 task_id 是否在 benchmark 的有效范围内
- 使用上面的代码查看可用的任务

**问题3: "CUDA out of memory"**
- 减少 `rollouts_per_env`
- 设置 `num_parallel_envs=1`
- 或者使用 CPU: `--device cpu`

**问题4: 模型加载失败**
- 确保 checkpoint 与当前代码版本兼容
- 检查 shape_meta 配置是否正确
