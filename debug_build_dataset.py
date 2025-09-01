import os
from quest.utils.libero_utils import build_dataset

from torch.utils.data import ConcatDataset

def main():
    """
    这是一个用于调试和测试 build_dataset 函数的脚本。
    您可以在下方的 `params_1` 和 `params_2` 字典中填充两组不同的参数，
    脚本将分别用于构建 `dataset_1` 和 `dataset_2`。
    """

    # --- 数据集1的参数 ---
    params_1 = {
        "data_prefix": "./data",
        "suite_name": "libero",
        "benchmark_name": "libero_object",
        "mode": "fewshot",
        "seq_len": 16,
        "frame_stack": 1,
        "shape_meta": {
            'action_dim': 7,
            'observation': {
                'rgb': {
                    'agentview_rgb': (3, 128, 128),
                    'eye_in_hand_rgb': (3, 128, 128)
                },
                'lowdim': {
                    'joint_states': 7,
                    'ee_pos': 3,
                    'gripper_states': 2
                },
                'task': {
                    'type': 'vector',
                    'dim': 512
                }
            }
        },
        "n_demos": 10,
        "task_ids": None,
        "obs_seq_len": 1,
        "load_obs": True,
        "task_embedding_format": "clip",
    }

    # --- 数据集2的参数 ---
    # 您可以根据需要修改这些参数以构建不同的数据集
    params_2 = {
        "data_prefix": "./data",
        "suite_name": "libero",
        "benchmark_name": "libero_spatial",  # 例如，使用不同的 benchmark
        "mode": "fewshot",  # 例如，使用 'train' 模式
        "seq_len": 16,  # 例如，不同的序列长度
        "frame_stack": 1,
        "shape_meta": {
            'action_dim': 7,
            'observation': {
                'rgb': {
                    'agentview_rgb': (3, 128, 128),
                    'eye_in_hand_rgb': (3, 128, 128)
                },
                'lowdim': {
                    'joint_states': 7,
                    'ee_pos': 3,
                    'gripper_states': 2
                },
                'task': {
                    'type': 'vector',
                    'dim': 512
                }
            }
        },
        "n_demos": 10,  # 'train' 模式下通常不使用 n_demos
        "task_ids": [1],
        "obs_seq_len": 1,
        "load_obs": True,
        "task_embedding_format": "clip",
    }

    # 检查参数是否有效
    for i, params in enumerate([params_1, params_2], 1):
        if not all(params.get(key) for key in ["data_prefix", "suite_name", "benchmark_name"]):
            print(f"请打开 `debug_build_dataset.py` 文件并为 `params_{i}` 填写 `data_prefix`, `suite_name`, 和 `benchmark_name` 变量。")
            return
        
        shape_meta = params.get("shape_meta", {})
        if not shape_meta.get('observation', {}).get('rgb') and not shape_meta.get('observation', {}).get('lowdim'):
            print(f"请在 `debug_build_dataset.py` 文件中为 `params_{i}` 配置 `shape_meta`。")
            return

    # --- 构建数据集 ---

    print("开始构建数据集1...")
    try:
        dataset_1 = build_dataset(**params_1)
        print("\n数据集1构建成功!")
        print(f"数据集1类型: {type(dataset_1)}")
        print(f"数据集1长度: {len(dataset_1)}")

    except Exception as e:
        print(f"\n构建数据集1时发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50 + "\n")

    print("开始构建数据集2...")
    try:
        dataset_2 = build_dataset(**params_2)
        print("\n数据集2构建成功!")
        print(f"数据集2类型: {type(dataset_2)}")
        print(f"数据集2长度: {len(dataset_2)}")

    except Exception as e:
        print(f"\n构建数据集2时发生错误: {e}")
        import traceback
        traceback.print_exc()

    dataset = ConcatDataset([dataset_1, dataset_2])

    # --- 验证拼接后的数据集 ---
    if 'dataset_1' in locals() and 'dataset_2' in locals():
        print("\n" + "="*50 + "\n")
        print("开始验证拼接后的数据集...")

        dataset = ConcatDataset([dataset_1, dataset_2])
        print(f"拼接后的数据集总长度: {len(dataset)}")
        
        # 检查总长度是否为两个数据集长度之和
        try:
            assert len(dataset) == len(dataset_1) + len(dataset_2)
            print("总长度验证成功!")
        except AssertionError:
            print(f"错误：拼接后长度 ({len(dataset)}) 与期望长度 ({len(dataset_1) + len(dataset_2)}) 不符！")


        def print_sample_structure(sample, sample_name):
            """递归打印样本结构。"""
            print(f"\n--- {sample_name} 结构 ---")
            if isinstance(sample, dict):
                for key, value in sample.items():
                    if isinstance(value, dict):
                        print(f"  - {key}:")
                        for sub_key, sub_value in value.items():
                            if hasattr(sub_value, 'shape'):
                                print(f"    - {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                            else:
                                print(f"    - {sub_key}: type={type(sub_value)}")
                    elif hasattr(value, 'shape'):
                        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  - {key}: type={type(value)}")
            else:
                print(f"样本类型: {type(sample)}")

        def compare_structures(sample1, sample2):
            """递归比较两个样本的结构。"""
            if type(sample1) != type(sample2):
                return False
            if isinstance(sample1, dict):
                if sample1.keys() != sample2.keys():
                    return False
                for key in sample1:
                    if not compare_structures(sample1[key], sample2[key]):
                        return False
            # 可以根据需要添加对 tensor 形状等的比较
            # if hasattr(sample1, 'shape') and hasattr(sample2, 'shape'):
            #     if sample1.shape != sample2.shape:
            #         return False
            return True

        # 比较 dataset_1[0] 和 dataset[0]
        if len(dataset_1) > 0:
            sample_1 = dataset_1[0]
            sample_concat_1 = dataset[0]
            print_sample_structure(sample_1, "原始数据集1的第一个样本 (dataset_1[0])")
            print_sample_structure(sample_concat_1, "拼接后数据集的第一个样本 (dataset[0])")
            
            print("\n--> 正在比较 dataset_1[0] 和 dataset[0] 的结构...")
            if compare_structures(sample_1, sample_concat_1):
                print("--> 结构一致性检查通过！")
            else:
                print("--> 警告：结构不一致！")
        else:
            print("\n数据集1为空，跳过第一个比较。")


        # 比较 dataset_2[0] 和 dataset[len(dataset_1)]
        if len(dataset_2) > 0:
            sample_2 = dataset_2[0]
            sample_concat_2 = dataset[len(dataset_1)]
            print_sample_structure(sample_2, f"原始数据集2的第一个样本 (dataset_2[0])")
            print_sample_structure(sample_concat_2, f"拼接后数据集的对应样本 (dataset[{len(dataset_1)}])")

            print(f"\n--> 正在比较 dataset_2[0] 和 dataset[{len(dataset_1)}] 的结构...")
            if compare_structures(sample_2, sample_concat_2):
                print("--> 结构一致性检查通过！")
            else:
                print("--> 警告：结构不一致！")
        else:
            print("\n数据集2为空，跳过第二个比较。")



if __name__ == "__main__":
    main()
