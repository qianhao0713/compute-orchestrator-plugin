"""
python autotest.py # 跑所有模型的所有任务
python autotest.py --model llama3,qwen25,qwen3 --size 8b,70b --task pretrain,sft,cpt --dry-run
"""

import argparse
import os
import subprocess
from pathlib import Path

TASK_TYPES = ["pretrain", "sft", "cpt"]

MODEL_CONFIG = {
    'deepseek_v2': {
        'sizes': ['16b', '236b'],
        'prefix': 'xmegatron_deepseek_v2',
        'tasks': ['pretrain', 'sft'],
    },
    'llama3': {
        'sizes': ['8b', '70b'],
        'prefix': 'xmegatron_llama3',
        'tasks': ['pretrain', 'sft', 'cpt'],
    },
    'qwen2_5': {
        'sizes': ['7b', '72b'],
        'prefix': 'xmegatron_qwen2.5',
        'tasks': ['pretrain', 'sft', 'cpt'],
    },
    'qwen3': {
        'sizes': ['8b', '32b'],
        'prefix': 'xmegatron_qwen3',
        'tasks': ['pretrain', 'sft', 'cpt'],
    },
}


def scan_available_scripts():
    """扫描当前目录下的模型文件夹，检测可用的脚本和配置"""
    config = {}
    for model in os.listdir('.'):
        if model in MODEL_CONFIG:
            config[model] = {'sizes': set(), 'tasks': set()}
            for root, _, files in os.walk(model):
                for file in files:
                    if file.endswith('.sh'):
                        path = Path(root) / file
                        # 检测模型尺寸
                        for size in MODEL_CONFIG[model]['sizes']:
                            if f"_{size}_" in str(path) or f"_{size}." in str(path):
                                config[model]['sizes'].add(size)
                        # 检测任务类型
                        for task in TASK_TYPES:
                            if task in str(path):
                                config[model]['tasks'].add(task)
    return config


def execute_scripts(model, size, task):
    """执行指定模型、尺寸和任务的脚本"""
    prefix = f"{MODEL_CONFIG[model]['prefix']}_{size}" if size else MODEL_CONFIG[model]['prefix']
    scripts = []

    for root, _, files in os.walk(model):
        for file in files:
            path = Path(root) / file
            if (
                file.startswith(prefix)
                and file.endswith('.sh')
                and any(task in str(path).lower() for task in TASK_TYPES)
            ):
                if os.path.exists(str(path)):
                    scripts.append(str(path))

    if not scripts:
        print(f"No {task} scripts found for {model}-{size if size else 'all'}")
        return False

    print(f"Executing {len(scripts)} {task} scripts for {model}-{size if size else 'all'}:")
    for script in sorted(scripts):
        print(f"bash {script}")
        dir, script = script.split('/')
        try:
            subprocess.run(['bash', script], check=True, cwd=dir, env=os.environ)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
            return False
    return True


def main():
    """主函数，解析命令行参数并执行相应的脚本"""
    parser = argparse.ArgumentParser(description='全自动模型任务执行器')
    parser.add_argument('--model', help='指定模型名称')
    parser.add_argument('--size', help='指定模型尺寸')
    parser.add_argument('--task', help='指定任务类型')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要执行的脚本')

    args = parser.parse_args()
    available_configs = scan_available_scripts()

    # 自动填充未指定的参数
    models = args.model.split(',') if args.model else available_configs.keys()
    tasks = [args.task] if args.task else list(TASK_TYPES)

    for model in models:
        if model not in available_configs:
            continue

        sizes = args.size.split(',') if args.size else list(available_configs[model]['sizes'])
        if not sizes:
            sizes = [None]  # 表示不限制尺寸

        for size in sizes:
            for task in tasks:
                if task not in available_configs[model]['tasks']:
                    continue

                if args.dry_run:
                    print(f"\n[DRY RUN] Would execute {task} for {model}-{size if size else 'all'}")
                else:
                    execute_scripts(model, size, task)


if __name__ == '__main__':
    main()
