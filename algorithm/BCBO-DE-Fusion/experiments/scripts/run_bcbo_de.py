#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行BCBO-DE融合算法实验
"""

import argparse
import sys
import os
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from core.bcbo_de_embedded import BCBO_DE_Embedded


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行BCBO-DE融合实验')
    parser.add_argument('--M', type=int, default=100, help='任务数量')
    parser.add_argument('--N', type=int, default=20, help='VM数量')
    parser.add_argument('--n', type=int, default=50, help='种群大小')
    parser.add_argument('--iterations', type=int, default=100, help='迭代次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--verbose', action='store_true', default=True, help='输出详细信息')
    return parser.parse_args()


def run_single_experiment(M, N, n, iterations, seed, verbose=True):
    """
    运行单次实验

    参数:
        M: 任务数
        N: VM数
        n: 种群大小
        iterations: 迭代次数
        seed: 随机种子
        verbose: 是否输出详细信息

    返回:
        result: 实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"开始实验: M={M}, N={N}, n={n}, iterations={iterations}, seed={seed}")
    print(f"{'='*60}\n")

    # 创建优化器
    optimizer = BCBO_DE_Embedded(
        M=M, N=N, n=n, iterations=iterations,
        random_seed=seed,
        verbose=verbose
    )

    # 运行优化
    result = optimizer.run_fusion_optimization()

    # 添加实验参数到结果
    result['experiment_params'] = {
        'M': M,
        'N': N,
        'n': n,
        'iterations': iterations,
        'seed': seed
    }

    return result


def save_result(result, output_path):
    """保存结果到JSON文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 转换不可序列化的对象
    result_serializable = {
        'best_fitness': result['best_fitness'],
        'best_solution': result['best_solution'],
        'experiment_params': result['experiment_params'],
        'summary': result['summary'],
        'diagnosis': result['diagnosis'],
        'history': {
            'iteration': result['history']['iteration'],
            'best_fitness': result['history']['best_fitness'],
            'avg_fitness': result['history']['avg_fitness'],
            'bcbo_ratio': result['history']['bcbo_ratio'],
            'phase': result['history']['phase']
        }
    }

    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    args = parse_arguments()

    # 运行实验
    result = run_single_experiment(
        M=args.M, N=args.N, n=args.n,
        iterations=args.iterations, seed=args.seed,
        verbose=args.verbose
    )

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        # 默认保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(__file__),
            '../../results/experiments/raw_data'
        )
        output_path = os.path.join(
            output_dir,
            f'bcbo_de_M{args.M}_N{args.N}_iter{args.iterations}_{timestamp}.json'
        )

    save_result(result, output_path)

    print(f"\n{'='*60}")
    print("实验完成!".center(60))
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
