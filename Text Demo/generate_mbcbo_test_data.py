#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBCBO测试数据生成脚本
========================================
为MBCBO生成与BCBO相同场景的测试数据，用于公平对比两种算法的性能

输出位置：
- Text Demo/RAW_data/mbcbo_comprehensive_test_results.json
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
import time
from typing import Dict, List

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加算法路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
ALGORITHM_PATH = os.path.join(PARENT_DIR, 'algorithm')

if ALGORITHM_PATH not in sys.path:
    sys.path.insert(0, ALGORITHM_PATH)
    print(f"[OK] 添加算法路径: {ALGORITHM_PATH}")

from BCBO.bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from MBCBO.mbcbo_cloud_scheduler import MBCBO_CloudScheduler


# NumPy类型JSON序列化编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# 输出目录配置
OUTPUT_DIR = os.path.join(BASE_DIR, 'Text Demo', 'RAW_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_mbcbo_test_data():
    """生成MBCBO测试数据，与现有的BCBO数据对比"""

    print("\n" + "="*70)
    print("MBCBO测试数据生成".center(70))
    print("="*70)

    # 定义测试场景（与chart_set中的参数保持一致）
    test_scenarios = {
        'chart_1_迭代测试': {
            'M': 100,
            'N': 20,
            'n': 50,
            'iterations': [20, 40, 60, 80, 100],
            'description': '不同迭代次数下的性能'
        },
        'chart_2_任务规模': {
            'iterations': 80,
            'N': 20,
            'n': 100,
            'M_values': [100, 200, 300, 500, 1000],
            'description': '不同任务规模下的性能'
        },
        'quick_test': {
            'scenarios': [
                {'name': '小规模', 'M': 50, 'N': 10, 'n': 20, 'iterations': 30},
                {'name': '中规模', 'M': 100, 'N': 20, 'n': 40, 'iterations': 50},
                {'name': '大规模', 'M': 200, 'N': 40, 'n': 60, 'iterations': 80}
            ],
            'description': '快速测试不同规模'
        }
    }

    all_results = {}

    # 1. 迭代测试
    print("\n[1/3] 迭代测试...")
    print("-"*50)
    iter_results = []

    for iterations in test_scenarios['chart_1_迭代测试']['iterations']:
        print(f"  测试迭代数: {iterations}")

        # 设置随机种子
        np.random.seed(42)

        # BCBO测试
        bcbo = BCBO_CloudScheduler(
            M=test_scenarios['chart_1_迭代测试']['M'],
            N=test_scenarios['chart_1_迭代测试']['N'],
            n=test_scenarios['chart_1_迭代测试']['n'],
            iterations=iterations
        )
        bcbo_result = bcbo.run_complete_algorithm()

        # 重置种子
        np.random.seed(42)

        # MBCBO测试
        mbcbo = MBCBO_CloudScheduler(
            M=test_scenarios['chart_1_迭代测试']['M'],
            N=test_scenarios['chart_1_迭代测试']['N'],
            n=test_scenarios['chart_1_迭代测试']['n'],
            iterations=iterations,
            verbose=False
        )
        mbcbo_result = mbcbo.optimize()

        iter_results.append({
            'iterations': iterations,
            'bcbo_fitness': bcbo_result['best_fitness'],
            'mbcbo_fitness': mbcbo_result['best_fitness'],
            'improvement': ((mbcbo_result['best_fitness'] - bcbo_result['best_fitness'])
                          / abs(bcbo_result['best_fitness']) * 100),
            'mbcbo_strategies': mbcbo_result.get('strategy_contributions', {})
        })

        print(f"    BCBO: {bcbo_result['best_fitness']:.4f}")
        print(f"    MBCBO: {mbcbo_result['best_fitness']:.4f}")
        print(f"    改进: {iter_results[-1]['improvement']:+.2f}%")

    all_results['iteration_test'] = iter_results

    # 2. 任务规模测试
    print("\n[2/3] 任务规模测试...")
    print("-"*50)
    scale_results = []

    for M in test_scenarios['chart_2_任务规模']['M_values']:
        print(f"  测试任务数M: {M}")

        # 调整种群大小
        n = min(100, M // 2)

        # 设置随机种子
        np.random.seed(42)

        # BCBO测试
        bcbo = BCBO_CloudScheduler(
            M=M,
            N=test_scenarios['chart_2_任务规模']['N'],
            n=n,
            iterations=test_scenarios['chart_2_任务规模']['iterations']
        )
        start_time = time.time()
        bcbo_result = bcbo.run_complete_algorithm()
        bcbo_time = time.time() - start_time

        # 重置种子
        np.random.seed(42)

        # MBCBO测试
        mbcbo = MBCBO_CloudScheduler(
            M=M,
            N=test_scenarios['chart_2_任务规模']['N'],
            n=n,
            iterations=test_scenarios['chart_2_任务规模']['iterations'],
            verbose=False
        )
        start_time = time.time()
        mbcbo_result = mbcbo.optimize()
        mbcbo_time = time.time() - start_time

        scale_results.append({
            'M': M,
            'bcbo_fitness': bcbo_result['best_fitness'],
            'mbcbo_fitness': mbcbo_result['best_fitness'],
            'bcbo_time': bcbo_time,
            'mbcbo_time': mbcbo_time,
            'improvement': ((mbcbo_result['best_fitness'] - bcbo_result['best_fitness'])
                          / abs(bcbo_result['best_fitness']) * 100),
            'time_ratio': mbcbo_time / bcbo_time,
            'mbcbo_strategies': mbcbo_result.get('strategy_contributions', {})
        })

        print(f"    BCBO: {bcbo_result['best_fitness']:.4f} (时间: {bcbo_time:.2f}s)")
        print(f"    MBCBO: {mbcbo_result['best_fitness']:.4f} (时间: {mbcbo_time:.2f}s)")
        print(f"    改进: {scale_results[-1]['improvement']:+.2f}%")

    all_results['scale_test'] = scale_results

    # 3. 快速综合测试
    print("\n[3/3] 快速综合测试...")
    print("-"*50)
    quick_results = []

    for scenario in test_scenarios['quick_test']['scenarios']:
        print(f"  测试场景: {scenario['name']}")

        # 设置随机种子
        np.random.seed(42)

        # BCBO测试
        bcbo = BCBO_CloudScheduler(
            M=scenario['M'],
            N=scenario['N'],
            n=scenario['n'],
            iterations=scenario['iterations']
        )
        bcbo_result = bcbo.run_complete_algorithm()

        # 重置种子
        np.random.seed(42)

        # MBCBO测试
        mbcbo = MBCBO_CloudScheduler(
            M=scenario['M'],
            N=scenario['N'],
            n=scenario['n'],
            iterations=scenario['iterations'],
            verbose=False
        )
        mbcbo_result = mbcbo.optimize()

        quick_results.append({
            'scenario': scenario['name'],
            'M': scenario['M'],
            'N': scenario['N'],
            'bcbo_fitness': bcbo_result['best_fitness'],
            'mbcbo_fitness': mbcbo_result['best_fitness'],
            'improvement': ((mbcbo_result['best_fitness'] - bcbo_result['best_fitness'])
                          / abs(bcbo_result['best_fitness']) * 100),
            'mbcbo_strategies': mbcbo_result.get('strategy_contributions', {})
        })

        print(f"    BCBO: {bcbo_result['best_fitness']:.4f}")
        print(f"    MBCBO: {mbcbo_result['best_fitness']:.4f}")
        print(f"    改进: {quick_results[-1]['improvement']:+.2f}%")

    all_results['quick_test'] = quick_results

    # 保存结果到输出目录
    output_file = os.path.join(OUTPUT_DIR, 'mbcbo_comprehensive_test_results.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

        print(f"\n[OK] 结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n[ERROR] 保存失败: {e}")

    print("\n" + "="*70)
    print("测试完成！".center(70))
    print("="*70)

    return all_results


def analyze_results(results):
    """分析测试结果"""
    print("\n" + "="*70)
    print("MBCBO性能分析".center(70))
    print("="*70)

    # 分析迭代测试
    print("\n1. 迭代测试分析:")
    print("-"*50)
    iter_improvements = [r['improvement'] for r in results['iteration_test']]
    print(f"  平均改进: {np.mean(iter_improvements):+.2f}%")
    print(f"  最大改进: {max(iter_improvements):+.2f}%")
    print(f"  最小改进: {min(iter_improvements):+.2f}%")

    # 分析规模测试
    print("\n2. 任务规模测试分析:")
    print("-"*50)
    scale_improvements = [r['improvement'] for r in results['scale_test']]
    time_ratios = [r['time_ratio'] for r in results['scale_test']]
    print(f"  平均改进: {np.mean(scale_improvements):+.2f}%")
    print(f"  平均时间比: {np.mean(time_ratios):.2f} (MBCBO/BCBO)")

    # 分析快速测试
    print("\n3. 快速综合测试分析:")
    print("-"*50)
    quick_improvements = [r['improvement'] for r in results['quick_test']]
    print(f"  平均改进: {np.mean(quick_improvements):+.2f}%")

    # 总体结论
    all_improvements = iter_improvements + scale_improvements + quick_improvements
    avg_improvement = np.mean(all_improvements)

    print("\n" + "="*70)
    print("总体结论".center(70))
    print("="*70)
    print(f"\n总体平均改进率: {avg_improvement:+.2f}%")

    if avg_improvement > 2.0:
        print("\n【判断】MBCBO显著优于BCBO")
        print("  - 性能提升明显")
        print("  - 适合发表期刊论文")
        print("  - 建议采用MBCBO作为主要算法")
    elif avg_improvement > 0:
        print("\n【判断】MBCBO略优于BCBO")
        print("  - 有一定性能提升")
        print("  - 可以发表，需强调创新性")
        print("  - 建议进一步优化参数")
    else:
        print("\n【判断】MBCBO性能需要改进")
        print("  - 未能超越BCBO")
        print("  - 需要调整策略权重")
        print("  - 建议优化信息交换机制")

    # 时间效率分析
    if 'scale_test' in results:
        avg_time_ratio = np.mean(time_ratios)
        if avg_time_ratio < 0.5:
            print("\n【时间优势】MBCBO执行速度快于BCBO 50%以上")
        elif avg_time_ratio < 1.0:
            print("\n【时间优势】MBCBO执行速度快于BCBO")
        else:
            print("\n【时间劣势】MBCBO执行速度慢于BCBO")

    return avg_improvement > 0


if __name__ == "__main__":
    # 生成测试数据
    results = generate_mbcbo_test_data()

    # 分析结果
    is_better = analyze_results(results)

    if is_better:
        print("\n" + "="*70)
        print("✓ MBCBO算法更优！".center(70))
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠ MBCBO需要进一步优化".center(70))
        print("="*70)