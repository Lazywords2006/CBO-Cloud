#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO与MBCBO算法快速对比测试
用于快速判断算法优劣
"""

import numpy as np
import json
import time
import sys
import os

# 添加算法路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
algorithm_path = os.path.join(parent_dir, 'algorithm')
sys.path.insert(0, algorithm_path)

from BCBO.bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from MBCBO.mbcbo_cloud_scheduler import MBCBO_CloudScheduler


def quick_comparison():
    """快速对比测试"""
    print("\n" + "="*60)
    print("BCBO vs MBCBO 快速对比测试".center(60))
    print("="*60)

    # 测试场景（缩小规模以加快速度）
    scenarios = [
        {'name': '小规模', 'M': 50, 'N': 10, 'n': 20, 'iterations': 20},
        {'name': '中规模', 'M': 100, 'N': 20, 'n': 30, 'iterations': 30},
        {'name': '大规模', 'M': 200, 'N': 40, 'n': 40, 'iterations': 40}
    ]

    results = []

    for scenario in scenarios:
        print(f"\n测试场景: {scenario['name']} (M={scenario['M']}, N={scenario['N']})")
        print("-" * 40)

        # 设置相同的随机种子
        np.random.seed(42)

        # 测试BCBO
        print("  运行BCBO...")
        start_time = time.time()
        bcbo = BCBO_CloudScheduler(
            M=scenario['M'],
            N=scenario['N'],
            n=scenario['n'],
            iterations=scenario['iterations']
        )

        bcbo_result = bcbo.run_complete_algorithm()
        bcbo_time = time.time() - start_time
        bcbo_fitness = bcbo_result['best_fitness']

        # 重置随机种子
        np.random.seed(42)

        # 测试MBCBO
        print("  运行MBCBO...")
        start_time = time.time()
        mbcbo = MBCBO_CloudScheduler(
            M=scenario['M'],
            N=scenario['N'],
            n=scenario['n'],
            iterations=scenario['iterations'],
            verbose=False
        )

        mbcbo_result = mbcbo.optimize()
        mbcbo_time = time.time() - start_time
        mbcbo_fitness = mbcbo_result['best_fitness']

        # 计算改进率
        improvement = ((mbcbo_fitness - bcbo_fitness) / abs(bcbo_fitness)) * 100

        result = {
            'scenario': scenario['name'],
            'M': scenario['M'],
            'N': scenario['N'],
            'bcbo_fitness': bcbo_fitness,
            'mbcbo_fitness': mbcbo_fitness,
            'improvement': improvement,
            'bcbo_time': bcbo_time,
            'mbcbo_time': mbcbo_time
        }
        results.append(result)

        # 显示结果
        print(f"\n  结果:")
        print(f"    BCBO适应度:  {bcbo_fitness:.4f} (用时: {bcbo_time:.2f}秒)")
        print(f"    MBCBO适应度: {mbcbo_fitness:.4f} (用时: {mbcbo_time:.2f}秒)")
        print(f"    改进率: {improvement:+.2f}%")

        if improvement > 0:
            print(f"    >>> MBCBO更优 (+{improvement:.2f}%)")
        elif improvement < -0.1:
            print(f"    >>> BCBO更优 ({improvement:.2f}%)")
        else:
            print(f"    >>> 性能相当")

    # 汇总分析
    print("\n" + "="*60)
    print("汇总分析".center(60))
    print("="*60)

    print("\n性能对比表:")
    print("-"*60)
    print(f"{'场景':<10} {'BCBO':<12} {'MBCBO':<12} {'改进率':<10} {'结论':<10}")
    print("-"*60)

    mbcbo_wins = 0
    bcbo_wins = 0
    ties = 0

    for r in results:
        if r['improvement'] > 0.1:
            conclusion = "MBCBO胜"
            mbcbo_wins += 1
        elif r['improvement'] < -0.1:
            conclusion = "BCBO胜"
            bcbo_wins += 1
        else:
            conclusion = "平局"
            ties += 1

        print(f"{r['scenario']:<10} {r['bcbo_fitness']:<12.4f} {r['mbcbo_fitness']:<12.4f} "
              f"{r['improvement']:+.2f}%{' '*5} {conclusion:<10}")

    # 计算平均改进率
    avg_improvement = np.mean([r['improvement'] for r in results])

    print("\n统计分析:")
    print("-"*40)
    print(f"  MBCBO获胜: {mbcbo_wins} 次")
    print(f"  BCBO获胜:  {bcbo_wins} 次")
    print(f"  平局:      {ties} 次")
    print(f"  平均改进率: {avg_improvement:+.2f}%")

    # 时间效率对比
    print("\n时间效率:")
    print("-"*40)
    for r in results:
        time_ratio = (r['mbcbo_time'] / r['bcbo_time'] - 1) * 100
        print(f"  {r['scenario']}: MBCBO用时 {time_ratio:+.1f}% {'更多' if time_ratio > 0 else '更少'}")

    # 最终结论
    print("\n" + "="*60)
    print("最终结论".center(60))
    print("="*60)

    if avg_improvement > 1.0:
        print("\n【结论】MBCBO显著优于BCBO")
        print(f"  - 平均性能提升: {avg_improvement:+.2f}%")
        print("  - 适合期刊发表: 是")
        print("  - 推荐使用场景: 所有场景")
    elif avg_improvement > 0:
        print("\n【结论】MBCBO略优于BCBO")
        print(f"  - 平均性能提升: {avg_improvement:+.2f}%")
        print("  - 适合期刊发表: 是（需要进一步优化）")
        print("  - 推荐使用场景: 大规模问题")
    elif avg_improvement > -1.0:
        print("\n【结论】两种算法性能相当")
        print(f"  - 平均性能差异: {avg_improvement:+.2f}%")
        print("  - 适合期刊发表: 是（强调理论创新）")
        print("  - 推荐使用场景: 根据具体需求选择")
    else:
        print("\n【结论】BCBO优于MBCBO")
        print(f"  - 平均性能差异: {avg_improvement:+.2f}%")
        print("  - 适合期刊发表: 需要改进MBCBO")
        print("  - 推荐使用场景: 使用BCBO")

    # 保存结果
    with open('quick_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[OK] 结果已保存到 quick_comparison_results.json")

    return results, avg_improvement


if __name__ == "__main__":
    results, avg_improvement = quick_comparison()

    # 判断是否MBCBO更优
    if avg_improvement > 0:
        print("\n" + "="*60)
        print("[OK] MBCBO算法更优，适合期刊发表！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[INFO] MBCBO需要进一步优化")
        print("="*60)