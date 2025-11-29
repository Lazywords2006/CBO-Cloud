#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBCBO算法性能测试与对比
用于验证MBCBO相对于BCBO的性能提升
"""

import sys
import os
import numpy as np
import time
import json
from typing import Dict, List

# 添加算法路径
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_dir = os.path.join(current_dir, '..', 'algorithm')
bcbo_path = os.path.join(algorithm_dir, 'BCBO')
mbcbo_path = os.path.join(algorithm_dir, 'MBCBO')

for path in [bcbo_path, mbcbo_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from mbcbo_cloud_scheduler import MBCBO_CloudScheduler


def test_mbcbo_performance():
    """测试MBCBO算法性能"""

    print("="*70)
    print("MBCBO vs BCBO 性能对比测试".center(70))
    print("="*70)

    # 测试配置
    test_configs = [
        {'M': 100, 'N': 20, 'n': 50, 'iterations': 50, 'name': '小规模'},
        {'M': 500, 'N': 50, 'n': 100, 'iterations': 80, 'name': '中规模'},
        {'M': 1000, 'N': 100, 'n': 150, 'iterations': 100, 'name': '大规模'},
    ]

    results = []

    for config in test_configs:
        print(f"\n测试场景: {config['name']} (M={config['M']}, N={config['N']})")
        print("-"*50)

        # 设置随机种子保证公平对比
        np.random.seed(42)

        # 测试BCBO
        print("运行BCBO...")
        start_time = time.time()
        bcbo = BCBO_CloudScheduler(
            M=config['M'],
            N=config['N'],
            n=config['n'],
            iterations=config['iterations']
        )
        bcbo_population = bcbo.initialize_population()

        # 运行BCBO优化
        for i in range(config['iterations']):
            if i < config['iterations'] * 0.5:
                bcbo_population = bcbo.dynamic_search_phase(
                    bcbo_population, i, config['iterations']
                )
            else:
                bcbo_population = bcbo.static_search_phase(
                    bcbo_population, i
                )

        # 获取BCBO最优解
        bcbo_fitness = max([bcbo.comprehensive_fitness(ind) for ind in bcbo_population])
        bcbo_time = time.time() - start_time

        # 测试MBCBO
        print("运行MBCBO...")
        np.random.seed(42)  # 重置随机种子
        start_time = time.time()
        mbcbo = MBCBO_CloudScheduler(
            M=config['M'],
            N=config['N'],
            n=config['n'],
            iterations=config['iterations'],
            verbose=False
        )
        mbcbo_result = mbcbo.optimize()
        mbcbo_fitness = mbcbo_result['best_fitness']
        mbcbo_time = time.time() - start_time

        # 计算改进率
        improvement = ((mbcbo_fitness - bcbo_fitness) / bcbo_fitness) * 100

        # 保存结果
        result = {
            'scenario': config['name'],
            'M': config['M'],
            'N': config['N'],
            'bcbo_fitness': bcbo_fitness,
            'mbcbo_fitness': mbcbo_fitness,
            'improvement': improvement,
            'bcbo_time': bcbo_time,
            'mbcbo_time': mbcbo_time,
            'strategy_contributions': mbcbo_result['strategy_contributions']
        }
        results.append(result)

        # 打印结果
        print(f"\n结果:")
        print(f"  BCBO适应度: {bcbo_fitness:.4f}")
        print(f"  MBCBO适应度: {mbcbo_fitness:.4f}")
        print(f"  改进率: {improvement:+.2f}%")
        print(f"  运行时间: BCBO={bcbo_time:.2f}s, MBCBO={mbcbo_time:.2f}s")
        print(f"  策略贡献: {mbcbo_result['strategy_contributions']}")

        if improvement > 0:
            print(f"  [SUCCESS] MBCBO优于BCBO")
        else:
            print(f"  [FAIL] MBCBO未能超越BCBO")

    # 总结
    print("\n" + "="*70)
    print("测试总结".center(70))
    print("="*70)

    avg_improvement = np.mean([r['improvement'] for r in results])
    win_rate = sum(1 for r in results if r['improvement'] > 0) / len(results)

    print(f"\n整体统计:")
    print(f"  平均改进率: {avg_improvement:+.2f}%")
    print(f"  获胜率: {win_rate*100:.0f}%")

    print(f"\n各场景改进率:")
    for r in results:
        status = "SUCCESS" if r['improvement'] > 0 else "FAIL"
        print(f"  {r['scenario']}: {r['improvement']:+.2f}% [{status}]")

    # 保存详细结果
    with open('mbcbo_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存至: mbcbo_test_results.json")

    # 期刊论文建议
    if avg_improvement > 3:
        print("\n" + "="*70)
        print("恭喜！MBCBO算法表现优秀，适合发表期刊论文")
        print("="*70)
        print("\n发表建议:")
        print("1. 算法名称: MBCBO (Multi-Strategy Collaborative BCBO)")
        print("2. 创新点:")
        print("   - 多策略并行进化机制")
        print("   - 动态资源分配策略")
        print("   - 子种群信息交换机制")
        print("3. 实验结果: 平均性能提升{:.2f}%".format(avg_improvement))
        print("4. 目标期刊: IEEE TEC, Information Sciences, ASOC")
    else:
        print("\n注意: 当前性能提升不够显著，建议进一步优化参数")

    return results


def analyze_strategy_effectiveness():
    """分析各策略的有效性"""
    print("\n" + "="*70)
    print("策略有效性分析".center(70))
    print("="*70)

    # 运行一个详细的测试
    mbcbo = MBCBO_CloudScheduler(
        M=200, N=30, n=80, iterations=100, verbose=True
    )

    result = mbcbo.optimize()

    print("\n策略贡献度分析:")
    contributions = result['strategy_contributions']

    # 排序策略
    sorted_strategies = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    for i, (strategy, contrib) in enumerate(sorted_strategies, 1):
        bar_length = int(contrib / max(contributions.values()) * 30) if max(contributions.values()) > 0 else 0
        bar = '#' * bar_length
        print(f"  {i}. {strategy:10s}: {bar} {contrib:.4f}")

    print("\n分析结论:")
    best_strategy = sorted_strategies[0][0]
    print(f"  最有效策略: {best_strategy}")
    print(f"  建议: 可以增加{best_strategy}的初始资源分配比例")


if __name__ == "__main__":
    # 运行性能测试
    results = test_mbcbo_performance()

    # 分析策略有效性
    analyze_strategy_effectiveness()