#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-DE v3.0 改进方案测试脚本

测试四维自适应融合策略:
1. 迭代自适应融合强度（余弦衰减）
2. 规模自适应参数调整
3. 收敛状态监控与自适应
4. 增强精英保护机制

对比测试:
- 原版BCBO
- 原版BCBO-DE v2.0
- 改进版BCBO-DE v3.0
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入算法
from core.bcbo_de_embedded import BCBO_DE_Embedded

# 导入BCBO
algorithm_dir = os.path.dirname(current_dir)
bcbo_path = os.path.join(algorithm_dir, 'BCBO')
if bcbo_path not in sys.path:
    sys.path.insert(0, bcbo_path)
from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler


def test_single_scenario(M, N, n, iterations, algorithm='BCBO-DE-v3', random_seed=42):
    """
    测试单个场景

    参数:
        M: 任务数
        N: 虚拟机数
        n: 种群大小
        iterations: 迭代次数
        algorithm: 'BCBO' | 'BCBO-DE-v2' | 'BCBO-DE-v3'
        random_seed: 随机种子

    返回:
        result: 包含性能指标的字典
    """
    print(f"\n{'='*80}")
    print(f"测试配置: M={M}, N={N}, n={n}, iterations={iterations}")
    print(f"算法: {algorithm}, 随机种子: {random_seed}")
    print(f"{'='*80}")

    if algorithm == 'BCBO':
        # 原版BCBO
        optimizer = BCBO_CloudScheduler(M=M, N=N, n=n, iterations=iterations)
        result = optimizer.run_optimization()

    elif algorithm == 'BCBO-DE-v3':
        # 改进版BCBO-DE v3.0（使用新的自适应策略）
        optimizer = BCBO_DE_Embedded(
            M=M, N=N, n=n, iterations=iterations,
            random_seed=random_seed,
            verbose=True,
            print_interval=max(1, iterations // 10)
        )
        result = optimizer.run_fusion_optimization()

    else:
        raise ValueError(f"未知算法: {algorithm}")

    return {
        'algorithm': algorithm,
        'M': M,
        'N': N,
        'n': n,
        'iterations': iterations,
        'best_fitness': result['best_fitness'],
        'history': result.get('history', []),
        'summary': result.get('summary', {}),
        'diagnosis': result.get('diagnosis', '')
    }


def run_comparison_tests():
    """运行对比测试"""
    print("\n" + "="*80)
    print("BCBO-DE v3.0 改进方案对比测试".center(80))
    print("="*80)

    # 测试场景（对应性能分析报告中的问题场景）
    test_scenarios = [
        # Chart Set 1: 小规模，变化迭代次数（问题1：后期收敛不足）
        {'name': 'Chart_Set_1', 'M': 100, 'N': 20, 'n': 50, 'iterations': 100},

        # Chart Set 2: 中等规模（问题2：中等规模性能差）
        {'name': 'Chart_Set_2_M200', 'M': 200, 'N': 20, 'n': 100, 'iterations': 80},
        {'name': 'Chart_Set_2_M500', 'M': 500, 'N': 20, 'n': 100, 'iterations': 80},
        {'name': 'Chart_Set_2_M1000', 'M': 1000, 'N': 20, 'n': 100, 'iterations': 80},

        # Chart Set 3: 大规模，变化迭代次数
        {'name': 'Chart_Set_3', 'M': 1000, 'N': 20, 'n': 150, 'iterations': 100},
    ]

    # 对比算法
    algorithms = ['BCBO', 'BCBO-DE-v3']

    # 存储结果
    all_results = []

    # 运行测试
    for scenario in test_scenarios:
        print(f"\n{'#'*80}")
        print(f"# 场景: {scenario['name']}")
        print(f"{'#'*80}")

        scenario_results = {
            'scenario': scenario['name'],
            'config': scenario,
            'algorithms': {}
        }

        for algo in algorithms:
            try:
                result = test_single_scenario(
                    M=scenario['M'],
                    N=scenario['N'],
                    n=scenario['n'],
                    iterations=scenario['iterations'],
                    algorithm=algo,
                    random_seed=42
                )

                scenario_results['algorithms'][algo] = result

                print(f"\n{algo} 结果:")
                print(f"  最优适应度: {result['best_fitness']:.6f}")
                print(f"  诊断: {result.get('diagnosis', 'N/A')}")

            except Exception as e:
                print(f"\n{algo} 运行失败: {e}")
                import traceback
                traceback.print_exc()

        # 对比分析
        if 'BCBO' in scenario_results['algorithms'] and 'BCBO-DE-v3' in scenario_results['algorithms']:
            bcbo_fitness = scenario_results['algorithms']['BCBO']['best_fitness']
            bcbo_de_fitness = scenario_results['algorithms']['BCBO-DE-v3']['best_fitness']

            improvement = ((bcbo_de_fitness - bcbo_fitness) / bcbo_fitness) * 100

            print(f"\n{'='*80}")
            print(f"场景 {scenario['name']} 对比结果:")
            print(f"  BCBO:        {bcbo_fitness:.6f}")
            print(f"  BCBO-DE v3:  {bcbo_de_fitness:.6f}")
            print(f"  改进幅度:    {improvement:+.2f}%")

            if improvement > 0:
                print(f"  ✓ BCBO-DE v3 胜出")
            elif improvement < -0.5:
                print(f"  ✗ BCBO-DE v3 落后")
            else:
                print(f"  ≈ 性能接近")
            print(f"{'='*80}")

            scenario_results['improvement'] = improvement

        all_results.append(scenario_results)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(current_dir, f'test_results_v3_{timestamp}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*80}")
    print(f"测试结果已保存至: {output_file}")
    print(f"{'='*80}")

    # 生成汇总报告
    generate_summary_report(all_results)

    return all_results


def generate_summary_report(all_results):
    """生成汇总报告"""
    print("\n" + "="*80)
    print("汇总报告".center(80))
    print("="*80)

    improvements = [r['improvement'] for r in all_results if 'improvement' in r]

    if improvements:
        avg_improvement = np.mean(improvements)
        win_count = sum(1 for imp in improvements if imp > 0)
        total_count = len(improvements)
        win_rate = (win_count / total_count) * 100

        print(f"\n整体性能指标:")
        print(f"  测试场景数:     {total_count}")
        print(f"  BCBO-DE v3 胜出: {win_count}/{total_count} ({win_rate:.0f}%)")
        print(f"  平均改进幅度:   {avg_improvement:+.2f}%")

        print(f"\n各场景详情:")
        for result in all_results:
            if 'improvement' in result:
                status = "✓" if result['improvement'] > 0 else "✗" if result['improvement'] < -0.5 else "≈"
                print(f"  {status} {result['scenario']:20s}: {result['improvement']:+6.2f}%")

        print(f"\n{'='*80}")

        # 评估改进效果
        if win_rate >= 60 and avg_improvement > 0:
            print("✅ 改进方案有效！BCBO-DE v3 显著优于 BCBO")
        elif win_rate >= 40:
            print("⚠️ 改进方案部分有效，但仍需优化")
        else:
            print("❌ 改进方案效果不佳，需要重新设计")

    else:
        print("没有足够的对比数据生成报告")


def test_adaptive_strategies():
    """测试自适应策略的效果"""
    print("\n" + "="*80)
    print("自适应策略单元测试".center(80))
    print("="*80)

    from utils.adaptive_strategies import (
        get_adaptive_fusion_intensity,
        get_scale_adaptive_params,
        ConvergenceMonitor
    )

    # 测试1: 迭代自适应融合强度
    print("\n测试1: 迭代自适应融合强度（余弦衰减）")
    print("-" * 80)
    total_iters = 100
    phase = 'attack_static'

    print(f"阶段: {phase}, 总迭代: {total_iters}")
    for iter in [0, 25, 50, 75, 100]:
        intensity = get_adaptive_fusion_intensity(phase, iter, total_iters)
        print(f"  迭代 {iter:3d}: 融合强度 = {intensity:.4f}")

    # 测试2: 规模自适应参数
    print("\n测试2: 规模自适应参数")
    print("-" * 80)
    for M in [100, 300, 500, 1000]:
        params = get_scale_adaptive_params(M, 20)
        print(f"  M={M:4d}: elite_ratio={params['elite_ratio']:.2%}, "
              f"intensity_scale={params['intensity_scale']:.2f}")

    # 测试3: 收敛监控器
    print("\n测试3: 收敛监控器")
    print("-" * 80)
    monitor = ConvergenceMonitor(patience=5)

    fitness_sequence = [100, 105, 110, 110, 110, 110, 110, 110, 115]
    for i, fitness in enumerate(fitness_sequence):
        monitor.update(fitness)
        adj = monitor.get_adaptive_adjustment()
        print(f"  迭代 {i}: fitness={fitness}, 停滞={monitor.stagnation_count}, "
              f"动作={adj['action']}, 调整={adj['intensity_adjust']:+.2f}")

    print("\n" + "="*80)
    print("单元测试完成")
    print("="*80)


if __name__ == '__main__':
    import argparse

    print("\nBCBO-DE v3.0 改进方案测试程序")
    print("=" * 80)

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='BCBO-DE v3.0 测试程序')
    parser.add_argument('--mode', type=str, default='3', choices=['1', '2', '3'],
                        help='测试模式: 1=对比测试, 2=单元测试, 3=全部测试')
    args = parser.parse_args()

    choice = args.mode

    print(f"\n测试模式: {choice}")
    if choice == "1":
        print("运行对比测试（BCBO vs BCBO-DE v3）")
        run_comparison_tests()
    elif choice == "2":
        print("运行自适应策略单元测试")
        test_adaptive_strategies()
    elif choice == "3":
        print("运行全部测试")
        test_adaptive_strategies()
        print("\n\n")
        run_comparison_tests()
    else:
        print("无效选择")

    print("\n测试程序结束")
