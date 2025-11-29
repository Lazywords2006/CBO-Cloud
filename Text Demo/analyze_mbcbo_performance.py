#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBCBO性能分析工具
分析Text Demo/RAW_data中的数据，对比BCBO、BCBO-DE和MBCBO的性能
"""

import json
import os
import numpy as np
from typing import Dict, List
import sys

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_chart_data(chart_set_name: str, data_dir: str) -> Dict:
    """加载指定图表集的数据"""
    file_path = os.path.join(data_dir, f'{chart_set_name}_bcbo_comparison.json')

    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_improvement(baseline_value: float, compare_value: float) -> float:
    """计算改进率（百分比）"""
    if baseline_value == 0:
        return 0.0
    return ((baseline_value - compare_value) / abs(baseline_value)) * 100


def analyze_algorithm_performance(data: Dict, algorithm_name: str) -> Dict:
    """分析单个算法的性能统计"""
    if 'algorithms' not in data or algorithm_name not in data['algorithms']:
        return None

    results = data['algorithms'][algorithm_name].get('results', [])
    if not results:
        return None

    # 提取所有指标
    total_costs = [r.get('total_cost', 0) for r in results]
    execution_times = [r.get('execution_time', 0) for r in results]
    load_balances = [r.get('load_balance', 0) for r in results]
    price_efficiencies = [r.get('price_efficiency', 0) for r in results]

    return {
        'total_cost': {
            'mean': np.mean(total_costs),
            'std': np.std(total_costs),
            'min': np.min(total_costs),
            'max': np.max(total_costs),
            'final': total_costs[-1] if total_costs else 0
        },
        'execution_time': {
            'mean': np.mean(execution_times),
            'std': np.std(execution_times),
            'min': np.min(execution_times),
            'max': np.max(execution_times),
            'final': execution_times[-1] if execution_times else 0
        },
        'load_balance': {
            'mean': np.mean(load_balances),
            'std': np.std(load_balances),
            'min': np.min(load_balances),
            'max': np.max(load_balances),
            'final': load_balances[-1] if load_balances else 0
        },
        'price_efficiency': {
            'mean': np.mean(price_efficiencies),
            'std': np.std(price_efficiencies),
            'min': np.min(price_efficiencies),
            'max': np.max(price_efficiencies),
            'final': price_efficiencies[-1] if price_efficiencies else 0
        },
        'data_points': len(results)
    }


def compare_algorithms(bcbo_stats: Dict, mbcbo_stats: Dict, bcbode_stats: Dict = None) -> Dict:
    """对比算法性能"""
    comparison = {}

    # MBCBO vs BCBO
    if bcbo_stats and mbcbo_stats:
        comparison['mbcbo_vs_bcbo'] = {
            'total_cost_improvement': calculate_improvement(
                bcbo_stats['total_cost']['final'],
                mbcbo_stats['total_cost']['final']
            ),
            'execution_time_improvement': calculate_improvement(
                bcbo_stats['execution_time']['final'],
                mbcbo_stats['execution_time']['final']
            ),
            'load_balance_improvement': (
                (mbcbo_stats['load_balance']['final'] - bcbo_stats['load_balance']['final']) /
                (bcbo_stats['load_balance']['final'] + 1e-6) * 100
            ),
            'price_efficiency_improvement': (
                (mbcbo_stats['price_efficiency']['final'] - bcbo_stats['price_efficiency']['final']) /
                (bcbo_stats['price_efficiency']['final'] + 1e-6) * 100
            )
        }

    # MBCBO vs BCBO-DE
    if bcbode_stats and mbcbo_stats:
        comparison['mbcbo_vs_bcbode'] = {
            'total_cost_improvement': calculate_improvement(
                bcbode_stats['total_cost']['final'],
                mbcbo_stats['total_cost']['final']
            ),
            'execution_time_improvement': calculate_improvement(
                bcbode_stats['execution_time']['final'],
                mbcbo_stats['execution_time']['final']
            ),
            'load_balance_improvement': (
                (mbcbo_stats['load_balance']['final'] - bcbode_stats['load_balance']['final']) /
                (bcbode_stats['load_balance']['final'] + 1e-6) * 100
            ),
            'price_efficiency_improvement': (
                (mbcbo_stats['price_efficiency']['final'] - bcbode_stats['price_efficiency']['final']) /
                (bcbode_stats['price_efficiency']['final'] + 1e-6) * 100
            )
        }

    # BCBO-DE vs BCBO
    if bcbo_stats and bcbode_stats:
        comparison['bcbode_vs_bcbo'] = {
            'total_cost_improvement': calculate_improvement(
                bcbo_stats['total_cost']['final'],
                bcbode_stats['total_cost']['final']
            ),
            'execution_time_improvement': calculate_improvement(
                bcbo_stats['execution_time']['final'],
                bcbode_stats['execution_time']['final']
            ),
            'load_balance_improvement': (
                (bcbode_stats['load_balance']['final'] - bcbo_stats['load_balance']['final']) /
                (bcbo_stats['load_balance']['final'] + 1e-6) * 100
            ),
            'price_efficiency_improvement': (
                (bcbode_stats['price_efficiency']['final'] - bcbo_stats['price_efficiency']['final']) /
                (bcbo_stats['price_efficiency']['final'] + 1e-6) * 100
            )
        }

    return comparison


def print_algorithm_stats(algo_name: str, stats: Dict):
    """打印算法统计信息"""
    print(f"\n{'='*60}")
    print(f"{algo_name} 性能统计")
    print(f"{'='*60}")
    print(f"数据点数: {stats['data_points']}")
    print(f"\n总成本 (Total Cost):")
    print(f"  平均值: {stats['total_cost']['mean']:.2f}")
    print(f"  标准差: {stats['total_cost']['std']:.2f}")
    print(f"  最终值: {stats['total_cost']['final']:.2f}")
    print(f"\n执行时间 (Execution Time):")
    print(f"  平均值: {stats['execution_time']['mean']:.2f}")
    print(f"  标准差: {stats['execution_time']['std']:.2f}")
    print(f"  最终值: {stats['execution_time']['final']:.2f}")
    print(f"\n负载均衡 (Load Balance):")
    print(f"  平均值: {stats['load_balance']['mean']:.4f}")
    print(f"  标准差: {stats['load_balance']['std']:.4f}")
    print(f"  最终值: {stats['load_balance']['final']:.4f}")
    print(f"\n价格效率 (Price Efficiency):")
    print(f"  平均值: {stats['price_efficiency']['mean']:.6f}")
    print(f"  标准差: {stats['price_efficiency']['std']:.6f}")
    print(f"  最终值: {stats['price_efficiency']['final']:.6f}")


def print_comparison(comparison: Dict):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print("算法对比分析")
    print(f"{'='*60}")

    if 'mbcbo_vs_bcbo' in comparison:
        print(f"\n【MBCBO vs BCBO】")
        comp = comparison['mbcbo_vs_bcbo']
        print(f"  总成本改进: {comp['total_cost_improvement']:+.2f}%")
        print(f"  执行时间改进: {comp['execution_time_improvement']:+.2f}%")
        print(f"  负载均衡改进: {comp['load_balance_improvement']:+.2f}%")
        print(f"  价格效率改进: {comp['price_efficiency_improvement']:+.2f}%")

        # 判断MBCBO是否更优
        cost_better = comp['total_cost_improvement'] > 0
        time_better = comp['execution_time_improvement'] > 0
        balance_better = comp['load_balance_improvement'] > 0
        efficiency_better = comp['price_efficiency_improvement'] > 0

        wins = sum([cost_better, time_better, balance_better, efficiency_better])

        print(f"\n  综合评价: MBCBO在4个指标中有{wins}个优于BCBO")
        if wins >= 3:
            print(f"  ✅ MBCBO显著优于BCBO")
        elif wins >= 2:
            print(f"  ✓ MBCBO整体优于BCBO")
        else:
            print(f"  ⚠ MBCBO需要进一步优化")

    if 'mbcbo_vs_bcbode' in comparison:
        print(f"\n【MBCBO vs BCBO-DE】")
        comp = comparison['mbcbo_vs_bcbode']
        print(f"  总成本改进: {comp['total_cost_improvement']:+.2f}%")
        print(f"  执行时间改进: {comp['execution_time_improvement']:+.2f}%")
        print(f"  负载均衡改进: {comp['load_balance_improvement']:+.2f}%")
        print(f"  价格效率改进: {comp['price_efficiency_improvement']:+.2f}%")

        # 判断MBCBO是否更优
        cost_better = comp['total_cost_improvement'] > 0
        time_better = comp['execution_time_improvement'] > 0
        balance_better = comp['load_balance_improvement'] > 0
        efficiency_better = comp['price_efficiency_improvement'] > 0

        wins = sum([cost_better, time_better, balance_better, efficiency_better])

        print(f"\n  综合评价: MBCBO在4个指标中有{wins}个优于BCBO-DE")
        if wins >= 3:
            print(f"  ✅ MBCBO显著优于BCBO-DE")
        elif wins >= 2:
            print(f"  ✓ MBCBO整体优于BCBO-DE")
        else:
            print(f"  ⚠ MBCBO表现不如BCBO-DE")

    if 'bcbode_vs_bcbo' in comparison:
        print(f"\n【BCBO-DE vs BCBO】")
        comp = comparison['bcbode_vs_bcbo']
        print(f"  总成本改进: {comp['total_cost_improvement']:+.2f}%")
        print(f"  执行时间改进: {comp['execution_time_improvement']:+.2f}%")
        print(f"  负载均衡改进: {comp['load_balance_improvement']:+.2f}%")
        print(f"  价格效率改进: {comp['price_efficiency_improvement']:+.2f}%")


def analyze_all_chart_sets(data_dir: str):
    """分析所有图表集的数据"""
    print("\n" + "="*80)
    print("MBCBO性能分析报告".center(80))
    print("="*80)

    chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']

    all_mbcbo_vs_bcbo = []
    all_mbcbo_vs_bcbode = []

    for chart_set in chart_sets:
        print(f"\n\n{'#'*80}")
        print(f"分析 {chart_set}".center(80))
        print(f"{'#'*80}")

        data = load_chart_data(chart_set, data_dir)
        if not data:
            continue

        # 分析各算法性能
        bcbo_stats = analyze_algorithm_performance(data, 'BCBO')
        mbcbo_stats = analyze_algorithm_performance(data, 'MBCBO')
        bcbode_stats = analyze_algorithm_performance(data, 'BCBO-DE')

        if bcbo_stats:
            print_algorithm_stats('BCBO', bcbo_stats)

        if bcbode_stats:
            print_algorithm_stats('BCBO-DE', bcbode_stats)

        if mbcbo_stats:
            print_algorithm_stats('MBCBO', mbcbo_stats)

        # 对比分析
        if bcbo_stats and mbcbo_stats:
            comparison = compare_algorithms(bcbo_stats, mbcbo_stats, bcbode_stats)
            print_comparison(comparison)

            # 收集数据用于总体分析
            if 'mbcbo_vs_bcbo' in comparison:
                all_mbcbo_vs_bcbo.append(comparison['mbcbo_vs_bcbo'])
            if 'mbcbo_vs_bcbode' in comparison:
                all_mbcbo_vs_bcbode.append(comparison['mbcbo_vs_bcbode'])

    # 总体分析
    print(f"\n\n{'='*80}")
    print("总体性能分析".center(80))
    print(f"{'='*80}")

    if all_mbcbo_vs_bcbo:
        print(f"\n【MBCBO vs BCBO 总体表现】")
        avg_cost_imp = np.mean([c['total_cost_improvement'] for c in all_mbcbo_vs_bcbo])
        avg_time_imp = np.mean([c['execution_time_improvement'] for c in all_mbcbo_vs_bcbo])
        avg_balance_imp = np.mean([c['load_balance_improvement'] for c in all_mbcbo_vs_bcbo])
        avg_efficiency_imp = np.mean([c['price_efficiency_improvement'] for c in all_mbcbo_vs_bcbo])

        print(f"  平均总成本改进: {avg_cost_imp:+.2f}%")
        print(f"  平均执行时间改进: {avg_time_imp:+.2f}%")
        print(f"  平均负载均衡改进: {avg_balance_imp:+.2f}%")
        print(f"  平均价格效率改进: {avg_efficiency_imp:+.2f}%")

        # 综合判断
        positive_count = sum([
            avg_cost_imp > 0,
            avg_time_imp > 0,
            avg_balance_imp > 0,
            avg_efficiency_imp > 0
        ])

        print(f"\n  综合评价: MBCBO在4个指标中有{positive_count}个平均优于BCBO")

        if positive_count >= 3:
            print(f"\n  ✅ 结论: MBCBO显著优于BCBO，推荐使用")
        elif positive_count >= 2:
            print(f"\n  ✓ 结论: MBCBO整体优于BCBO，可以使用")
        elif avg_cost_imp > -1.0 and positive_count >= 1:
            print(f"\n  ⚠ 结论: MBCBO性能接近BCBO，可强调其他优势（如时间效率）")
        else:
            print(f"\n  ✗ 结论: MBCBO需要进一步优化")

    if all_mbcbo_vs_bcbode:
        print(f"\n【MBCBO vs BCBO-DE 总体表现】")
        avg_cost_imp = np.mean([c['total_cost_improvement'] for c in all_mbcbo_vs_bcbode])
        avg_time_imp = np.mean([c['execution_time_improvement'] for c in all_mbcbo_vs_bcbode])
        avg_balance_imp = np.mean([c['load_balance_improvement'] for c in all_mbcbo_vs_bcbode])
        avg_efficiency_imp = np.mean([c['price_efficiency_improvement'] for c in all_mbcbo_vs_bcbode])

        print(f"  平均总成本改进: {avg_cost_imp:+.2f}%")
        print(f"  平均执行时间改进: {avg_time_imp:+.2f}%")
        print(f"  平均负载均衡改进: {avg_balance_imp:+.2f}%")
        print(f"  平均价格效率改进: {avg_efficiency_imp:+.2f}%")

        # 综合判断
        positive_count = sum([
            avg_cost_imp > 0,
            avg_time_imp > 0,
            avg_balance_imp > 0,
            avg_efficiency_imp > 0
        ])

        print(f"\n  综合评价: MBCBO在4个指标中有{positive_count}个平均优于BCBO-DE")

        if positive_count >= 3:
            print(f"\n  ✅ 结论: MBCBO显著优于BCBO-DE")
        elif positive_count >= 2:
            print(f"\n  ✓ 结论: MBCBO整体优于BCBO-DE")
        else:
            print(f"\n  ⚠ 结论: MBCBO表现不如BCBO-DE")

    # 最终建议
    print(f"\n{'='*80}")
    print("最终建议".center(80))
    print(f"{'='*80}")

    if all_mbcbo_vs_bcbo:
        avg_cost_imp = np.mean([c['total_cost_improvement'] for c in all_mbcbo_vs_bcbo])
        positive_count = sum([
            np.mean([c['total_cost_improvement'] for c in all_mbcbo_vs_bcbo]) > 0,
            np.mean([c['execution_time_improvement'] for c in all_mbcbo_vs_bcbo]) > 0,
            np.mean([c['load_balance_improvement'] for c in all_mbcbo_vs_bcbo]) > 0,
            np.mean([c['price_efficiency_improvement'] for c in all_mbcbo_vs_bcbo]) > 0
        ])

        if positive_count >= 3:
            print("\n✅ MBCBO是优秀的算法改进，强烈推荐用于论文发表")
            print("\n优势:")
            print("  1. 多个性能指标显著优于BCBO")
            print("  2. 并行多策略协同框架具有理论创新性")
            print("  3. 适合大规模云任务调度场景")
        elif positive_count >= 2:
            print("\n✓ MBCBO是有价值的算法改进，推荐用于论文发表")
            print("\n优势:")
            print("  1. 整体性能优于BCBO")
            print("  2. 多策略协同机制具有创新性")
            print("  3. 可强调时间-性能权衡策略")
        elif avg_cost_imp > -1.0:
            print("\n⚠ MBCBO性能接近BCBO，可考虑发表但需强调其他优势")
            print("\n建议:")
            print("  1. 强调时间效率优势（如果有）")
            print("  2. 突出理论创新：并行多策略协同框架")
            print("  3. 明确适用场景：时间敏感的云调度")
            print("  4. 诚实报告性能权衡")
        else:
            print("\n✗ MBCBO需要进一步优化后再考虑发表")
            print("\n建议:")
            print("  1. 调整策略参数")
            print("  2. 优化资源分配机制")
            print("  3. 改进信息交换策略")


def main():
    """主函数"""
    # 数据目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Text Demo', 'RAW_data')

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return 1

    # 分析所有图表集
    analyze_all_chart_sets(data_dir)

    print("\n" + "="*80)
    print("分析完成".center(80))
    print("="*80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
