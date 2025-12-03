#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Chart Sets统计分析工具
对Chart Sets 1-3的10种子数据进行统计分析
"""

import os
import sys
import json
import numpy as np
from scipy import stats
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chart Set配置
CHART_SETS = {
    1: {
        'name': 'Chart Set 1 - 小规模迭代测试',
        'description': 'M=100, N=20, iterations=100',
        'validation_dir': 'multi_seed_validation',
        'metric_type': 'iterations'  # 迭代次数 vs 性能
    },
    2: {
        'name': 'Chart Set 2 - 中等规模任务扫描',
        'description': 'M=100-1000, N=20, iterations=80',
        'validation_dir': 'multi_seed_validation_set2',
        'metric_type': 'scale'  # 任务规模 vs 性能
    },
    3: {
        'name': 'Chart Set 3 - 大规模迭代测试',
        'description': 'M=1000, N=20, iterations=100',
        'validation_dir': 'multi_seed_validation_set3',
        'metric_type': 'iterations'  # 迭代次数 vs 性能
    }
}

def auto_detect_seeds(chart_set):
    """自动检测可用种子"""
    import glob
    import re

    validation_dir = os.path.join(BASE_DIR, CHART_SETS[chart_set]['validation_dir'])
    pattern = os.path.join(validation_dir, f'chart_set_{chart_set}_seed_*.json')
    files = glob.glob(pattern)

    seeds = []
    for filepath in files:
        match = re.search(r'seed_(\d+)\.json$', filepath)
        if match:
            seeds.append(int(match.group(1)))

    seeds.sort()
    return seeds

def load_chart_set_data(chart_set, seeds):
    """加载指定Chart Set的所有种子数据"""
    validation_dir = os.path.join(BASE_DIR, CHART_SETS[chart_set]['validation_dir'])
    data = {}

    print(f"\n[INFO] 加载 {CHART_SETS[chart_set]['name']}")
    print(f"       {CHART_SETS[chart_set]['description']}")

    for seed in seeds:
        filepath = os.path.join(validation_dir, f'chart_set_{chart_set}_seed_{seed}.json')

        if not os.path.exists(filepath):
            print(f"  [WARN] Seed {seed}: 文件不存在")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            data[seed] = {
                'BCBO': file_data['algorithms']['BCBO']['results'],
                'BCBO-GA': file_data['algorithms']['BCBO-GA']['results']
            }

            if CHART_SETS[chart_set]['metric_type'] == 'iterations':
                print(f"  [OK] Seed {seed}: {len(data[seed]['BCBO'])} iterations")
            else:
                print(f"  [OK] Seed {seed}: {len(data[seed]['BCBO'])} data points")

        except Exception as e:
            print(f"  [ERROR] Seed {seed}: {e}")

    return data

def compute_final_iteration_statistics(chart_set, data):
    """计算最终迭代/数据点的统计指标"""
    print(f"\n[INFO] 计算最终数据点统计...")

    if not data:
        print("[ERROR] 无数据")
        return None

    metrics = ['total_cost', 'execution_time', 'load_balance']
    results = {}

    for algorithm in ['BCBO', 'BCBO-GA']:
        results[algorithm] = {}

        for metric in metrics:
            values = []
            for seed, seed_data in data.items():
                try:
                    # 取最后一个数据点
                    final_value = seed_data[algorithm][-1][metric]
                    values.append(final_value)
                except (IndexError, KeyError):
                    continue

            if len(values) > 0:
                results[algorithm][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }

    print(f"  [OK] 统计完成（n={len(data)} seeds）")
    return results

def paired_t_test(statistics):
    """执行配对t检验"""
    print(f"\n[INFO] 执行配对t检验...")

    if not statistics:
        print("[ERROR] 无统计数据")
        return None

    results = {}
    metrics = ['total_cost', 'execution_time', 'load_balance']

    for metric in metrics:
        try:
            bcbo_values = statistics['BCBO'][metric]['values']
            bcbo_ga_values = statistics['BCBO-GA'][metric]['values']

            t_stat, p_value = stats.ttest_rel(bcbo_values, bcbo_ga_values)

            bcbo_mean = np.mean(bcbo_values)
            bcbo_ga_mean = np.mean(bcbo_ga_values)

            if metric == 'load_balance':
                improvement = (bcbo_ga_mean - bcbo_mean) / bcbo_mean * 100
            else:
                improvement = (bcbo_mean - bcbo_ga_mean) / bcbo_mean * 100

            results[metric] = {
                'bcbo_mean': bcbo_mean,
                'bcbo_ga_mean': bcbo_ga_mean,
                'improvement_%': improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            sig_mark = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
            print(f"  {metric}: {improvement:+.2f}%, p={p_value:.4f} {sig_mark}")

        except Exception as e:
            print(f"  [ERROR] {metric}: {e}")

    return results

def comprehensive_improvement_rate(statistics):
    """计算综合改进率"""
    print(f"\n[INFO] 计算综合改进率...")

    try:
        bcbo_cost = statistics['BCBO']['total_cost']['mean']
        bcbo_time = statistics['BCBO']['execution_time']['mean']
        bcbo_balance = statistics['BCBO']['load_balance']['mean']

        bcbo_ga_cost = statistics['BCBO-GA']['total_cost']['mean']
        bcbo_ga_time = statistics['BCBO-GA']['execution_time']['mean']
        bcbo_ga_balance = statistics['BCBO-GA']['load_balance']['mean']

        cost_imp = (bcbo_cost - bcbo_ga_cost) / bcbo_cost * 100
        time_imp = (bcbo_time - bcbo_ga_time) / bcbo_time * 100
        balance_imp = (bcbo_ga_balance - bcbo_balance) / bcbo_balance * 100

        comprehensive = time_imp * 0.5 + balance_imp * 0.3 + cost_imp * 0.2

        print(f"  Cost: {cost_imp:+.2f}%")
        print(f"  Time: {time_imp:+.2f}%")
        print(f"  Balance: {balance_imp:+.2f}%")
        print(f"  Comprehensive: {comprehensive:+.2f}%")

        return {
            'cost_improvement_%': cost_imp,
            'time_improvement_%': time_imp,
            'balance_improvement_%': balance_imp,
            'comprehensive_%': comprehensive
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def generate_report(chart_set, seeds, statistics, t_test_results, comp_improvement):
    """生成报告"""
    validation_dir = os.path.join(BASE_DIR, CHART_SETS[chart_set]['validation_dir'])
    report_path = os.path.join(validation_dir, f'statistical_analysis_report_set{chart_set}.txt')

    print(f"\n[INFO] 生成报告...")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{CHART_SETS[chart_set]['name']} - 统计分析报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置: {CHART_SETS[chart_set]['description']}\n")
        f.write(f"随机种子: {seeds}\n")
        f.write(f"样本数量: {len(seeds)}\n")
        f.write("="*80 + "\n\n")

        # 1. 最终数据点统计
        f.write("1. 最终数据点统计摘要\n")
        f.write("-"*80 + "\n\n")

        for algorithm in ['BCBO', 'BCBO-GA']:
            f.write(f"{algorithm}:\n")
            for metric in ['total_cost', 'execution_time', 'load_balance']:
                stats_data = statistics[algorithm][metric]
                f.write(f"  {metric}: {stats_data['mean']:.2f} ± {stats_data['std']:.2f}\n")
                f.write(f"    Range: [{stats_data['min']:.2f}, {stats_data['max']:.2f}]\n")
            f.write("\n")

        # 2. 配对t检验
        f.write("\n2. 配对t检验结果\n")
        f.write("-"*80 + "\n\n")

        if t_test_results:
            f.write(f"{'指标':<20} {'改进率':>10} {'p值':>10} {'显著性':>10}\n")
            f.write("-"*80 + "\n")
            for metric, result in t_test_results.items():
                significance = '***' if result['p_value'] < 0.001 else \
                               '**' if result['p_value'] < 0.01 else \
                               '*' if result['p_value'] < 0.05 else 'n.s.'
                f.write(f"{metric:<20} {result['improvement_%']:>+9.2f}% {result['p_value']:>10.4f} {significance:>10}\n")

            f.write("\n注: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\n")

        # 3. 综合改进率
        f.write("\n\n3. 综合改进率\n")
        f.write("-"*80 + "\n\n")

        if comp_improvement:
            f.write(f"  成本改进率: {comp_improvement['cost_improvement_%']:+.2f}%\n")
            f.write(f"  时间改进率: {comp_improvement['time_improvement_%']:+.2f}%\n")
            f.write(f"  负载均衡改进率: {comp_improvement['balance_improvement_%']:+.2f}%\n")
            f.write(f"\n  综合改进率: {comp_improvement['comprehensive_%']:+.2f}%\n")
            f.write(f"  (公式: time*0.5 + balance*0.3 + cost*0.2)\n")

        # 4. 结论
        f.write("\n\n4. 统计验证结论\n")
        f.write("-"*80 + "\n\n")

        if t_test_results:
            significant_count = sum(1 for r in t_test_results.values() if r['significant'])
            total_metrics = len(t_test_results)

            f.write(f"显著性检验通过率: {significant_count}/{total_metrics} ({significant_count/total_metrics*100:.1f}%)\n\n")

            is_significant = significant_count >= 2
            is_positive = comp_improvement and comp_improvement['comprehensive_%'] > 0

            if is_significant and is_positive:
                f.write(f"[结论] BCBO-GA在统计学上显著优于BCBO ✓\n")
                f.write(f"       综合改进率为 {comp_improvement['comprehensive_%']:+.2f}%\n")
                f.write(f"       {significant_count}/{total_metrics} 指标达到显著性水平 (p<0.05)\n")
            elif is_positive and not is_significant:
                f.write(f"[结论] BCBO-GA数值上优于BCBO，但统计学上不显著 ⚠\n")
                f.write(f"       综合改进率为 {comp_improvement['comprehensive_%']:+.2f}%\n")
                f.write(f"       显著性检验: {significant_count}/{total_metrics} 指标显著\n")
            else:
                f.write(f"[结论] BCBO-GA性能未达预期 ✗\n")
                f.write(f"       综合改进率为 {comp_improvement['comprehensive_%']:+.2f}%\n")

        f.write("\n" + "="*80 + "\n")

    print(f"  [OK] 报告已保存: {report_path}")
    return report_path

def main():
    """主函数"""
    print("="*80)
    print("多Chart Sets统计分析")
    print("="*80)

    all_results = {}

    for chart_set in [1, 2, 3]:
        print(f"\n\n{'#'*80}")
        print(f"# 处理 Chart Set {chart_set}")
        print(f"{'#'*80}")

        # 检测种子
        seeds = auto_detect_seeds(chart_set)

        if not seeds:
            print(f"[ERROR] Chart Set {chart_set} 无数据")
            continue

        print(f"\n[INFO] 检测到 {len(seeds)} 个种子: {seeds}")

        # 加载数据
        data = load_chart_set_data(chart_set, seeds)

        if not data:
            print(f"[ERROR] Chart Set {chart_set} 数据加载失败")
            continue

        # 计算统计指标
        stats = compute_final_iteration_statistics(chart_set, data)

        if not stats:
            print(f"[ERROR] Chart Set {chart_set} 统计计算失败")
            continue

        # t检验
        t_test_results = paired_t_test(stats)

        # 综合改进率
        comp_improvement = comprehensive_improvement_rate(stats)

        # 生成报告
        report_path = generate_report(chart_set, seeds, stats, t_test_results, comp_improvement)

        all_results[chart_set] = {
            'seeds': len(seeds),
            'significant_metrics': sum(1 for r in (t_test_results or {}).values() if r['significant']),
            'comprehensive_improvement': comp_improvement['comprehensive_%'] if comp_improvement else None,
            'report': report_path
        }

    # 最终总结
    print("\n\n" + "="*80)
    print("多Chart Sets统计分析完成")
    print("="*80)

    for chart_set, results in all_results.items():
        print(f"\nChart Set {chart_set}:")
        print(f"  种子数: {results['seeds']}")
        print(f"  显著指标: {results['significant_metrics']}/3")
        if results['comprehensive_improvement'] is not None:
            print(f"  综合改进: {results['comprehensive_improvement']:+.2f}%")
        print(f"  报告: {results['report']}")

    print("\n" + "="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
