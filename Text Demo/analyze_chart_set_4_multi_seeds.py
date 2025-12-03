#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chart Set 4 多种子统计分析脚本

分析Chart Set 4（超大规模任务扫描）的10个随机种子数据
"""

import json
import os
import numpy as np
from scipy import stats
from datetime import datetime


def load_set4_data(base_dir='multi_seed_validation_set4'):
    """加载Chart Set 4的所有种子数据（仅保留完整数据）"""
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    data_by_seed = {}
    incomplete_seeds = []

    for seed in seeds:
        filename = f'{base_dir}/chart_set_4_seed_{seed}.json'
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查数据完整性：BCBO和BCBO-GA都必须有M=5000的数据点
            bcbo_results = data['algorithms']['BCBO']['results']
            bcbo_ga_results = data['algorithms']['BCBO-GA']['results']

            bcbo_has_5000 = any(p['M'] == 5000 for p in bcbo_results)
            bcbo_ga_has_5000 = any(p['M'] == 5000 for p in bcbo_ga_results)

            if bcbo_has_5000 and bcbo_ga_has_5000:
                data_by_seed[seed] = data
                print(f"  [OK] Seed {seed}: {len(bcbo_results)} BCBO points, {len(bcbo_ga_results)} BCBO-GA points")
            else:
                incomplete_seeds.append(seed)
                print(f"  [SKIP] Seed {seed}: 数据不完整 (BCBO M=5000: {bcbo_has_5000}, BCBO-GA M=5000: {bcbo_ga_has_5000})")
        else:
            print(f"  [ERROR] Seed {seed}: 文件不存在")

    if incomplete_seeds:
        print(f"\n  [WARN] 跳过 {len(incomplete_seeds)} 个不完整的种子: {incomplete_seeds}")
        print(f"  [INFO] 使用 {len(data_by_seed)} 个完整种子进行分析")

    return data_by_seed


def extract_final_metrics(data_by_seed):
    """提取每个种子的最终数据点（M=5000）的指标"""
    metrics = {
        'BCBO': {'total_cost': [], 'execution_time': [], 'load_balance': []},
        'BCBO-GA': {'total_cost': [], 'execution_time': [], 'load_balance': []}
    }

    for seed, data in sorted(data_by_seed.items()):
        for algo_name in ['BCBO', 'BCBO-GA']:
            results = data['algorithms'][algo_name]['results']

            # 找到M=5000的数据点
            final_point = None
            for point in results:
                if point['M'] == 5000:
                    final_point = point
                    break

            if final_point is None:
                raise ValueError(f"Seed {seed} {algo_name}: 缺少M=5000数据点")

            metrics[algo_name]['total_cost'].append(final_point['total_cost'])
            metrics[algo_name]['execution_time'].append(final_point['execution_time'])
            metrics[algo_name]['load_balance'].append(final_point['load_balance'])

    # 转换为NumPy数组
    for algo in metrics:
        for metric in metrics[algo]:
            metrics[algo][metric] = np.array(metrics[algo][metric])

    return metrics


def perform_statistical_analysis(metrics):
    """执行统计分析（配对t检验）"""
    results = {}
    metric_names = ['total_cost', 'execution_time', 'load_balance']

    for metric in metric_names:
        bcbo_values = metrics['BCBO'][metric]
        bcbo_ga_values = metrics['BCBO-GA'][metric]

        # 配对t检验
        t_stat, p_value = stats.ttest_rel(bcbo_ga_values, bcbo_values)

        # 计算改进率
        mean_bcbo = np.mean(bcbo_values)
        mean_bcbo_ga = np.mean(bcbo_ga_values)

        if metric == 'load_balance':
            # 负载均衡越高越好
            improvement = ((mean_bcbo_ga - mean_bcbo) / mean_bcbo) * 100
        else:
            # 成本和时间越低越好
            improvement = ((mean_bcbo - mean_bcbo_ga) / mean_bcbo) * 100

        results[metric] = {
            'bcbo_mean': mean_bcbo,
            'bcbo_std': np.std(bcbo_values, ddof=1),
            'bcbo_ga_mean': mean_bcbo_ga,
            'bcbo_ga_std': np.std(bcbo_ga_values, ddof=1),
            'improvement': improvement,
            'p_value': p_value,
            't_stat': t_stat
        }

    return results


def generate_report(metrics, stats_results, output_file, num_seeds):
    """生成统计分析报告"""
    report = []
    report.append("=" * 80)
    report.append("Chart Set 4 - 超大规模任务扫描 - 统计分析报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("配置: M=1000-5000, N=20, iterations=50")
    seeds_list = list(metrics['BCBO']['total_cost'])  # 获取实际使用的种子数
    report.append(f"随机种子: 45-51 (共{num_seeds}个完整种子)")
    report.append(f"样本数量: {num_seeds}")
    report.append("=" * 80)
    report.append("")

    # 1. 统计摘要
    report.append("1. 最终数据点统计摘要 (M=5000)")
    report.append("-" * 80)
    report.append("")

    for algo in ['BCBO', 'BCBO-GA']:
        report.append(f"{algo}:")
        for metric in ['total_cost', 'execution_time', 'load_balance']:
            mean = stats_results[metric][f'{algo.lower().replace("-", "_")}_mean']
            std = stats_results[metric][f'{algo.lower().replace("-", "_")}_std']
            values = metrics[algo][metric]
            report.append(f"  {metric}: {mean:.2f} +- {std:.2f}")
            report.append(f"    Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
        report.append("")

    # 2. 配对t检验结果
    report.append("2. 配对t检验结果")
    report.append("-" * 80)
    report.append("")
    report.append(f"{'指标':<25} {'改进率':>12} {'p值':>12} {'显著性':>10}")
    report.append("-" * 80)

    significant_count = 0
    for metric in ['total_cost', 'execution_time', 'load_balance']:
        improvement = stats_results[metric]['improvement']
        p_value = stats_results[metric]['p_value']

        # 判断显著性
        if p_value < 0.001:
            sig_marker = '***'
            significant_count += 1
        elif p_value < 0.01:
            sig_marker = '**'
            significant_count += 1
        elif p_value < 0.05:
            sig_marker = '*'
            significant_count += 1
        else:
            sig_marker = 'n.s.'

        report.append(f"{metric:<25} {improvement:>+11.2f}% {p_value:>11.4f} {sig_marker:>10}")

    report.append("")
    report.append("注: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    report.append("")
    report.append("")

    # 3. 综合改进率
    cost_imp = stats_results['total_cost']['improvement']
    time_imp = stats_results['execution_time']['improvement']
    balance_imp = stats_results['load_balance']['improvement']

    comprehensive_improvement = time_imp * 0.5 + balance_imp * 0.3 + cost_imp * 0.2

    report.append("3. 综合改进率")
    report.append("-" * 80)
    report.append("")
    report.append(f"  成本改进率: {cost_imp:+.2f}%")
    report.append(f"  时间改进率: {time_imp:+.2f}%")
    report.append(f"  负载均衡改进率: {balance_imp:+.2f}%")
    report.append("")
    report.append(f"  综合改进率: {comprehensive_improvement:+.2f}%")
    report.append("  (公式: time*0.5 + balance*0.3 + cost*0.2)")
    report.append("")
    report.append("")

    # 4. 结论
    report.append("4. 统计验证结论")
    report.append("-" * 80)
    report.append("")
    report.append(f"显著性检验通过率: {significant_count}/3 ({significant_count/3*100:.1f}%)")
    report.append("")

    if significant_count >= 2:
        conclusion = "[结论] BCBO-GA在统计学上显著优于BCBO"
    elif comprehensive_improvement > 0:
        conclusion = "[结论] BCBO-GA数值上优于BCBO，但统计学上不显著"
    else:
        conclusion = "[结论] BCBO-GA未能改进BCBO性能"

    report.append(conclusion)
    report.append(f"       综合改进率为 {comprehensive_improvement:+.2f}%")
    if significant_count > 0:
        report.append(f"       {significant_count}/3 指标达到显著性水平 (p<0.05)")
    else:
        report.append(f"       显著性检验: {significant_count}/3 指标显著")
    report.append("")
    report.append("=" * 80)

    # 写入文件（使用UTF-8）
    report_text = '\n'.join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 打印到控制台（避免特殊字符）
    print(report_text.replace('±', '+-'))
    return comprehensive_improvement, significant_count


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Chart Set 4 多种子统计分析")
    print("=" * 80)
    print()

    # 1. 加载数据
    print("[INFO] 加载Chart Set 4数据...")
    data_by_seed = load_set4_data()
    print(f"  [OK] 成功加载 {len(data_by_seed)} 个种子的数据")
    print()

    # 2. 提取最终指标
    print("[INFO] 提取最终数据点指标 (M=5000)...")
    metrics = extract_final_metrics(data_by_seed)
    print(f"  [OK] 已提取 {len(metrics['BCBO']['total_cost'])} 个样本")
    print()

    # 3. 统计分析
    print("[INFO] 执行配对t检验...")
    stats_results = perform_statistical_analysis(metrics)
    for metric, result in stats_results.items():
        sig = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'n.s.'
        print(f"  {metric}: {result['improvement']:+.2f}%, p={result['p_value']:.4f} {sig}")
    print()

    # 4. 生成报告
    print("[INFO] 生成统计分析报告...")
    output_file = 'multi_seed_validation_set4/statistical_analysis_report_set4.txt'
    comprehensive_imp, sig_count = generate_report(metrics, stats_results, output_file, len(data_by_seed))
    print(f"  [OK] 报告已保存: {output_file}")
    print()

    # 5. 总结
    print("=" * 80)
    print("分析完成")
    print("=" * 80)
    print(f"综合改进率: {comprehensive_imp:+.2f}%")
    print(f"显著性指标数: {sig_count}/3")
    print()


if __name__ == '__main__':
    main()
