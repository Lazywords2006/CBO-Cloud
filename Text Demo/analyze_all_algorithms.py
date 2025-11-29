#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全算法对比数据分析脚本
分析四组图表数据,评估BCBO-DE vs BCBO性能,以及数据是否适合发表
"""

import json
import os
import sys

def load_chart_data(chart_set_num):
    """加载图表数据"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'RAW_data')
    filename = f'chart_set_{chart_set_num}_merged_results.json'
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"[ERROR] 文件不存在: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def analyze_bcbo_vs_bcbode(chart_data):
    """分析BCBO vs BCBO-DE性能对比"""
    chart_set = chart_data['chart_set']
    config = chart_data['config']

    bcbo_results = chart_data['algorithms'].get('BCBO', {}).get('results', [])
    bcbode_results = chart_data['algorithms'].get('BCBO-DE', {}).get('results', [])

    if not bcbo_results or not bcbode_results:
        return None

    # 计算各项指标的平均值
    analysis = {
        'chart_set': chart_set,
        'config_name': config['name'],
        'data_points': len(bcbo_results),
        'comparisons': []
    }

    total_cost_wins = 0
    load_balance_wins = 0

    for i in range(min(len(bcbo_results), len(bcbode_results))):
        bcbo = bcbo_results[i]
        bcbode = bcbode_results[i]

        # 成本对比 (越低越好)
        cost_diff = (bcbode['total_cost'] - bcbo['total_cost']) / bcbo['total_cost'] * 100

        # 负载均衡对比 (越高越好)
        lb_diff = (bcbode['load_balance'] - bcbo['load_balance']) / bcbo['load_balance'] * 100

        # 执行时间对比 (越低越好)
        time_diff = (bcbode['execution_time'] - bcbo['execution_time']) / bcbo['execution_time'] * 100

        if cost_diff < 0:
            total_cost_wins += 1
        if lb_diff > 0:
            load_balance_wins += 1

        comparison = {
            'point_idx': i + 1,
            'bcbo_cost': bcbo['total_cost'],
            'bcbode_cost': bcbode['total_cost'],
            'cost_diff_pct': cost_diff,
            'bcbo_lb': bcbo['load_balance'],
            'bcbode_lb': bcbode['load_balance'],
            'lb_diff_pct': lb_diff,
            'time_diff_pct': time_diff,
            'bcbode_wins_cost': cost_diff < 0,
            'bcbode_wins_lb': lb_diff > 0
        }

        analysis['comparisons'].append(comparison)

    # 汇总统计
    analysis['summary'] = {
        'total_cost_wins': total_cost_wins,
        'total_cost_win_rate': total_cost_wins / len(analysis['comparisons']) * 100,
        'load_balance_wins': load_balance_wins,
        'load_balance_win_rate': load_balance_wins / len(analysis['comparisons']) * 100,
        'avg_cost_diff_pct': sum(c['cost_diff_pct'] for c in analysis['comparisons']) / len(analysis['comparisons']),
        'avg_lb_diff_pct': sum(c['lb_diff_pct'] for c in analysis['comparisons']) / len(analysis['comparisons']),
        'avg_time_diff_pct': sum(c['time_diff_pct'] for c in analysis['comparisons']) / len(analysis['comparisons'])
    }

    return analysis

def analyze_algorithm_ranking(chart_data):
    """分析所有算法的排名"""
    algorithms = chart_data['algorithms']

    # 对最后一个数据点进行排名
    rankings = []

    for algo_name, algo_data in algorithms.items():
        results = algo_data.get('results', [])
        if not results:
            continue

        last_result = results[-1]
        rankings.append({
            'algorithm': algo_name,
            'total_cost': last_result['total_cost'],
            'load_balance': last_result['load_balance'],
            'execution_time': last_result['execution_time']
        })

    # 按total_cost排序 (越低越好)
    rankings_by_cost = sorted(rankings, key=lambda x: x['total_cost'])

    # 按load_balance排序 (越高越好)
    rankings_by_lb = sorted(rankings, key=lambda x: x['load_balance'], reverse=True)

    return {
        'by_cost': rankings_by_cost,
        'by_load_balance': rankings_by_lb
    }

def check_publication_suitability(all_analyses):
    """检查数据是否适合发表"""
    issues = []
    warnings = []
    passed = []

    # 检查1: BCBO-DE是否总体优于BCBO
    overall_cost_wins = 0
    overall_cost_total = 0

    for analysis in all_analyses:
        if analysis:
            summary = analysis['summary']
            overall_cost_wins += summary['total_cost_wins']
            overall_cost_total += analysis['data_points']

    overall_win_rate = overall_cost_wins / overall_cost_total * 100 if overall_cost_total > 0 else 0

    if overall_win_rate >= 70:
        passed.append(f"[OK] BCBO-DE成本胜率 {overall_win_rate:.1f}% (>=70%标准)")
    elif overall_win_rate >= 50:
        warnings.append(f"[WARN] BCBO-DE成本胜率 {overall_win_rate:.1f}% (50-70%,需说明)")
    else:
        issues.append(f"[ISSUE] BCBO-DE成本胜率仅 {overall_win_rate:.1f}% (<50%,不推荐发表)")

    # 检查2: 平均成本差距
    for analysis in all_analyses:
        if analysis:
            avg_cost_diff = analysis['summary']['avg_cost_diff_pct']
            chart_set = analysis['chart_set']

            if avg_cost_diff < -1.0:
                passed.append(f"[OK] {chart_set}: 平均成本优势 {-avg_cost_diff:.2f}%")
            elif avg_cost_diff < 1.0:
                warnings.append(f"[WARN] {chart_set}: 平均成本差距 {avg_cost_diff:.2f}% (接近持平)")
            else:
                issues.append(f"[ISSUE] {chart_set}: BCBO-DE成本更高 +{avg_cost_diff:.2f}%")

    # 检查3: 负载均衡
    for analysis in all_analyses:
        if analysis:
            avg_lb_diff = analysis['summary']['avg_lb_diff_pct']
            chart_set = analysis['chart_set']

            if avg_lb_diff > 1.0:
                passed.append(f"[OK] {chart_set}: 负载均衡优势 +{avg_lb_diff:.2f}%")
            elif avg_lb_diff > -5.0:
                warnings.append(f"[WARN] {chart_set}: 负载均衡略降 {avg_lb_diff:.2f}%")
            else:
                issues.append(f"[ISSUE] {chart_set}: 负载均衡显著降低 {avg_lb_diff:.2f}%")

    # 检查4: 数据一致性 (runs_per_point)
    for analysis in all_analyses:
        if analysis:
            chart_set = analysis['chart_set']
            warnings.append(f"[WARN] {chart_set}: 仅单次运行 (runs_per_point=1),建议30次")

    return {
        'issues': issues,
        'warnings': warnings,
        'passed': passed,
        'overall_win_rate': overall_win_rate,
        'suitable_for_publication': len(issues) == 0
    }

def print_analysis_report(all_analyses, publication_check):
    """打印分析报告"""
    print("=" * 80)
    print("全算法对比数据分析报告")
    print("=" * 80)

    # 1. 各图表集BCBO vs BCBO-DE对比
    print("\n[1] BCBO vs BCBO-DE 详细对比")
    print("-" * 80)

    for analysis in all_analyses:
        if not analysis:
            continue

        print(f"\n{analysis['config_name']}")
        print(f"数据点数: {analysis['data_points']}")

        summary = analysis['summary']
        print(f"  成本胜率: {summary['total_cost_win_rate']:.1f}% ({summary['total_cost_wins']}/{analysis['data_points']})")
        print(f"  平均成本差距: {summary['avg_cost_diff_pct']:+.2f}%")
        print(f"  负载均衡胜率: {summary['load_balance_win_rate']:.1f}% ({summary['load_balance_wins']}/{analysis['data_points']})")
        print(f"  平均负载差距: {summary['avg_lb_diff_pct']:+.2f}%")
        print(f"  平均时间差距: {summary['avg_time_diff_pct']:+.2f}%")

        # 显示前3个和后3个数据点
        print("\n  关键数据点:")
        for i in [0, 1, 2, -3, -2, -1]:
            if i < len(analysis['comparisons']) and i >= -len(analysis['comparisons']):
                comp = analysis['comparisons'][i]
                print(f"    点{comp['point_idx']}: 成本{comp['cost_diff_pct']:+.2f}%, 负载{comp['lb_diff_pct']:+.2f}%")

    # 2. 发表适用性检查
    print("\n" + "=" * 80)
    print("[2] 期刊发表适用性评估")
    print("-" * 80)

    print(f"\n整体胜率: {publication_check['overall_win_rate']:.1f}%")
    print(f"是否适合发表: {'是' if publication_check['suitable_for_publication'] else '否'}")

    if publication_check['passed']:
        print("\n[通过项]")
        for item in publication_check['passed']:
            print(f"  {item}")

    if publication_check['warnings']:
        print("\n[警告项]")
        for item in publication_check['warnings']:
            print(f"  {item}")

    if publication_check['issues']:
        print("\n[问题项]")
        for item in publication_check['issues']:
            print(f"  {item}")

    # 3. 建议
    print("\n" + "=" * 80)
    print("[3] 建议与下一步")
    print("-" * 80)

    if publication_check['suitable_for_publication']:
        print("\n[OK] 数据质量符合发表标准!")
        print("\n建议:")
        print("  1. 将 runs_per_point 从 1 改为 30,增加统计可靠性")
        print("  2. 重新生成全部4组数据")
        print("  3. 计算标准差和置信区间")
        print("  4. 准备图表可视化")
    else:
        print("\n[ISSUE] 当前数据存在问题,不建议直接发表")
        print("\n需要:")
        print("  1. 检查BCBO-DE算法实现")
        print("  2. 确认负载均衡修复机制正常工作")
        print("  3. 验证问题实例共享机制")
        print("  4. 解决上述问题项后重新生成数据")

    print("\n" + "=" * 80)

def main():
    """主函数"""
    all_analyses = []

    print("正在加载数据...")
    for i in range(1, 5):
        print(f"  加载 Chart Set {i}...")
        chart_data = load_chart_data(i)
        if chart_data:
            analysis = analyze_bcbo_vs_bcbode(chart_data)
            all_analyses.append(analysis)

            # 打印算法排名
            rankings = analyze_algorithm_ranking(chart_data)
            print(f"    成本排名前3: {', '.join([r['algorithm'] for r in rankings['by_cost'][:3]])}")

    # 发表适用性检查
    publication_check = check_publication_suitability(all_analyses)

    # 打印完整报告
    print_analysis_report(all_analyses, publication_check)

if __name__ == "__main__":
    main()
