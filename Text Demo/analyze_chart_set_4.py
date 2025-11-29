#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案D v2效果分析脚本
对比BCBO vs BCBO-DE在Chart Set 4的表现
"""

import json
import os

def analyze_bcbo_vs_bcbode():
    """分析BCBO vs BCBO-DE在旧数据中的表现"""

    # 读取旧数据
    data_file = 'RAW_data/chart_set_4_merged_results.json'

    if not os.path.exists(data_file):
        print(f"[ERROR] 数据文件不存在: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bcbo_results = data['algorithms']['BCBO']['results']
    bcbode_results = data['algorithms']['BCBO-DE']['results']

    print("="*80)
    print("Chart Set 4 数据分析 (旧版本 - 2025-11-28 16:28)")
    print("="*80)
    print(f"数据来源: {data['timestamp']}")
    print(f"配置: {data['config']['fixed_params']}")
    print(f"M范围: {data['config']['values']}")
    print()

    # 核心指标对比
    print("="*80)
    print("BCBO vs BCBO-DE 负载均衡对比")
    print("="*80)
    print(f"{'M':>6} | {'BCBO LB':>10} | {'BCBO-DE LB':>12} | {'差异':>10} | {'差异%':>10} | {'达标(≥0.85)':>12}")
    print("-"*80)

    results_summary = []

    for bcbo_point, bcbode_point in zip(bcbo_results, bcbode_results):
        M = bcbo_point['M']
        bcbo_lb = bcbo_point['load_balance']
        bcbode_lb = bcbode_point['load_balance']

        lb_diff = bcbode_lb - bcbo_lb
        lb_diff_pct = (lb_diff / bcbo_lb) * 100

        meets_threshold = "OK" if bcbode_lb >= 0.85 else "FAIL"

        results_summary.append({
            'M': M,
            'bcbo_lb': bcbo_lb,
            'bcbode_lb': bcbode_lb,
            'lb_diff': lb_diff,
            'lb_diff_pct': lb_diff_pct,
            'meets_085': bcbode_lb >= 0.85
        })

        print(f"{M:6d} | {bcbo_lb:10.4f} | {bcbode_lb:12.4f} | {lb_diff:+10.4f} | {lb_diff_pct:+9.2f}% | {meets_threshold:>12}")

    # 成本对比
    print()
    print("="*80)
    print("BCBO vs BCBO-DE 成本对比")
    print("="*80)
    print(f"{'M':>6} | {'BCBO Cost':>12} | {'BCBO-DE Cost':>14} | {'差异':>10} | {'差异%':>10}")
    print("-"*80)

    for i, (bcbo_point, bcbode_point) in enumerate(zip(bcbo_results, bcbode_results)):
        M = bcbo_point['M']
        bcbo_cost = bcbo_point['total_cost']
        bcbode_cost = bcbode_point['total_cost']

        cost_diff = bcbode_cost - bcbo_cost
        cost_diff_pct = (cost_diff / bcbo_cost) * 100

        results_summary[i]['bcbo_cost'] = bcbo_cost
        results_summary[i]['bcbode_cost'] = bcbode_cost
        results_summary[i]['cost_diff_pct'] = cost_diff_pct

        print(f"{M:6d} | {bcbo_cost:12.2f} | {bcbode_cost:14.2f} | {cost_diff:+10.2f} | {cost_diff_pct:+9.2f}%")

    # 综合评估
    print()
    print("="*80)
    print("综合评估 (旧版本)")
    print("="*80)

    avg_lb_diff_pct = sum(r['lb_diff_pct'] for r in results_summary) / len(results_summary)
    avg_cost_diff_pct = sum(r['cost_diff_pct'] for r in results_summary) / len(results_summary)

    all_lb_above_85 = all(r['meets_085'] for r in results_summary)
    all_lb_drop_below_5 = all(abs(r['lb_diff_pct']) < 5 for r in results_summary)
    all_lb_drop_below_10 = all(abs(r['lb_diff_pct']) < 10 for r in results_summary)
    all_cost_reasonable = all(abs(r['cost_diff_pct']) < 3 for r in results_summary)

    cost_win_count = sum(1 for r in results_summary if r['cost_diff_pct'] < 0)
    cost_win_rate = cost_win_count / len(results_summary) * 100

    print(f"平均负载均衡差异: {avg_lb_diff_pct:+.2f}%")
    print(f"平均成本差异: {avg_cost_diff_pct:+.2f}%")
    print(f"成本胜率: {cost_win_rate:.1f}% ({cost_win_count}/{len(results_summary)})")
    print()

    print("发表标准验证:")
    print(f"  [{'OK' if all_lb_above_85 else 'FAIL'}] 所有场景负载均衡 >= 0.85")
    print(f"  [{'OK' if all_lb_drop_below_5 else 'FAIL'}] 所有场景负载均衡降幅 < 5%")
    print(f"  [{'OK' if all_lb_drop_below_10 else 'FAIL'}] 所有场景负载均衡降幅 < 10%")
    print(f"  [{'OK' if all_cost_reasonable else 'FAIL'}] 所有场景成本差异 < 3%")
    print(f"  [{'OK' if cost_win_rate >= 40 else 'FAIL'}] 成本胜率 >= 40%")
    print()

    # 问题场景识别
    print("="*80)
    print("问题场景识别")
    print("="*80)

    failed_085 = [r for r in results_summary if not r['meets_085']]
    high_drop = [r for r in results_summary if abs(r['lb_diff_pct']) >= 10]
    high_cost = [r for r in results_summary if abs(r['cost_diff_pct']) >= 5]

    if failed_085:
        print(f"未达0.85阈值的场景: {[r['M'] for r in failed_085]}")
        for r in failed_085:
            print(f"  M={r['M']}: LB={r['bcbode_lb']:.4f} (vs BCBO {r['bcbo_lb']:.4f})")

    if high_drop:
        print(f"\n负载均衡降幅≥10%的场景: {[r['M'] for r in high_drop]}")
        for r in high_drop:
            print(f"  M={r['M']}: 降幅={r['lb_diff_pct']:.2f}%")

    if high_cost:
        print(f"\n成本差异≥5%的场景: {[r['M'] for r in high_cost]}")
        for r in high_cost:
            print(f"  M={r['M']}: 差异={r['cost_diff_pct']:.2f}%")

    print()
    print("="*80)
    print("结论 (旧版本)")
    print("="*80)

    if all_lb_above_85 and all_lb_drop_below_5 and all_cost_reasonable:
        print("[SUCCESS] 完全达标！所有指标符合理想发表标准")
        print("[建议] 可以直接进行论文发表")
    elif all_lb_above_85 and all_lb_drop_below_10:
        print("[PARTIAL] 基本达标，达到可接受发表标准")
        print("[建议] 可以发表，但需在论文中说明负载均衡的权衡")
    else:
        print("[FAIL] 未达标，需要进一步优化")
        print("[建议] 实施方案D v2 (beta=500, gamma=10)")

    print()
    print("="*80)

    return results_summary

if __name__ == "__main__":
    analyze_bcbo_vs_bcbode()
