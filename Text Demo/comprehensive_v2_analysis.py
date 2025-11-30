#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-GA v2.0 综合性能分析 - 所有图表集
分析v2.0自适应参数机制在不同规模下的表现
"""

import json
import numpy as np
import os

def load_chart_data(chart_set_name):
    """加载图表集数据"""
    filename = f"BCBO_vs_BCBO_GA_Data/{chart_set_name}_bcbo_comparison.json"
    if not os.path.exists(filename):
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_chart_set(data, chart_set_num):
    """分析单个图表集"""
    if not data:
        return None

    config = data['config']
    timestamp = data['timestamp']
    bcbo_results = data['algorithms']['BCBO']['results']
    bcbo_ga_results = data['algorithms']['BCBO-GA']['results']

    # 获取最终数据点
    bcbo_final = bcbo_results[-1]
    ga_final = bcbo_ga_results[-1]

    # 计算改进率
    metrics = {}

    # Total Cost
    bcbo_cost = bcbo_final['total_cost']
    ga_cost = ga_final['total_cost']
    metrics['cost_improvement'] = ((bcbo_cost - ga_cost) / bcbo_cost) * 100

    # Execution Time
    bcbo_time = bcbo_final['execution_time']
    ga_time = ga_final['execution_time']
    metrics['time_improvement'] = ((bcbo_time - ga_time) / bcbo_time) * 100

    # Load Balance
    bcbo_lb = bcbo_final['load_balance']
    ga_lb = ga_final['load_balance']
    metrics['lb_improvement'] = ((ga_lb - bcbo_lb) / bcbo_lb) * 100

    # Best Fitness (if available)
    if 'best_fitness' in bcbo_final and 'best_fitness' in ga_final:
        bcbo_fitness = bcbo_final['best_fitness']
        ga_fitness = ga_final['best_fitness']
        metrics['fitness_improvement'] = ((ga_fitness - bcbo_fitness) / bcbo_fitness) * 100
        has_fitness = True
    else:
        metrics['fitness_improvement'] = 0.0
        has_fitness = False

    # 综合平均改进
    if has_fitness:
        metrics['avg_improvement'] = (
            metrics['cost_improvement'] +
            metrics['time_improvement'] +
            metrics['lb_improvement'] +
            metrics['fitness_improvement']
        ) / 4
    else:
        # For task_scale type charts, use only cost, time, and load_balance
        metrics['avg_improvement'] = (
            metrics['cost_improvement'] +
            metrics['time_improvement'] +
            metrics['lb_improvement']
        ) / 3

    metrics['has_fitness'] = has_fitness

    # 元数据
    metrics['timestamp'] = timestamp
    metrics['config'] = config
    metrics['data_points'] = len(bcbo_results)
    metrics['bcbo_final'] = bcbo_final
    metrics['ga_final'] = ga_final

    return metrics

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_section(title):
    """打印章节"""
    print(f"\n{title}")
    print("-" * 80)

# 主分析
print_header("BCBO-GA v2.0 Comprehensive Performance Analysis")
print(f"\nAnalyzing all 4 chart sets to verify v2.0 adaptive parameters...")

# v2.0预期性能
v2_targets = {
    1: {"name": "Small Scale (M=100)", "M": 100, "target": 0.20, "description": "迭代次数 vs 性能"},
    2: {"name": "Medium Scale (M=100-1000)", "M": "100-1000", "target": 0.10, "description": "任务规模 vs 成本"},
    3: {"name": "Large Scale (M=1000)", "M": 1000, "target": -0.10, "description": "迭代次数 vs 性能"},
    4: {"name": "Super-Large Scale (M=1000-5000)", "M": "1000-5000", "target": -0.50, "description": "任务规模 vs 成本"}
}

# 收集所有结果
results = {}
for i in range(1, 5):
    chart_set_name = f"chart_set_{i}"
    data = load_chart_data(chart_set_name)
    if data:
        results[i] = analyze_chart_set(data, i)
    else:
        results[i] = None

# 打印详细分析
print_header("Individual Chart Set Analysis")

for i in range(1, 5):
    if not results[i]:
        print(f"\n[WARNING] Chart Set {i} data not found - skipping")
        continue

    metrics = results[i]
    target_info = v2_targets[i]

    print_section(f"Chart Set {i}: {target_info['name']}")

    print(f"Description: {target_info['description']}")
    print(f"Scale: M = {target_info['M']}, N = {metrics['config']['fixed_params'].get('N', 'N/A')}")
    print(f"Data generated: {metrics['timestamp']}")
    print(f"Total iterations: {metrics['data_points']}")

    print(f"\nPerformance Metrics:")
    print(f"  Total Cost:       {metrics['cost_improvement']:+.2f}%")
    print(f"  Execution Time:   {metrics['time_improvement']:+.2f}%")
    print(f"  Load Balance:     {metrics['lb_improvement']:+.2f}%")
    if metrics.get('has_fitness', False):
        print(f"  Best Fitness:     {metrics['fitness_improvement']:+.2f}%")
    print(f"  ----------------------------------------")
    print(f"  AVERAGE:          {metrics['avg_improvement']:+.2f}%")

    # 与v2.0目标对比
    target = target_info['target']
    actual = metrics['avg_improvement']
    gap = actual - target

    print(f"\nv2.0 Target Comparison:")
    print(f"  Expected (v2.0):  {target:+.2f}%")
    print(f"  Actual (v2.0):    {actual:+.2f}%")
    print(f"  Gap:              {gap:+.2f}%")

    if actual >= target:
        print(f"  Status:           [SUCCESS] Exceeds target!")
    elif actual >= target - 0.10:
        print(f"  Status:           [OK] Close to target")
    else:
        print(f"  Status:           [WARNING] Below target")

# 综合总结
print_header("Overall v2.0 Performance Summary")

summary_table = []
all_targets_met = True

print(f"\n{'Chart Set':<15} {'Scale':<20} {'v2.0 Target':<15} {'Actual':<15} {'Gap':<15} {'Status':<15}")
print("-" * 100)

for i in range(1, 5):
    if not results[i]:
        print(f"Chart Set {i:<4} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} [MISSING]")
        all_targets_met = False
        continue

    metrics = results[i]
    target_info = v2_targets[i]

    target = target_info['target']
    actual = metrics['avg_improvement']
    gap = actual - target

    if actual >= target:
        status = "[SUCCESS]"
    elif actual >= target - 0.10:
        status = "[OK]"
    else:
        status = "[WARNING]"
        all_targets_met = False

    scale_str = str(target_info['M'])

    print(f"Chart Set {i:<4} {scale_str:<20} {target:+.2f}%{'':<10} {actual:+.2f}%{'':<10} {gap:+.2f}%{'':<10} {status}")

    summary_table.append({
        'chart_set': i,
        'target': target,
        'actual': actual,
        'gap': gap,
        'status': status
    })

# 计算综合改进率
if all(results[i] is not None for i in range(1, 5)):
    overall_avg = np.mean([results[i]['avg_improvement'] for i in range(1, 5)])
    overall_target = np.mean([v2_targets[i]['target'] for i in range(1, 5)])

    print("-" * 100)
    print(f"{'OVERALL':<15} {'All Scales':<20} {overall_target:+.2f}%{'':<10} {overall_avg:+.2f}%{'':<10} {overall_avg - overall_target:+.2f}%{'':<10}", end="")

    if overall_avg >= overall_target:
        print(" [SUCCESS]")
    else:
        print(" [WARNING]")

# 最终结论
print_header("FINAL CONCLUSION")

if all(results[i] is not None for i in range(1, 5)):
    success_count = sum(1 for item in summary_table if '[SUCCESS]' in item['status'])
    ok_count = sum(1 for item in summary_table if '[OK]' in item['status'])
    warning_count = sum(1 for item in summary_table if '[WARNING]' in item['status'])

    print(f"\nResults Summary:")
    print(f"  [SUCCESS]: {success_count}/4 chart sets")
    print(f"  [OK]:      {ok_count}/4 chart sets")
    print(f"  [WARNING]: {warning_count}/4 chart sets")

    print(f"\nOverall v2.0 Performance: {overall_avg:+.2f}%")
    print(f"Overall v2.0 Target:      {overall_target:+.2f}%")

    if success_count >= 3:
        print("\n[SUCCESS] v2.0 adaptive parameter mechanism is WORKING correctly!")
        print("  - Majority of chart sets meet or exceed targets")
        print("  - Adaptive parameters successfully adjust to different scales")
        print("  - Ready for publication!")
    elif success_count + ok_count >= 3:
        print("\n[OK] v2.0 performance is acceptable")
        print("  - Most chart sets show good performance")
        print("  - Minor adjustments may improve results")
    else:
        print("\n[WARNING] v2.0 performance below expectations")
        print("  - Need to investigate why targets not met")
        print("  - Consider parameter tuning")
else:
    print("\n[WARNING] Not all chart sets generated")
    print("Missing chart sets - please generate all 4 sets for complete analysis")

# 关键发现
print_header("Key Findings")

if all(results[i] is not None for i in range(1, 5)):
    print("\n1. Scale-dependent Performance:")
    for i in range(1, 5):
        metrics = results[i]
        target_info = v2_targets[i]
        print(f"   Chart Set {i} ({target_info['name']}): {metrics['avg_improvement']:+.2f}%")

    print("\n2. Best Performing Scale:")
    best_chart = max(range(1, 5), key=lambda i: results[i]['avg_improvement'])
    print(f"   Chart Set {best_chart}: {v2_targets[best_chart]['name']}")
    print(f"   Improvement: {results[best_chart]['avg_improvement']:+.2f}%")

    print("\n3. Adaptive Parameter Impact:")
    small_scale = results[1]['avg_improvement']
    large_scale = results[3]['avg_improvement'] if results[3] else 0
    super_large_scale = results[4]['avg_improvement'] if results[4] else 0

    print(f"   Small scale (M=100):        {small_scale:+.2f}%")
    print(f"   Large scale (M=1000):       {large_scale:+.2f}%")
    print(f"   Super-large (M=1000-5000):  {super_large_scale:+.2f}%")

    if large_scale > -0.10 or super_large_scale > -0.50:
        print("\n   [SUCCESS] Adaptive parameters mitigate large-scale performance degradation!")

print("\n" + "=" * 80)
