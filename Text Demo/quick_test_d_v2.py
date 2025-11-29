#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案D v2快速验证测试
仅测试M=1000-3000，快速验证效果
"""

import sys
import os
import json
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
scripts_dir = os.path.join(current_dir, 'scripts')

sys.path.insert(0, project_root)
sys.path.insert(0, scripts_dir)

from scripts.real_algorithm_integration import RealAlgorithmIntegrator

def quick_test_solution_d_v2():
    """快速测试方案D v2 (M=1000-3000)"""
    print("="*80)
    print("方案D v2快速验证 (M=1000-3000)")
    print("="*80)
    print("权重: beta=500, gamma=10")
    print("阈值: mid_elites=0.85, top_elites=0.90")
    print("="*80)

    integrator = RealAlgorithmIntegrator()

    # 测试配置 (仅M=1000-3000，快速验证)
    test_configs = [
        {'M': 1000, 'N': 20, 'n': 100, 'iterations': 30},
        {'M': 2000, 'N': 20, 'n': 150, 'iterations': 30},
        {'M': 3000, 'N': 20, 'n': 200, 'iterations': 30},
    ]

    results_summary = []

    # 旧版本对比数据 (来自analyze_chart_set_4.py)
    old_version_lb = {
        1000: 0.8941,
        2000: 0.8435,
        3000: 0.7977
    }

    for config in test_configs:
        M = config['M']
        print(f"\n{'='*80}")
        print(f"测试场景: M={M}, N={config['N']}, iterations={config['iterations']}")
        print(f"{'='*80}")

        params = config.copy()
        params['random_seed'] = 42

        # 运行BCBO
        print(f"\n[BCBO] 运行中...")
        bcbo_result = integrator.run_algorithm('BCBO', params)

        # 运行BCBO-DE (方案D v2)
        print(f"[BCBO-DE v2] 运行中...")
        bcbode_result = integrator.run_algorithm('BCBO-DE', params)

        if bcbo_result and bcbode_result:
            cost_diff = (bcbode_result['total_cost'] - bcbo_result['total_cost']) / bcbo_result['total_cost'] * 100
            lb_diff = (bcbode_result['load_balance'] - bcbo_result['load_balance']) / bcbo_result['load_balance'] * 100

            old_lb = old_version_lb.get(M, 0)
            improvement = bcbode_result['load_balance'] - old_lb

            result = {
                'M': M,
                'bcbo_cost': bcbo_result['total_cost'],
                'bcbo_lb': bcbo_result['load_balance'],
                'bcbode_cost': bcbode_result['total_cost'],
                'bcbode_lb': bcbode_result['load_balance'],
                'cost_diff_pct': cost_diff,
                'lb_diff_pct': lb_diff,
                'old_lb': old_lb,
                'improvement': improvement
            }
            results_summary.append(result)

            print(f"\n结果对比 (M={M}):")
            print(f"  BCBO        - 成本: {bcbo_result['total_cost']:.2f}, 负载均衡: {bcbo_result['load_balance']:.4f}")
            print(f"  BCBO-DE v2  - 成本: {bcbode_result['total_cost']:.2f}, 负载均衡: {bcbode_result['load_balance']:.4f}")
            print(f"  vs BCBO     - 成本: {cost_diff:+.2f}%, 负载均衡: {lb_diff:+.2f}%")
            print(f"  vs 旧版本   - 改善: {improvement:+.4f} ({improvement/old_lb*100:+.2f}%)")

    # 生成对比报告
    print(f"\n{'='*80}")
    print("方案D v2综合验证报告")
    print(f"{'='*80}")
    print(f"{'M':>6} | {'BCBO LB':>10} | {'BCBO-DE v2':>12} | {'vs BCBO':>10} | {'旧版本LB':>10} | {'改善幅度':>10}")
    print("-"*80)

    for r in results_summary:
        print(f"{r['M']:6d} | {r['bcbo_lb']:10.4f} | {r['bcbode_lb']:12.4f} | {r['lb_diff_pct']:+9.2f}% | {r['old_lb']:10.4f} | {r['improvement']:+9.4f}")

    # 评估效果
    print(f"\n{'='*80}")
    print("方案D v2效果评估")
    print(f"{'='*80}")

    avg_cost_diff = sum(r['cost_diff_pct'] for r in results_summary) / len(results_summary)
    avg_lb_diff = sum(r['lb_diff_pct'] for r in results_summary) / len(results_summary)
    avg_improvement = sum(r['improvement'] for r in results_summary) / len(results_summary)

    all_lb_above_85 = all(r['bcbode_lb'] >= 0.85 for r in results_summary)
    all_lb_drop_below_5 = all(abs(r['lb_diff_pct']) < 5 for r in results_summary)
    all_lb_drop_below_10 = all(abs(r['lb_diff_pct']) < 10 for r in results_summary)
    all_cost_reasonable = all(abs(r['cost_diff_pct']) < 3 for r in results_summary)

    print(f"平均成本差异: {avg_cost_diff:+.2f}%")
    print(f"平均负载均衡差异: {avg_lb_diff:+.2f}%")
    print(f"平均改善幅度: {avg_improvement:+.4f} ({avg_improvement/sum(r['old_lb'] for r in results_summary)*len(results_summary)*100:+.2f}%)")

    print(f"\n理想发表标准:")
    print(f"  [{'OK' if all_lb_above_85 else 'FAIL'}] 所有场景负载均衡 >= 0.85")
    print(f"  [{'OK' if all_lb_drop_below_5 else 'FAIL'}] 所有场景负载均衡降幅 < 5%")
    print(f"  [{'OK' if all_cost_reasonable else 'WARN'}] 所有场景成本差异 < 3%")

    print(f"\n可接受发表标准:")
    print(f"  [{'OK' if all_lb_drop_below_10 else 'FAIL'}] 所有场景负载均衡降幅 < 10%")
    print(f"  负载均衡范围: {min(r['bcbode_lb'] for r in results_summary):.4f} - {max(r['bcbode_lb'] for r in results_summary):.4f}")

    if all_lb_above_85 and all_lb_drop_below_5:
        print(f"\n[SUCCESS] 方案D v2完全达标！")
        print(f"[RECOMMENDATION] 可以生成完整数据用于论文发表")
    elif all_lb_drop_below_10:
        print(f"\n[PARTIAL SUCCESS] 方案D v2基本达标")
        print(f"[RECOMMENDATION] 可以发表，但需说明权衡")
    else:
        print(f"\n[FAIL] 方案D v2仍未达标")
        print(f"[RECOMMENDATION] 考虑方案D v3 (beta=1000, gamma=5)")

    # 保存结果
    output_file = os.path.join(current_dir, 'RAW_data', 'solution_d_v2_quick_test.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output_data = {
        'test_date': datetime.now().isoformat(),
        'strategy': 'Solution D v2: Aggressive Balance-Oriented Fitness (Quick Test)',
        'weights': {'alpha': 0.001, 'beta': 500, 'gamma': 10},
        'thresholds': {'mid_elites': 0.85, 'top_elites': 0.90},
        'test_configs': test_configs,
        'results': results_summary,
        'summary': {
            'avg_cost_diff_pct': avg_cost_diff,
            'avg_lb_diff_pct': avg_lb_diff,
            'avg_improvement': avg_improvement,
            'all_lb_above_85': all_lb_above_85,
            'all_lb_drop_below_5': all_lb_drop_below_5,
            'all_lb_drop_below_10': all_lb_drop_below_10,
            'all_cost_reasonable': all_cost_reasonable
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] 验证结果已保存: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    quick_test_solution_d_v2()
