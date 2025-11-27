#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期刊论文数据质量分析工具
========================================
分析单次运行的实验数据是否符合SCI Q3/Q4期刊发表标准

评估指标：
1. 性能改进幅度（是否显著）
2. 数据可靠性（单次vs多次运行）
3. 统计显著性要求
4. 收敛性分析
5. 对比基准算法的竞争力
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

def analyze_single_run_data(data_file):
    """分析单次运行数据的质量"""

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("="*80)
    print("期刊论文数据质量分析报告")
    print("="*80)
    print(f"数据文件: {data_file}")
    print(f"配置信息: {data['config']['name']}")
    print(f"运行次数: {data['config']['runs_per_point']}次")
    print()

    # 提取算法结果
    algorithms = data['algorithms']
    bcbo_results = algorithms['BCBO']['results']
    bcbo_de_results = algorithms['BCBO-DE']['results']

    # ===== 1. 性能改进分析 =====
    print("="*80)
    print("1. 性能改进分析")
    print("="*80)

    # 分析不同迭代点的改进
    analysis_points = [
        ('早期 (iter=20)', min(19, len(bcbo_results)-1)),
        ('中期 (iter=50)', min(49, len(bcbo_results)-1)),
        ('后期 (iter=80)', min(79, len(bcbo_results)-1)),
        ('最终 (iter=100)', len(bcbo_results)-1)
    ]

    improvements = []
    for label, idx in analysis_points:
        if idx < len(bcbo_results):
            bcbo_val = bcbo_results[idx]['execution_time']
            bcbo_de_val = bcbo_de_results[idx]['execution_time']
            improvement = (bcbo_val - bcbo_de_val) / bcbo_val * 100
            improvements.append(improvement)

            status = "[PASS]" if improvement >= 8 else ("[OK]" if improvement >= 5 else ("[WARN]" if improvement >= 3 else "[FAIL]"))

            print(f"{label:15s}: BCBO={bcbo_val:7.2f}, BCBO-DE={bcbo_de_val:7.2f}, "
                  f"改进={improvement:+6.2f}% {status}")

    avg_improvement = np.mean(improvements)
    print(f"\n平均改进幅度: {avg_improvement:+.2f}%")
    print()

    # ===== 2. 期刊发表标准评估 =====
    print("="*80)
    print("2. 期刊发表标准评估")
    print("="*80)

    criteria = []

    # 2.1 性能改进幅度
    print("\n【标准1】性能改进幅度")
    if avg_improvement >= 8:
        print(f"  ✓ 优秀 - 改进{avg_improvement:.2f}% >= 8%（Q3/Q4期刊要求>=5%）")
        criteria.append(("性能改进", "优秀", True))
    elif avg_improvement >= 5:
        print(f"  ✓ 合格 - 改进{avg_improvement:.2f}% >= 5%（达到Q3/Q4期刊最低要求）")
        criteria.append(("性能改进", "合格", True))
    elif avg_improvement >= 3:
        print(f"  ⚠ 边缘 - 改进{avg_improvement:.2f}% >= 3%（可能需要补充实验）")
        criteria.append(("性能改进", "边缘", False))
    else:
        print(f"  ✗ 不足 - 改进{avg_improvement:.2f}% < 3%（不符合期刊要求）")
        criteria.append(("性能改进", "不足", False))

    # 2.2 数据可靠性（单次运行）
    print("\n【标准2】数据可靠性")
    runs = data['config']['runs_per_point']
    if runs >= 30:
        print(f"  ✓ 优秀 - {runs}次独立运行（满足统计学要求）")
        criteria.append(("数据可靠性", "优秀", True))
    elif runs >= 20:
        print(f"  ✓ 良好 - {runs}次独立运行（基本满足要求）")
        criteria.append(("数据可靠性", "良好", True))
    elif runs >= 10:
        print(f"  ⚠ 勉强 - {runs}次独立运行（建议增加到30次）")
        criteria.append(("数据可靠性", "勉强", False))
    else:
        print(f"  ✗ 不足 - {runs}次运行（无法进行统计显著性检验）")
        print(f"     ⚠ 警告: 单次运行数据无法：")
        print(f"        - 计算标准差和置信区间")
        print(f"        - 进行Wilcoxon秩和检验")
        print(f"        - 验证结果可重复性")
        print(f"     ⚠ 建议: 必须增加到30次运行才能发表")
        criteria.append(("数据可靠性", "严重不足", False))

    # 2.3 收敛性分析
    print("\n【标准3】收敛性分析")
    # 检查最后10次迭代的改进幅度
    if len(bcbo_de_results) >= 10:
        last_10_bcbo_de = [bcbo_de_results[i]['execution_time'] for i in range(-10, 0)]
        convergence_improvement = (max(last_10_bcbo_de) - min(last_10_bcbo_de)) / max(last_10_bcbo_de) * 100

        if convergence_improvement < 1:
            print(f"  ✓ 良好收敛 - 最后10次迭代波动 {convergence_improvement:.2f}% < 1%")
            criteria.append(("收敛性", "良好", True))
        elif convergence_improvement < 5:
            print(f"  ⚠ 收敛较慢 - 最后10次迭代波动 {convergence_improvement:.2f}%")
            criteria.append(("收敛性", "一般", True))
        else:
            print(f"  ✗ 未收敛 - 最后10次迭代波动 {convergence_improvement:.2f}% >= 5%")
            criteria.append(("收敛性", "差", False))

    # 2.4 对比其他算法
    print("\n【标准4】与其他基准算法对比")
    other_algos = ['GA', 'PSO', 'ACO', 'FA', 'CS', 'GWO']
    bcbo_de_final = bcbo_de_results[-1]['execution_time']

    better_count = 0
    for algo in other_algos:
        if algo in algorithms:
            algo_final = algorithms[algo]['results'][-1]['execution_time']
            if bcbo_de_final < algo_final:
                better_count += 1
                improvement_vs = (algo_final - bcbo_de_final) / algo_final * 100
                print(f"  ✓ BCBO-DE优于{algo}: {improvement_vs:+.2f}%")

    if better_count >= 5:
        print(f"\n  ✓ 优秀 - BCBO-DE优于{better_count}/{len(other_algos)}个基准算法")
        criteria.append(("算法竞争力", "优秀", True))
    elif better_count >= 3:
        print(f"\n  ⚠ 一般 - BCBO-DE优于{better_count}/{len(other_algos)}个基准算法")
        criteria.append(("算法竞争力", "一般", True))
    else:
        print(f"\n  ✗ 不足 - BCBO-DE仅优于{better_count}/{len(other_algos)}个基准算法")
        criteria.append(("算法竞争力", "不足", False))

    print()

    # ===== 3. 总体评估 =====
    print("="*80)
    print("3. 总体评估与建议")
    print("="*80)

    passed = sum(1 for _, _, p in criteria if p)
    total = len(criteria)

    print(f"\n通过标准: {passed}/{total}")
    print("\n详细评分:")
    for name, rating, passed_flag in criteria:
        status = "✓" if passed_flag else "✗"
        print(f"  {status} {name:12s}: {rating}")

    print("\n" + "="*80)
    print("最终结论")
    print("="*80)

    # 判断是否可以发表
    critical_pass = False
    reliability_pass = False

    for name, rating, passed_flag in criteria:
        if name == "性能改进" and passed_flag:
            critical_pass = True
        if name == "数据可靠性" and passed_flag:
            reliability_pass = True

    if runs == 1:
        print("\n❌ 当前状态: 不适合发表")
        print("\n【关键问题】")
        print("  • 只有1次运行，无法进行统计分析")
        print("  • 缺少标准差、置信区间等关键指标")
        print("  • 无法证明结果可重复性")
        print("  • 无法通过同行评审的统计学要求")

        print("\n【必须改进】")
        print("  1. ⚠ 紧急: 将runs_per_point改为30")
        print("  2. ⚠ 紧急: 重新运行所有chart_set生成多次运行数据")
        print("  3. ⚠ 紧急: 添加统计显著性检验（Wilcoxon test）")
        print("  4. ⚠ 紧急: 计算并报告标准差和95%置信区间")

        print("\n【单次运行的价值】")
        print("  ✓ 可用于快速参数调优")
        print("  ✓ 可验证算法基本功能")
        print("  ✓ 可初步评估改进方向")
        print("  ✗ 不能用于期刊论文发表")

        print("\n【时间估算】")
        print(f"  • 单次运行耗时: ~100秒")
        print(f"  • 30次运行耗时: ~50分钟（30x100秒）")
        print(f"  • 4个chart_set总耗时: ~3-4小时")

        print("\n【下一步行动】")
        print("  1. 修改generate_data_for_charts_optimized.py")
        print("     将所有'runs_per_point': 1 改为 'runs_per_point': 30")
        print("  2. 运行: python generate_data_for_charts_optimized.py --all")
        print("  3. 运行统计分析脚本生成Wilcoxon检验结果")
        print("  4. 生成带置信区间的论文图表")

    elif critical_pass and reliability_pass:
        print("\n✓ 当前状态: 基本符合期刊发表要求")
        print("\n【优势】")
        print(f"  • 性能改进显著 ({avg_improvement:.2f}%)")
        print(f"  • 数据可靠性充足 ({runs}次运行)")
        print("  • 可进行统计显著性检验")

        print("\n【建议补充】")
        print("  1. 进行Wilcoxon秩和检验")
        print("  2. 计算Cohen's d效应量")
        print("  3. 添加箱线图展示数据分布")
        print("  4. 进行消融实验验证各组件贡献")

    else:
        print("\n⚠ 当前状态: 需要改进后才能发表")
        print("\n【需要改进的方面】")
        for name, rating, passed_flag in criteria:
            if not passed_flag:
                print(f"  • {name}: {rating}")

    print("\n" + "="*80)
    print()

    return {
        'average_improvement': avg_improvement,
        'criteria': criteria,
        'passed': passed,
        'total': total,
        'runs': runs,
        'publishable': critical_pass and reliability_pass
    }

if __name__ == '__main__':
    data_file = Path('RAW_data/chart_set_1_merged_results.json')

    if not data_file.exists():
        print(f"错误: 找不到数据文件 {data_file}")
        print("请先运行 generate_data_for_charts_optimized.py 生成数据")
    else:
        result = analyze_single_run_data(data_file)

        # 保存分析结果
        output = {
            'data_file': str(data_file),
            'results': result
        }

        with open('data_quality_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("分析结果已保存到: data_quality_analysis.json")
