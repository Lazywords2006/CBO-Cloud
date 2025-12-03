#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多种子验证结果可视化

生成4个图表集的综合对比可视化，包括：
1. 综合改进率对比柱状图
2. 统计显著性热力图
3. 性能分布箱线图
4. 规模-性能趋势图
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_set_statistics(set_number, base_dir_map):
    """加载指定图表集的统计数据"""
    import re

    base_dir = base_dir_map[set_number]
    report_file = f"{base_dir}/statistical_analysis_report_set{set_number}.txt"

    if not os.path.exists(report_file):
        print(f"[WARN] {report_file} 不存在")
        return None

    # 解析统计报告提取关键数据
    data = {
        'set_number': set_number,
        'cost_improvement': 0.0,
        'time_improvement': 0.0,
        'balance_improvement': 0.0,
        'comprehensive_improvement': 0.0,
        'cost_pvalue': 1.0,
        'time_pvalue': 1.0,
        'balance_pvalue': 1.0,
        'significant_count': 0
    }

    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式解析
    # 1. 综合改进率
    pattern = r'综合改进率:\s*([+-]?\d+\.\d+)%'
    match = re.search(pattern, content)
    if match:
        data['comprehensive_improvement'] = float(match.group(1))

    # 2. 各指标改进率
    pattern_cost = r'成本改进率:\s*([+-]?\d+\.\d+)%'
    pattern_time = r'时间改进率:\s*([+-]?\d+\.\d+)%'
    pattern_balance = r'负载均衡改进率:\s*([+-]?\d+\.\d+)%'

    cost_match = re.search(pattern_cost, content)
    time_match = re.search(pattern_time, content)
    balance_match = re.search(pattern_balance, content)

    if cost_match:
        data['cost_improvement'] = float(cost_match.group(1))
    if time_match:
        data['time_improvement'] = float(time_match.group(1))
    if balance_match:
        data['balance_improvement'] = float(balance_match.group(1))

    # 3. p值（从配对t检验结果表格中提取）
    # 格式: total_cost               -0.06%     0.9542       n.s.
    pattern_pvalue = r'(total_cost|execution_time|load_balance)\s+[+-]?\d+\.\d+%\s+(\d+\.\d+)\s+'
    for match in re.finditer(pattern_pvalue, content):
        metric = match.group(1)
        pvalue = float(match.group(2))

        if metric == 'total_cost':
            data['cost_pvalue'] = pvalue
        elif metric == 'execution_time':
            data['time_pvalue'] = pvalue
        elif metric == 'load_balance':
            data['balance_pvalue'] = pvalue

    # 4. 显著性检验通过率
    pattern_sig = r'显著性检验通过率:\s*(\d+)/3'
    sig_match = re.search(pattern_sig, content)
    if sig_match:
        data['significant_count'] = int(sig_match.group(1))

    return data


def load_all_statistics():
    """加载所有4个图表集的统计数据"""
    base_dir_map = {
        1: 'multi_seed_validation',
        2: 'multi_seed_validation_set2',
        3: 'multi_seed_validation_set3',
        4: 'multi_seed_validation_set4'
    }

    all_data = []
    for set_num in [1, 2, 3, 4]:
        data = load_set_statistics(set_num, base_dir_map)
        if data:
            all_data.append(data)

    return all_data


def plot_comprehensive_improvement_bar(all_data, output_dir):
    """绘制综合改进率对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    set_names = ['Set 1\n(M=100)', 'Set 2\n(M=100-1000)', 'Set 3\n(M=1000)', 'Set 4\n(M=5000)']
    improvements = [d['comprehensive_improvement'] for d in all_data]

    # 根据改进率设置颜色（正向=绿色，负向=红色）
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

    bars = ax.bar(range(len(set_names)), improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold')

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('图表集 (任务规模)', fontsize=14, fontweight='bold')
    ax.set_ylabel('综合改进率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('BCBO-GA vs BCBO 多种子验证综合改进率对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(set_names)))
    ax.set_xticklabels(set_names, fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加图例
    green_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='BCBO-GA优于BCBO')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='BCBO-GA劣于BCBO')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'comprehensive_improvement_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] 已保存: {output_file}")
    plt.close()


def plot_significance_heatmap(all_data, output_dir):
    """绘制统计显著性热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    metrics = ['Total Cost', 'Execution Time', 'Load Balance']
    set_names = ['Set 1\n(M=100)', 'Set 2\n(M=100-1k)', 'Set 3\n(M=1000)', 'Set 4\n(M=5000)']

    # 构建热力图数据矩阵 (p值)
    heatmap_data = np.zeros((len(metrics), len(all_data)))
    improvements = np.zeros((len(metrics), len(all_data)))

    for i, data in enumerate(all_data):
        heatmap_data[0, i] = data['cost_pvalue'] if data['cost_pvalue'] is not None else 1.0
        heatmap_data[1, i] = data['time_pvalue'] if data['time_pvalue'] is not None else 1.0
        heatmap_data[2, i] = data['balance_pvalue'] if data['balance_pvalue'] is not None else 1.0

        improvements[0, i] = data['cost_improvement'] if data['cost_improvement'] is not None else 0
        improvements[1, i] = data['time_improvement'] if data['time_improvement'] is not None else 0
        improvements[2, i] = data['balance_improvement'] if data['balance_improvement'] is not None else 0

    # 创建显著性标记矩阵（用于颜色映射）
    # 0: 不显著(n.s.), 1: *(p<0.05), 2: **(p<0.01), 3: ***(p<0.001)
    sig_levels = np.zeros_like(heatmap_data)
    sig_levels[heatmap_data < 0.05] = 1
    sig_levels[heatmap_data < 0.01] = 2
    sig_levels[heatmap_data < 0.001] = 3

    # 创建颜色映射（基于改进方向和显著性）
    # 绿色：正向改进，红色：负向改进，亮度：显著性
    color_matrix = np.zeros((*heatmap_data.shape, 3))
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            imp = improvements[i, j]
            sig = sig_levels[i, j]

            if imp > 0:
                # 绿色系
                intensity = 0.3 + sig * 0.2  # 0.3, 0.5, 0.7, 0.9
                color_matrix[i, j] = [0.2, intensity, 0.2]
            else:
                # 红色系
                intensity = 0.3 + sig * 0.2
                color_matrix[i, j] = [intensity, 0.2, 0.2]

    # 绘制热力图
    im = ax.imshow(color_matrix, aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(len(set_names)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(set_names, fontsize=11)
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')

    # 添加文本标注（改进率 + 显著性标记）
    for i in range(len(metrics)):
        for j in range(len(all_data)):
            imp = improvements[i, j]
            p_val = heatmap_data[i, j]

            # 显著性标记
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
            else:
                sig_marker = 'n.s.'

            text = f'{imp:+.2f}%\n{sig_marker}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white', fontsize=10, fontweight='bold')

    ax.set_title('BCBO-GA性能改进统计显著性热力图\n(绿色=改进, 红色=退化, 亮度=显著性)',
                fontsize=14, fontweight='bold', pad=15)

    # 添加颜色图例
    legend_elements = [
        mpatches.Patch(facecolor='#2d7a2d', label='*** (p<0.001, 高显著)'),
        mpatches.Patch(facecolor='#47a047', label='** (p<0.01, 显著)'),
        mpatches.Patch(facecolor='#61c661', label='* (p<0.05, 边界显著)'),
        mpatches.Patch(facecolor='#4d4d4d', label='n.s. (p≥0.05, 不显著)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'significance_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] 已保存: {output_file}")
    plt.close()


def plot_metric_breakdown(all_data, output_dir):
    """绘制各指标改进率分解图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    set_names = ['Set 1', 'Set 2', 'Set 3', 'Set 4']
    metrics = [
        ('cost_improvement', 'Total Cost Improvement', 'Cost (%)'),
        ('time_improvement', 'Execution Time Improvement', 'Time (%)'),
        ('balance_improvement', 'Load Balance Improvement', 'Balance (%)')
    ]

    for idx, (metric_key, title, ylabel) in enumerate(metrics):
        ax = axes[idx]

        improvements = [d[metric_key] for d in all_data]
        pvalues = [d[metric_key.replace('improvement', 'pvalue')] for d in all_data]

        # 根据显著性设置颜色
        colors = []
        for imp, p in zip(improvements, pvalues):
            if imp > 0:
                if p < 0.001:
                    colors.append('#1e7b1e')  # 深绿
                elif p < 0.01:
                    colors.append('#2ecc71')  # 中绿
                elif p < 0.05:
                    colors.append('#7dda7d')  # 浅绿
                else:
                    colors.append('#b8e6b8')  # 很浅绿
            else:
                if p < 0.001:
                    colors.append('#b71c1c')  # 深红
                elif p < 0.01:
                    colors.append('#e74c3c')  # 中红
                elif p < 0.05:
                    colors.append('#f08080')  # 浅红
                else:
                    colors.append('#f5b7b1')  # 很浅红

        bars = ax.bar(range(len(set_names)), improvements, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for i, (bar, imp, p) in enumerate(zip(bars, improvements, pvalues)):
            height = bar.get_height()
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:+.2f}%\n{sig}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Chart Set', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(set_names)))
        ax.set_xticklabels(set_names)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metric_breakdown.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] 已保存: {output_file}")
    plt.close()


def plot_comprehensive_dashboard(all_data, output_dir):
    """绘制综合仪表盘（4合1）"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 综合改进率柱状图 (左上)
    ax1 = fig.add_subplot(gs[0, :])
    set_names = ['Set 1\n(M=100)', 'Set 2\n(M=100-1k)', 'Set 3\n(M=1000)', 'Set 4\n(M=5000)']
    improvements = [d['comprehensive_improvement'] for d in all_data]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax1.bar(range(len(set_names)), improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Comprehensive Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) 综合改进率对比', fontsize=13, fontweight='bold', loc='left')
    ax1.set_xticks(range(len(set_names)))
    ax1.set_xticklabels(set_names, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 2. 显著性指标数 (右上)
    ax2 = fig.add_subplot(gs[1, 0])
    sig_counts = [d['significant_count'] for d in all_data]
    total_metrics = 3
    colors_sig = ['#27ae60' if count >= 2 else '#f39c12' if count == 1 else '#95a5a6' for count in sig_counts]
    bars = ax2.bar(range(len(set_names)), sig_counts, color=colors_sig, alpha=0.7, edgecolor='black')
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='显著性阈值 (2/3)')
    for bar, count in zip(bars, sig_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}/3', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Significant Metrics Count', fontsize=11, fontweight='bold')
    ax2.set_title('(B) 显著性指标数量', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xticks(range(len(set_names)))
    ax2.set_xticklabels(['Set 1', 'Set 2', 'Set 3', 'Set 4'], fontsize=9)
    ax2.set_ylim([0, 3.5])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. 各指标改进率雷达图 (右中)
    ax3 = fig.add_subplot(gs[1, 1], projection='polar')
    categories = ['Cost', 'Time', 'Balance']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 为每个Set绘制一条线
    set_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4']
    set_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    for i, (data, color, label) in enumerate(zip(all_data, set_colors, set_labels)):
        values = [
            data['cost_improvement'],
            data['time_improvement'],
            data['balance_improvement']
        ]
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax3.fill(angles, values, alpha=0.15, color=color)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_title('(C) 各指标改进率雷达图', fontsize=12, fontweight='bold', pad=20, loc='left', y=1.08)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax3.grid(True)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 4. 性能退化分析 (下方)
    ax4 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(len(set_names))
    width = 0.25

    cost_imps = [d['cost_improvement'] for d in all_data]
    time_imps = [d['time_improvement'] for d in all_data]
    balance_imps = [d['balance_improvement'] for d in all_data]

    bars1 = ax4.bar(x_pos - width, cost_imps, width, label='Cost', color='#3498db', alpha=0.7)
    bars2 = ax4.bar(x_pos, time_imps, width, label='Time', color='#2ecc71', alpha=0.7)
    bars3 = ax4.bar(x_pos + width, balance_imps, width, label='Balance', color='#f39c12', alpha=0.7)

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Chart Set', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) 各指标详细改进率对比', fontsize=13, fontweight='bold', loc='left')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(set_names, fontsize=10)
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(axis='y', alpha=0.3)

    # 添加总标题
    fig.suptitle('BCBO-GA vs BCBO 多种子验证综合仪表盘',
                fontsize=16, fontweight='bold', y=0.98)

    output_file = os.path.join(output_dir, 'comprehensive_dashboard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] 已保存: {output_file}")
    plt.close()


def main():
    """主函数"""
    print("\n" + "="*80)
    print("多种子验证结果可视化")
    print("="*80)
    print()

    # 创建输出目录
    output_dir = 'multi_seed_visualization'
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {output_dir}")
    print()

    # 1. 加载所有统计数据
    print("[INFO] 加载统计数据...")
    all_data = load_all_statistics()
    print(f"  [OK] 成功加载 {len(all_data)} 个图表集的数据")
    print()

    # 2. 生成可视化
    print("[INFO] 生成可视化图表...")

    print("  [1/4] 生成综合改进率对比柱状图...")
    plot_comprehensive_improvement_bar(all_data, output_dir)

    print("  [2/4] 生成统计显著性热力图...")
    plot_significance_heatmap(all_data, output_dir)

    print("  [3/4] 生成各指标改进率分解图...")
    plot_metric_breakdown(all_data, output_dir)

    print("  [4/4] 生成综合仪表盘...")
    plot_comprehensive_dashboard(all_data, output_dir)

    print()
    print("="*80)
    print("可视化完成")
    print("="*80)
    print(f"所有图表已保存到: {output_dir}/")
    print()
    print("生成的文件:")
    print("  - comprehensive_improvement_comparison.png  (综合改进率对比)")
    print("  - significance_heatmap.png                 (显著性热力图)")
    print("  - metric_breakdown.png                     (指标分解图)")
    print("  - comprehensive_dashboard.png              (综合仪表盘)")
    print()


if __name__ == '__main__':
    main()
