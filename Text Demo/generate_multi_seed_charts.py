#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种子验证数据可视化工具
========================================
为三种子数据生成带误差线的专业对比图表

图表类型：
1. 迭代收敛曲线（均值±标准差阴影）
2. 最终迭代对比柱状图（带误差棒）
3. 箱线图（显示数据分布）

Author: Multi-seed Validation Team
Date: 2025-12-02
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(BASE_DIR, 'multi_seed_validation')
OUTPUT_DIR = os.path.join(VALIDATION_DIR, 'charts')


class MultiSeedChartGenerator:
    """多种子数据图表生成器"""

    def __init__(self, stats_json_path=None):
        """
        初始化图表生成器

        参数:
            stats_json_path: 统计数据JSON文件路径
        """
        if stats_json_path is None:
            stats_json_path = os.path.join(VALIDATION_DIR, 'multi_seed_statistics.json')

        self.stats_json_path = stats_json_path
        self.stats_data = None
        self.output_dir = OUTPUT_DIR

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def load_statistics(self):
        """加载统计数据"""
        print(f"\n[INFO] 加载统计数据: {self.stats_json_path}")

        try:
            with open(self.stats_json_path, 'r', encoding='utf-8') as f:
                self.stats_data = json.load(f)

            num_iterations = len(self.stats_data['statistics'])
            print(f"  [OK] 已加载 {num_iterations} 个迭代的统计数据")
            return True

        except Exception as e:
            print(f"[ERROR] 加载失败: {e}")
            return False

    def plot_convergence_with_errorbars(self, metric='best_fitness'):
        """
        绘制收敛曲线（均值±标准差阴影）

        参数:
            metric: 要绘制的指标名称
        """
        print(f"\n[INFO] 绘制收敛曲线: {metric}")

        if not self.stats_data:
            print(f"[ERROR] 未加载统计数据")
            return None

        # 提取数据
        iterations = []
        bcbo_means = []
        bcbo_stds = []
        bcbo_ga_means = []
        bcbo_ga_stds = []

        for iter_str, iter_data in sorted(self.stats_data['statistics'].items(), key=lambda x: int(x[0])):
            iter_num = int(iter_str)
            iterations.append(iter_num)

            bcbo_means.append(iter_data['BCBO'][metric]['mean'])
            bcbo_stds.append(iter_data['BCBO'][metric]['std'])

            bcbo_ga_means.append(iter_data['BCBO-GA'][metric]['mean'])
            bcbo_ga_stds.append(iter_data['BCBO-GA'][metric]['std'])

        iterations = np.array(iterations)
        bcbo_means = np.array(bcbo_means)
        bcbo_stds = np.array(bcbo_stds)
        bcbo_ga_means = np.array(bcbo_ga_means)
        bcbo_ga_stds = np.array(bcbo_ga_stds)

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 8))

        # BCBO曲线
        ax.plot(iterations, bcbo_means, 'b-', linewidth=2, label='BCBO', alpha=0.8)
        ax.fill_between(iterations,
                        bcbo_means - bcbo_stds,
                        bcbo_means + bcbo_stds,
                        alpha=0.2, color='blue', label='BCBO ±1 std')

        # BCBO-GA曲线
        ax.plot(iterations, bcbo_ga_means, 'r-', linewidth=2, label='BCBO-GA', alpha=0.8)
        ax.fill_between(iterations,
                        bcbo_ga_means - bcbo_ga_stds,
                        bcbo_ga_means + bcbo_ga_stds,
                        alpha=0.2, color='red', label='BCBO-GA ±1 std')

        # 设置标签和标题
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')

        metric_labels = {
            'best_fitness': 'Best Fitness',
            'total_cost': 'Total Cost',
            'execution_time': 'Execution Time (s)',
            'load_balance': 'Load Balance'
        }
        ylabel = metric_labels.get(metric, metric)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

        title = f'{ylabel} Convergence (Mean ± Std, n={self.stats_data["num_seeds"]} seeds)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 保存
        filename = f'convergence_{metric}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 已保存: {filename}")
        return filepath

    def plot_final_comparison_bars(self, iteration_num=100):
        """
        绘制最终迭代对比柱状图（带误差棒）

        参数:
            iteration_num: 迭代编号
        """
        print(f"\n[INFO] 绘制最终迭代对比柱状图 (Iteration {iteration_num})")

        if not self.stats_data:
            print(f"[ERROR] 未加载统计数据")
            return None

        iter_str = str(iteration_num)
        if iter_str not in self.stats_data['statistics']:
            print(f"[ERROR] 迭代 {iteration_num} 无数据")
            return None

        iter_data = self.stats_data['statistics'][iter_str]

        # 指标列表
        metrics = ['total_cost', 'execution_time', 'load_balance']
        metric_labels = ['Total Cost', 'Execution Time (s)', 'Load Balance']

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            # 提取数据
            bcbo_mean = iter_data['BCBO'][metric]['mean']
            bcbo_std = iter_data['BCBO'][metric]['std']
            bcbo_ga_mean = iter_data['BCBO-GA'][metric]['mean']
            bcbo_ga_std = iter_data['BCBO-GA'][metric]['std']

            # 柱状图
            x = np.array([0, 1])
            means = [bcbo_mean, bcbo_ga_mean]
            stds = [bcbo_std, bcbo_ga_std]
            colors = ['#2E86C1', '#E74C3C']

            bars = ax.bar(x, means, yerr=stds, capsize=10,
                         color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            # 设置标签
            ax.set_xticks(x)
            ax.set_xticklabels(['BCBO', 'BCBO-GA'], fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')

            # 添加数值标注
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02 * mean, f'{mean:.2f}±{std:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.suptitle(f'Final Iteration ({iteration_num}) Performance Comparison (n={self.stats_data["num_seeds"]} seeds)',
                    fontsize=16, fontweight='bold', y=1.02)

        # 保存
        filename = f'final_comparison_bars_iter{iteration_num}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 已保存: {filename}")
        return filepath

    def plot_boxplots(self, seeds=[42, 43, 44], iteration_num=100):
        """
        绘制箱线图（显示数据分布）

        参数:
            seeds: 种子列表
            iteration_num: 迭代编号
        """
        print(f"\n[INFO] 绘制箱线图 (Iteration {iteration_num})")

        # 加载原始数据（非统计数据）
        raw_data = {}
        for seed in seeds:
            filepath = os.path.join(VALIDATION_DIR, f'chart_set_1_seed_{seed}.json')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_data[seed] = data
            except Exception as e:
                print(f"  [WARNING] 加载seed {seed}失败: {e}")
                continue

        if len(raw_data) == 0:
            print(f"[ERROR] 无原始数据")
            return None

        # 提取最终迭代数据
        metrics = ['total_cost', 'execution_time', 'load_balance']
        metric_labels = ['Total Cost', 'Execution Time (s)', 'Load Balance']

        bcbo_data = {m: [] for m in metrics}
        bcbo_ga_data = {m: [] for m in metrics}

        for seed, data in raw_data.items():
            bcbo_results = data['algorithms']['BCBO']['results']
            bcbo_ga_results = data['algorithms']['BCBO-GA']['results']

            # 获取最终迭代
            bcbo_final = bcbo_results[iteration_num - 1] if len(bcbo_results) >= iteration_num else bcbo_results[-1]
            bcbo_ga_final = bcbo_ga_results[iteration_num - 1] if len(bcbo_ga_results) >= iteration_num else bcbo_ga_results[-1]

            for metric in metrics:
                bcbo_data[metric].append(bcbo_final[metric])
                bcbo_ga_data[metric].append(bcbo_ga_final[metric])

        # 绘制箱线图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            # 准备数据
            box_data = [bcbo_data[metric], bcbo_ga_data[metric]]

            # 箱线图
            bp = ax.boxplot(box_data, labels=['BCBO', 'BCBO-GA'],
                           patch_artist=True, widths=0.6,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))

            # 着色
            colors = ['#2E86C1', '#E74C3C']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # 添加散点（原始数据点）
            for i, data in enumerate(box_data):
                x = np.random.normal(i+1, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.6, s=80, c='black', zorder=10)

            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.suptitle(f'Data Distribution Boxplots (Iteration {iteration_num}, n={len(seeds)} seeds)',
                    fontsize=16, fontweight='bold', y=1.02)

        # 保存
        filename = f'boxplots_iter{iteration_num}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 已保存: {filename}")
        return filepath

    def generate_all_charts(self):
        """生成所有图表"""
        print("\n" + "="*80)
        print("开始生成多种子验证图表")
        print("="*80)

        generated_files = []

        # 1. 收敛曲线
        for metric in ['best_fitness', 'total_cost', 'execution_time', 'load_balance']:
            filepath = self.plot_convergence_with_errorbars(metric=metric)
            if filepath:
                generated_files.append(filepath)

        # 2. 最终迭代柱状图
        filepath = self.plot_final_comparison_bars(iteration_num=100)
        if filepath:
            generated_files.append(filepath)

        # 3. 箱线图
        filepath = self.plot_boxplots(seeds=[42, 43, 44], iteration_num=100)
        if filepath:
            generated_files.append(filepath)

        print("\n" + "="*80)
        print(f"[SUCCESS] 图表生成完成！共生成 {len(generated_files)} 个文件")
        print("="*80)
        print(f"输出目录: {self.output_dir}")
        for f in generated_files:
            print(f"  - {os.path.basename(f)}")
        print("="*80)

        return generated_files


def main():
    """主函数"""
    print("="*80)
    print("多种子验证数据可视化工具")
    print("="*80)

    # 创建图表生成器
    generator = MultiSeedChartGenerator()

    # 加载统计数据
    if not generator.load_statistics():
        print("\n[ERROR] 统计数据加载失败，退出")
        return 1

    # 生成所有图表
    generated_files = generator.generate_all_charts()

    return 0 if len(generated_files) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
