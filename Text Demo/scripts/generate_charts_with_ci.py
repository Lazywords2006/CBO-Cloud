#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强图表生成器 - 添加置信区间
==============================
在收敛曲线中添加95%置信区间阴影,展示算法稳定性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# 获取项目根目录
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent


class EnhancedChartGenerator:
    """带置信区间的图表生成器"""

    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        初始化生成器

        参数:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / 'RAW_data'
        else:
            self.data_dir = Path(data_dir)

        if output_dir is None:
            self.output_dir = PROJECT_ROOT / 'publication_charts_with_ci'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 算法颜色配置
        self.algo_colors = {
            'BCBO': '#377eb8',
            'BCBO-DE': '#e41a1c',
            'GA': '#4daf4a',
            'PSO': '#984ea3',
            'ACO': '#ff7f00',
            'FA': '#ffff33',
            'CS': '#a65628',
            'GWO': '#f781bf'
        }

    def load_data(self, chart_set_name: str) -> Dict:
        """
        加载数据

        参数:
            chart_set_name: 图表集名称

        返回:
            数据字典
        """
        json_file = self.data_dir / f"{chart_set_name}_merged_results.json"

        if not json_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def calculate_confidence_interval(self,
                                     data: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算置信区间

        参数:
            data: 数据数组
            confidence: 置信水平 (默认0.95)

        返回:
            (下界, 上界)
        """
        mean = np.mean(data)
        sem = stats.sem(data)  # 标准误差
        interval = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)

        return mean - interval, mean + interval

    def plot_convergence_with_ci(self,
                                 chart_set_name: str,
                                 metric: str = 'makespan',
                                 algorithms: List[str] = None,
                                 confidence: float = 0.95):
        """
        绘制带置信区间的收敛曲线

        参数:
            chart_set_name: 图表集名称
            metric: 指标名称
            algorithms: 算法列表
            confidence: 置信水平
        """
        if algorithms is None:
            algorithms = ['BCBO', 'BCBO-DE', 'GA', 'PSO', 'ACO']

        # 加载数据
        data = self.load_data(chart_set_name)

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_labels = {
            'makespan': 'Makespan (time units)',
            'total_cost': 'Total Cost (normalized)',
            'load_balance': 'Load Balance Index',
            'energy': 'Energy Consumption (kWh)'
        }

        for algo_name in algorithms:
            if algo_name not in data:
                continue

            algo_data = data[algo_name].get(metric, [])

            if not algo_data or len(algo_data) == 0:
                continue

            # 假设数据格式: [[run1_iter1, run1_iter2, ...], [run2_iter1, run2_iter2, ...], ...]
            # 转换为numpy数组: shape = (n_runs, n_iterations)
            if isinstance(algo_data[0], list):
                data_array = np.array(algo_data)
            else:
                # 如果是单次运行,扩展维度
                data_array = np.array([algo_data])

            n_runs, n_iterations = data_array.shape

            # 计算均值
            mean_values = np.mean(data_array, axis=0)

            # 计算置信区间
            ci_lower = []
            ci_upper = []

            for i in range(n_iterations):
                values_at_iter = data_array[:, i]
                lower, upper = self.calculate_confidence_interval(values_at_iter, confidence)
                ci_lower.append(lower)
                ci_upper.append(upper)

            ci_lower = np.array(ci_lower)
            ci_upper = np.array(ci_upper)

            iterations = np.arange(1, n_iterations + 1)

            # 绘制主曲线
            color = self.algo_colors.get(algo_name, '#000000')

            ax.plot(iterations, mean_values,
                   color=color,
                   linewidth=2.5 if algo_name == 'BCBO-DE' else 2.0,
                   label=algo_name,
                   zorder=10 if algo_name == 'BCBO-DE' else 5)

            # 填充置信区间
            ax.fill_between(iterations, ci_lower, ci_upper,
                           color=color,
                           alpha=0.2,
                           zorder=1)

            print(f"  {algo_name}: {n_runs} runs, {n_iterations} iterations")

        # 图表装饰
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
        ax.set_title(f'{chart_set_name} - {metric.capitalize()} Convergence with 95% CI',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

        # 添加说明文本
        textstr = f'Shaded areas represent {int(confidence*100)}% confidence intervals'
        ax.text(0.95, 0.05, textstr,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{chart_set_name}_{metric}_with_ci_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[保存] 置信区间图表: {output_path}")

        return str(output_path)

    def plot_4_metrics_with_ci(self, chart_set_name: str, algorithms: List[str] = None):
        """
        绘制4个指标的收敛曲线(带置信区间)

        参数:
            chart_set_name: 图表集名称
            algorithms: 算法列表
        """
        if algorithms is None:
            algorithms = ['BCBO', 'BCBO-DE', 'GA', 'PSO', 'ACO']

        metrics = ['makespan', 'total_cost', 'load_balance', 'energy']
        metric_labels = {
            'makespan': 'Makespan (time units)',
            'total_cost': 'Total Cost (normalized)',
            'load_balance': 'Load Balance Index',
            'energy': 'Energy Consumption (kWh)'
        }

        # 加载数据
        data = self.load_data(chart_set_name)

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{chart_set_name} - Multi-Metric Convergence with 95% CI',
                    fontsize=16, fontweight='bold')

        for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
            for algo_name in algorithms:
                if algo_name not in data:
                    continue

                algo_data = data[algo_name].get(metric, [])

                if not algo_data or len(algo_data) == 0:
                    continue

                # 转换数据
                if isinstance(algo_data[0], list):
                    data_array = np.array(algo_data)
                else:
                    data_array = np.array([algo_data])

                n_runs, n_iterations = data_array.shape

                # 计算均值和置信区间
                mean_values = np.mean(data_array, axis=0)

                ci_lower = []
                ci_upper = []

                for i in range(n_iterations):
                    values_at_iter = data_array[:, i]
                    lower, upper = self.calculate_confidence_interval(values_at_iter, 0.95)
                    ci_lower.append(lower)
                    ci_upper.append(upper)

                ci_lower = np.array(ci_lower)
                ci_upper = np.array(ci_upper)

                iterations = np.arange(1, n_iterations + 1)

                # 绘制
                color = self.algo_colors.get(algo_name, '#000000')

                ax.plot(iterations, mean_values,
                       color=color,
                       linewidth=2.5 if algo_name == 'BCBO-DE' else 1.8,
                       label=algo_name,
                       zorder=10 if algo_name == 'BCBO-DE' else 5)

                ax.fill_between(iterations, ci_lower, ci_upper,
                               color=color,
                               alpha=0.15,
                               zorder=1)

            # 图表装饰
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {metric.capitalize()}', fontsize=12)
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{chart_set_name}_4metrics_with_ci_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[保存] 4指标置信区间图表: {output_path}")

        return str(output_path)

    def generate_all_charts(self):
        """生成所有图表集的置信区间图表"""
        chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']

        for chart_set in chart_sets:
            print(f"\n{'='*80}")
            print(f"生成 {chart_set} 置信区间图表".center(80))
            print(f"{'='*80}")

            try:
                # 生成4指标综合图
                self.plot_4_metrics_with_ci(chart_set)

                # 生成单个makespan图表(更详细)
                self.plot_convergence_with_ci(chart_set, 'makespan')

            except FileNotFoundError as e:
                print(f"错误: {e}")
                continue


def main():
    """主函数"""
    print("="*80)
    print("增强图表生成器 - 添加置信区间".center(80))
    print("="*80)

    generator = EnhancedChartGenerator()
    generator.generate_all_charts()

    print("\n" + "="*80)
    print("置信区间图表生成完成!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
