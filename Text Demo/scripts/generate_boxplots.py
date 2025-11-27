#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
箱线图生成工具
============
生成算法稳定性对比的箱线图(Box Plot)

箱线图展示:
- 中位数
- 四分位数 (Q1, Q3)
- 离群值
- 算法稳定性 (IQR越小越稳定)
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent


class BoxPlotGenerator:
    """箱线图生成器"""

    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        初始化生成器

        参数:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / 'Text Demo' / 'RAW_data'
        else:
            self.data_dir = Path(data_dir)

        if output_dir is None:
            self.output_dir = PROJECT_ROOT / 'Text Demo' / 'boxplots'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

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

    def extract_final_values(self, data: Dict, metric: str) -> Dict:
        """
        提取每次运行的最终值

        参数:
            data: 原始数据
            metric: 指标名称

        返回:
            {算法名: [最终值列表]}
        """
        results = {}

        for algo_name, algo_data in data.items():
            if metric in algo_data:
                metric_data = algo_data[metric]

                # 提取最终值
                if isinstance(metric_data, list):
                    if len(metric_data) > 0 and isinstance(metric_data[0], list):
                        # 每次运行是一个列表,取最后一个值
                        final_values = [run[-1] for run in metric_data]
                    else:
                        # 已经是最终值列表
                        final_values = metric_data
                else:
                    continue

                results[algo_name] = final_values

        return results

    def generate_boxplot(self,
                        chart_set_name: str,
                        metrics: List[str] = None,
                        algorithms: List[str] = None):
        """
        生成箱线图

        参数:
            chart_set_name: 图表集名称
            metrics: 要绘制的指标列表
            algorithms: 要对比的算法列表
        """
        if metrics is None:
            metrics = ['makespan', 'total_cost']

        if algorithms is None:
            algorithms = ['BCBO', 'BCBO-DE', 'GA', 'PSO', 'ACO', 'FA', 'CS', 'GWO']

        # 加载数据
        data = self.load_data(chart_set_name)

        # 创建子图
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(7.0 * n_metrics, 5.5))

        if n_metrics == 1:
            axes = [axes]

        metric_labels = {
            'makespan': 'Makespan (time units)',
            'total_cost': 'Total Cost (normalized)',
            'load_balance': 'Load Balance Index',
            'energy': 'Energy Consumption (kWh)'
        }

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # 提取数据
            metric_data = self.extract_final_values(data, metric)

            # 准备箱线图数据
            box_data = []
            box_labels = []

            for algo_name in algorithms:
                if algo_name in metric_data:
                    box_data.append(metric_data[algo_name])
                    box_labels.append(algo_name)

            if not box_data:
                print(f"警告: {metric}没有数据")
                continue

            # 绘制箱线图
            bp = ax.boxplot(box_data,
                           labels=box_labels,
                           patch_artist=True,
                           showmeans=True,
                           meanline=True,
                           widths=0.6)

            # 颜色配置
            colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                     '#9B59B6', '#1ABC9C', '#E67E22', '#95A5A6']

            # 设置箱体颜色
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)

            # 设置中位数线颜色
            for median in bp['medians']:
                median.set_color('#C0392B')
                median.set_linewidth(2.5)

            # 设置均值线颜色
            for mean in bp['means']:
                mean.set_color('#27AE60')
                mean.set_linewidth(2.0)
                mean.set_linestyle('--')

            # 设置须线颜色
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1.2)
                whisker.set_linestyle('--')

            # 设置帽子颜色
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.2)

            # 设置离群点
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markerfacecolor('red')
                flier.set_markersize(5)
                flier.set_alpha(0.5)

            # 图表装饰
            ax.set_ylabel(metric_labels.get(metric, metric),
                         fontsize=12, fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {metric.capitalize()} Distribution',
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_xticklabels(box_labels, rotation=45, ha='right')

            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#C0392B', linewidth=2.5, label='Median'),
                Line2D([0], [0], color='#27AE60', linewidth=2.0,
                      linestyle='--', label='Mean'),
                Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='red', markersize=7, label='Outliers')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            # 计算并显示统计信息
            print(f"\n[{metric}] 稳定性分析:")
            for algo_name, values in zip(box_labels, box_data):
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                median = np.median(values)
                mean = np.mean(values)
                std = np.std(values, ddof=1)

                print(f"  {algo_name:12s}: Median={median:.2f}, "
                     f"Mean={mean:.2f}±{std:.2f}, IQR={iqr:.2f}")

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{chart_set_name}_boxplot_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[保存] 箱线图: {output_path}")

        return str(output_path)

    def generate_statistical_summary(self,
                                    chart_set_name: str,
                                    metric: str = 'makespan') -> str:
        """
        生成统计摘要表格

        参数:
            chart_set_name: 图表集名称
            metric: 指标名称

        返回:
            Markdown格式的表格
        """
        data = self.load_data(chart_set_name)
        metric_data = self.extract_final_values(data, metric)

        md = f"# {chart_set_name} - {metric}稳定性分析\n\n"
        md += "| 算法 | 中位数 | 均值±标准差 | Q1 | Q3 | IQR | 变异系数 | 稳定性 |\n"
        md += "|------|-------|-----------|-----|-----|-----|---------|-------|\n"

        for algo_name, values in metric_data.items():
            values_arr = np.array(values)

            median = np.median(values_arr)
            mean = np.mean(values_arr)
            std = np.std(values_arr, ddof=1)
            q1 = np.percentile(values_arr, 25)
            q3 = np.percentile(values_arr, 75)
            iqr = q3 - q1
            cv = (std / mean) * 100 if mean != 0 else 0  # 变异系数

            # 判断稳定性
            if cv < 5:
                stability = "优秀"
            elif cv < 10:
                stability = "良好"
            elif cv < 15:
                stability = "一般"
            else:
                stability = "较差"

            md += f"| {algo_name} | {median:.2f} | {mean:.2f}±{std:.2f} | "
            md += f"{q1:.2f} | {q3:.2f} | {iqr:.2f} | {cv:.2f}% | {stability} |\n"

        md += "\n**说明**:\n"
        md += "- **IQR (Interquartile Range)**: 四分位距,越小表示数据越集中\n"
        md += "- **变异系数 (CV)**: 标准差/均值,越小表示越稳定\n"
        md += "- **稳定性评级**: CV<5%为优秀, 5-10%为良好, 10-15%为一般, >15%为较差\n"

        return md

    def generate_all_boxplots(self):
        """生成所有图表集的箱线图"""
        chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']

        for chart_set in chart_sets:
            print(f"\n{'='*80}")
            print(f"生成 {chart_set} 箱线图".center(80))
            print(f"{'='*80}")

            try:
                self.generate_boxplot(chart_set)

                # 生成统计摘要
                summary = self.generate_statistical_summary(chart_set, 'makespan')
                summary_path = self.output_dir / f"{chart_set}_statistical_summary.md"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"[保存] 统计摘要: {summary_path}")

            except FileNotFoundError as e:
                print(f"错误: {e}")
                continue


def main():
    """主函数"""
    print("="*80)
    print("箱线图生成工具".center(80))
    print("="*80)

    generator = BoxPlotGenerator()
    generator.generate_all_boxplots()

    print("\n" + "="*80)
    print("箱线图生成完成!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
