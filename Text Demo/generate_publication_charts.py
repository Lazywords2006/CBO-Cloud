#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多算法对比 - 论文级图表生成
=========================================
基于数据生成工具的RAW_data，生成高质量的多算法对比图表
支持: BCBO, GA, PSO, ACO, FA, CS, GWO, BCBO-GA-Hybrid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime
import glob

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'RAW_data')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'publication_charts')

# 设置中文字体和样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 算法配置
ALGORITHM_CONFIG = {
    'BCBO': {'color': '#1f77b4', 'label': 'BCBO', 'linestyle': '-', 'marker': 'o'},
    'GA': {'color': '#ff7f0e', 'label': 'GA', 'linestyle': '--', 'marker': 's'},
    'PSO': {'color': '#2ca02c', 'label': 'PSO', 'linestyle': '-.', 'marker': '^'},
    'ACO': {'color': '#d62728', 'label': 'ACO', 'linestyle': ':', 'marker': 'D'},
    'FA': {'color': '#9467bd', 'label': 'FA', 'linestyle': '-', 'marker': 'v'},
    'CS': {'color': '#8c564b', 'label': 'CS', 'linestyle': '--', 'marker': 'p'},
    'GWO': {'color': '#e377c2', 'label': 'GWO', 'linestyle': '-.', 'marker': '*'},
    'BCBO-DE': {'color': '#17becf', 'label': 'BCBO-DE', 'linestyle': '-', 'marker': 'H'},
}


class MultiAlgorithmChartGenerator:
    """多算法对比图表生成器"""

    def __init__(self, raw_data_dir, chart_set='chart_set_1', output_dir='publication_charts',
                 max_points=20):
        """
        初始化

        参数:
            raw_data_dir: RAW_data目录路径
            chart_set: 图表集名称 (chart_set_1, chart_set_2, chart_set_3, chart_set_4)
            output_dir: 输出目录
            max_points: 每条曲线的最大显示点数（默认20，让曲线更清晰）
        """
        self.raw_data_dir = raw_data_dir
        self.chart_set = chart_set
        self.output_dir = output_dir
        self.max_points = max_points
        os.makedirs(output_dir, exist_ok=True)

        # 加载所有算法数据
        self.algorithm_data = {}
        self._load_all_algorithms()

        print(f"数据加载完成 ({chart_set}):")
        for algo, data in self.algorithm_data.items():
            print(f"  {algo}: {len(data)} 数据点")

    def _load_all_algorithms(self):
        """加载所有算法的数据"""
        # 优先尝试加载合并文件
        merged_file = os.path.join(self.raw_data_dir, f'{self.chart_set}_merged_results.json')

        if os.path.exists(merged_file):
            print(f"找到合并文件: {merged_file}")
            try:
                with open(merged_file, 'r', encoding='utf-8') as f:
                    merged_data = json.load(f)
                    algorithms_data = merged_data.get('algorithms', {})

                    for algo_name, algo_info in algorithms_data.items():
                        results = algo_info.get('results', [])
                        if results:
                            self.algorithm_data[algo_name] = results
                            print(f"  从合并文件加载: {algo_name} ({len(results)} 数据点)")

                    if self.algorithm_data:
                        print(f"成功从合并文件加载 {len(self.algorithm_data)} 个算法")
                        return
            except Exception as e:
                print(f"警告: 无法加载合并文件: {e}")

        # 如果合并文件不存在或加载失败，尝试加载单独文件
        print(f"尝试加载单独的算法文件...")
        pattern = os.path.join(self.raw_data_dir, f'{self.chart_set}_*_results.json')
        json_files = glob.glob(pattern)

        # 排除合并文件
        json_files = [f for f in json_files if 'merged' not in os.path.basename(f)]

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    algorithm = data['algorithm']
                    results = data['results']
                    self.algorithm_data[algorithm] = results
                    print(f"  加载: {algorithm} ({len(results)} 数据点)")
            except Exception as e:
                print(f"警告: 无法加载 {json_file}: {e}")

        # 检查是否成功加载至少一个算法
        if not self.algorithm_data:
            error_msg = (
                f"\n{'='*80}\n"
                f"[错误] 在 {self.raw_data_dir} 中找不到任何算法数据文件\n"
                f"{'='*80}\n"
                f"可能的原因:\n"
                f"1. RAW_data 目录为空 - 请先运行数据生成脚本\n"
                f"2. 文件名格式不匹配 - 期望格式: {self.chart_set}_*.json\n"
                f"3. 路径错误 - 请检查数据目录路径是否正确\n"
                f"\n建议操作:\n"
                f"  cd 'Text Demo'\n"
                f"  python generate_data_for_charts_optimized.py --chart-set 1\n"
                f"{'='*80}\n"
            )
            raise RuntimeError(error_msg)

    def _downsample_data(self, data, max_points=None):
        """
        对数据进行下采样，减少显示的点数

        策略：
        - 总是保留第一个和最后一个点
        - 在中间均匀采样
        - 确保关键转折点被保留

        参数:
            data: 原始数据点列表
            max_points: 最大点数，如果为None则使用self.max_points

        返回:
            下采样后的数据点列表
        """
        if max_points is None:
            max_points = self.max_points

        if len(data) <= max_points:
            return data

        # 使用numpy进行均匀采样
        indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
        sampled_data = [data[i] for i in indices]

        return sampled_data

    def generate_comprehensive_comparison(self, selected_algorithms=None):
        """
        生成综合对比图（4子图）

        参数:
            selected_algorithms: 选择要对比的算法列表，None表示使用所有可用算法
        """
        if selected_algorithms is None:
            selected_algorithms = list(self.algorithm_data.keys())

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 标题
        chart_set_titles = {
            'chart_set_1': 'Iteration Performance Comparison (100 tasks, 20 VMs)',
            'chart_set_2': 'Task Scale Cost Comparison (100-1000 tasks)',
            'chart_set_3': 'Iteration Performance Comparison (1000 tasks, 50 VMs)',
            'chart_set_4': 'Large Scale Cost Comparison (1000-5000 tasks)'
        }
        title = chart_set_titles.get(self.chart_set, 'Multi-Algorithm Performance Comparison')
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

        # 1. Execution Time对比（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_execution_time_comparison(ax1, selected_algorithms)

        # 2. Total Cost对比（右上）
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cost_comparison(ax2, selected_algorithms)

        # 3. Load Balance对比（左下）
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_load_balance_comparison(ax3, selected_algorithms)

        # 4. Price Efficiency对比（右下）
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_price_efficiency_comparison(ax4, selected_algorithms)

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_suffix = '_'.join(selected_algorithms[:3]) if len(selected_algorithms) <= 3 else 'multi_algo'
        filepath = os.path.join(self.output_dir,
                                f'{self.chart_set}_comprehensive_{algo_suffix}_{timestamp}.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"综合对比图已保存: {filepath}")

        # 同时保存数据到CSV
        self._save_comparison_data(selected_algorithms, timestamp)

        return filepath

    def _plot_execution_time_comparison(self, ax, selected_algorithms):
        """绘制Execution Time对比"""
        xlabel = 'Iteration'  # 默认xlabel

        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue

            data = self.algorithm_data[algo]
            config = ALGORITHM_CONFIG.get(algo, {})

            # 下采样数据
            sampled_data = self._downsample_data(data)

            # 提取x轴和y轴数据
            if 'iteration' in sampled_data[0]:
                x = [point['iteration'] for point in sampled_data]
                xlabel = 'Iteration'
            else:
                x = [point.get('M', point.get('task_count', i)) for i, point in enumerate(sampled_data)]
                xlabel = 'Task Count'

            y = [point['execution_time'] for point in sampled_data]

            # 绘制曲线
            ax.plot(x, y,
                   color=config.get('color', 'blue'),
                   linestyle=config.get('linestyle', '-'),
                   marker=config.get('marker', 'o'),
                   linewidth=2.5,
                   markersize=8,
                   label=config.get('label', algo),
                   alpha=0.85)

        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (Time Units)', fontsize=12, fontweight='bold')
        ax.set_title('(a) Execution Time Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def _plot_cost_comparison(self, ax, selected_algorithms):
        """绘制Total Cost对比"""
        xlabel = 'Iteration'  # 默认xlabel

        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue

            data = self.algorithm_data[algo]
            config = ALGORITHM_CONFIG.get(algo, {})

            # 下采样数据
            sampled_data = self._downsample_data(data)

            # 提取x轴和y轴数据
            if 'iteration' in sampled_data[0]:
                x = [point['iteration'] for point in sampled_data]
                xlabel = 'Iteration'
            else:
                x = [point.get('M', point.get('task_count', i)) for i, point in enumerate(sampled_data)]
                xlabel = 'Task Count'

            y = [point['total_cost'] for point in sampled_data]

            # 绘制曲线
            ax.plot(x, y,
                   color=config.get('color', 'blue'),
                   linestyle=config.get('linestyle', '-'),
                   marker=config.get('marker', 'o'),
                   linewidth=2.5,
                   markersize=8,
                   label=config.get('label', algo),
                   alpha=0.85)

        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
        ax.set_title('(b) Total Cost Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def _plot_load_balance_comparison(self, ax, selected_algorithms):
        """绘制Load Balance对比"""
        xlabel = 'Iteration'  # 默认xlabel

        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue

            data = self.algorithm_data[algo]
            config = ALGORITHM_CONFIG.get(algo, {})

            # 下采样数据
            sampled_data = self._downsample_data(data)

            # 提取x轴和y轴数据
            if 'iteration' in sampled_data[0]:
                x = [point['iteration'] for point in sampled_data]
                xlabel = 'Iteration'
            else:
                x = [point.get('M', point.get('task_count', i)) for i, point in enumerate(sampled_data)]
                xlabel = 'Task Count'

            y = [point['load_balance'] for point in sampled_data]

            # 绘制曲线
            ax.plot(x, y,
                   color=config.get('color', 'blue'),
                   linestyle=config.get('linestyle', '-'),
                   marker=config.get('marker', 'o'),
                   linewidth=2.5,
                   markersize=8,
                   label=config.get('label', algo),
                   alpha=0.85)

        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Load Balance (Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_title('(c) Load Balance Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def _plot_price_efficiency_comparison(self, ax, selected_algorithms):
        """绘制Price Efficiency对比"""
        xlabel = 'Iteration'  # 默认xlabel

        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue

            data = self.algorithm_data[algo]
            config = ALGORITHM_CONFIG.get(algo, {})

            # 下采样数据
            sampled_data = self._downsample_data(data)

            # 提取x轴和y轴数据
            if 'iteration' in sampled_data[0]:
                x = [point['iteration'] for point in sampled_data]
                xlabel = 'Iteration'
            else:
                x = [point.get('M', point.get('task_count', i)) for i, point in enumerate(sampled_data)]
                xlabel = 'Task Count'

            y = [point['price_efficiency'] for point in sampled_data]

            # 绘制曲线
            ax.plot(x, y,
                   color=config.get('color', 'blue'),
                   linestyle=config.get('linestyle', '-'),
                   marker=config.get('marker', 'o'),
                   linewidth=2.5,
                   markersize=8,
                   label=config.get('label', algo),
                   alpha=0.85)

        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Price Efficiency (Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_title('(d) Price Efficiency Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def _save_comparison_data(self, selected_algorithms, timestamp):
        """保存对比数据到CSV和Excel"""
        # 准备数据
        all_data = []
        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue

            data = self.algorithm_data[algo]
            for point in data:
                row = point.copy()
                row['algorithm'] = algo
                all_data.append(row)

        # 转换为DataFrame
        df = pd.DataFrame(all_data)

        # 保存CSV
        csv_path = os.path.join(self.output_dir,
                                f'{self.chart_set}_comparison_data_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"对比数据已保存 (CSV): {csv_path}")

        # 保存Excel
        try:
            excel_path = os.path.join(self.output_dir,
                                     f'{self.chart_set}_comparison_data_{timestamp}.xlsx')
            df.to_excel(excel_path, index=False, sheet_name='Comparison Data')
            print(f"对比数据已保存 (Excel): {excel_path}")
        except Exception as e:
            print(f"警告: 无法保存Excel文件: {e}")

    def generate_individual_metric_charts(self, selected_algorithms=None):
        """生成单独的指标对比图（每个指标一张图）"""
        if selected_algorithms is None:
            selected_algorithms = list(self.algorithm_data.keys())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepaths = []

        metrics = {
            'execution_time': ('Execution Time', 'Execution Time (Time Units)', '(a)'),
            'total_cost': ('Total Cost', 'Total Cost', '(b)'),
            'load_balance': ('Load Balance', 'Load Balance (Higher is Better)', '(c)'),
            'price_efficiency': ('Price Efficiency', 'Price Efficiency (Higher is Better)', '(d)')
        }

        for metric_key, (title, ylabel, label_prefix) in metrics.items():
            fig, ax = plt.subplots(figsize=(12, 8))

            for algo in selected_algorithms:
                if algo not in self.algorithm_data:
                    continue

                data = self.algorithm_data[algo]
                config = ALGORITHM_CONFIG.get(algo, {})

                # 下采样数据
                sampled_data = self._downsample_data(data)

                # 提取数据
                if 'iteration' in sampled_data[0]:
                    x = [point['iteration'] for point in sampled_data]
                    xlabel = 'Iteration'
                else:
                    x = [point.get('M', point.get('task_count', i)) for i, point in enumerate(sampled_data)]
                    xlabel = 'Task Count'

                y = [point[metric_key] for point in sampled_data]

                # 绘制
                ax.plot(x, y,
                       color=config.get('color', 'blue'),
                       linestyle=config.get('linestyle', '-'),
                       marker=config.get('marker', 'o'),
                       linewidth=2.5,
                       markersize=7,
                       label=config.get('label', algo),
                       alpha=0.85)

            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
            ax.set_title(f'{label_prefix} {title} Comparison - {self.chart_set}',
                        fontsize=15, fontweight='bold')
            ax.legend(fontsize=12, loc='best', framealpha=0.95)
            ax.grid(True, alpha=0.4, linestyle='--')

            # 保存
            filepath = os.path.join(self.output_dir,
                                   f'{self.chart_set}_{metric_key}_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(fig)
            filepaths.append(filepath)
            print(f"单指标图表已保存: {filepath}")

        return filepaths



def generate_chart_for_set(raw_data_dir, chart_set, output_dir, max_points=20,
                          algorithms=None, individual=False):
    """
    为指定图表集生成图表

    参数:
        raw_data_dir: RAW数据目录路径
        chart_set: 图表集名称
        output_dir: 输出目录
        max_points: 最大显示点数
        algorithms: 选择要对比的算法列表
        individual: 是否同时生成单独的指标图表

    返回:
        是否成功生成
    """
    try:
        generator = MultiAlgorithmChartGenerator(
            raw_data_dir,
            chart_set=chart_set,
            output_dir=output_dir,
            max_points=max_points
        )

        # 生成综合对比图
        print("\n生成综合对比图...")
        filepath1 = generator.generate_comprehensive_comparison(algorithms)

        # 生成单独指标图（如果启用）
        if individual:
            print("\n生成单独指标图...")
            filepaths = generator.generate_individual_metric_charts(algorithms)
            print(f"\n生成了 {len(filepaths)} 个单独指标图")

        print(f"\n{chart_set} 图表生成完成！")
        return True
    except Exception as e:
        print(f"\n错误: 处理 {chart_set} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='多算法对比图表生成工具')
    parser.add_argument('--raw-data-dir', type=str,
                       default=DEFAULT_RAW_DATA_DIR,
                       help=f'RAW_data目录路径 (默认: {DEFAULT_RAW_DATA_DIR})')
    parser.add_argument('--chart-set', type=int, choices=[1, 2, 3, 4],
                       help='指定要生成图表的图表集编号')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       help='选择要对比的算法（空格分隔）')
    parser.add_argument('--all', action='store_true',
                       help='生成所有图表集的图表')
    parser.add_argument('--individual', action='store_true',
                       help='同时生成单独的指标图表')
    parser.add_argument('--max-points', type=int, default=20,
                       help='每条曲线的最大显示点数（默认20，减少密度让曲线更清晰）')

    args = parser.parse_args()

    print("="*80)
    print("多算法对比 - 论文级图表生成工具")
    print("="*80)
    print(f"RAW数据目录: {args.raw_data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大显示点数: {args.max_points} (从原始数据中采样)")

    # 验证目录存在
    if not os.path.exists(args.raw_data_dir):
        print(f"\n[ERROR] RAW数据目录不存在: {args.raw_data_dir}")
        print(f"[提示] 请确保数据文件在正确的位置")
        return 1

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[OK] 输出目录已准备: {args.output_dir}")
    print("="*80)

    # 命令行参数模式
    if args.chart_set:
        # 生成指定的图表集
        chart_set_name = f'chart_set_{args.chart_set}'

        print(f"\n[TARGET] 生成图表: chart set {args.chart_set}")
        if args.algorithms:
            print(f"[ALGORITHM] 指定算法: {', '.join(args.algorithms)}")

        success = generate_chart_for_set(
            args.raw_data_dir,
            chart_set_name,
            args.output_dir,
            args.max_points,
            args.algorithms,
            args.individual
        )

        return 0 if success else 1

    elif args.all:
        # 生成所有图表集
        print("\n[START] 生成所有图表集的图表")
        chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']

        for chart_set in chart_sets:
            print(f"\n{'='*80}")
            print(f"处理图表集: {chart_set}")
            print(f"{'='*80}")

            generate_chart_for_set(
                args.raw_data_dir,
                chart_set,
                args.output_dir,
                args.max_points,
                args.algorithms,
                args.individual
            )

        print("\n[SUCCESS] 所有图表生成完成!")
        return 0

    else:
        # 交互式模式
        while True:
            print("\n请选择要生成的图表集:")
            print("1. 图表集1 - 迭代次数 vs 性能指标 (100任务, 20虚拟机)")
            print("2. 图表集2 - 任务规模 vs 成本 (100-1000任务)")
            print("3. 图表集3 - 迭代次数 vs 性能指标 (1000任务, 50虚拟机)")
            print("4. 图表集4 - 任务规模 vs 成本 (1000-5000任务)")
            print("5. 生成所有图表集")
            print("0. 退出")
            print("-" * 60)

            try:
                choice = input("请输入选项 (0-5): ").strip()
                choice = int(choice)

                if choice == 0:
                    print("[EXIT] 退出程序")
                    break

                elif choice in [1, 2, 3, 4]:
                    chart_set_name = f'chart_set_{choice}'

                    # 询问是否生成单独指标图
                    individual_choice = input("\n是否同时生成单独的指标图表? (y/n, 默认n): ").strip().lower()
                    individual = (individual_choice == 'y')

                    generate_chart_for_set(
                        args.raw_data_dir,
                        chart_set_name,
                        args.output_dir,
                        args.max_points,
                        args.algorithms,
                        individual
                    )

                elif choice == 5:
                    # 生成所有图表集
                    print("\n[START] 生成所有图表集的图表")

                    # 询问是否生成单独指标图
                    individual_choice = input("\n是否同时生成单独的指标图表? (y/n, 默认n): ").strip().lower()
                    individual = (individual_choice == 'y')

                    chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']
                    for chart_set in chart_sets:
                        print(f"\n{'='*80}")
                        print(f"处理图表集: {chart_set}")
                        print(f"{'='*80}")

                        generate_chart_for_set(
                            args.raw_data_dir,
                            chart_set,
                            args.output_dir,
                            args.max_points,
                            args.algorithms,
                            individual
                        )

                    print("\n[SUCCESS] 所有图表生成完成!")

                else:
                    print("[ERROR] 无效选项，请重新选择")

            except (ValueError, KeyboardInterrupt):
                print("\n[EXIT] 退出程序")
                break

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
