#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文级图表生成工具 - 简化版
=========================================
只生成两种核心图表:
1. BCBO vs MBCBO 对比图
2. 所有算法对比图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime
import glob
import re
import sys

# 设置标准输出编码为 UTF-8，避免 Windows 下的编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'RAW_data')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'publication_charts')

# 设置期刊出版级样式
# 使用标准学术字体：Times New Roman (衬线) + Arial (无衬线)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'SimSun']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用STIX字体
plt.rcParams['axes.unicode_minus'] = False

# 高分辨率设置 - 期刊要求至少300 DPI
plt.rcParams['figure.dpi'] = 100  # 屏幕显示
plt.rcParams['savefig.dpi'] = 600  # 保存为600 DPI，超过期刊要求

# 线条和标记设置 - 清晰可见
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6

# 图表网格和边框
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# 字体大小 - 适合期刊出版
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# 图例设置
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = '0.8'

# PDF输出设置
plt.rcParams['pdf.fonttype'] = 42  # TrueType字体，便于编辑
plt.rcParams['ps.fonttype'] = 42

# SVG输出设置
plt.rcParams['svg.fonttype'] = 'none'  # 保持文本可编辑

# 算法配置 - 期刊出版级颜色方案
# 使用ColorBrewer的高对比度配色，在灰度打印时仍可区分
# 同时使用不同的线型和标记确保黑白打印时的可区分性
ALGORITHM_CONFIG = {
    'BCBO': {
        'color': '#377eb8',      # 深蓝色 - 基准算法
        'label': 'BCBO', 
        'linestyle': '-', 
        'marker': 'o',
        'markersize': 7
    },
    'MBCBO': {
        'color': '#e41a1c',      # 鲜红色 - 提出的算法（强调）
        'label': 'MBCBO',
        'linestyle': '-',
        'marker': 's',
        'markersize': 8,
        'linewidth': 2.5          # 加粗以强调
    },
    'GA': {
        'color': '#4daf4a',      # 绿色
        'label': 'GA', 
        'linestyle': '--', 
        'marker': '^',
        'markersize': 7
    },
    'PSO': {
        'color': '#984ea3',      # 紫色
        'label': 'PSO', 
        'linestyle': '-.', 
        'marker': 'v',
        'markersize': 7
    },
    'ACO': {
        'color': '#ff7f00',      # 橙色
        'label': 'ACO', 
        'linestyle': ':', 
        'marker': 'D',
        'markersize': 6
    },
    'FA': {
        'color': '#a65628',      # 棕色
        'label': 'FA', 
        'linestyle': '--', 
        'marker': 'p',
        'markersize': 7
    },
    'CS': {
        'color': '#f781bf',      # 粉色
        'label': 'CS', 
        'linestyle': '-.', 
        'marker': 'h',
        'markersize': 7
    },
    'GWO': {
        'color': '#999999',      # 灰色
        'label': 'GWO', 
        'linestyle': ':', 
        'marker': '*',
        'markersize': 8
    },
}

# 图表集标题配置
CHART_SET_TITLES = {
    'chart_set_1': 'Iteration Performance Comparison (100 tasks, 20 VMs)',
    'chart_set_2': 'Task Scale Cost Comparison (100-1000 tasks)',
    'chart_set_3': 'Iteration Performance Comparison (1000 tasks, 50 VMs)',
    'chart_set_4': 'Large Scale Cost Comparison (1000-5000 tasks)'
}


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, raw_data_dir, chart_set, output_dir, max_points=20):
        self.raw_data_dir = raw_data_dir
        self.chart_set = chart_set
        self.output_dir = output_dir
        self.max_points = max_points
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.algorithm_data = {}
        self._load_data()
        
    def _load_data(self):
        """加载算法数据"""
        merged_file = os.path.join(self.raw_data_dir, f'{self.chart_set}_merged_results.json')
        
        if not os.path.exists(merged_file):
            raise FileNotFoundError(f"找不到数据文件: {merged_file}")
        
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
            algorithms_data = merged_data.get('algorithms', {})
            
            for algo_name, algo_info in algorithms_data.items():
                results = algo_info.get('results', [])
                if results:
                    self.algorithm_data[algo_name] = results
        
        print(f"✓ 加载 {self.chart_set}: {len(self.algorithm_data)} 个算法")
    
    def _downsample_data(self, data):
        """下采样数据"""
        if len(data) <= self.max_points:
            return data
        indices = np.linspace(0, len(data) - 1, self.max_points, dtype=int)
        return [data[i] for i in indices]
    
    def _plot_metric(self, ax, selected_algorithms, metric_key, ylabel, title):
        """绘制单个指标 - 期刊出版级质量"""
        xlabel = 'Iteration'
        
        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue
            
            data = self.algorithm_data[algo]
            config = ALGORITHM_CONFIG.get(algo, {})
            sampled_data = self._downsample_data(data)
            
            # 提取x轴和y轴数据
            if 'iteration' in sampled_data[0]:
                x = [point['iteration'] for point in sampled_data]
                xlabel = 'Iteration'
            else:
                x = [point.get('M', point.get('task_count', i)) for i, point in enumerate(sampled_data)]
                xlabel = 'Number of Tasks'
            
            y = [point[metric_key] for point in sampled_data]
            
            # 绘制曲线 - 使用配置中的所有参数
            ax.plot(x, y,
                   color=config.get('color', '#000000'),
                   linestyle=config.get('linestyle', '-'),
                   marker=config.get('marker', 'o'),
                   linewidth=config.get('linewidth', 1.5),
                   markersize=config.get('markersize', 6),
                   label=config.get('label', algo),
                   markerfacecolor=config.get('color', '#000000'),
                   markeredgewidth=0.5,
                   markeredgecolor='white',
                   alpha=0.9,
                   zorder=3)  # 确保线条在网格之上
        
        # 设置轴标签 - 使用标准学术格式
        ax.set_xlabel(xlabel, fontsize=11, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='normal')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # 图例设置 - 紧凑且专业，缩短标记线
        # 使用半透明背景，避免完全遮挡数据线
        legend = ax.legend(
            fontsize=9,
            loc='best',
            framealpha=0.7,          # 降低不透明度至70%，使背景数据线可见
            edgecolor='0.8',
            fancybox=False,
            shadow=False,
            ncol=1 if len(selected_algorithms) <= 4 else 2,
            handlelength=1.5,        # 缩短图例中的线条长度
            handleheight=0.7,        # 减小图例标记高度
            columnspacing=1.0,       # 列间距
            labelspacing=0.5,        # 行间距
            facecolor='white'        # 明确设置白色背景
        )
        # 设置图例框的圆角和边框样式，使其更加轻量化
        legend.get_frame().set_linewidth(0.5)  # 更细的边框
        legend.get_frame().set_boxstyle('round,pad=0.3')  # 圆角边框，减小内边距
        
        # 网格设置 - 细微辅助线
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)  # 网格在数据下方
        
        # 边框设置
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('0.5')
    
    def generate_comprehensive_comparison(self, selected_algorithms, output_suffix):
        """生成综合对比图（4子图）- 期刊出版级质量"""

        # 期刊标准尺寸：增大尺寸以确保完整显示
        # IEEE标准：单栏 3.5", 双栏 7.16"
        fig_width = 10.0  # 英寸，增大宽度
        fig_height = 8.5  # 英寸，增大高度以容纳标题和图例

        fig = plt.figure(figsize=(fig_width, fig_height))
        # 调整布局：增加所有边距，确保标题、标签和图例完全可见
        # 增大子图间距，避免重叠
        gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40,
                     left=0.10, right=0.95, top=0.90, bottom=0.10)
        
        # 主标题 - 使用标准学术格式，调整位置确保不重叠
        title = CHART_SET_TITLES.get(self.chart_set, 'Multi-Algorithm Performance Comparison')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
        
        # 四个子图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric(ax1, selected_algorithms, 'execution_time',
                         'Execution Time (s)', '(a) Execution Time')
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric(ax2, selected_algorithms, 'total_cost',
                         'Total Cost ($)', '(b) Total Cost')
        
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_metric(ax3, selected_algorithms, 'load_balance',
                         'Load Balance Index', '(c) Load Balance')
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_metric(ax4, selected_algorithms, 'price_efficiency',
                         'Price Efficiency Index', '(d) Price Efficiency')
        
        # 保存为多种格式 - 期刊投稿需要
        # 为每个图表集创建独立子目录
        chart_set_dir = os.path.join(self.output_dir, self.chart_set)
        os.makedirs(chart_set_dir, exist_ok=True)
        base_path = os.path.join(chart_set_dir, output_suffix)
        
        # 1. PNG格式 - 高分辨率，用于预览和论文初稿
        png_path = f'{base_path}.png'
        plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none', 
                   pil_kwargs={'optimize': True})
        print(f"  ✓ 保存PNG: {output_suffix}.png (600 DPI)")
        
        # 2. PDF格式 - 矢量图，最常用于论文发表
        pdf_path = f'{base_path}.pdf'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   metadata={'Creator': 'Publication Chart Generator',
                            'Author': 'Research Team'})
        print(f"  ✓ 保存PDF: {output_suffix}.pdf (矢量)")
        
        # 4. EPS格式 - 部分期刊要求的格式
        eps_path = f'{base_path}.eps'
        try:
            plt.savefig(eps_path, format='eps', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  ✓ 保存EPS: {output_suffix}.eps (期刊格式)")
        except Exception as e:
            print(f"  ⚠ EPS保存失败 (可选格式): {e}")
        
        # 关闭当前图表
        plt.close(fig)
        
        # 3. SVG格式 - 使用更大尺寸以便于查看和编辑
        # 为SVG单独生成更大的图表（1.5倍尺寸，而非2倍，避免过大）
        svg_width = fig_width * 1.5  # 15英寸
        svg_height = fig_height * 1.5  # 12.75英寸

        fig_svg = plt.figure(figsize=(svg_width, svg_height))
        # SVG使用相同的优化布局比例
        gs_svg = GridSpec(2, 2, figure=fig_svg, hspace=0.45, wspace=0.40,
                         left=0.10, right=0.95, top=0.90, bottom=0.10)

        fig_svg.suptitle(title, fontsize=21, fontweight='bold', y=0.96)
        
        # 重新绘制四个子图（字体会自动按比例放大）
        ax1_svg = fig_svg.add_subplot(gs_svg[0, 0])
        self._plot_metric(ax1_svg, selected_algorithms, 'execution_time',
                         'Execution Time (s)', '(a) Execution Time')
        
        ax2_svg = fig_svg.add_subplot(gs_svg[0, 1])
        self._plot_metric(ax2_svg, selected_algorithms, 'total_cost',
                         'Total Cost ($)', '(b) Total Cost')
        
        ax3_svg = fig_svg.add_subplot(gs_svg[1, 0])
        self._plot_metric(ax3_svg, selected_algorithms, 'load_balance',
                         'Load Balance Index', '(c) Load Balance')
        
        ax4_svg = fig_svg.add_subplot(gs_svg[1, 1])
        self._plot_metric(ax4_svg, selected_algorithms, 'price_efficiency',
                         'Price Efficiency Index', '(d) Price Efficiency')
        
        # 保存SVG
        svg_path = f'{base_path}.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"  ✓ 保存SVG: {output_suffix}.svg (15×12.75英寸, 可编辑)")
        
        plt.close(fig_svg)
        
        # 保存数据
        self._save_data(selected_algorithms, output_suffix)
        
        return png_path
    
    def _save_data(self, selected_algorithms, output_suffix):
        """保存对比数据到Excel"""
        all_data = []
        for algo in selected_algorithms:
            if algo not in self.algorithm_data:
                continue

            data = self.algorithm_data[algo]
            for point in data:
                row = point.copy()
                row['algorithm'] = algo
                all_data.append(row)

        df = pd.DataFrame(all_data)
        # 数据文件也保存在对应的子目录中
        chart_set_dir = os.path.join(self.output_dir, self.chart_set)
        os.makedirs(chart_set_dir, exist_ok=True)
        excel_path = os.path.join(chart_set_dir, f'{output_suffix}_data.xlsx')
        df.to_excel(excel_path, index=False, sheet_name='Comparison Data')
        print(f"  ✓ 保存数据: {output_suffix}_data.xlsx")
    
    def cleanup_old_files(self):
        """清理旧的时间戳文件"""
        pattern = os.path.join(self.output_dir, f'{self.chart_set}_*_2025*.png')
        old_files = glob.glob(pattern)
        
        for file in old_files:
            try:
                os.remove(file)
            except OSError:
                pass


def generate_charts_for_set(chart_set_name, raw_data_dir, output_dir):
    """为指定图表集生成所有必要的图表"""
    print(f"\n{'='*60}")
    print(f"处理: {chart_set_name}")
    print(f"{'='*60}")
    
    try:
        generator = ChartGenerator(raw_data_dir, chart_set_name, output_dir)
        
        # 生成 BCBO vs MBCBO 对比图
        print("生成 BCBO vs MBCBO 对比图...")
        generator.generate_comprehensive_comparison(['BCBO', 'MBCBO'], 'BCBO_vs_MBCBO')
        
        # 生成所有算法对比图
        print("生成所有算法对比图...")
        all_algos = list(generator.algorithm_data.keys())
        generator.generate_comprehensive_comparison(all_algos, 'All_Algorithms')
        
        # 清理旧文件
        generator.cleanup_old_files()
        
        print(f"✓ {chart_set_name} 完成")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='论文级图表生成工具')
    parser.add_argument('--raw-data-dir', type=str, default=DEFAULT_RAW_DATA_DIR,
                       help='RAW_data目录路径')
    parser.add_argument('--chart-set', type=int, choices=[1, 2, 3, 4],
                       help='指定要生成图表的图表集编号')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--all', action='store_true',
                       help='生成所有图表集的图表')
    
    args = parser.parse_args()
    
    print("="*60)
    print("论文级图表生成工具 - 简化版")
    print("="*60)
    print(f"RAW数据目录: {args.raw_data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 验证目录存在
    if not os.path.exists(args.raw_data_dir):
        print(f"\n✗ 错误: RAW数据目录不存在: {args.raw_data_dir}")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成图表
    if args.chart_set:
        # 生成指定的图表集
        chart_set_name = f'chart_set_{args.chart_set}'
        success = generate_charts_for_set(chart_set_name, args.raw_data_dir, args.output_dir)
        return 0 if success else 1
        
    elif args.all:
        # 生成所有图表集
        print("\n开始生成所有图表集...")
        for i in range(1, 5):
            chart_set_name = f'chart_set_{i}'
            generate_charts_for_set(chart_set_name, args.raw_data_dir, args.output_dir)
        print("\n✓ 所有图表生成完成!")
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
                    print("退出程序")
                    break
                    
                elif choice in [1, 2, 3, 4]:
                    chart_set_name = f'chart_set_{choice}'
                    generate_charts_for_set(chart_set_name, args.raw_data_dir, args.output_dir)
                    
                elif choice == 5:
                    for i in range(1, 5):
                        chart_set_name = f'chart_set_{i}'
                        generate_charts_for_set(chart_set_name, args.raw_data_dir, args.output_dir)
                    print("\n✓ 所有图表生成完成!")
                    
                else:
                    print("✗ 无效选项，请重新选择")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n退出程序")
                break
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
