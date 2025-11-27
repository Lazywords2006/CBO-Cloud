#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
能耗分析增强工具
==============
强化能耗指标分析,包含CO₂排放和经济分析

分析内容:
1. 能耗对比 (kWh)
2. CO₂排放量 (kg)
3. 经济成本分析 ($)
4. 绿色计算指标
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# 获取项目根目录
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent


class EnergyAnalysisEnhancer:
    """能耗分析增强器"""

    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        初始化分析器

        参数:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / 'Text Demo' / 'RAW_data'
        else:
            self.data_dir = Path(data_dir)

        if output_dir is None:
            self.output_dir = PROJECT_ROOT / 'Text Demo' / 'energy_analysis'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 能耗和排放系数 (可根据实际情况调整)
        self.CO2_FACTOR = 0.5  # kg CO₂/kWh (数据中心平均值)
        self.ENERGY_COST = 0.10  # $/kWh (工业电价)

    def load_data(self, chart_set_name: str) -> Dict:
        """加载数据"""
        json_file = self.data_dir / f"{chart_set_name}_merged_results.json"

        if not json_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def extract_energy_data(self, data: Dict) -> Dict:
        """
        提取能耗数据

        参数:
            data: 原始数据

        返回:
            {算法名: [能耗值列表]}
        """
        results = {}

        for algo_name, algo_data in data.items():
            if 'energy' in algo_data:
                energy_data = algo_data['energy']

                # 提取最终值
                if isinstance(energy_data, list):
                    if len(energy_data) > 0:
                        if isinstance(energy_data[0], list):
                            # 每次运行是一个列表,取最后一个值
                            final_values = [run[-1] if run else 0 for run in energy_data]
                        else:
                            # 已经是最终值列表
                            final_values = energy_data
                    else:
                        continue
                else:
                    continue

                results[algo_name] = final_values

        return results

    def calculate_co2_emissions(self, energy_kwh: float) -> float:
        """
        计算CO₂排放量

        参数:
            energy_kwh: 能耗 (kWh)

        返回:
            CO₂排放量 (kg)
        """
        return energy_kwh * self.CO2_FACTOR

    def calculate_energy_cost(self, energy_kwh: float) -> float:
        """
        计算能耗成本

        参数:
            energy_kwh: 能耗 (kWh)

        返回:
            成本 ($)
        """
        return energy_kwh * self.ENERGY_COST

    def generate_energy_analysis_table(self, chart_set_name: str) -> pd.DataFrame:
        """
        生成能耗分析表格

        参数:
            chart_set_name: 图表集名称

        返回:
            能耗分析DataFrame
        """
        data = self.load_data(chart_set_name)
        energy_data = self.extract_energy_data(data)

        results = []

        for algo_name, energy_values in energy_data.items():
            energy_arr = np.array(energy_values)

            mean_energy = np.mean(energy_arr)
            std_energy = np.std(energy_arr, ddof=1)

            # 计算CO₂排放
            co2_emissions = self.calculate_co2_emissions(mean_energy)
            co2_std = self.calculate_co2_emissions(std_energy)

            # 计算成本
            energy_cost = self.calculate_energy_cost(mean_energy)
            cost_std = self.calculate_energy_cost(std_energy)

            # 计算每任务能耗 (假设从chart_set名称获取任务数)
            # 简化处理: chart_set_1和3是100任务, chart_set_2是变化的, chart_set_4是大规模
            if 'chart_set_1' in chart_set_name or 'chart_set_3' in chart_set_name:
                n_tasks = 100
            elif 'chart_set_2' in chart_set_name:
                n_tasks = 500  # 取平均值
            elif 'chart_set_4' in chart_set_name:
                n_tasks = 3000  # 取平均值
            else:
                n_tasks = 100

            energy_per_task = mean_energy / n_tasks if n_tasks > 0 else 0

            results.append({
                'Algorithm': algo_name,
                'Mean Energy (kWh)': mean_energy,
                'Std Energy (kWh)': std_energy,
                'CO₂ Emissions (kg)': co2_emissions,
                'CO₂ Std (kg)': co2_std,
                'Energy Cost ($)': energy_cost,
                'Cost Std ($)': cost_std,
                'Energy per Task (Wh)': energy_per_task * 1000,  # 转换为Wh
                'n_samples': len(energy_values)
            })

        df = pd.DataFrame(results)

        # 计算相对于基准的改进
        if 'BCBO' in df['Algorithm'].values:
            baseline_energy = df[df['Algorithm'] == 'BCBO']['Mean Energy (kWh)'].values[0]

            df['Energy Improvement (%)'] = df['Mean Energy (kWh)'].apply(
                lambda x: ((baseline_energy - x) / baseline_energy) * 100
            )
        else:
            df['Energy Improvement (%)'] = 0.0

        return df

    def generate_markdown_report(self, df: pd.DataFrame, chart_set_name: str) -> str:
        """生成Markdown格式的能耗分析报告"""
        md = f"# {chart_set_name} 能耗分析报告\n\n"
        md += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        md += "## 能耗对比表\n\n"
        md += "| 算法 | 能耗 (kWh) | CO₂排放 (kg) | 能耗成本 ($) | 每任务能耗 (Wh) | 改进幅度 |\n"
        md += "|------|-----------|------------|------------|---------------|----------|\n"

        for _, row in df.iterrows():
            md += f"| {row['Algorithm']} | "
            md += f"{row['Mean Energy (kWh)']:.2f}±{row['Std Energy (kWh)']:.2f} | "
            md += f"{row['CO₂ Emissions (kg)']:.2f}±{row['CO₂ Std (kg)']:.2f} | "
            md += f"${row['Energy Cost ($)']:.2f}±{row['Cost Std ($)']:.2f} | "
            md += f"{row['Energy per Task (Wh)']:.3f} | "

            improvement = row['Energy Improvement (%)']
            if improvement > 0:
                md += f"↓{improvement:.1f}% ✅ |\n"
            elif improvement < 0:
                md += f"↑{abs(improvement):.1f}% ❌ |\n"
            else:
                md += "— (基准) |\n"

        md += "\n## 关键发现\n\n"

        # 找到最优算法
        best_algo_idx = df['Mean Energy (kWh)'].idxmin()
        best_algo = df.loc[best_algo_idx, 'Algorithm']
        best_energy = df.loc[best_algo_idx, 'Mean Energy (kWh)']
        best_improvement = df.loc[best_algo_idx, 'Energy Improvement (%)']

        md += f"1. **最优算法**: {best_algo}\n"
        md += f"   - 平均能耗: {best_energy:.2f} kWh\n"
        md += f"   - 相比基准改进: {best_improvement:.1f}%\n\n"

        # 计算总体节省
        if 'BCBO-DE' in df['Algorithm'].values:
            bcbo_de_row = df[df['Algorithm'] == 'BCBO-DE'].iloc[0]
            annual_tasks = 10000 * 365  # 假设每天处理10000个任务

            energy_saving_per_task = (
                df[df['Algorithm'] == 'BCBO']['Energy per Task (Wh)'].values[0] -
                bcbo_de_row['Energy per Task (Wh)']
            ) / 1000  # 转换为kWh

            annual_energy_saving = energy_saving_per_task * annual_tasks
            annual_co2_saving = self.calculate_co2_emissions(annual_energy_saving)
            annual_cost_saving = self.calculate_energy_cost(annual_energy_saving)

            md += f"2. **BCBO-DE经济效益** (假设每天10,000任务):\n"
            md += f"   - 年度节能: {annual_energy_saving:,.0f} kWh\n"
            md += f"   - 年度减排: {annual_co2_saving:,.0f} kg CO₂\n"
            md += f"   - 年度节约: ${annual_cost_saving:,.2f}\n\n"

        md += "## 环境影响\n\n"
        md += "**CO₂排放系数**: 0.5 kg CO₂/kWh (数据中心平均值)\n\n"

        md += "使用BCBO-DE算法相当于:\n"
        if 'BCBO-DE' in df['Algorithm'].values:
            bcbo_de_co2 = df[df['Algorithm'] == 'BCBO-DE']['CO₂ Emissions (kg)'].values[0]
            bcbo_co2 = df[df['Algorithm'] == 'BCBO']['CO₂ Emissions (kg)'].values[0]
            co2_reduction = bcbo_co2 - bcbo_de_co2

            # 等效森林面积 (1棵树每年吸收约21.77 kg CO₂)
            trees_equivalent = annual_co2_saving / 21.77 if 'annual_co2_saving' in locals() else co2_reduction / 21.77

            md += f"- 每年减少 {annual_co2_saving if 'annual_co2_saving' in locals() else co2_reduction:,.0f} kg CO₂排放\n"
            md += f"- 相当于种植 {trees_equivalent:,.0f} 棵树的碳吸收量\n\n"

        md += "## 说明\n\n"
        md += "- **能耗**: 基于虚拟机执行任务的总能耗\n"
        md += "- **CO₂排放**: 基于数据中心平均碳排放系数计算\n"
        md += "- **能耗成本**: 基于工业电价 $0.10/kWh 计算\n"
        md += "- **改进幅度**: 相对于BCBO基准算法的改进百分比\n"

        return md

    def plot_energy_comparison(self, df: pd.DataFrame, chart_set_name: str):
        """绘制能耗对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{chart_set_name} - Energy Analysis', fontsize=16, fontweight='bold')

        algorithms = df['Algorithm'].tolist()
        colors = ['#2E86AB', '#C73E1D', '#6A994E', '#F18F01', '#BC4B51', '#8B5A3C', '#5F9EA0', '#D2691E']

        # 子图1: 能耗对比
        ax1 = axes[0, 0]
        energy_means = df['Mean Energy (kWh)'].tolist()
        energy_stds = df['Std Energy (kWh)'].tolist()

        bars1 = ax1.bar(range(len(algorithms)), energy_means, yerr=energy_stds,
                       capsize=5, color=colors[:len(algorithms)], alpha=0.8,
                       edgecolor='black', linewidth=1.2)

        # 高亮BCBO-DE
        if 'BCBO-DE' in algorithms:
            idx = algorithms.index('BCBO-DE')
            bars1[idx].set_facecolor('#C73E1D')
            bars1[idx].set_edgecolor('#8B0000')
            bars1[idx].set_linewidth(2.5)

        ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Energy Consumption', fontsize=12)
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # 子图2: CO₂排放
        ax2 = axes[0, 1]
        co2_means = df['CO₂ Emissions (kg)'].tolist()
        co2_stds = df['CO₂ Std (kg)'].tolist()

        bars2 = ax2.bar(range(len(algorithms)), co2_means, yerr=co2_stds,
                       capsize=5, color=colors[:len(algorithms)], alpha=0.8,
                       edgecolor='black', linewidth=1.2)

        if 'BCBO-DE' in algorithms:
            idx = algorithms.index('BCBO-DE')
            bars2[idx].set_facecolor('#C73E1D')
            bars2[idx].set_edgecolor('#8B0000')
            bars2[idx].set_linewidth(2.5)

        ax2.set_ylabel('CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Carbon Emissions', fontsize=12)
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # 子图3: 能耗成本
        ax3 = axes[1, 0]
        cost_means = df['Energy Cost ($)'].tolist()
        cost_stds = df['Cost Std ($)'].tolist()

        bars3 = ax3.bar(range(len(algorithms)), cost_means, yerr=cost_stds,
                       capsize=5, color=colors[:len(algorithms)], alpha=0.8,
                       edgecolor='black', linewidth=1.2)

        if 'BCBO-DE' in algorithms:
            idx = algorithms.index('BCBO-DE')
            bars3[idx].set_facecolor('#C73E1D')
            bars3[idx].set_edgecolor('#8B0000')
            bars3[idx].set_linewidth(2.5)

        ax3.set_ylabel('Energy Cost ($)', fontsize=12, fontweight='bold')
        ax3.set_title('(c) Economic Cost', fontsize=12)
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels(algorithms, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # 子图4: 改进幅度
        ax4 = axes[1, 1]
        improvements = df['Energy Improvement (%)'].tolist()

        bar_colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
        bars4 = ax4.bar(range(len(algorithms)), improvements,
                       color=bar_colors, alpha=0.7,
                       edgecolor='black', linewidth=1.2)

        if 'BCBO-DE' in algorithms:
            idx = algorithms.index('BCBO-DE')
            bars4[idx].set_facecolor('darkgreen' if improvements[idx] > 0 else 'darkred')
            bars4[idx].set_linewidth(2.5)

        ax4.set_ylabel('Energy Improvement (%)', fontsize=12, fontweight='bold')
        ax4.set_title('(d) Energy Efficiency Improvement', fontsize=12)
        ax4.set_xticks(range(len(algorithms)))
        ax4.set_xticklabels(algorithms, rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{chart_set_name}_energy_analysis_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[保存] 能耗分析图表: {output_path}")

        return str(output_path)

    def analyze_all_chart_sets(self):
        """分析所有图表集的能耗"""
        chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']

        for chart_set in chart_sets:
            print(f"\n{'='*80}")
            print(f"分析 {chart_set} 能耗".center(80))
            print(f"{'='*80}")

            try:
                # 生成分析表格
                df = self.generate_energy_analysis_table(chart_set)

                # 保存CSV
                csv_path = self.output_dir / f"{chart_set}_energy_analysis.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"[保存] CSV文件: {csv_path}")

                # 保存Excel
                excel_path = self.output_dir / f"{chart_set}_energy_analysis.xlsx"
                df.to_excel(excel_path, index=False, engine='openpyxl')
                print(f"[保存] Excel文件: {excel_path}")

                # 生成Markdown报告
                md_content = self.generate_markdown_report(df, chart_set)
                md_path = self.output_dir / f"{chart_set}_energy_report.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                print(f"[保存] Markdown报告: {md_path}")

                # 绘制图表
                self.plot_energy_comparison(df, chart_set)

            except FileNotFoundError as e:
                print(f"错误: {e}")
                continue


def main():
    """主函数"""
    print("="*80)
    print("能耗分析增强工具".center(80))
    print("="*80)

    analyzer = EnergyAnalysisEnhancer()
    analyzer.analyze_all_chart_sets()

    print("\n" + "="*80)
    print("能耗分析完成!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
