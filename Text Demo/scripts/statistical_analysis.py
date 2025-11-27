#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计显著性检验分析工具
===================
用于分析BCBO-DE与基准算法之间的性能差异的统计显著性

提供功能:
1. Wilcoxon signed-rank test (配对非参数检验)
2. Mann-Whitney U test (独立非参数检验)
3. Cohen's d效应量计算
4. 生成论文级别的统计分析表格
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import json
import os
from pathlib import Path


class StatisticalAnalyzer:
    """统计分析器"""

    def __init__(self, data_dir: str):
        """
        初始化统计分析器

        参数:
            data_dir: RAW_data目录路径
        """
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_merged_data(self, chart_set_name: str) -> Dict:
        """
        加载合并后的实验数据

        参数:
            chart_set_name: 图表集名称 (如 'chart_set_1')

        返回:
            包含所有算法数据的字典
        """
        json_file = self.data_dir / f"{chart_set_name}_merged_results.json"

        if not json_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    @staticmethod
    def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        计算Cohen's d效应量

        Cohen's d解释:
        - |d| < 0.2: 微小效应 (trivial)
        - 0.2 ≤ |d| < 0.5: 小效应 (small)
        - 0.5 ≤ |d| < 0.8: 中等效应 (medium)
        - |d| ≥ 0.8: 大效应 (large)

        参数:
            group1: 第一组数据
            group2: 第二组数据

        返回:
            Cohen's d值
        """
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)

        if pooled_std == 0:
            return 0.0

        return mean_diff / pooled_std

    @staticmethod
    def effect_size_interpretation(d: float) -> str:
        """
        解释效应量大小

        参数:
            d: Cohen's d值

        返回:
            效应量解释
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Trivial"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"

    def compare_algorithms(self,
                          data1: List[float],
                          data2: List[float],
                          paired: bool = False) -> Dict:
        """
        对比两个算法的性能,进行统计检验

        参数:
            data1: BCBO-DE的数据
            data2: 基准算法的数据
            paired: 是否为配对数据

        返回:
            包含统计检验结果的字典
        """
        arr1 = np.array(data1)
        arr2 = np.array(data2)

        # 计算基础统计量
        mean1, std1 = np.mean(arr1), np.std(arr1, ddof=1)
        mean2, std2 = np.mean(arr2), np.std(arr2, ddof=1)

        # 计算改进百分比
        improvement = ((mean2 - mean1) / mean2) * 100 if mean2 != 0 else 0

        # 选择合适的统计检验
        if paired:
            # Wilcoxon signed-rank test (配对非参数检验)
            if len(arr1) < 20:
                statistic, p_value = stats.wilcoxon(arr1, arr2, alternative='less')
                test_name = "Wilcoxon"
            else:
                # 样本量足够,使用t检验
                statistic, p_value = stats.ttest_rel(arr1, arr2, alternative='less')
                test_name = "Paired t-test"
        else:
            # Mann-Whitney U test (独立非参数检验)
            if len(arr1) < 20 or len(arr2) < 20:
                statistic, p_value = stats.mannwhitneyu(arr1, arr2, alternative='less')
                test_name = "Mann-Whitney U"
            else:
                # 样本量足够,使用独立样本t检验
                statistic, p_value = stats.ttest_ind(arr1, arr2, alternative='less')
                test_name = "Independent t-test"

        # 计算Cohen's d
        cohen_d_value = self.cohen_d(arr1, arr2)
        effect_size = self.effect_size_interpretation(cohen_d_value)

        # 判断显著性
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"

        return {
            'mean_bcbo_de': mean1,
            'std_bcbo_de': std1,
            'mean_baseline': mean2,
            'std_baseline': std2,
            'improvement_%': improvement,
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significance': significance,
            'cohen_d': cohen_d_value,
            'effect_size': effect_size
        }

    def analyze_chart_set(self,
                         chart_set_name: str,
                         metric: str = 'makespan',
                         baseline_algorithms: List[str] = None) -> pd.DataFrame:
        """
        分析特定图表集的统计显著性

        参数:
            chart_set_name: 图表集名称
            metric: 要分析的指标 (makespan, total_cost, energy等)
            baseline_algorithms: 基准算法列表,默认使用所有算法

        返回:
            包含统计分析结果的DataFrame
        """
        data = self.load_merged_data(chart_set_name)

        if baseline_algorithms is None:
            baseline_algorithms = ['BCBO', 'GA', 'PSO', 'ACO', 'FA', 'CS', 'GWO']

        results = []

        # 获取BCBO-DE的数据
        bcbo_de_data = data.get('BCBO-DE', {}).get(metric, [])

        if not bcbo_de_data:
            print(f"警告: BCBO-DE在{chart_set_name}中没有{metric}数据")
            return pd.DataFrame()

        # 对每个基准算法进行对比
        for algo_name in baseline_algorithms:
            algo_data = data.get(algo_name, {}).get(metric, [])

            if not algo_data:
                print(f"警告: {algo_name}在{chart_set_name}中没有{metric}数据")
                continue

            # 提取最终值(假设每次运行都有多个迭代,取最后一个)
            bcbo_de_finals = [run[-1] if isinstance(run, list) else run for run in bcbo_de_data]
            algo_finals = [run[-1] if isinstance(run, list) else run for run in algo_data]

            # 确保数据长度一致(配对检验)
            min_len = min(len(bcbo_de_finals), len(algo_finals))
            bcbo_de_finals = bcbo_de_finals[:min_len]
            algo_finals = algo_finals[:min_len]

            # 进行统计检验
            result = self.compare_algorithms(bcbo_de_finals, algo_finals, paired=True)
            result['comparison'] = f"BCBO-DE vs {algo_name}"
            result['n_samples'] = min_len

            results.append(result)

        # 创建DataFrame
        df = pd.DataFrame(results)

        # 重新排列列顺序
        columns_order = [
            'comparison',
            'n_samples',
            'mean_bcbo_de',
            'std_bcbo_de',
            'mean_baseline',
            'std_baseline',
            'improvement_%',
            'test_name',
            'p_value',
            'significance',
            'cohen_d',
            'effect_size'
        ]

        df = df[columns_order]

        return df

    def generate_latex_table(self, df: pd.DataFrame, caption: str = "") -> str:
        """
        生成LaTeX格式的表格

        参数:
            df: 统计分析结果DataFrame
            caption: 表格标题

        返回:
            LaTeX表格代码
        """
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += "\\begin{tabular}{lcccccc}\n"
        latex += "\\hline\n"
        latex += "Comparison & Mean±Std (BCBO-DE) & Mean±Std (Baseline) & Improvement & p-value & Cohen's d & Effect \\\\\n"
        latex += "\\hline\n"

        for _, row in df.iterrows():
            latex += f"{row['comparison']} & "
            latex += f"{row['mean_bcbo_de']:.2f}±{row['std_bcbo_de']:.2f} & "
            latex += f"{row['mean_baseline']:.2f}±{row['std_baseline']:.2f} & "
            latex += f"{row['improvement_%']:.1f}\\% & "
            latex += f"{row['p_value']:.4f}{row['significance']} & "
            latex += f"{row['cohen_d']:.2f} & "
            latex += f"{row['effect_size']} \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\label{tab:statistical_test}\n"
        latex += "\\end{table}\n"

        latex += "\n% Note: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant\n"

        return latex

    def generate_markdown_table(self, df: pd.DataFrame) -> str:
        """
        生成Markdown格式的表格

        参数:
            df: 统计分析结果DataFrame

        返回:
            Markdown表格文本
        """
        md = "| Comparison | Mean±Std (BCBO-DE) | Mean±Std (Baseline) | Improvement | p-value | Cohen's d | Effect Size |\n"
        md += "|------------|-------------------|---------------------|-------------|---------|-----------|-------------|\n"

        for _, row in df.iterrows():
            md += f"| {row['comparison']} | "
            md += f"{row['mean_bcbo_de']:.2f}±{row['std_bcbo_de']:.2f} | "
            md += f"{row['mean_baseline']:.2f}±{row['std_baseline']:.2f} | "
            md += f"{row['improvement_%']:.1f}% | "
            md += f"{row['p_value']:.4f}{row['significance']} | "
            md += f"{row['cohen_d']:.2f} | "
            md += f"{row['effect_size']} |\n"

        md += "\n**Note**: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant\n"
        md += "\n**Cohen's d interpretation**: Trivial (<0.2), Small (0.2-0.5), Medium (0.5-0.8), Large (≥0.8)\n"

        return md

    def save_analysis_report(self,
                            chart_set_name: str,
                            df: pd.DataFrame,
                            output_dir: str = None):
        """
        保存统计分析报告

        参数:
            chart_set_name: 图表集名称
            df: 统计分析结果DataFrame
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.data_dir.parent / 'statistical_analysis'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True, parents=True)

        # 保存CSV
        csv_path = output_dir / f"{chart_set_name}_statistical_analysis.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[保存] CSV文件: {csv_path}")

        # 保存Excel
        excel_path = output_dir / f"{chart_set_name}_statistical_analysis.xlsx"
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"[保存] Excel文件: {excel_path}")

        # 保存Markdown
        md_path = output_dir / f"{chart_set_name}_statistical_analysis.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {chart_set_name}统计显著性分析\n\n")
            f.write(self.generate_markdown_table(df))
        print(f"[保存] Markdown文件: {md_path}")

        # 保存LaTeX
        latex_path = output_dir / f"{chart_set_name}_statistical_analysis.tex"
        caption = f"Statistical Significance Test for {chart_set_name}"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_latex_table(df, caption))
        print(f"[保存] LaTeX文件: {latex_path}")


def main():
    """主函数 - 运行完整的统计分析"""
    import sys

    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    data_dir = project_root / 'RAW_data'

    print("="*80)
    print("统计显著性检验分析工具".center(80))
    print("="*80)
    print()

    # 初始化分析器
    analyzer = StatisticalAnalyzer(str(data_dir))

    # 分析所有图表集
    chart_sets = ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']
    metrics = ['makespan', 'total_cost', 'load_balance', 'energy']

    for chart_set in chart_sets:
        print(f"\n{'='*80}")
        print(f"分析 {chart_set}".center(80))
        print(f"{'='*80}")

        try:
            # 针对每个指标进行分析
            for metric in metrics:
                print(f"\n[分析] 指标: {metric}")

                df = analyzer.analyze_chart_set(chart_set, metric=metric)

                if not df.empty:
                    # 显示结果
                    print(f"\n{analyzer.generate_markdown_table(df)}")

                    # 保存报告
                    analyzer.save_analysis_report(
                        f"{chart_set}_{metric}",
                        df
                    )
                else:
                    print(f"警告: {chart_set}的{metric}数据为空")

        except FileNotFoundError as e:
            print(f"错误: {e}")
            continue

    print("\n" + "="*80)
    print("统计分析完成!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
