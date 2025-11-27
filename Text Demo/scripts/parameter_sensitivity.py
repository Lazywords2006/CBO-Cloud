#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数敏感性分析工具
================
分析BCBO-DE关键参数对性能的影响

关键参数:
1. F_max: DE变异因子最大值
2. CR_max: DE交叉概率最大值
3. elite_ratio: 精英比例
4. bcbo_ratio: BCBO/DE融合比例
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加算法路径
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
ALGORITHM_ROOT = PROJECT_ROOT / 'algorithm'

BCBO_DE_DIR = ALGORITHM_ROOT / 'BCBO-DE-Fusion'
BCBO_DE_CORE = BCBO_DE_DIR / 'core'

for path in [str(BCBO_DE_CORE)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from bcbo_de_embedded import BCBO_DE_Embedded


class ParameterSensitivityAnalyzer:
    """参数敏感性分析器"""

    def __init__(self, output_dir: str = None):
        """
        初始化分析器

        参数:
            output_dir: 输出目录
        """
        if output_dir is None:
            self.output_dir = PROJECT_ROOT / 'Text Demo' / 'parameter_sensitivity'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}

    def analyze_parameter(self,
                         param_name: str,
                         param_values: List[float],
                         M: int = 100,
                         N: int = 20,
                         n: int = 50,
                         iterations: int = 100,
                         runs: int = 10) -> Dict:
        """
        分析单个参数的敏感性

        参数:
            param_name: 参数名称 ('F_max', 'CR_max', 'elite_ratio', 'bcbo_ratio')
            param_values: 参数取值列表
            M: 任务数
            N: 虚拟机数
            n: 种群大小
            iterations: 迭代次数
            runs: 每个参数值的运行次数

        返回:
            实验结果字典
        """
        print(f"\n{'='*80}")
        print(f"参数敏感性分析: {param_name}".center(80))
        print(f"{'='*80}")
        print(f"参数取值: {param_values}")
        print(f"实验配置: M={M}, N={N}, n={n}, iterations={iterations}, runs={runs}\n")

        results = {
            'param_name': param_name,
            'param_values': param_values,
            'makespans': [],
            'mean_makespans': [],
            'std_makespans': [],
            'convergence_iters': [],
            'execution_times': []
        }

        for param_value in param_values:
            print(f"\n[测试] {param_name} = {param_value}")

            makespans = []
            conv_iters = []
            exec_times = []

            for run in range(runs):
                # 设置默认参数
                kwargs = {
                    'M': M,
                    'N': N,
                    'n': n,
                    'F_max': 0.4,
                    'F_min': 0.15,
                    'CR_max': 0.7,
                    'CR_min': 0.3,
                    'elite_ratio': 0.2,
                    'bcbo_ratio': 0.85,
                    'fusion_mode': 'late'
                }

                # 更新目标参数
                if param_name == 'F_max':
                    kwargs['F_max'] = param_value
                    kwargs['F_min'] = param_value * 0.375  # 保持比例
                elif param_name == 'CR_max':
                    kwargs['CR_max'] = param_value
                    kwargs['CR_min'] = param_value * 0.43  # 保持比例
                elif param_name == 'elite_ratio':
                    kwargs['elite_ratio'] = param_value
                elif param_name == 'bcbo_ratio':
                    kwargs['bcbo_ratio'] = param_value

                # 创建调度器
                scheduler = BCBO_DE_Embedded(**kwargs)

                import time
                start_time = time.time()

                best_solution, best_fitness, history = scheduler.optimize(max_iterations=iterations)

                end_time = time.time()

                makespans.append(best_fitness)
                exec_times.append(end_time - start_time)

                # 计算收敛迭代次数
                if history:
                    final_fitness = history[-1]['best_fitness']
                    target_fitness = final_fitness * 0.95

                    converged_iter = iterations
                    for i, record in enumerate(history):
                        if record['best_fitness'] <= target_fitness:
                            converged_iter = i + 1
                            break

                    conv_iters.append(converged_iter)

                print(f"  Run {run+1}/{runs}: Makespan={best_fitness:.2f}", end='\r')

            print()  # 换行

            results['makespans'].append(makespans)
            results['mean_makespans'].append(np.mean(makespans))
            results['std_makespans'].append(np.std(makespans, ddof=1))
            results['convergence_iters'].append(np.mean(conv_iters))
            results['execution_times'].append(np.mean(exec_times))

            print(f"  均值: {np.mean(makespans):.2f} ± {np.std(makespans, ddof=1):.2f}")

        # 找到最优参数值
        optimal_idx = np.argmin(results['mean_makespans'])
        optimal_value = param_values[optimal_idx]
        optimal_makespan = results['mean_makespans'][optimal_idx]

        results['optimal_value'] = optimal_value
        results['optimal_makespan'] = optimal_makespan

        print(f"\n[最优] {param_name} = {optimal_value}, Makespan = {optimal_makespan:.2f}")

        return results

    def run_full_sensitivity_analysis(self,
                                     M: int = 100,
                                     N: int = 20,
                                     n: int = 50,
                                     iterations: int = 100,
                                     runs: int = 10):
        """
        运行完整的参数敏感性分析

        参数:
            M: 任务数
            N: 虚拟机数
            n: 种群大小
            iterations: 迭代次数
            runs: 每个参数值的运行次数
        """
        print("="*80)
        print("BCBO-DE参数敏感性分析".center(80))
        print("="*80)

        # 定义参数范围
        param_configs = {
            'F_max': np.linspace(0.2, 0.6, 9),
            'CR_max': np.linspace(0.5, 0.9, 9),
            'elite_ratio': np.linspace(0.10, 0.30, 9),
            'bcbo_ratio': np.linspace(0.70, 0.95, 9)
        }

        # 分析每个参数
        for param_name, param_values in param_configs.items():
            result = self.analyze_parameter(
                param_name,
                param_values.tolist(),
                M, N, n, iterations, runs
            )
            self.results[param_name] = result

    def plot_sensitivity_results(self):
        """绘制参数敏感性分析图表"""
        if not self.results:
            print("错误: 没有分析结果")
            return

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BCBO-DE Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

        param_names = list(self.results.keys())
        param_labels = {
            'F_max': 'F_max (Mutation Factor)',
            'CR_max': 'CR_max (Crossover Probability)',
            'elite_ratio': 'Elite Ratio',
            'bcbo_ratio': 'BCBO Ratio'
        }

        for idx, (ax, param_name) in enumerate(zip(axes.flatten(), param_names)):
            result = self.results[param_name]

            param_values = result['param_values']
            mean_makespans = result['mean_makespans']
            std_makespans = result['std_makespans']
            optimal_value = result['optimal_value']

            # 绘制主曲线
            ax.plot(param_values, mean_makespans,
                   marker='o', markersize=8, linewidth=2.5,
                   color='#2E86AB', label='Mean Makespan')

            # 填充标准差区域
            ax.fill_between(param_values,
                           np.array(mean_makespans) - np.array(std_makespans),
                           np.array(mean_makespans) + np.array(std_makespans),
                           alpha=0.2, color='#2E86AB', label='±1 Std Dev')

            # 标注最优值
            optimal_idx = param_values.index(optimal_value)
            ax.scatter([optimal_value], [mean_makespans[optimal_idx]],
                      s=200, marker='*', color='#C73E1D',
                      edgecolor='#8B0000', linewidth=2, zorder=5,
                      label=f'Optimal: {optimal_value:.2f}')

            # 图表装饰
            ax.set_xlabel(param_labels[param_name], fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Makespan', fontsize=11, fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {param_labels[param_name]} Sensitivity', fontsize=11)
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(fontsize=9, loc='best')

            # 添加参考线
            ax.axhline(y=mean_makespans[optimal_idx], color='red',
                      linestyle='--', linewidth=1, alpha=0.5)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"parameter_sensitivity_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[保存] 参数敏感性图表: {output_path}")

        return str(output_path)

    def generate_summary_table(self) -> str:
        """生成参数敏感性分析摘要表格"""
        if not self.results:
            return "错误: 没有分析结果"

        md = "# BCBO-DE参数敏感性分析结果\n\n"
        md += "## 最优参数配置\n\n"
        md += "| 参数 | 测试范围 | 最优值 | 最优Makespan | 影响程度 |\n"
        md += "|------|---------|--------|-------------|----------|\n"

        for param_name, result in self.results.items():
            param_range = f"[{min(result['param_values']):.2f}, {max(result['param_values']):.2f}]"
            optimal_value = result['optimal_value']
            optimal_makespan = result['optimal_makespan']

            # 计算影响程度 (最大makespan与最小makespan的差异百分比)
            max_makespan = max(result['mean_makespans'])
            min_makespan = min(result['mean_makespans'])
            impact = ((max_makespan - min_makespan) / min_makespan) * 100

            if impact > 5.0:
                impact_level = "高"
            elif impact > 2.0:
                impact_level = "中"
            else:
                impact_level = "低"

            md += f"| {param_name} | {param_range} | **{optimal_value:.2f}** | "
            md += f"{optimal_makespan:.2f} | {impact_level} ({impact:.1f}%) |\n"

        md += "\n## 推荐配置\n\n"
        md += "基于参数敏感性分析,推荐使用以下参数配置:\n\n"
        md += "```python\n"
        md += "BCBO_DE_Embedded(\n"
        for param_name, result in self.results.items():
            md += f"    {param_name}={result['optimal_value']:.2f},\n"
        md += "    # ... 其他参数\n"
        md += ")\n"
        md += "```\n"

        return md

    def save_results(self):
        """保存分析结果"""
        if not self.results:
            print("错误: 没有分析结果")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON
        json_path = self.output_dir / f"parameter_sensitivity_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"[保存] JSON结果: {json_path}")

        # 保存Markdown报告
        md_content = self.generate_summary_table()
        md_path = self.output_dir / f"parameter_sensitivity_report_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"[保存] Markdown报告: {md_path}")

        # 绘制并保存图表
        self.plot_sensitivity_results()


def main():
    """主函数"""
    print("="*80)
    print("BCBO-DE参数敏感性分析工具".center(80))
    print("="*80)

    # 初始化分析器
    analyzer = ParameterSensitivityAnalyzer()

    # 运行完整分析
    # 注意: 10次运行 x 9个参数值 x 4个参数 = 360次实验
    analyzer.run_full_sensitivity_analysis(
        M=100,
        N=20,
        n=50,
        iterations=100,
        runs=10
    )

    # 保存结果
    analyzer.save_results()

    print("\n" + "="*80)
    print("参数敏感性分析完成!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
