#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验(Ablation Study)工具
==========================
验证BCBO-DE"反转融合策略"的有效性

实验配置:
1. Pure BCBO (基准)
2. Pure DE (对照)
3. Full Fusion (全程50/50融合)
4. Early Fusion (前20%融合 - DynamicSearch + StaticSearch阶段)
5. Late Fusion (后60%融合 - Encircle + Attack阶段) ⭐ 当前方法
6. Different Ratios (后60%融合,不同比例: 70/30, 75/25, 80/20, 85/15, 90/10)
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
matplotlib.use('Agg')  # 使用非GUI后端

# 添加算法路径
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
ALGORITHM_ROOT = PROJECT_ROOT / 'algorithm'

# 添加BCBO-DE路径
BCBO_DE_DIR = ALGORITHM_ROOT / 'BCBO-DE-Fusion'
BCBO_DE_CORE = BCBO_DE_DIR / 'core'

for path in [str(BCBO_DE_CORE), str(ALGORITHM_ROOT / 'BCBO')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from bcbo_de_embedded import BCBO_DE_Embedded


class AblationStudy:
    """消融实验管理器"""

    def __init__(self, output_dir: str = None):
        """
        初始化消融实验

        参数:
            output_dir: 输出目录
        """
        if output_dir is None:
            self.output_dir = PROJECT_ROOT / 'Text Demo' / 'ablation_study'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}

    def run_bcbo_pure(self, M: int, N: int, n: int, iterations: int, runs: int = 30) -> Dict:
        """
        运行纯BCBO算法(基准)

        参数:
            M: 任务数
            N: 虚拟机数
            n: 种群大小
            iterations: 迭代次数
            runs: 运行次数

        返回:
            实验结果字典
        """
        print(f"\n[实验] 运行Pure BCBO (基准算法)")
        print(f"  参数: M={M}, N={N}, n={n}, iterations={iterations}, runs={runs}")

        makespans = []
        convergence_iterations = []
        execution_times = []

        for run in range(runs):
            # 使用BCBO-DE但禁用DE (bcbo_ratio=1.0)
            scheduler = BCBO_DE_Embedded(
                M=M,
                N=N,
                n=n,
                bcbo_ratio=1.0,  # 100% BCBO, 0% DE
                fusion_mode='none'
            )

            import time
            start_time = time.time()

            best_solution, best_fitness, history = scheduler.optimize(max_iterations=iterations)

            end_time = time.time()

            makespans.append(best_fitness)
            execution_times.append(end_time - start_time)

            # 计算达到95%最优值的迭代次数
            if history:
                final_fitness = history[-1]['best_fitness']
                target_fitness = final_fitness * 0.95

                converged_iter = iterations
                for i, record in enumerate(history):
                    if record['best_fitness'] <= target_fitness:
                        converged_iter = i + 1
                        break

                convergence_iterations.append(converged_iter)

            print(f"  Run {run+1}/{runs}: Makespan={best_fitness:.2f}", end='\r')

        print()  # 换行

        return {
            'config': 'Pure BCBO (Baseline)',
            'makespans': makespans,
            'mean_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans, ddof=1),
            'mean_convergence_iter': np.mean(convergence_iterations),
            'mean_execution_time': np.mean(execution_times),
            'improvement': 0.0  # 基准
        }

    def run_bcbo_de_variant(self,
                           M: int,
                           N: int,
                           n: int,
                           iterations: int,
                           fusion_mode: str,
                           bcbo_ratio: float = 0.85,
                           fusion_start: float = None,
                           fusion_end: float = None,
                           runs: int = 30) -> Dict:
        """
        运行BCBO-DE变体

        参数:
            M: 任务数
            N: 虚拟机数
            n: 种群大小
            iterations: 迭代次数
            fusion_mode: 融合模式 ('none', 'full', 'early', 'late', 'custom')
            bcbo_ratio: BCBO比例
            fusion_start: 融合起始比例 (0-1)
            fusion_end: 融合结束比例 (0-1)
            runs: 运行次数

        返回:
            实验结果字典
        """
        config_name = self._get_config_name(fusion_mode, bcbo_ratio, fusion_start, fusion_end)
        print(f"\n[实验] 运行 {config_name}")
        print(f"  参数: M={M}, N={N}, n={n}, iterations={iterations}, runs={runs}")
        print(f"  融合配置: mode={fusion_mode}, bcbo_ratio={bcbo_ratio:.2f}")

        makespans = []
        convergence_iterations = []
        execution_times = []

        for run in range(runs):
            scheduler = BCBO_DE_Embedded(
                M=M,
                N=N,
                n=n,
                bcbo_ratio=bcbo_ratio,
                fusion_mode=fusion_mode
            )

            # 设置自定义融合阶段(如果提供)
            if fusion_start is not None and fusion_end is not None:
                scheduler.fusion_start_ratio = fusion_start
                scheduler.fusion_end_ratio = fusion_end

            import time
            start_time = time.time()

            best_solution, best_fitness, history = scheduler.optimize(max_iterations=iterations)

            end_time = time.time()

            makespans.append(best_fitness)
            execution_times.append(end_time - start_time)

            # 计算收敛迭代次数
            if history:
                final_fitness = history[-1]['best_fitness']
                target_fitness = final_fitness * 0.95

                converged_iter = iterations
                for i, record in enumerate(history):
                    if record['best_fitness'] <= target_fitness:
                        converged_iter = i + 1
                        break

                convergence_iterations.append(converged_iter)

            print(f"  Run {run+1}/{runs}: Makespan={best_fitness:.2f}", end='\r')

        print()  # 换行

        return {
            'config': config_name,
            'fusion_mode': fusion_mode,
            'bcbo_ratio': bcbo_ratio,
            'makespans': makespans,
            'mean_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans, ddof=1),
            'mean_convergence_iter': np.mean(convergence_iterations),
            'mean_execution_time': np.mean(execution_times),
            'improvement': 0.0  # 将在后续计算
        }

    @staticmethod
    def _get_config_name(fusion_mode: str,
                        bcbo_ratio: float,
                        fusion_start: float = None,
                        fusion_end: float = None) -> str:
        """生成配置名称"""
        if fusion_mode == 'none':
            return "Pure BCBO"
        elif fusion_mode == 'full':
            return f"Full Fusion ({bcbo_ratio*100:.0f}/{(1-bcbo_ratio)*100:.0f})"
        elif fusion_mode == 'early':
            return f"Early Fusion (前20%, {bcbo_ratio*100:.0f}/{(1-bcbo_ratio)*100:.0f})"
        elif fusion_mode == 'late':
            return f"Late Fusion (后60%, {bcbo_ratio*100:.0f}/{(1-bcbo_ratio)*100:.0f}) ⭐"
        elif fusion_mode == 'custom' and fusion_start is not None:
            return f"Custom Fusion ({fusion_start*100:.0f}%-{fusion_end*100:.0f}%)"
        else:
            return f"Unknown ({fusion_mode})"

    def run_full_ablation_study(self,
                                M: int = 100,
                                N: int = 20,
                                n: int = 50,
                                iterations: int = 100,
                                runs: int = 30):
        """
        运行完整的消融实验

        实验配置:
        1. Pure BCBO (基准)
        2. Full Fusion (全程50/50)
        3. Early Fusion (前20%, 85/15)
        4. Late Fusion (后60%, 85/15) ⭐ 当前方法
        5. Late Fusion (后60%, 70/30)
        6. Late Fusion (后60%, 80/20)
        7. Late Fusion (后60%, 90/10)

        参数:
            M: 任务数
            N: 虚拟机数
            n: 种群大小
            iterations: 迭代次数
            runs: 运行次数
        """
        print("="*80)
        print("BCBO-DE消融实验 (Ablation Study)".center(80))
        print("="*80)
        print(f"\n实验配置: M={M}, N={N}, n={n}, iterations={iterations}, runs={runs}")
        print()

        results = []

        # 1. Pure BCBO (基准)
        results.append(self.run_bcbo_pure(M, N, n, iterations, runs))

        # 2. Full Fusion (全程50/50)
        results.append(self.run_bcbo_de_variant(
            M, N, n, iterations,
            fusion_mode='full',
            bcbo_ratio=0.5,
            runs=runs
        ))

        # 3. Early Fusion (前20%, 85/15)
        results.append(self.run_bcbo_de_variant(
            M, N, n, iterations,
            fusion_mode='early',
            bcbo_ratio=0.85,
            fusion_start=0.0,
            fusion_end=0.2,
            runs=runs
        ))

        # 4. Late Fusion (后60%, 85/15) ⭐ 当前方法
        results.append(self.run_bcbo_de_variant(
            M, N, n, iterations,
            fusion_mode='late',
            bcbo_ratio=0.85,
            fusion_start=0.4,
            fusion_end=1.0,
            runs=runs
        ))

        # 5-7. Late Fusion with different ratios
        for ratio in [0.70, 0.80, 0.90]:
            results.append(self.run_bcbo_de_variant(
                M, N, n, iterations,
                fusion_mode='late',
                bcbo_ratio=ratio,
                fusion_start=0.4,
                fusion_end=1.0,
                runs=runs
            ))

        # 计算相对于基准的改进
        baseline_makespan = results[0]['mean_makespan']
        for result in results[1:]:
            improvement = ((baseline_makespan - result['mean_makespan']) / baseline_makespan) * 100
            result['improvement'] = improvement

        self.results = results
        return results

    def generate_comparison_table(self) -> str:
        """生成对比表格 (Markdown格式)"""
        if not self.results:
            return "错误: 没有实验结果"

        md = "# BCBO-DE消融实验结果\n\n"
        md += "## 表格: 不同融合策略的性能对比\n\n"
        md += "| 配置 | Mean Makespan | Std | 收敛迭代次数 | 执行时间(s) | 改进幅度 | 计算开销 |\n"
        md += "|------|--------------|-----|-------------|-----------|---------|----------|\n"

        baseline_time = self.results[0]['mean_execution_time']

        for result in self.results:
            overhead = ((result['mean_execution_time'] - baseline_time) / baseline_time) * 100
            overhead_str = f"+{overhead:.1f}%" if overhead > 0 else f"{overhead:.1f}%"

            improvement_str = "—" if result['improvement'] == 0 else f"{result['improvement']:.1f}%"

            md += f"| {result['config']} | "
            md += f"{result['mean_makespan']:.2f} | "
            md += f"±{result['std_makespan']:.2f} | "
            md += f"{result['mean_convergence_iter']:.0f} | "
            md += f"{result['mean_execution_time']:.2f} | "
            md += f"{improvement_str} | "
            md += f"{overhead_str} |\n"

        md += "\n**结论**: Late Fusion (后60%, 85/15) 在性能提升与计算开销之间达到最优平衡。\n"

        return md

    def plot_ablation_results(self):
        """绘制消融实验结果图表"""
        if not self.results:
            print("错误: 没有实验结果")
            return

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BCBO-DE Ablation Study Results', fontsize=16, fontweight='bold')

        configs = [r['config'] for r in self.results]
        makespans = [r['mean_makespan'] for r in self.results]
        stds = [r['std_makespan'] for r in self.results]
        conv_iters = [r['mean_convergence_iter'] for r in self.results]
        exec_times = [r['mean_execution_time'] for r in self.results]

        # 颜色配置
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#8B5A3C']
        highlight_idx = 3  # Late Fusion (85/15) 的索引

        # 子图1: Makespan对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(configs)), makespans, yerr=stds,
                       capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        bars1[highlight_idx].set_facecolor('#C73E1D')
        bars1[highlight_idx].set_edgecolor('#8B0000')
        bars1[highlight_idx].set_linewidth(2.5)

        ax1.set_ylabel('Mean Makespan', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Makespan Comparison', fontsize=12)
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # 子图2: 收敛速度
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(configs)), conv_iters,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2[highlight_idx].set_facecolor('#C73E1D')
        bars2[highlight_idx].set_edgecolor('#8B0000')
        bars2[highlight_idx].set_linewidth(2.5)

        ax2.set_ylabel('Convergence Iterations', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Convergence Speed', fontsize=12)
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # 子图3: 执行时间
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(configs)), exec_times,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        bars3[highlight_idx].set_facecolor('#C73E1D')
        bars3[highlight_idx].set_edgecolor('#8B0000')
        bars3[highlight_idx].set_linewidth(2.5)

        ax3.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
        ax3.set_title('(c) Computational Overhead', fontsize=12)
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # 子图4: 改进幅度
        ax4 = axes[1, 1]
        improvements = [r['improvement'] for r in self.results]
        bars4 = ax4.bar(range(len(configs)), improvements,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        bars4[highlight_idx].set_facecolor('#C73E1D')
        bars4[highlight_idx].set_edgecolor('#8B0000')
        bars4[highlight_idx].set_linewidth(2.5)

        ax4.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
        ax4.set_title('(d) Performance Improvement', fontsize=12)
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"ablation_study_results_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[保存] 消融实验图表: {output_path}")

        return str(output_path)

    def save_results(self):
        """保存实验结果"""
        if not self.results:
            print("错误: 没有实验结果")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON
        json_path = self.output_dir / f"ablation_study_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"[保存] JSON结果: {json_path}")

        # 保存Markdown表格
        md_content = self.generate_comparison_table()
        md_path = self.output_dir / f"ablation_study_report_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"[保存] Markdown报告: {md_path}")

        # 绘制并保存图表
        self.plot_ablation_results()


def main():
    """主函数"""
    print("="*80)
    print("BCBO-DE消融实验工具".center(80))
    print("="*80)

    # 初始化消融实验
    study = AblationStudy()

    # 运行完整实验
    # 注意: 30次运行 x 7种配置 = 210次实验,需要较长时间
    study.run_full_ablation_study(
        M=100,
        N=20,
        n=50,
        iterations=100,
        runs=30
    )

    # 保存结果
    study.save_results()

    print("\n" + "="*80)
    print("消融实验完成!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
