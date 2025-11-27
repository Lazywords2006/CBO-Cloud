#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控器

记录和分析优化过程中的性能指标。
"""

from typing import Dict, List
import numpy as np


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        """初始化监控器"""
        self.history = {
            'iteration': [],       # 迭代次数
            'best_fitness': [],    # 最优适应度
            'avg_fitness': [],     # 平均适应度
            'diversity': [],       # 种群多样性
            'bcbo_ratio': [],      # BCBO组比例
            'phase': [],           # 当前阶段
            'best_solution': []    # 最优解历史
        }

    def record(self, iteration: int, population: List, best_fitness: float,
               bcbo_ratio: float, phase: str, fitness_func=None, best_solution=None):
        """
        记录每代的性能指标

        参数:
            iteration: 当前迭代次数
            population: 当前种群
            best_fitness: 最优适应度
            bcbo_ratio: BCBO组比例
            phase: 当前阶段
            fitness_func: 适应度函数
            best_solution: 当前最优解 (可选)
        """
        self.history['iteration'].append(iteration)
        self.history['best_fitness'].append(best_fitness)
        self.history['bcbo_ratio'].append(bcbo_ratio)
        self.history['phase'].append(phase)
        self.history['best_solution'].append(best_solution)

        # 计算平均适应度
        if fitness_func and population:
            avg_fitness = np.mean([fitness_func(ind) for ind in population])
            self.history['avg_fitness'].append(avg_fitness)
        else:
            self.history['avg_fitness'].append(best_fitness)

    def get_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.history['best_fitness']:
            return {}

        summary = {
            'final_best_fitness': self.history['best_fitness'][-1],
            'initial_best_fitness': self.history['best_fitness'][0],
            'improvement': None,
            'total_iterations': len(self.history['iteration']),
        }

        if summary['initial_best_fitness'] and summary['final_best_fitness']:
            summary['improvement'] = (
                (summary['final_best_fitness'] - summary['initial_best_fitness']) /
                abs(summary['initial_best_fitness']) * 100
            )

        return summary

    def diagnose(self) -> str:
        """诊断优化状态"""
        if len(self.history['best_fitness']) < 10:
            return "数据不足,无法诊断"

        recent_fitness = self.history['best_fitness'][-10:]
        fitness_std = np.std(recent_fitness)
        fitness_change = abs(recent_fitness[-1] - recent_fitness[0])

        if fitness_std < 0.01 and fitness_change < 0.01:
            return "已收敛"
        elif fitness_std < 0.05:
            return "可能停滞"
        elif fitness_std > 0.1:
            return "振荡中"
        else:
            return "正常优化"

    def plot_convergence(self, save_path=None):
        """绘制收敛曲线"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.history['iteration'], self.history['best_fitness'],
                    label='最优适应度', marker='o', markersize=3)
            if self.history['avg_fitness']:
                plt.plot(self.history['iteration'], self.history['avg_fitness'],
                        label='平均适应度', marker='s', markersize=3, alpha=0.7)

            plt.xlabel('迭代次数', fontsize=12)
            plt.ylabel('适应度', fontsize=12)
            plt.title('BCBO-DE收敛曲线', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"收敛曲线已保存到: {save_path}")
            else:
                plt.show()
        except ImportError:
            print("警告: 需要安装matplotlib才能绘制图表")
