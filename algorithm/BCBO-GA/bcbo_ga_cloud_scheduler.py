#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-GA: Multi-Strategy Collaborative BCBO Algorithm
多策略协同BCBO算法 - 适合期刊发表的创新算法

作者: [您的名字]
版本: v1.0
日期: 2025-11-29

创新点:
1. 并行子种群进化
2. 多策略协同（原始BCBO、Levy飞行、混沌映射、量子行为）
3. 动态资源分配
4. 信息交换机制
"""

import numpy as np
import copy
import sys
import os
from typing import List, Dict, Tuple
import math

# 添加BCBO路径
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_dir = os.path.join(current_dir, '..', '..', 'algorithm')
bcbo_path = os.path.join(algorithm_dir, 'BCBO')
if bcbo_path not in sys.path:
    sys.path.insert(0, bcbo_path)

from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler


class BCBO_GA_CloudScheduler:
    """多策略协同BCBO云任务调度算法"""

    def __init__(self, M: int, N: int, n: int, iterations: int, **kwargs):
        """
        初始化BCBO-GA算法

        参数:
            M: 任务数量
            N: 虚拟机数量
            n: 总种群大小
            iterations: 迭代次数
        """
        self.M = M
        self.N = N
        self.n = n
        self.iterations = iterations
        self.verbose = kwargs.get('verbose', True)

        # 创建基础BCBO实例
        self.bcbo = BCBO_CloudScheduler(M=M, N=N, n=n, iterations=iterations)

        # 子种群配置
        self.num_strategies = 4
        self.strategy_names = ['Original', 'Levy', 'Chaos', 'Quantum']

        # 初始资源分配（优化版：给表现好的策略更多资源）
        self.subpop_ratios = np.array([0.25, 0.25, 0.25, 0.25])  # 均衡分配
        self.subpop_sizes = self._calculate_subpop_sizes()

        # 性能跟踪
        self.strategy_performance = np.zeros(self.num_strategies)
        self.performance_window = 5   # 降低窗口大小，更快响应
        self.performance_history = {i: [] for i in range(self.num_strategies)}

        # 信息交换参数（加强）
        self.exchange_rate = 0.2     # 提高到20%
        self.exchange_interval = 3   # 更频繁交换

        # 混沌映射参数
        self.chaos_value = 0.7  # 初始混沌值

        # Levy飞行参数（优化）
        self.levy_beta = 1.8    # 提高beta值

        # 量子参数（优化）
        self.quantum_delta = 0.1 * np.pi  # 增大旋转角

        # 全局最优
        self.global_best_solution = None
        self.global_best_fitness = float('-inf')

        # 添加收敛历史记录
        self.fitness_history = []

        if self.verbose:
            print(f"BCBO-GA多策略协同算法初始化完成")
            print(f"  任务数M={M}, VM数N={N}, 总种群n={n}")
            print(f"  策略: {self.strategy_names}")
            print(f"  初始分配: {self.subpop_ratios}")

    def _calculate_subpop_sizes(self) -> List[int]:
        """根据比例计算子种群大小"""
        sizes = (self.subpop_ratios * self.n).astype(int)
        # 确保总数等于n
        sizes[-1] = self.n - np.sum(sizes[:-1])
        return sizes.tolist()

    def optimize(self) -> Dict:
        """
        执行BCBO-GA优化

        返回:
            result: 包含最优解和性能指标的字典
        """
        if self.verbose:
            print("\n" + "="*60)
            print("开始BCBO-GA多策略协同优化".center(60))
            print("="*60)

        # 初始化子种群
        subpopulations = self._initialize_subpopulations()

        # 主循环
        for iteration in range(self.iterations):
            # 1. 并行执行各策略
            for i in range(self.num_strategies):
                subpopulations[i] = self._execute_strategy(
                    i, subpopulations[i], iteration
                )

            # 2. 更新全局最优
            self._update_global_best(subpopulations)

            # 记录收敛历史
            self.fitness_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.global_best_fitness,
                'best_solution': copy.deepcopy(self.global_best_solution),
                'global_best_fitness': self.global_best_fitness
            })

            # 3. 信息交换（每隔一定代数）
            if (iteration + 1) % self.exchange_interval == 0:
                subpopulations = self._information_exchange(subpopulations)

            # 4. 动态资源分配（基于性能）
            if iteration > 0 and (iteration + 1) % self.performance_window == 0:
                self._dynamic_resource_allocation(subpopulations)

            # 5. 打印进度
            if self.verbose and iteration % 10 == 0:
                self._print_progress(iteration)

        # 生成结果
        result = {
            'best_solution': self.global_best_solution,
            'best_fitness': self.global_best_fitness,
            'strategy_contributions': self._calculate_contributions(),
            'final_ratios': self.subpop_ratios.tolist(),
            'convergence_history': self.fitness_history  # 添加收敛历史
        }

        if self.verbose:
            print("\n" + "="*60)
            print("优化完成".center(60))
            print("="*60)
            print(f"最优适应度: {self.global_best_fitness:.6f}")
            print(f"策略贡献度: {result['strategy_contributions']}")

        return result

    def _initialize_subpopulations(self) -> List[List]:
        """初始化子种群"""
        full_population = self.bcbo.initialize_population()
        subpopulations = []

        start_idx = 0
        for size in self.subpop_sizes:
            end_idx = start_idx + size
            subpopulations.append(full_population[start_idx:end_idx])
            start_idx = end_idx

        return subpopulations

    def _execute_strategy(self, strategy_id: int, population: List,
                         iteration: int) -> List:
        """执行特定策略"""
        if strategy_id == 0:
            # 策略1: 原始BCBO
            return self._bcbo_original(population, iteration)
        elif strategy_id == 1:
            # 策略2: BCBO + Levy飞行
            return self._bcbo_levy(population, iteration)
        elif strategy_id == 2:
            # 策略3: BCBO + 混沌映射
            return self._bcbo_chaos(population, iteration)
        else:
            # 策略4: BCBO + 量子行为
            return self._bcbo_quantum(population, iteration)

    def _bcbo_original(self, population: List, iteration: int) -> List:
        """原始BCBO策略"""
        # 使用BCBO的标准更新
        phase = self._determine_phase(iteration)
        if 'dynamic' in phase:
            return self.bcbo.dynamic_search_phase(population, iteration, self.iterations)
        else:
            return self.bcbo.static_search_phase(population, iteration)

    def _bcbo_levy(self, population: List, iteration: int) -> List:
        """BCBO + Levy飞行策略（增强版）"""
        # 先执行标准BCBO更新
        population = self._bcbo_original(population, iteration)

        # 应用增强的Levy飞行
        best_individual = max(population, key=lambda x: self.bcbo.comprehensive_fitness(x))

        for i in range(len(population)):
            if np.random.random() < 0.4:  # 提高到40%概率
                levy_step = self._levy_flight()

                # 向最优解方向飞行
                for j in range(self.M):
                    if np.random.random() < 0.6:  # 60%的任务受影响
                        # 结合最优解和Levy飞行
                        if best_individual[j] != population[i][j]:
                            # 有概率向最优解靠拢
                            if np.random.random() < 0.5:
                                population[i][j] = best_individual[j]
                            else:
                                # Levy飞行探索
                                new_vm = int(population[i][j] + levy_step * self.N) % self.N
                                population[i][j] = new_vm

        return population

    def _bcbo_chaos(self, population: List, iteration: int) -> List:
        """BCBO + 混沌映射策略"""
        # 先执行标准BCBO更新
        population = self._bcbo_original(population, iteration)

        # 应用混沌映射增强多样性
        for i in range(len(population)):
            if np.random.random() < 0.25:  # 25%概率应用混沌
                # Logistic混沌映射
                self.chaos_value = 4 * self.chaos_value * (1 - self.chaos_value)

                # 基于混沌值调整
                chaos_positions = int(self.chaos_value * self.M)
                for _ in range(min(5, chaos_positions)):
                    j = np.random.randint(0, self.M)
                    population[i][j] = np.random.randint(0, self.N)

        return population

    def _bcbo_quantum(self, population: List, iteration: int) -> List:
        """BCBO + 量子行为策略"""
        # 先执行标准BCBO更新
        population = self._bcbo_original(population, iteration)

        # 应用量子行为
        for i in range(len(population)):
            if np.random.random() < 0.2:  # 20%概率应用量子行为
                # 量子旋转门操作
                for j in range(self.M):
                    if np.random.random() < 0.3:
                        # 量子坍缩到新状态
                        alpha = np.cos(self.quantum_delta)
                        beta = np.sin(self.quantum_delta)

                        # 概率性选择新VM
                        if np.random.random() < beta**2:
                            population[i][j] = np.random.randint(0, self.N)

        return population

    def _levy_flight(self) -> float:
        """生成Levy飞行步长"""
        # Mantegna算法生成Levy分布
        sigma = (math.gamma(1 + self.levy_beta) * np.sin(np.pi * self.levy_beta / 2) /
                (math.gamma((1 + self.levy_beta) / 2) * self.levy_beta *
                 2**((self.levy_beta - 1) / 2)))**(1 / self.levy_beta)

        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)

        step = u / abs(v)**(1 / self.levy_beta)
        return step

    def _information_exchange(self, subpopulations: List[List]) -> List[List]:
        """子种群间信息交换"""
        # 每个子种群选出最优个体
        best_individuals = []
        for subpop in subpopulations:
            best_idx = np.argmax([self.bcbo.comprehensive_fitness(ind) for ind in subpop])
            best_individuals.append(copy.deepcopy(subpop[best_idx]))

        # 将最优个体复制到其他子种群
        for i, subpop in enumerate(subpopulations):
            # 替换最差的个体
            fitness_values = [self.bcbo.comprehensive_fitness(ind) for ind in subpop]
            worst_indices = np.argsort(fitness_values)[:len(best_individuals)-1]

            idx = 0
            for j in range(len(best_individuals)):
                if j != i:  # 不替换自己的最优
                    if idx < len(worst_indices):
                        subpop[worst_indices[idx]] = copy.deepcopy(best_individuals[j])
                        idx += 1

        return subpopulations

    def _dynamic_resource_allocation(self, subpopulations: List[List]):
        """动态资源分配（增强版）"""
        # 计算各策略的平均性能
        avg_performance = np.zeros(self.num_strategies)
        for i, subpop in enumerate(subpopulations):
            fitness_values = [self.bcbo.comprehensive_fitness(ind) for ind in subpop]
            avg_performance[i] = np.mean(fitness_values)

        # 基于性能调整比例（更激进的调整）
        performance_rank = np.argsort(avg_performance)[::-1]  # 降序排列

        # 调整策略：最好的+15%，次好的+5%，次差的-5%，最差的-15%
        adjustments = [0.15, 0.05, -0.05, -0.15]

        for i, rank_idx in enumerate(performance_rank):
            self.subpop_ratios[rank_idx] = np.clip(
                self.subpop_ratios[rank_idx] + adjustments[i],
                0.1,  # 最小10%
                0.5   # 最大50%
            )

        # 归一化
        self.subpop_ratios = self.subpop_ratios / np.sum(self.subpop_ratios)

        # 记录性能
        for i in range(self.num_strategies):
            self.performance_history[i].append(avg_performance[i])

    def _update_global_best(self, subpopulations: List[List]):
        """更新全局最优解"""
        for subpop in subpopulations:
            for individual in subpop:
                fitness = self.bcbo.comprehensive_fitness(individual)
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = copy.deepcopy(individual)

    def _determine_phase(self, iteration: int) -> str:
        """确定当前阶段"""
        progress = iteration / self.iterations
        if progress < 0.5:
            return 'dynamic'
        else:
            return 'static'

    def _print_progress(self, iteration: int):
        """打印进度信息"""
        print(f"迭代 {iteration:3d}/{self.iterations} | "
              f"最优: {self.global_best_fitness:.4f} | "
              f"资源分配: {self.subpop_ratios}")

    def _calculate_contributions(self) -> Dict[str, float]:
        """计算各策略的贡献度"""
        contributions = {}
        for i, name in enumerate(self.strategy_names):
            if len(self.performance_history[i]) > 0:
                contributions[name] = np.mean(self.performance_history[i][-10:])
            else:
                contributions[name] = 0.0
        return contributions


# 测试代码
if __name__ == '__main__':
    print("="*60)
    print("BCBO-GA多策略协同算法测试".center(60))
    print("="*60)

    # 创建BCBO-GA实例
    bcbo_ga = BCBO_GA_CloudScheduler(
        M=100,          # 100个任务
        N=20,           # 20个虚拟机
        n=50,           # 种群大小50
        iterations=50,  # 50次迭代
        verbose=True
    )

    # 执行优化
    result = bcbo_ga.optimize()

    # 显示结果
    print("\n测试结果:")
    print(f"  最优适应度: {result['best_fitness']:.6f}")
    print(f"  最优解前10个分配: {result['best_solution'][:10]}")
    print(f"  策略贡献度:")
    for strategy, contrib in result['strategy_contributions'].items():
        print(f"    {strategy}: {contrib:.4f}")
    print(f"  最终资源分配: {result['final_ratios']}")