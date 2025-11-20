#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE变异策略模块
==============
实现多种差分进化变异策略

包含策略:
- DE/rand/1: 随机基向量变异
- DE/best/1: 最优解基向量变异
- DE/current-to-best/1: 引导式变异
- AdaptiveDE: 自适应变异(基于HDE论文)

参考文献:
[1] Storn & Price (1997): Differential Evolution
[2] Mohamed et al. (2020): HDE - Hybrid Differential Evolution
"""

import numpy as np
import random
from typing import List, Optional


class MutationStrategy:
    """变异策略基类"""

    def __init__(self, F: float = 0.5):
        """
        初始化变异策略

        参数:
            F: 缩放因子 (scaling factor), 控制差分向量的缩放强度
        """
        self.F = F

    def mutate(self,
               population: List[List[int]],
               target: List[int],
               best_solution: Optional[List[int]] = None,
               N: int = None,
               **kwargs) -> List[int]:
        """
        执行变异操作

        参数:
            population: 当前种群
            target: 目标个体
            best_solution: 当前最优解
            N: VM数量 (用于边界处理)
            **kwargs: 其他参数

        返回:
            mutant: 变异向量
        """
        raise NotImplementedError("子类必须实现mutate方法")


class DE_Rand_1(MutationStrategy):
    """
    DE/rand/1变异策略

    公式: V_i = X_r1 + F * (X_r2 - X_r3)

    特点:
    - 随机选择基向量,探索能力强
    - 适合多峰函数优化
    - 对参数不敏感
    - 收敛速度中等

    适用场景:
    - 云任务调度的全局搜索阶段
    - BCBO的弱搜索阶段增强
    """

    def mutate(self,
               population: List[List[int]],
               target: List[int],
               best_solution: Optional[List[int]] = None,
               N: int = None,
               **kwargs) -> List[int]:
        """
        执行DE/rand/1变异

        步骤:
        1. 随机选择三个不同的个体 r1, r2, r3
        2. 计算差分向量: V = r1 + F * (r2 - r3)
        3. 边界处理: 确保VM索引在[0, N-1]范围内

        参数:
            population: 种群
            target: 目标个体 (用于排除,不参与变异)
            N: VM数量

        返回:
            mutant: 变异个体
        """
        # 排除目标个体,从剩余个体中选择
        candidates = [ind for ind in population if ind != target]

        # 如果候选个体不足3个,返回目标个体的副本
        if len(candidates) < 3:
            return target.copy()

        # 随机选择三个不同的个体
        r1, r2, r3 = random.sample(candidates, 3)

        M = len(target)  # 任务数量
        mutant = []

        # 对每个任务执行差分变异
        for j in range(M):
            # 差分变异公式: V[j] = r1[j] + F * (r2[j] - r3[j])
            mutant_value = r1[j] + self.F * (r2[j] - r3[j])

            # 边界处理: 限制在[0, N-1]范围,并转换为整数
            if N is not None:
                mutant_value = int(np.clip(mutant_value, 0, N - 1))
            else:
                mutant_value = int(mutant_value)

            mutant.append(mutant_value)

        return mutant


class DE_Best_1(MutationStrategy):
    """
    DE/best/1变异策略

    公式: V_i = X_best + F * (X_r1 - X_r2)

    特点:
    - 以最优解为基向量,收敛快
    - 开发能力强,探索能力弱
    - 容易陷入局部最优
    - 适合后期精细调优

    适用场景:
    - 收敛阶段加速
    - 已找到较好解,需要精确优化
    """

    def mutate(self,
               population: List[List[int]],
               target: List[int],
               best_solution: List[int],
               N: int = None,
               **kwargs) -> List[int]:
        """
        执行DE/best/1变异

        步骤:
        1. 使用当前最优解作为基向量
        2. 随机选择两个不同个体 r1, r2
        3. 计算: V = best + F * (r1 - r2)
        """
        # 如果没有提供最优解,降级为DE/rand/1
        if best_solution is None:
            return DE_Rand_1(self.F).mutate(population, target, N=N)

        # 随机选择两个不同个体
        candidates = [ind for ind in population if ind != target]
        if len(candidates) < 2:
            return target.copy()

        r1, r2 = random.sample(candidates, 2)

        M = len(target)
        mutant = []

        for j in range(M):
            # 以最优解为基: V[j] = best[j] + F * (r1[j] - r2[j])
            mutant_value = best_solution[j] + self.F * (r1[j] - r2[j])

            if N is not None:
                mutant_value = int(np.clip(mutant_value, 0, N - 1))
            else:
                mutant_value = int(mutant_value)

            mutant.append(mutant_value)

        return mutant


class DE_Current_To_Best_1(MutationStrategy):
    """
    DE/current-to-best/1变异策略

    公式: V_i = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)

    特点:
    - 引导当前解向最优解靠拢
    - 平衡探索和开发
    - 适合动态优化问题
    - 对当前解的改进更有方向性

    适用场景:
    - 需要快速收敛但保持一定多样性
    - 中后期优化阶段
    """

    def mutate(self,
               population: List[List[int]],
               target: List[int],
               best_solution: List[int],
               N: int = None,
               **kwargs) -> List[int]:
        """
        执行DE/current-to-best/1变异

        步骤:
        1. 计算从当前解到最优解的向量
        2. 添加随机差分扰动
        3. V = current + F*(best - current) + F*(r1 - r2)
        """
        if best_solution is None:
            return DE_Rand_1(self.F).mutate(population, target, N=N)

        candidates = [ind for ind in population if ind != target]
        if len(candidates) < 2:
            return target.copy()

        r1, r2 = random.sample(candidates, 2)

        M = len(target)
        mutant = []

        for j in range(M):
            # V[j] = current[j] + F*(best[j] - current[j]) + F*(r1[j] - r2[j])
            mutant_value = (target[j] +
                          self.F * (best_solution[j] - target[j]) +
                          self.F * (r1[j] - r2[j]))

            if N is not None:
                mutant_value = int(np.clip(mutant_value, 0, N - 1))
            else:
                mutant_value = int(mutant_value)

            mutant.append(mutant_value)

        return mutant


class AdaptiveDE(MutationStrategy):
    """
    自适应DE变异策略

    基于HDE论文的自适应缩放因子:
    F' = alpha * (T - t) / T

    特点:
    - 早期迭代: F大,探索能力强
    - 后期迭代: F小,开发能力强
    - 自动平衡探索和开发
    - 减少参数调优工作

    参考: Mohamed et al. (2020) - HDE论文
    """

    def __init__(self, F_base: float = 0.5, alpha: float = 0.1):
        """
        初始化自适应DE

        参数:
            F_base: 基础缩放因子
            alpha: 自适应系数 (建议范围: 0.05-0.2)
        """
        super().__init__(F_base)
        self.alpha = alpha
        self.base_strategy = DE_Rand_1(F_base)

    def adaptive_F(self, iteration: int, total_iterations: int) -> float:
        """
        计算自适应缩放因子

        公式: F' = alpha * (T - t) / T

        其中:
        - T: 总迭代次数
        - t: 当前迭代次数
        - alpha: 自适应系数

        效果:
        - 迭代0: F' = alpha * 1.0 (最大,探索)
        - 迭代T/2: F' = alpha * 0.5 (中等)
        - 迭代T: F' = alpha * 0.0 (最小,开发)

        参数:
            iteration: 当前迭代次数
            total_iterations: 总迭代次数

        返回:
            F': 自适应缩放因子
        """
        if total_iterations == 0:
            return self.F

        return self.alpha * (total_iterations - iteration) / total_iterations

    def mutate(self,
               population: List[List[int]],
               target: List[int],
               iteration: int = 0,
               total_iterations: int = 100,
               **kwargs) -> List[int]:
        """
        执行自适应变异

        步骤:
        1. 根据当前迭代计算自适应F
        2. 使用DE/rand/1策略进行变异

        参数:
            population: 种群
            target: 目标个体
            iteration: 当前迭代次数
            total_iterations: 总迭代次数

        返回:
            mutant: 变异个体
        """
        # 计算自适应F
        adaptive_f = self.adaptive_F(iteration, total_iterations)

        # 更新基础策略的F值
        self.base_strategy.F = adaptive_f

        # 使用更新后的F进行变异
        return self.base_strategy.mutate(population, target, **kwargs)


# 导出所有策略
__all__ = [
    'MutationStrategy',
    'DE_Rand_1',
    'DE_Best_1',
    'DE_Current_To_Best_1',
    'AdaptiveDE'
]
