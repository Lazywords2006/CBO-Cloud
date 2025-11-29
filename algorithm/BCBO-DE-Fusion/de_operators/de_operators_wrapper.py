#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE算子统一接口

为了兼容不同的调用方式,提供统一的DEOperators类。

方案C修改 (2025-11-28):
========================================
添加负载均衡约束到DE交叉操作中
- 目标: 解决BCBO-DE负载均衡退化问题
- 方法: 在交叉后检查负载均衡度，如果低于阈值则拒绝trial
- 触发条件: M>=1000时启用
"""

import random
import numpy as np
from typing import List, Callable, Optional
from .mutation_strategies import DE_Rand_1
from .crossover_strategies import BinomialCrossover
from .selection import greedy_selection


class DEOperators:
    """
    DE算子统一封装类

    提供变异、交叉、选择的统一接口。

    方案C增强版 (2025-11-28):
    - 交叉操作支持负载均衡约束检查
    """

    def __init__(self, M: int, N: int, F: float = 0.5, CR: float = 0.8,
                 execution_time: Optional[np.ndarray] = None):
        """
        初始化DE算子

        参数:
            M: 任务数量
            N: 虚拟机数量
            F: 缩放因子
            CR: 交叉概率
            execution_time: 任务执行时间矩阵 (用于负载均衡计算)
        """
        self.M = M
        self.N = N
        self.F = F
        self.CR = CR
        self.execution_time = execution_time

        # 初始化策略
        self.mutation_strategy = DE_Rand_1()
        self.crossover_strategy = BinomialCrossover(CR)

    def mutate(self, population: List[List[int]], target: List[int], F: float = None) -> List[int]:
        """
        变异操作

        参数:
            population: 当前种群
            target: 目标个体
            F: 缩放因子(可选,默认使用初始化时的F)

        返回:
            mutant: 变异向量
        """
        if F is None:
            F = self.F

        return self.mutation_strategy.mutate(population, target, F, self.N)

    def crossover(self, target: List[int], mutant: List[int], CR: float = None,
                  enable_balance_check: bool = True, balance_threshold: float = 0.75) -> List[int]:
        """
        交叉操作 (方案C增强版)

        参数:
            target: 目标个体
            mutant: 变异向量
            CR: 交叉概率(可选,默认使用初始化时的CR)
            enable_balance_check: 是否启用负载均衡检查 (方案C)
            balance_threshold: 负载均衡阈值，低于此值拒绝trial (方案C)

        返回:
            trial: 试验向量
        """
        if CR is None:
            CR = self.CR

        # 如果CR与当前策略的CR不同,创建新的交叉策略实例
        if CR != self.crossover_strategy.CR:
            self.crossover_strategy = BinomialCrossover(CR)

        trial = self.crossover_strategy.crossover(target, mutant)

        # 方案C: 负载均衡约束检查
        if enable_balance_check and self.execution_time is not None and self.M >= 1000:
            trial_balance = self._calculate_load_balance(trial)

            # 如果负载均衡低于阈值，拒绝trial，保留target
            if trial_balance < balance_threshold:
                return target  # 保持原解

        return trial

    def select(self, target: List[int], trial: List[int], fitness_func: Callable) -> List[int]:
        """
        选择操作(贪婪选择)

        参数:
            target: 目标个体
            trial: 试验个体
            fitness_func: 适应度函数

        返回:
            selected: 被选中的个体
        """
        return greedy_selection(target, trial, fitness_func)

    def _calculate_workloads(self, solution: List[int]) -> List[float]:
        """
        计算每台VM的工作负载 (方案C辅助方法)

        参数:
            solution: 任务分配方案

        返回:
            workloads: 每台VM的负载列表
        """
        if self.execution_time is None:
            return [0.0] * self.N

        workloads = [0.0] * self.N
        for task_id, vm_id in enumerate(solution):
            if task_id < len(self.execution_time) and vm_id < self.N:
                workloads[vm_id] += self.execution_time[task_id][vm_id]
        return workloads

    def _calculate_load_balance(self, solution: List[int]) -> float:
        """
        计算负载均衡度 (方案C核心方法)

        参数:
            solution: 任务分配方案

        返回:
            load_balance: 0-1之间,越接近1越均衡
        """
        workloads = self._calculate_workloads(solution)
        max_load = max(workloads)
        min_load = min(workloads)

        if max_load == 0:
            return 1.0

        # 负载均衡度 = 1 - (最大负载-最小负载)/最大负载
        balance = 1.0 - (max_load - min_load) / max_load
        return balance

