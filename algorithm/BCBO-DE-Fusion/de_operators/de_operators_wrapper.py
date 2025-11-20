#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE算子统一接口

为了兼容不同的调用方式,提供统一的DEOperators类。
"""

import random
from typing import List, Callable
from .mutation_strategies import DE_Rand_1
from .crossover_strategies import BinomialCrossover
from .selection import greedy_selection


class DEOperators:
    """
    DE算子统一封装类

    提供变异、交叉、选择的统一接口。
    """

    def __init__(self, M: int, N: int, F: float = 0.5, CR: float = 0.8):
        """
        初始化DE算子

        参数:
            M: 任务数量
            N: 虚拟机数量
            F: 缩放因子
            CR: 交叉概率
        """
        self.M = M
        self.N = N
        self.F = F
        self.CR = CR

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

    def crossover(self, target: List[int], mutant: List[int], CR: float = None) -> List[int]:
        """
        交叉操作

        参数:
            target: 目标个体
            mutant: 变异向量
            CR: 交叉概率(可选,默认使用初始化时的CR)

        返回:
            trial: 试验向量
        """
        if CR is None:
            CR = self.CR

        # 如果CR与当前策略的CR不同,创建新的交叉策略实例
        if CR != self.crossover_strategy.CR:
            self.crossover_strategy = BinomialCrossover(CR)

        return self.crossover_strategy.crossover(target, mutant)

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
