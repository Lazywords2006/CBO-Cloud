#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE算子库
========
提供差分进化算法的核心算子

模块包含:
- mutation_strategies: 变异策略 (DE/rand/1, DE/best/1等)
- crossover_strategies: 交叉策略 (二项式, 指数)
- selection: 选择策略 (贪婪选择, 锦标赛选择)
"""

from .mutation_strategies import (
    MutationStrategy,
    DE_Rand_1,
    DE_Best_1,
    DE_Current_To_Best_1,
    AdaptiveDE
)

from .crossover_strategies import (
    CrossoverStrategy,
    BinomialCrossover,
    ExponentialCrossover,
    AdaptiveCrossover
)

from .selection import (
    greedy_selection,
    tournament_selection,
    stochastic_universal_sampling
)

from .de_operators_wrapper import DEOperators

__all__ = [
    # 变异策略
    'MutationStrategy',
    'DE_Rand_1',
    'DE_Best_1',
    'DE_Current_To_Best_1',
    'AdaptiveDE',

    # 交叉策略
    'CrossoverStrategy',
    'BinomialCrossover',
    'ExponentialCrossover',
    'AdaptiveCrossover',

    # 选择策略
    'greedy_selection',
    'tournament_selection',
    'stochastic_universal_sampling',

    # 统一接口
    'DEOperators'
]

__version__ = '1.0.0'
__author__ = 'BCBO-DE Project'
