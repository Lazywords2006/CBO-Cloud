#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE选择策略模块
==============
实现差分进化的选择操作

选择策略:
- greedy_selection: 贪婪选择 (DE标准策略)
- tournament_selection: 锦标赛选择
- stochastic_universal_sampling: 随机遍历抽样

选择的作用:
- 决定哪些个体进入下一代
- 保证种群质量单调不减
- 控制选择压力
"""

import random
from typing import List, Callable


def greedy_selection(target: List[int],
                    trial: List[int],
                    fitness_func: Callable) -> List[int]:
    """
    贪婪选择策略 (Greedy Selection)

    这是DE的标准选择策略,也是最简单有效的策略

    规则:
    - 如果trial的适应度 > target的适应度: 选择trial
    - 否则: 保留target

    特点:
    - 精英保留: 永远选择更优的个体
    - 单调性: 种群质量单调不减
    - 简单高效: 仅需比较一次适应度
    - 无需额外参数

    数学表示:
    X_i^(t+1) = {
        U_i,  if f(U_i) >= f(X_i)  (试验向量更优)
        X_i,  otherwise             (保持不变)
    }

    适用场景:
    - DE的标准选择
    - 云任务调度优化 (始终保留更优调度方案)

    参数:
        target: 目标个体 (当前解)
        trial: 试验个体 (候选解)
        fitness_func: 适应度函数 (越大越好)

    返回:
        selected: 被选中的个体 (target或trial)

    示例:
        >>> def fitness(sol):
        >>>     return -calculate_makespan(sol)  # 负makespan,越大越好
        >>>
        >>> target = [0, 1, 0, 2, 1]
        >>> trial = [1, 0, 2, 1, 0]
        >>> selected = greedy_selection(target, trial, fitness)
    """
    # 计算适应度
    fitness_trial = fitness_func(trial)
    fitness_target = fitness_func(target)

    # 贪婪选择: 保留更优的个体
    if fitness_trial > fitness_target:
        return trial.copy()  # trial更优,选择trial
    else:
        return target.copy()  # target更优或相等,保持不变


def tournament_selection(population: List[List[int]],
                        fitness_func: Callable,
                        k: int = 3) -> List[int]:
    """
    锦标赛选择 (Tournament Selection)

    从种群中随机选择k个个体,返回其中最优的

    原理:
    1. 从种群中随机抽取k个个体
    2. 计算这k个个体的适应度
    3. 返回适应度最高的个体

    特点:
    - 选择压力可调: k越大,选择压力越大
    - 保持多样性: 即使是次优个体也有机会被选中
    - 计算高效: 只需评估k个个体
    - 适合大规模种群

    参数:
        population: 当前种群
        fitness_func: 适应度函数
        k: 锦标赛规模 (默认3)
            - k=2: 低选择压力,保持高多样性
            - k=3-5: 中等选择压力 (推荐)
            - k>5: 高选择压力,快速收敛

    返回:
        winner: 锦标赛获胜者

    适用场景:
    - 需要从种群中选择个体进行操作
    - 保持种群多样性的同时有一定选择压力

    示例:
        >>> population = [[0,1,2], [1,0,2], [2,1,0]]
        >>> winner = tournament_selection(population, fitness_func, k=2)
    """
    # 确保k不超过种群大小
    k = min(k, len(population))

    # 随机选择k个个体组成锦标赛
    tournament = random.sample(population, k)

    # 返回锦标赛中适应度最高的个体
    winner = max(tournament, key=fitness_func)

    return winner.copy()


def stochastic_universal_sampling(population: List[List[int]],
                                  fitness_values: List[float],
                                  n_select: int) -> List[List[int]]:
    """
    随机遍历抽样 (Stochastic Universal Sampling, SUS)

    一种低方差的比例选择方法,改进自轮盘赌选择

    原理:
    1. 将种群按适应度比例分配到一条线上
    2. 使用等间距的多个指针同时选择
    3. 保证选择比例接近适应度比例,且方差低

    特点:
    - 低方差: 相比轮盘赌选择,选择结果更稳定
    - 比例选择: 适应度高的个体被选中的概率更大
    - 高效: 一次遍历完成所有选择
    - 公平: 避免轮盘赌的随机性波动

    算法步骤:
    1. 计算总适应度: F_total = Σf_i
    2. 计算选择步长: step = F_total / n_select
    3. 随机选择起始点: start ∈ [0, step)
    4. 放置n_select个等间距指针: pointer_i = start + i*step
    5. 每个指针选择对应区间的个体

    参数:
        population: 当前种群
        fitness_values: 对应的适应度值列表
        n_select: 需要选择的个体数量

    返回:
        selected: 被选中的个体列表

    示例:
        >>> population = [[0,1], [1,0], [0,0], [1,1]]
        >>> fitness_values = [10.0, 15.0, 5.0, 20.0]  # 总和=50
        >>> selected = stochastic_universal_sampling(population, fitness_values, 2)
        >>> # step = 50/2 = 25
        >>> # 假设start=5, 则指针位置: 5, 30
        >>> # 个体累积: [0,10), [10,25), [25,30), [30,50)
        >>> # 指针5选中第1个, 指针30选中第4个

    注意:
    - 要求所有适应度值 >= 0
    - 如果有负适应度,需要先平移到非负区间
    """
    # 如果种群为空或选择数为0,返回空列表
    if len(population) == 0 or n_select == 0:
        return []

    # 计算总适应度
    total_fitness = sum(fitness_values)

    # 如果总适应度为0,随机选择
    if total_fitness == 0:
        return random.sample(population, min(n_select, len(population)))

    # 计算选择步长
    step = total_fitness / n_select

    # 随机选择起始点 (在[0, step)范围内)
    start = random.uniform(0, step)

    # 选择个体
    selected = []
    cumulative = 0  # 累积适应度
    i = 0           # 当前个体索引

    # 遍历n_select个指针
    for n in range(n_select):
        # 计算当前指针位置
        pointer = start + n * step

        # 移动到指针位置对应的个体
        while cumulative < pointer and i < len(population):
            cumulative += fitness_values[i]
            i += 1

        # 选择当前个体 (如果已遍历完,选择最后一个)
        if i > 0:
            selected.append(population[i - 1].copy())
        else:
            selected.append(population[0].copy())

    return selected


def rank_selection(population: List[List[int]],
                  fitness_func: Callable,
                  n_select: int,
                  pressure: float = 2.0) -> List[List[int]]:
    """
    排名选择 (Rank Selection)

    基于个体的排名而非适应度值进行选择

    原理:
    1. 按适应度排序种群
    2. 根据排名分配选择概率
    3. 线性排名: P(i) = (2-s)/n + 2i(s-1)/(n(n-1))

    特点:
    - 避免适应度值差异过大的问题
    - 选择压力可控 (通过pressure参数)
    - 适合适应度值范围很大的问题

    参数:
        population: 当前种群
        fitness_func: 适应度函数
        n_select: 需要选择的个体数量
        pressure: 选择压力 (1.0-2.0)
            - 1.0: 均匀选择
            - 2.0: 最大选择压力 (默认)

    返回:
        selected: 被选中的个体列表
    """
    # 按适应度排序 (从小到大)
    sorted_population = sorted(population, key=fitness_func)

    n = len(population)
    selected = []

    # 计算每个个体的选择概率
    for _ in range(n_select):
        # 生成随机数
        rand = random.random()

        # 累积概率选择
        cumulative_prob = 0
        for i, ind in enumerate(sorted_population):
            # 线性排名选择概率
            rank = i + 1  # 排名从1开始
            prob = (2 - pressure) / n + 2 * rank * (pressure - 1) / (n * (n - 1))

            cumulative_prob += prob

            if rand <= cumulative_prob:
                selected.append(ind.copy())
                break

    return selected


# 导出所有选择策略
__all__ = [
    'greedy_selection',
    'tournament_selection',
    'stochastic_universal_sampling',
    'rank_selection'
]
