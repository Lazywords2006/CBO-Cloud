#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE交叉策略模块
==============
实现差分进化的交叉操作

包含策略:
- BinomialCrossover: 二项式交叉 (最常用)
- ExponentialCrossover: 指数交叉
- AdaptiveCrossover: 自适应交叉

交叉的作用:
- 将变异向量的信息融入目标向量
- 控制新解与原解的相似度
- 通过CR参数调节探索和开发的平衡
"""

import random
from typing import List


class CrossoverStrategy:
    """交叉策略基类"""

    def __init__(self, CR: float = 0.8):
        """
        初始化交叉策略

        参数:
            CR: 交叉概率 (crossover rate)
                - CR=1.0: 完全继承变异向量
                - CR=0.0: 完全保留目标向量
                - 推荐值: 0.7-0.9
        """
        self.CR = CR

    def crossover(self, target: List[int], mutant: List[int]) -> List[int]:
        """
        执行交叉操作

        参数:
            target: 目标向量 (原始个体)
            mutant: 变异向量 (通过变异算子生成)

        返回:
            trial: 试验向量 (交叉后的新个体)
        """
        raise NotImplementedError("子类必须实现crossover方法")


class BinomialCrossover(CrossoverStrategy):
    """
    二项式交叉 (Binomial Crossover)

    这是DE中最常用的交叉策略

    原理:
    - 每个基因位(任务分配)独立地以概率CR从mutant继承
    - 保证至少有一个基因位来自mutant (通过j_rand)
    - 其余基因位从target继承

    伪代码:
    for j in range(M):
        if rand() < CR or j == j_rand:
            trial[j] = mutant[j]  # 来自变异向量
        else:
            trial[j] = target[j]  # 来自目标向量

    特点:
    - 简单高效
    - 保持种群多样性
    - CR可调节继承比例
    """

    def crossover(self, target: List[int], mutant: List[int]) -> List[int]:
        """
        执行二项式交叉

        步骤:
        1. 随机选择一个必须继承mutant的位置j_rand
        2. 对每个位置,以概率CR决定是否继承mutant
        3. j_rand位置强制继承mutant,确保至少一个基因来自变异

        参数:
            target: 目标向量 [vm1, vm2, ..., vmM]
            mutant: 变异向量 [vm1', vm2', ..., vmM']

        返回:
            trial: 试验向量 [vm1'', vm2'', ..., vmM'']
        """
        M = len(target)
        trial = []

        # 随机选择一个位置,确保至少一个基因来自mutant
        # 这是DE算法的关键:避免trial完全等于target
        j_rand = random.randint(0, M - 1)

        for j in range(M):
            if random.random() < self.CR or j == j_rand:
                # 条件1: 随机数 < CR -> 继承mutant
                # 条件2: j == j_rand -> 强制继承mutant
                trial.append(mutant[j])
            else:
                # 保留target的基因
                trial.append(target[j])

        return trial


class ExponentialCrossover(CrossoverStrategy):
    """
    指数交叉 (Exponential Crossover)

    也称为"两点交叉"或"连续交叉"

    原理:
    - 从随机位置开始,连续继承mutant的基因
    - 继承长度L服从几何分布: P(L=k) = CR^(k-1) * (1-CR)
    - 形成一段连续的变异基因

    特点:
    - 连续性: 相邻任务的分配可能来自同一来源
    - 较少使用,但在某些问题上效果更好
    - 对于有相关性的任务调度可能更有效

    适用场景:
    - 任务之间有依赖关系
    - 希望保持局部结构
    """

    def crossover(self, target: List[int], mutant: List[int]) -> List[int]:
        """
        执行指数交叉

        步骤:
        1. 随机选择起始位置n
        2. 从n开始连续替换,直到随机数 >= CR 或遍历完所有位置
        3. 形成一段连续的变异区域

        伪代码:
        n = random_start
        L = 0  # 连续长度
        do:
            trial[n] = mutant[n]
            n = (n + 1) % M
            L += 1
        while (rand() < CR and L < M)

        参数:
            target: 目标向量
            mutant: 变异向量

        返回:
            trial: 试验向量
        """
        M = len(target)
        trial = target.copy()  # 先复制target

        # 随机起始位置
        n = random.randint(0, M - 1)
        L = 0  # 已替换的长度

        # 连续替换,直到概率条件不满足或遍历完所有位置
        while True:
            trial[n] = mutant[n]  # 替换当前位置
            n = (n + 1) % M       # 循环到下一个位置
            L += 1

            # 终止条件:
            # 1. 随机数 >= CR (概率终止)
            # 2. L >= M (已遍历所有位置)
            if not (random.random() < self.CR and L < M):
                break

        return trial


class AdaptiveCrossover(CrossoverStrategy):
    """
    自适应交叉策略

    根据种群多样性动态调整交叉概率CR

    策略:
    - 多样性低 (种群趋同) -> CR高 -> 引入更多变异信息
    - 多样性高 (种群分散) -> CR低 -> 保留更多原始信息

    公式:
    CR = CR_min + (CR_max - CR_min) * (1 - diversity)

    特点:
    - 自动平衡探索和开发
    - 无需手动调参
    - 适应优化过程的不同阶段

    适用场景:
    - 不确定最优CR值
    - 希望算法自适应调整
    - 动态优化问题
    """

    def __init__(self, CR_min: float = 0.5, CR_max: float = 0.9):
        """
        初始化自适应交叉

        参数:
            CR_min: 最小交叉概率 (多样性高时使用)
            CR_max: 最大交叉概率 (多样性低时使用)
        """
        super().__init__((CR_min + CR_max) / 2)
        self.CR_min = CR_min
        self.CR_max = CR_max

    def adaptive_CR(self, diversity: float) -> float:
        """
        计算自适应交叉概率

        公式: CR = CR_min + (CR_max - CR_min) * (1 - diversity)

        示例:
        - diversity=0.2 (低) -> CR = 0.5 + 0.4*0.8 = 0.82 (高CR,多引入变异)
        - diversity=0.8 (高) -> CR = 0.5 + 0.4*0.2 = 0.58 (低CR,多保留原信息)

        参数:
            diversity: 种群多样性 (范围: 0-1)
                - 0: 完全相同
                - 1: 完全不同

        返回:
            CR: 自适应交叉概率
        """
        # 确保diversity在[0, 1]范围内
        diversity = max(0.0, min(1.0, diversity))

        # 计算自适应CR
        adaptive_cr = self.CR_min + (self.CR_max - self.CR_min) * (1 - diversity)

        return adaptive_cr

    def crossover(self,
                 target: List[int],
                 mutant: List[int],
                 diversity: float = 0.5) -> List[int]:
        """
        执行自适应交叉

        步骤:
        1. 根据当前种群多样性计算自适应CR
        2. 使用二项式交叉策略

        参数:
            target: 目标向量
            mutant: 变异向量
            diversity: 当前种群多样性 (默认0.5)

        返回:
            trial: 试验向量
        """
        # 更新交叉概率
        self.CR = self.adaptive_CR(diversity)

        # 使用二项式交叉
        return BinomialCrossover(self.CR).crossover(target, mutant)


# 导出所有策略
__all__ = [
    'CrossoverStrategy',
    'BinomialCrossover',
    'ExponentialCrossover',
    'AdaptiveCrossover'
]
