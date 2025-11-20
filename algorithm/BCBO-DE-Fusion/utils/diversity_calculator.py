#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种群多样性计算器

提供多种种群多样性度量方法,用于评估种群的分散程度。
"""

from typing import List
import numpy as np


class DiversityCalculator:
    """
    种群多样性计算器

    提供基于汉明距离和信息熵的多样性计算方法。
    """

    @staticmethod
    def hamming_distance_diversity(population: List[List[int]]) -> float:
        """
        基于汉明距离计算多样性

        参数:
            population: 种群(个体列表),每个个体是VM分配方案

        返回:
            diversity: 多样性值(0.0-1.0之间)
                - 0.0: 所有个体完全相同(无多样性)
                - 1.0: 所有个体完全不同(最大多样性)

        公式:
            diversity = (1 / (n*(n-1))) * Σ(i<j) hamming_distance(i, j) / M
            其中:
                n: 种群大小
                M: 任务数量(个体长度)
                hamming_distance(i, j): 个体i和j之间不同位置的数量

        示例:
            >>> population = [[0, 1, 2], [0, 1, 2], [1, 2, 0]]
            >>> diversity = DiversityCalculator.hamming_distance_diversity(population)
            >>> print(f"多样性: {diversity:.4f}")
        """
        n = len(population)
        if n <= 1:
            return 0.0

        total_distance = 0
        count = 0

        # 计算所有配对的汉明距离
        for i in range(n):
            for j in range(i + 1, n):
                # 计算汉明距离(不同位置的数量)
                distance = sum(1 for k in range(len(population[i]))
                              if population[i][k] != population[j][k])
                total_distance += distance
                count += 1

        # 归一化(除以最大可能距离)
        max_distance = len(population[0])  # 任务数量
        avg_distance = total_distance / count if count > 0 else 0
        diversity = avg_distance / max_distance if max_distance > 0 else 0

        return diversity

    @staticmethod
    def entropy_diversity(population: List[List[int]]) -> float:
        """
        基于信息熵计算多样性

        参数:
            population: 种群(个体列表)

        返回:
            diversity: 多样性值(0.0-1.0之间)

        原理:
            计算每个任务位置上VM分配的熵值,熵值越大表示该位置的分配越分散。
            H(X) = -Σ p(x) * log2(p(x))

        示例:
            >>> population = [[0, 1, 2], [0, 2, 1], [1, 0, 2]]
            >>> diversity = DiversityCalculator.entropy_diversity(population)
        """
        n_individuals = len(population)
        if n_individuals <= 1:
            return 0.0

        n_tasks = len(population[0])

        total_entropy = 0

        # 对每个任务位置计算熵
        for task_idx in range(n_tasks):
            # 统计该任务位置上每个VM的频率
            vm_counts = {}
            for individual in population:
                vm = individual[task_idx]
                vm_counts[vm] = vm_counts.get(vm, 0) + 1

            # 计算熵
            entropy = 0
            for count in vm_counts.values():
                p = count / n_individuals
                if p > 0:
                    entropy -= p * np.log2(p)

            total_entropy += entropy

        # 归一化(除以最大可能熵)
        max_entropy = np.log2(n_individuals) * n_tasks
        diversity = total_entropy / max_entropy if max_entropy > 0 else 0

        return diversity

    @staticmethod
    def pairwise_distance_diversity(population: List[List[int]]) -> float:
        """
        基于两两距离计算多样性(平均欧氏距离)

        参数:
            population: 种群(个体列表)

        返回:
            diversity: 归一化的平均欧氏距离
        """
        n = len(population)
        if n <= 1:
            return 0.0

        total_distance = 0
        count = 0

        # 计算所有配对的欧氏距离
        for i in range(n):
            for j in range(i + 1, n):
                # 欧氏距离
                distance = np.sqrt(sum((population[i][k] - population[j][k])**2
                                      for k in range(len(population[i]))))
                total_distance += distance
                count += 1

        # 平均距离
        avg_distance = total_distance / count if count > 0 else 0

        # 归一化(估计最大可能距离)
        max_possible_distance = np.sqrt(len(population[0]))  # 简化估计
        diversity = avg_distance / max_possible_distance if max_possible_distance > 0 else 0

        return min(diversity, 1.0)  # 确保不超过1.0


# 测试代码
if __name__ == '__main__':
    print("种群多样性计算器测试")
    print("=" * 60)

    # 创建测试种群
    # 测试1: 完全相同的个体(无多样性)
    population_identical = [
        [0, 1, 2, 0, 1],
        [0, 1, 2, 0, 1],
        [0, 1, 2, 0, 1]
    ]

    # 测试2: 部分不同的个体(中等多样性)
    population_medium = [
        [0, 1, 2, 0, 1],
        [0, 1, 2, 1, 0],
        [1, 0, 2, 0, 1]
    ]

    # 测试3: 完全不同的个体(高多样性)
    population_diverse = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2]
    ]

    calculator = DiversityCalculator()

    print("\n【测试1: 完全相同的个体】")
    print(f"种群: {population_identical}")
    hamming_div = calculator.hamming_distance_diversity(population_identical)
    entropy_div = calculator.entropy_diversity(population_identical)
    print(f"汉明距离多样性: {hamming_div:.4f}")
    print(f"信息熵多样性: {entropy_div:.4f}")

    print("\n【测试2: 部分不同的个体】")
    print(f"种群: {population_medium}")
    hamming_div = calculator.hamming_distance_diversity(population_medium)
    entropy_div = calculator.entropy_diversity(population_medium)
    print(f"汉明距离多样性: {hamming_div:.4f}")
    print(f"信息熵多样性: {entropy_div:.4f}")

    print("\n【测试3: 完全不同的个体】")
    print(f"种群: {population_diverse}")
    hamming_div = calculator.hamming_distance_diversity(population_diverse)
    entropy_div = calculator.entropy_diversity(population_diverse)
    print(f"汉明距离多样性: {hamming_div:.4f}")
    print(f"信息熵多样性: {entropy_div:.4f}")

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
