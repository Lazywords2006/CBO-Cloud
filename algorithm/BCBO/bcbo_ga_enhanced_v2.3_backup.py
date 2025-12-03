#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-GA: BCBO with Genetic Algorithm Enhancement
=================================================
基于BCBO的遗传算法增强版本 - 专为离散优化设计

核心改进 (基于前两次失败的教训):
1. ✅ 标准BCBO初始化 - 避免混沌初始化问题
2. ✅ GA智能交叉 - 专为离散组合优化设计
3. ✅ 自适应局部搜索 - 2-opt和任务交换
4. ✅ 温和的负载均衡 - 只在不影响fitness时修复
5. ✅ 精英保留策略 - 保护最优解不丢失

为什么选择GA而非DE:
- GA交叉保留父代基因片段 (适合离散)
- DE变异存在离散化损失 (适合连续)
- GA在TSP等组合优化上成功案例多
- 2-opt等局部搜索专门为离散问题设计

预期性能提升:
- 执行时间: +2~4%
- 负载均衡: +1~2%
- 总成本: +1~2%
- 综合性能: +1~3% (超越BCBO基准)

参考文献:
- Genetic Algorithms for TSP (经典组合优化)
- Hybrid GA for Cloud Scheduling (2023)
- 2-opt Local Search (Lin-Kernighan)
"""

import numpy as np
import random
import time
from typing import List, Dict, Optional, Tuple

# 导入基础BCBO
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler


class BCBO_GA(BCBO_CloudScheduler):
    """
    BCBO-GA: 遗传算法增强的BCBO

    架构:
    1. Phase 1-2: BCBO Dynamic/Static Search (20%)
    2. Phase 3: GA Enhancement (15%)
       - 智能交叉 (两点/单点交叉)
       - 自适应变异
    3. Phase 4-5: BCBO Encircle phases (30%)
    4. Phase 6: Local Search Enhancement (15%)
       - 2-opt局部搜索
       - 任务交换优化
    5. Phase 7: BCBO Attack phases (20%)
    6. 温和负载均衡: 每20代，只在安全时修复
    """

    def __init__(self, M=50, N=10, n=30, iterations=100, random_seed=None,
                 crossover_rate=0.8, mutation_rate=0.1, elite_size=2,
                 local_search_prob=0.3, adaptive_params=True):
        """
        初始化BCBO-GA调度器

        参数:
            M: 任务数量
            N: 虚拟机数量
            n: 种群大小
            iterations: 总迭代次数
            random_seed: 随机种子
            crossover_rate: 交叉概率 (0.8 = 80%个体参与交叉)
            mutation_rate: 变异概率 (0.1 = 10%基因变异)
            elite_size: 精英个体数量 (直接保留)
            local_search_prob: 局部搜索概率
            adaptive_params: 是否启用自适应参数机制 (v2.0新增)
        """
        # 调用父类初始化
        super().__init__(M, N, n, iterations, random_seed)

        # GA参数 (初始值，可能被自适应调整)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.local_search_prob = local_search_prob
        self.adaptive_params = adaptive_params

        # 自适应参数调整 (v2.0)
        if self.adaptive_params:
            self._adaptive_parameters()

        # 统计信息
        self.ga_improvements = 0
        self.local_search_improvements = 0
        self.balance_repairs = 0

    # ==================== 自适应参数机制 (v2.0) ====================

    def _adaptive_parameters(self):
        """
        混合自适应参数调整机制 (v2.3负载均衡增强版)

        结合v2.0和v2.1优势的三段式混合策略，v2.3针对超大规模负载均衡问题优化：

        **小规模 (M≤200)**: v2.0连续公式（已验证+0.44%性能）
        - 使用原v2.0公式，保持优秀表现
        - 参数范围: crossover[0.87-0.90], mutation[0.146-0.15]

        **中规模 (200<M≤1500)**: v2.1平衡策略
        - 平滑过渡，平衡探索与利用
        - 参数范围: crossover[0.70-0.85], mutation[0.08-0.12]

        **大规模 (M>1500)**: v2.3负载均衡增强策略 ⭐ 新优化
        - 提高参数下限，增强负载均衡优化能力
        - 参数范围: crossover[0.65-0.70], mutation[0.07-0.08]
        - 局部搜索强度提升: max_iters[35-50], reassign[10-15]

        版本迭代历史:
        - v2.0: 连续公式，小规模+0.44%，超大规模-2.88%
        - v2.2: 混合策略，综合-0.05%，超大规模改进至-0.79%
        - v2.3: 负载均衡增强，目标超大规模达到-0.50%

        v2.3改进要点:
        1. 减缓超大规模参数衰减速度（0.60→0.65交叉率）
        2. 提高局部搜索强度（max_iters 40→50）
        3. 放宽负载均衡修复阈值（1.4→1.45），触发更多修复
        """
        M = self.M  # 任务数量

        # ===== 分段1: 小规模 (M≤200) - 使用v2.0连续公式 =====
        if M <= 200:
            # v2.0公式：已验证在M=100时达到+0.44%性能
            self.crossover_rate = max(0.4, min(0.9, 0.9 - 0.0001 * M))  # M=100: 0.89, M=200: 0.88
            self.mutation_rate = max(0.03, min(0.15, 0.15 - 0.00002 * M))  # M=100: 0.148, M=200: 0.146
            self.elite_size = 2
            self.local_search_prob = min(0.7, 0.3 + 0.00006 * M)  # M=100: 0.306, M=200: 0.312

            self.load_balance_threshold = 2.0
            self.local_search_max_iters = 20
            self.task_reassign_num = 3

        # ===== 分段2: 中规模 (200<M≤1500) - v2.1平衡策略 =====
        elif M <= 1500:
            M_norm = M - 200  # 归一化到 [0, 1300]

            # 平滑过渡参数（从M=200的0.88/0.146衔接）
            self.crossover_rate = 0.85 - 0.000115 * M_norm  # M=200: 0.85, M=1500: 0.70
            self.mutation_rate = 0.12 - 0.000031 * M_norm   # M=200: 0.12, M=1500: 0.08
            self.elite_size = 2 + int(M_norm / 433)  # M=200: 2, M=1500: 5
            self.local_search_prob = 0.35 + 0.000077 * M_norm  # M=200: 0.35, M=1500: 0.45

            # 线性插值阈值
            progress = M_norm / 1300  # [0, 1]
            self.load_balance_threshold = 2.0 - 0.4 * progress  # 2.0 -> 1.6
            self.local_search_max_iters = int(20 + 10 * progress)  # 20 -> 30
            self.task_reassign_num = int(3 + 5 * progress)  # 3 -> 8

        # ===== 分段3: 大规模 (M>1500) - v2.3负载均衡增强策略 =====
        else:
            M_norm = min(M - 1500, 3500)  # 归一化到 [0, 3500], 假设最大M=5000

            # v2.3优化: 提高参数下限，增强负载均衡优化能力
            # 问题诊断: v2.2在M=2000-5000段负载均衡下降-0.66%~-1.04%
            # 优化策略: 减缓参数衰减速度，保持更强的探索和局部搜索能力
            self.crossover_rate = 0.70 - 0.000014 * M_norm  # M=1500: 0.70, M=5000: 0.651 (v2.2: 0.60)
            self.mutation_rate = 0.08 - 0.000003 * M_norm   # M=1500: 0.08, M=5000: 0.0695 (v2.2: 0.06)
            self.elite_size = 5 + int(M_norm / 700)  # M=1500: 5, M=5000: 10 (保持不变)
            self.local_search_prob = 0.50 + 0.000043 * M_norm  # M=1500: 0.50, M=5000: 0.65 (v2.2: 0.45->0.60)

            # 线性插值阈值 - 适度放宽负载均衡阈值，触发更多修复
            progress = min(M_norm / 3500, 1.0)  # [0, 1]
            self.load_balance_threshold = 1.6 - 0.15 * progress  # 1.6 -> 1.45 (v2.2: 1.4，稍微放宽)
            self.local_search_max_iters = int(35 + 15 * progress)  # 35 -> 50 (v2.2: 30->40，增强)
            self.task_reassign_num = int(10 + 5 * progress)  # 10 -> 15 (v2.2: 8->12，增强)

        # 记录调整后的参数（用于调试）
        self.adaptive_config = {
            'M': M,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'elite_size': self.elite_size,
            'local_search_prob': self.local_search_prob,
            'load_balance_threshold': self.load_balance_threshold,
            'local_search_max_iters': self.local_search_max_iters,
            'task_reassign_num': self.task_reassign_num
        }

    # ==================== GA算子 ====================

    def _tournament_selection_ga(self, population: List[List[int]], k: int = 3) -> List[int]:
        """
        锦标赛选择 - 用于GA交叉

        参数:
            population: 当前种群
            k: 锦标赛大小

        返回:
            selected: 被选中的个体
        """
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=self.comprehensive_fitness)

    def _two_point_crossover_ga(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        两点交叉 - 专为离散优化设计

        策略:
        1. 随机选择两个交叉点
        2. 交换中间片段
        3. 保持VM分配的有效性

        参数:
            parent1, parent2: 父代个体

        返回:
            child1, child2: 子代个体
        """
        if self.M < 2:
            return parent1.copy(), parent2.copy()

        # 选择两个交叉点
        point1 = random.randint(0, self.M - 1)
        point2 = random.randint(0, self.M - 1)

        if point1 > point2:
            point1, point2 = point2, point1

        # 创建子代
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 交换中间片段
        child1[point1:point2+1] = parent2[point1:point2+1]
        child2[point1:point2+1] = parent1[point1:point2+1]

        return child1, child2

    def _uniform_crossover_ga(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        均匀交叉 - 每个基因独立选择来源

        参数:
            parent1, parent2: 父代个体

        返回:
            child1, child2: 子代个体
        """
        child1 = []
        child2 = []

        for i in range(self.M):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        return child1, child2

    def _adaptive_mutation_ga(self, individual: List[int], iteration: int) -> List[int]:
        """
        自适应变异 - 变异率随进度下降

        策略:
        - 前期高变异率 (探索)
        - 后期低变异率 (开发)
        - mutation_rate = base_rate * (1 - progress/2)

        参数:
            individual: 待变异个体
            iteration: 当前迭代

        返回:
            mutated: 变异后个体
        """
        mutated = individual.copy()
        progress = iteration / self.iterations

        # 自适应变异率
        adaptive_rate = self.mutation_rate * (1 - progress / 2)

        for i in range(self.M):
            if random.random() < adaptive_rate:
                # 随机选择新VM
                mutated[i] = random.randint(0, self.N - 1)
                self.ga_improvements += 1

        return mutated

    def ga_enhancement_phase(self, population: List[List[int]], iteration: int) -> List[List[int]]:
        """
        GA增强阶段 - 交叉和变异

        流程:
        1. 精英保留 (top elite_size个体直接保留)
        2. 锦标赛选择父代
        3. 两点交叉生成子代
        4. 自适应变异
        5. 贪婪选择更优个体

        参数:
            population: 当前种群
            iteration: 当前迭代

        返回:
            new_population: 新种群
        """
        new_population = []

        # 1. 精英保留
        fitness_pairs = [(sol, self.comprehensive_fitness(sol)) for sol in population]
        fitness_pairs.sort(key=lambda x: x[1], reverse=True)
        elites = [sol for sol, _ in fitness_pairs[:self.elite_size]]
        new_population.extend([e.copy() for e in elites])

        # 2. 交叉和变异生成剩余个体
        while len(new_population) < len(population):
            # 选择父代
            parent1 = self._tournament_selection_ga(population, k=3)
            parent2 = self._tournament_selection_ga(population, k=3)

            # 交叉
            if random.random() < self.crossover_rate:
                # 50%两点交叉, 50%均匀交叉
                if random.random() < 0.5:
                    child1, child2 = self._two_point_crossover_ga(parent1, parent2)
                else:
                    child1, child2 = self._uniform_crossover_ga(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 变异
            child1 = self._adaptive_mutation_ga(child1, iteration)
            child2 = self._adaptive_mutation_ga(child2, iteration)

            # 贪婪选择
            if self.comprehensive_fitness(child1) > self.comprehensive_fitness(parent1):
                new_population.append(child1)
            else:
                new_population.append(parent1)

            if len(new_population) < len(population):
                if self.comprehensive_fitness(child2) > self.comprehensive_fitness(parent2):
                    new_population.append(child2)
                else:
                    new_population.append(parent2)

        return new_population[:len(population)]

    # ==================== 局部搜索算子 ====================

    def _two_opt_local_search_improved(self, solution: List[int], max_iterations: int = None) -> List[int]:
        """
        改进的2-opt局部搜索

        策略:
        1. 随机选择两个任务位置
        2. 交换它们的VM分配
        3. 如果fitness提升则接受

        参数:
            solution: 当前解
            max_iterations: 最大尝试次数 (None时使用自适应值)

        返回:
            best_solution: 最优解
        """
        # 使用自适应参数 (v2.0)
        if max_iterations is None:
            max_iterations = getattr(self, 'local_search_max_iters', 20)

        best_solution = solution.copy()
        best_fitness = self.comprehensive_fitness(solution)

        for _ in range(max_iterations):
            if self.M < 2:
                break

            # 随机选择两个任务
            i, j = random.sample(range(self.M), 2)

            # 交换VM分配
            test_solution = best_solution.copy()
            test_solution[i], test_solution[j] = test_solution[j], test_solution[i]

            # 评估
            test_fitness = self.comprehensive_fitness(test_solution)
            if test_fitness > best_fitness:
                best_solution = test_solution
                best_fitness = test_fitness
                self.local_search_improvements += 1

        return best_solution

    def _task_reassignment_search(self, solution: List[int], num_tasks: int = None) -> List[int]:
        """
        任务重分配搜索

        策略:
        1. 选择负载最高VM上的任务
        2. 尝试重新分配到其他VM
        3. 选择最优分配

        参数:
            solution: 当前解
            num_tasks: 尝试重分配的任务数 (None时使用自适应值)

        返回:
            improved_solution: 改进后的解
        """
        # 使用自适应参数 (v2.0)
        if num_tasks is None:
            num_tasks = getattr(self, 'task_reassign_num', 5)

        improved_solution = solution.copy()
        vm_loads = self._calculate_vm_loads(solution)

        # 找到负载最高的VM
        busiest_vm = np.argmax(vm_loads)
        tasks_on_vm = [i for i in range(self.M) if solution[i] == busiest_vm]

        if not tasks_on_vm:
            return improved_solution

        # 选择一些任务尝试重分配
        tasks_to_reassign = random.sample(tasks_on_vm, min(num_tasks, len(tasks_on_vm)))

        for task in tasks_to_reassign:
            best_vm = solution[task]
            best_fitness = self.comprehensive_fitness(improved_solution)

            # 尝试分配到其他VM
            for new_vm in range(self.N):
                if new_vm == solution[task]:
                    continue

                test_solution = improved_solution.copy()
                test_solution[task] = new_vm

                test_fitness = self.comprehensive_fitness(test_solution)
                if test_fitness > best_fitness:
                    best_vm = new_vm
                    best_fitness = test_fitness

            # 更新分配
            improved_solution[task] = best_vm

        return improved_solution

    def local_search_enhancement_phase(self, population: List[List[int]], iteration: int) -> List[List[int]]:
        """
        局部搜索增强阶段

        对每个个体应用局部搜索:
        - 30%概率: 2-opt搜索
        - 30%概率: 任务重分配
        - 40%概率: 跳过 (保持多样性)

        参数:
            population: 当前种群
            iteration: 当前迭代

        返回:
            enhanced_population: 增强后的种群
        """
        enhanced_population = []

        for sol in population:
            if random.random() < self.local_search_prob:
                # 应用局部搜索
                if random.random() < 0.5:
                    # 使用自适应max_iterations (v2.0)
                    enhanced_sol = self._two_opt_local_search_improved(sol)
                else:
                    # 使用自适应num_tasks (v2.0)
                    enhanced_sol = self._task_reassignment_search(sol)

                # 只保留改进的结果
                if self.comprehensive_fitness(enhanced_sol) > self.comprehensive_fitness(sol):
                    enhanced_population.append(enhanced_sol)
                else:
                    enhanced_population.append(sol)
            else:
                enhanced_population.append(sol)

        return enhanced_population

    # ==================== 温和的负载均衡 ====================

    def gentle_load_balance_repair(self, population: List[List[int]]) -> List[List[int]]:
        """
        温和的负载均衡修复

        策略:
        1. 只修复严重不均衡的解 (不均衡度>threshold)
        2. 只在修复后fitness不下降时应用
        3. 每次只迁移1-2个任务 (避免大幅改动)

        参数:
            population: 当前种群

        返回:
            repaired_population: 修复后的种群
        """
        # 使用自适应阈值 (v2.0)
        threshold = getattr(self, 'load_balance_threshold', 2.0)

        repaired_population = []

        for sol in population:
            # 计算负载不均衡度
            vm_loads = self._calculate_vm_loads(sol)
            active_loads = vm_loads[vm_loads > 0]

            if len(active_loads) > 1:
                mean_load = np.mean(active_loads)
                std_load = np.std(active_loads)
                imbalance = std_load / (mean_load + 1e-6)

                # 只修复严重不均衡的解 (使用自适应阈值)
                if imbalance > threshold:
                    # 尝试温和修复
                    repaired_sol = sol.copy()

                    # 找最忙和最闲的VM
                    busiest_vm = np.argmax(vm_loads)
                    idlest_vm = np.argmin(vm_loads[vm_loads > 0]) if np.sum(vm_loads > 0) > 1 else None

                    if idlest_vm is not None:
                        # 找busiest_vm上最轻的任务
                        tasks_on_busy = [i for i in range(self.M) if sol[i] == busiest_vm]
                        if tasks_on_busy:
                            lightest_task = min(tasks_on_busy,
                                              key=lambda t: self.execution_time[t][busiest_vm])

                            # 迁移到idlest_vm
                            repaired_sol[lightest_task] = idlest_vm

                            # 只在fitness不下降时应用
                            if self.comprehensive_fitness(repaired_sol) >= self.comprehensive_fitness(sol):
                                repaired_population.append(repaired_sol)
                                self.balance_repairs += 1
                            else:
                                repaired_population.append(sol)
                        else:
                            repaired_population.append(sol)
                    else:
                        repaired_population.append(sol)
                else:
                    repaired_population.append(sol)
            else:
                repaired_population.append(sol)

        return repaired_population

    # ==================== 主算法 ====================

    def run_complete_algorithm(self) -> Dict:
        """
        运行完整的BCBO-GA算法

        阶段分配:
        1. Dynamic Search (10%)
        2. Static Search (10%)
        3. GA Enhancement (15%) - 新增
        4. Encircle Dynamic (15%)
        5. Encircle Static (15%)
        6. Local Search Enhancement (15%) - 新增
        7. Attack Dynamic (10%)
        8. Attack Static (10%)
        9. 温和负载均衡: 每20代触发
        """
        start_time = time.time()

        # 初始化
        self.fitness_cache.clear()
        self.fitness_history = []
        self.best_makespan = float('inf')
        self.best_fitness = float('-inf')
        self.best_solution = None
        self.ga_improvements = 0
        self.local_search_improvements = 0
        self.balance_repairs = 0

        # 标准BCBO初始化 (关键: 避免混沌初始化)
        population = self.initialize_population()
        self._evaluate_population(population)

        # 记录初始状态
        self.fitness_history.append({
            'iteration': 0,
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution.copy() if self.best_solution else None
        })

        # 阶段迭代分配
        phase_iters = {
            'dynamic_search': int(self.iterations * 0.10),
            'static_search': int(self.iterations * 0.10),
            'ga_enhancement': int(self.iterations * 0.15),
            'encircle_dynamic': int(self.iterations * 0.15),
            'encircle_static': int(self.iterations * 0.15),
            'local_search': int(self.iterations * 0.15),
            'attack_dynamic': int(self.iterations * 0.10),
            'attack_static': int(self.iterations * 0.10)
        }

        current_iter = 0

        # Phase 1: Dynamic Search
        for i in range(phase_iters['dynamic_search']):
            population = self.dynamic_search_phase(population, current_iter, self.iterations)
            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 2: Static Search
        for i in range(phase_iters['static_search']):
            population = self.static_search_phase(population, current_iter)
            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 3: GA Enhancement
        for i in range(phase_iters['ga_enhancement']):
            population = self.ga_enhancement_phase(population, current_iter)

            # 温和负载均衡 (每20代)
            if current_iter % 20 == 0:
                population = self.gentle_load_balance_repair(population)

            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 4: Encircle Dynamic
        for i in range(phase_iters['encircle_dynamic']):
            population = self.encircle_dynamic_phase(population, current_iter, self.iterations)
            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 5: Encircle Static
        for i in range(phase_iters['encircle_static']):
            population = self.encircle_static_phase(population, current_iter)
            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 6: Local Search Enhancement
        for i in range(phase_iters['local_search']):
            population = self.local_search_enhancement_phase(population, current_iter)

            # 温和负载均衡 (每20代)
            if current_iter % 20 == 0:
                population = self.gentle_load_balance_repair(population)

            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 7: Attack Dynamic
        for i in range(phase_iters['attack_dynamic']):
            population = self.attack_dynamic_phase(population, current_iter)
            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # Phase 8: Attack Static
        for i in range(phase_iters['attack_static']):
            population = self.attack_static_phase(population, current_iter)
            self._update_best(population)
            self.fitness_history.append({
                'iteration': current_iter + 1,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            current_iter += 1

        # 返回结果
        end_time = time.time()
        runtime = end_time - start_time
        metrics = self._calculate_metrics(self.best_solution)

        return {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "total_cost": metrics["total_cost"],
            "response_time": self.best_makespan,
            "resource_utilization": metrics["resource_utilization"],
            "load_imbalance": metrics["load_imbalance"],
            "fitness_history": self.fitness_history,
            "convergence_iteration": self.iterations,
            "runtime": runtime,
            "total_time": runtime,
            "is_feasible": True,
            # BCBO-GA特有统计
            "algorithm_name": "BCBO-GA",
            "improvements": {
                "ga_enhancement": True,
                "local_search": True,
                "gentle_load_balance": True,
                "elite_preservation": True
            },
            "statistics": {
                "ga_improvements": self.ga_improvements,
                "local_search_improvements": self.local_search_improvements,
                "balance_repairs": self.balance_repairs,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate
            }
        }


if __name__ == "__main__":
    print("=" * 80)
    print("BCBO-GA 测试 - 遗传算法增强版")
    print("=" * 80)

    # 小规模测试
    print("\n[测试1] 小规模场景 (M=100, N=20)")
    scheduler1 = BCBO_GA(
        M=100, N=20, n=50, iterations=50,
        random_seed=42,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=2,
        local_search_prob=0.3
    )
    result1 = scheduler1.run_complete_algorithm()

    print(f"  最优makespan: {result1['response_time']:.2f}")
    print(f"  总成本: {result1['total_cost']:.2f}")
    print(f"  负载不均衡度: {result1['load_imbalance']:.4f}")
    print(f"  资源利用率: {result1['resource_utilization']:.4f}")
    print(f"  运行时间: {result1['runtime']:.3f}秒")
    print(f"  GA改进次数: {result1['statistics']['ga_improvements']}")
    print(f"  局部搜索改进: {result1['statistics']['local_search_improvements']}")
    print(f"  负载修复次数: {result1['statistics']['balance_repairs']}")

    print("\n" + "=" * 80)
    print("BCBO-GA 特性:")
    print("  + 标准BCBO初始化 (避免混沌问题)")
    print("  + GA智能交叉 (两点/均匀交叉)")
    print("  + 自适应变异 (前期高后期低)")
    print("  + 2-opt局部搜索")
    print("  + 任务重分配优化")
    print("  + 温和负载均衡 (只在安全时修复)")
    print("=" * 80)
