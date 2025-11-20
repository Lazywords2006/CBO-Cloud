#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-GA分阶段混合算法
====================================
三阶段协作优化策略：
1. 阶段1 (0-30%): BCBO探索 - 快速找到高质量解空间
2. 阶段2 (30-70%): GA优化 - 优化成本和资源利用
3. 阶段3 (70-100%): 融合收敛 - 交替优化达到最优平衡

核心创新：
- 分阶段协作避免负向污染
- 智能信息交换机制
- 多目标感知的解融合
- 自适应参数控制
"""

import numpy as np
import random
import time
import sys
import os
from typing import List, Dict, Optional, Tuple

# 添加父目录到路径以导入BCBO和GA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../BCBO'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../other_algorithms'))

from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from genetic_algorithm_scheduler import GeneticAlgorithmScheduler


class BCBO_GA_StagedHybrid:
    """
    BCBO-GA分阶段混合算法
    Three-Stage Hybrid Algorithm combining BCBO and GA
    """

    def __init__(self, M=50, N=10, n=30, iterations=100, random_seed=None,
                 stage1_ratio=0.30, stage2_ratio=0.40, stage3_ratio=0.30,
                 elite_ratio=0.10, exchange_interval=5, bcbo_instance=None):
        """
        初始化混合算法

        参数:
            M: 任务数量
            N: 虚拟机数量
            n: 种群大小
            iterations: 总迭代次数
            random_seed: 随机种子
            stage1_ratio: 阶段1比例（BCBO探索）
            stage2_ratio: 阶段2比例（GA优化）
            stage3_ratio: 阶段3比例（融合收敛）
            elite_ratio: 精英保留比例
            exchange_interval: 信息交换间隔
            bcbo_instance: 外部提供的BCBO实例（用于确保数据一致性）
        """
        if M <= 0 or N <= 0 or n <= 0 or iterations <= 0:
            raise ValueError("M, N, n, iterations必须为正整数")
        if M < 2:
            raise ValueError("M (任务数量) 必须 >= 2")

        self.M = M
        self.N = N
        self.n = n
        self.iterations = iterations

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # 阶段配置
        self.stage1_ratio = stage1_ratio
        self.stage2_ratio = stage2_ratio
        self.stage3_ratio = stage3_ratio
        self.elite_ratio = elite_ratio
        self.exchange_interval = exchange_interval

        # 创建或使用BCBO实例
        if bcbo_instance is not None:
            # 使用外部提供的BCBO实例（推荐，确保数据一致）
            self.bcbo = bcbo_instance
        else:
            # 创建新的BCBO实例
            self.bcbo = BCBO_CloudScheduler(M, N, n, iterations, random_seed)

        # 创建GA实例
        self.ga = GeneticAlgorithmScheduler(M, N, random_seed)

        # 同步任务和VM数据（确保一致性）
        self.task_loads = self.bcbo.task_loads
        self.vm_caps = self.bcbo.vm_caps
        self.task_cpu = self.bcbo.task_cpu
        self.task_memory = self.bcbo.task_memory
        self.execution_time = self.bcbo.execution_time
        self.vm_cost = self.bcbo.vm_cost

        # 同步到GA
        self.ga.task_loads = self.task_loads
        self.ga.vm_caps = self.vm_caps
        if hasattr(self.ga, 'execution_time'):
            self.ga.execution_time = self.execution_time

        # 最优解
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.best_makespan = float('inf')

        # 历史记录
        self.fitness_history = []
        self.stage_history = []
        self.exchange_history = []

        print(f"BCBO-GA分阶段混合算法初始化完成")
        print(f"  任务数M={M}, VM数N={N}, 种群大小n={n}, 迭代次数={iterations}")
        print(f"  阶段划分: Stage1={stage1_ratio*100:.0f}%, Stage2={stage2_ratio*100:.0f}%, Stage3={stage3_ratio*100:.0f}%")
        print(f"  精英保留: {elite_ratio*100:.0f}%, 信息交换间隔: {exchange_interval}代")
        if bcbo_instance is not None:
            print(f"  使用外部BCBO实例（数据已同步）")

    def set_task_loads(self, task_loads: np.ndarray):
        """设置自定义任务负载"""
        self.bcbo.set_task_loads(task_loads)
        self.ga.task_loads = self.bcbo.task_loads
        self.task_loads = self.bcbo.task_loads
        self.execution_time = self.bcbo.execution_time

    def set_vm_capabilities(self, vm_caps: np.ndarray):
        """设置自定义虚拟机性能"""
        self.bcbo.set_vm_capabilities(vm_caps)
        self.ga.vm_caps = self.bcbo.vm_caps
        self.vm_caps = self.bcbo.vm_caps
        self.execution_time = self.bcbo.execution_time

    # ==================== 适应度评估（使用BCBO的评估函数） ====================

    def comprehensive_fitness(self, assignment: List[int]) -> float:
        """综合适应度评估（调用BCBO的评估）"""
        return self.bcbo.comprehensive_fitness(assignment)

    def _calculate_makespan(self, sol: List[int]) -> float:
        """计算makespan"""
        return self.bcbo._calculate_makespan(sol)

    def _calculate_cost(self, sol: List[int]) -> float:
        """计算总成本（与BCBO的_calculate_metrics完全一致）"""
        # 使用与BCBO完全相同的逻辑
        vm_time = np.zeros(self.N)
        for i, vm in enumerate(sol):
            if 0 <= vm < self.N and self.bcbo.vm_caps[vm] > 0:
                vm_time[vm] += self.bcbo.task_loads[i] / self.bcbo.vm_caps[vm]

        makespan = np.max(vm_time) if len(vm_time) > 0 else 0.0

        if makespan > 0:
            total_cost = makespan * np.mean(self.bcbo.vm_caps) * 0.05
        else:
            total_cost = 0.0

        return total_cost

    def _calculate_load_balance(self, sol: List[int]) -> float:
        """计算负载不均衡度"""
        vm_loads = self.bcbo._calculate_vm_loads(sol)
        active_vm_loads = vm_loads[vm_loads > 0]

        if len(active_vm_loads) > 1:
            return np.std(active_vm_loads) / (np.mean(active_vm_loads) + 1e-6)
        elif len(active_vm_loads) == 1:
            return 10.0
        else:
            return 0.0

    def _update_best(self, population: List[List[int]]):
        """更新最优解"""
        for sol in population:
            fitness = self.comprehensive_fitness(sol)
            makespan = self._calculate_makespan(sol)

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_makespan = makespan
                self.best_solution = sol.copy()

    # ==================== 阶段1: BCBO探索 ====================

    def stage_1_bcbo_exploration(self, total_iterations: int) -> List[List[int]]:
        """
        阶段1: BCBO快速探索可行解空间

        策略：
        - 使用BCBO的Dynamic Search和Encircle Dynamic阶段
        - 目标：快速找到高质量的初始解
        - 优势：利用BCBO的全局探索能力
        """
        stage_1_iters = int(total_iterations * self.stage1_ratio)

        print(f"\n{'='*60}")
        print(f"阶段1: BCBO探索 (0-{stage_1_iters}代)")
        print(f"目标: 快速探索高质量解空间")
        print(f"{'='*60}")

        # 初始化种群
        population = self.bcbo.initialize_population()
        self._update_best(population)

        current_iter = 0

        # 记录初始状态
        self.fitness_history.append({
            'iteration': current_iter,
            'stage': 'stage1_start',
            'best_fitness': self.best_makespan,
            'best_solution': self.best_solution.copy() if self.best_solution else None
        })

        # BCBO探索阶段
        for i in range(stage_1_iters):
            # Dynamic Search: 郊狼主导的全局探索
            population = self.bcbo.dynamic_search_phase(
                population, current_iter, total_iterations
            )

            # Encircle Dynamic: 协作包围优势区域
            population = self.bcbo.encircle_dynamic_phase(
                population, current_iter, total_iterations
            )

            # 更新最优解
            self._update_best(population)

            # 记录历史
            self.fitness_history.append({
                'iteration': current_iter,
                'stage': 'stage1_bcbo',
                'best_fitness': self.best_makespan,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })

            if (i + 1) % 10 == 0:
                print(f"  Stage1 - 迭代 {current_iter+1}/{stage_1_iters}: "
                      f"Makespan={self.best_makespan:.2f}, Fitness={self.best_fitness:.6f}")

            current_iter += 1

        print(f"阶段1完成: 最优Makespan={self.best_makespan:.2f}")

        return population

    # ==================== 阶段2: GA优化 ====================

    def stage_2_ga_optimization(self, population: List[List[int]],
                               start_iter: int, total_iterations: int) -> List[List[int]]:
        """
        阶段2: GA优化成本和负载均衡

        策略：
        - 保留BCBO找到的精英解
        - 使用GA的交叉变异优化成本
        - 重点降低总成本和改善负载均衡
        """
        stage_2_iters = int(total_iterations * self.stage2_ratio)

        print(f"\n{'='*60}")
        print(f"阶段2: GA优化 ({start_iter}-{start_iter+stage_2_iters}代)")
        print(f"目标: 优化成本和负载均衡")
        print(f"{'='*60}")

        current_iter = start_iter

        for i in range(stage_2_iters):
            # 精英保留策略
            elite_size = max(2, int(len(population) * self.elite_ratio))
            fitness_pairs = [(sol, self.comprehensive_fitness(sol)) for sol in population]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)
            elites = [sol for sol, _ in fitness_pairs[:elite_size]]

            # GA进化
            offspring = []

            # 精英直接保留
            offspring.extend([e.copy() for e in elites])

            # GA交叉变异生成新解
            while len(offspring) < len(population):
                # 锦标赛选择
                parent1 = self._tournament_selection(population, k=3)
                parent2 = self._tournament_selection(population, k=3)

                # 交叉
                if random.random() < 0.8:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # 成本感知的变异
                if random.random() < 0.15:
                    child = self._mutation_cost_aware(child)

                offspring.append(child)

            population = offspring[:len(population)]

            # 更新最优解
            self._update_best(population)

            # 记录历史
            self.fitness_history.append({
                'iteration': current_iter,
                'stage': 'stage2_ga',
                'best_fitness': self.best_makespan,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })

            if (i + 1) % 10 == 0:
                cost = self._calculate_cost(self.best_solution)
                load_balance = self._calculate_load_balance(self.best_solution)
                print(f"  Stage2 - 迭代 {current_iter+1}: "
                      f"Makespan={self.best_makespan:.2f}, "
                      f"Cost={cost:.2f}, LoadBalance={load_balance:.4f}")

            current_iter += 1

        cost = self._calculate_cost(self.best_solution)
        print(f"阶段2完成: 最优Makespan={self.best_makespan:.2f}, Cost={cost:.2f}")

        return population

    def _tournament_selection(self, population: List[List[int]], k: int = 3) -> List[int]:
        """锦标赛选择"""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=self.comprehensive_fitness)

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """两点交叉"""
        if self.M < 2:
            return parent1.copy()

        child = parent1.copy()

        # 按任务负载排序，在分组边界处交叉
        task_order = np.argsort(self.task_loads)

        # 选择2个交叉点
        num_segments = random.randint(2, 3)
        segment_size = self.M // num_segments

        for seg in range(num_segments):
            if random.random() < 0.5:  # 50%概率交换该段
                start = seg * segment_size
                end = (seg + 1) * segment_size if seg < num_segments - 1 else self.M
                for idx in range(start, end):
                    task_idx = task_order[idx]
                    child[task_idx] = parent2[task_idx]

        return child

    def _mutation_cost_aware(self, sol: List[int]) -> List[int]:
        """成本感知的变异"""
        new_sol = sol.copy()

        # 找到成本最高的VM
        vm_costs = np.zeros(self.N)
        for i, vm in enumerate(sol):
            if 0 <= vm < self.N:
                vm_costs[vm] += self.vm_cost[vm] * self.execution_time[i][vm]

        if np.max(vm_costs) > 0:
            costliest_vm = np.argmax(vm_costs)
            tasks_on_costly = [i for i in range(self.M) if sol[i] == costliest_vm]

            if tasks_on_costly:
                # 迁移一些任务到成本更低的VM
                num_to_move = min(3, len(tasks_on_costly))
                tasks_to_move = random.sample(tasks_on_costly, num_to_move)

                for task in tasks_to_move:
                    # 选择成本更低的VM
                    costs = [self.vm_cost[v] * self.execution_time[task][v]
                            for v in range(self.N)]
                    cheaper_vms = np.argsort(costs)[:min(3, self.N)]
                    new_sol[task] = random.choice(cheaper_vms)

        return new_sol

    # ==================== 阶段3: 融合收敛 ====================

    def stage_3_hybrid_refinement(self, population: List[List[int]],
                                  start_iter: int, total_iterations: int) -> List[List[int]]:
        """
        阶段3: BCBO和GA交替优化

        策略：
        - BCBO和GA交替执行
        - 智能信息交换
        - 多目标感知的解融合
        """
        stage_3_iters = int(total_iterations * self.stage3_ratio)

        print(f"\n{'='*60}")
        print(f"阶段3: 融合收敛 ({start_iter}-{start_iter+stage_3_iters}代)")
        print(f"目标: 综合两者优势，达到最优平衡")
        print(f"{'='*60}")

        current_iter = start_iter

        for i in range(stage_3_iters):
            if i % 2 == 0:
                # 偶数代：BCBO局部搜索
                population = self.bcbo.attack_static_phase(population, current_iter)
            else:
                # 奇数代：GA局部优化
                population = self._ga_local_search(population)

            # 定期信息交换
            if i % self.exchange_interval == 0 and i > 0:
                population = self._intelligent_information_exchange(population, current_iter)

            # 更新最优解
            self._update_best(population)

            # 记录历史
            self.fitness_history.append({
                'iteration': current_iter,
                'stage': 'stage3_hybrid',
                'best_fitness': self.best_makespan,
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })

            if (i + 1) % 10 == 0:
                cost = self._calculate_cost(self.best_solution)
                load_balance = self._calculate_load_balance(self.best_solution)
                print(f"  Stage3 - 迭代 {current_iter+1}: "
                      f"Makespan={self.best_makespan:.2f}, "
                      f"Cost={cost:.2f}, LoadBalance={load_balance:.4f}")

            current_iter += 1

        final_cost = self._calculate_cost(self.best_solution)
        final_load = self._calculate_load_balance(self.best_solution)
        print(f"阶段3完成: Makespan={self.best_makespan:.2f}, "
              f"Cost={final_cost:.2f}, LoadBalance={final_load:.4f}")

        return population

    def _ga_local_search(self, population: List[List[int]]) -> List[List[int]]:
        """GA局部搜索"""
        new_population = []

        for sol in population:
            # 2-opt局部搜索
            improved_sol = self._two_opt_search(sol)

            # 成本优化
            if random.random() < 0.3:
                improved_sol = self._mutation_cost_aware(improved_sol)

            new_population.append(improved_sol)

        return new_population

    def _two_opt_search(self, sol: List[int]) -> List[int]:
        """2-opt局部搜索"""
        best_sol = sol.copy()
        best_fitness = self.comprehensive_fitness(sol)

        for _ in range(min(10, self.M // 2)):
            if self.M < 2:
                break
            i, j = random.sample(range(self.M), 2)
            test_sol = sol.copy()
            test_sol[i], test_sol[j] = sol[j], sol[i]

            test_fitness = self.comprehensive_fitness(test_sol)
            if test_fitness > best_fitness:
                best_sol = test_sol
                best_fitness = test_fitness

        return best_sol

    def _intelligent_information_exchange(self, population: List[List[int]],
                                         iteration: int) -> List[List[int]]:
        """
        智能信息交换：只交换真正有价值的信息

        策略：
        - 找到makespan最优解（BCBO强项）
        - 找到cost最优解（GA强项）
        - 融合两个解的优点
        """
        # 评估每个解的多目标性能
        multi_obj_scores = []
        for sol in population:
            makespan = self._calculate_makespan(sol)
            cost = self._calculate_cost(sol)
            load_balance = self._calculate_load_balance(sol)

            multi_obj_scores.append({
                'solution': sol,
                'makespan': makespan,
                'cost': cost,
                'load_balance': load_balance,
                'comprehensive': self.comprehensive_fitness(sol)
            })

        # 找到各目标的最优解
        best_makespan_sol = min(multi_obj_scores, key=lambda x: x['makespan'])
        best_cost_sol = min(multi_obj_scores, key=lambda x: x['cost'])
        best_load_sol = min(multi_obj_scores, key=lambda x: x['load_balance'])

        # 融合解
        hybrid_sol = self._combine_solutions(
            best_makespan_sol['solution'],
            best_cost_sol['solution'],
            best_load_sol['solution']
        )

        # 记录交换信息
        self.exchange_history.append({
            'iteration': iteration,
            'makespan_before': best_makespan_sol['makespan'],
            'cost_before': best_cost_sol['cost'],
            'load_before': best_load_sol['load_balance']
        })

        # 替换种群中综合评分最差的解
        worst_idx = min(range(len(multi_obj_scores)),
                       key=lambda i: multi_obj_scores[i]['comprehensive'])
        population[worst_idx] = hybrid_sol

        return population

    def _combine_solutions(self, sol_makespan: List[int],
                          sol_cost: List[int],
                          sol_load: List[int]) -> List[int]:
        """
        融合三个解：平衡makespan、cost和load balance

        策略：
        - 对每个任务，评估三种分配的trade-off
        - 智能选择最优分配
        """
        new_sol = []

        for task_id in range(self.M):
            vm_makespan = sol_makespan[task_id]
            vm_cost = sol_cost[task_id]
            vm_load = sol_load[task_id]

            # 计算三种分配的性能
            candidates = [
                {'vm': vm_makespan, 'priority': 'makespan'},
                {'vm': vm_cost, 'priority': 'cost'},
                {'vm': vm_load, 'priority': 'load'}
            ]

            # 评估每个候选
            scores = []
            for cand in candidates:
                vm = cand['vm']

                # 计算该分配的综合评分
                exec_time = self.execution_time[task_id][vm]
                cost = self.vm_cost[vm] * exec_time

                # 归一化评分（越小越好）
                time_score = exec_time / (np.max(self.execution_time[task_id]) + 1e-6)
                cost_score = cost / (np.max([self.vm_cost[v] * self.execution_time[task_id][v]
                                             for v in range(self.N)]) + 1e-6)

                # 综合评分（时间40%，成本40%，随机性20%）
                comprehensive_score = 0.4 * time_score + 0.4 * cost_score + 0.2 * random.random()

                scores.append(comprehensive_score)

            # 选择评分最低的（最优的）
            best_idx = np.argmin(scores)
            new_sol.append(candidates[best_idx]['vm'])

        return new_sol

    # ==================== 主算法 ====================

    def run_complete_algorithm(self) -> Dict:
        """运行完整的三阶段混合算法"""
        start_time = time.time()

        print("\n" + "="*80)
        print("BCBO-GA三阶段混合算法开始运行")
        print("="*80)

        # 初始化
        self.fitness_history = []
        self.stage_history = []
        self.exchange_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.best_makespan = float('inf')

        current_iter = 0

        # === 阶段1: BCBO探索 (0-30%) ===
        population = self.stage_1_bcbo_exploration(self.iterations)
        stage1_end_iter = int(self.iterations * self.stage1_ratio)

        self.stage_history.append({
            'stage': 'stage1',
            'start_iter': 0,
            'end_iter': stage1_end_iter,
            'best_makespan': self.best_makespan,
            'best_fitness': self.best_fitness
        })

        # === 阶段2: GA优化 (30-70%) ===
        population = self.stage_2_ga_optimization(
            population, stage1_end_iter, self.iterations
        )
        stage2_end_iter = stage1_end_iter + int(self.iterations * self.stage2_ratio)

        self.stage_history.append({
            'stage': 'stage2',
            'start_iter': stage1_end_iter,
            'end_iter': stage2_end_iter,
            'best_makespan': self.best_makespan,
            'best_fitness': self.best_fitness
        })

        # === 阶段3: 融合收敛 (70-100%) ===
        population = self.stage_3_hybrid_refinement(
            population, stage2_end_iter, self.iterations
        )

        self.stage_history.append({
            'stage': 'stage3',
            'start_iter': stage2_end_iter,
            'end_iter': self.iterations,
            'best_makespan': self.best_makespan,
            'best_fitness': self.best_fitness
        })

        # 计算最终指标
        end_time = time.time()
        runtime = end_time - start_time

        final_cost = self._calculate_cost(self.best_solution)
        final_load = self._calculate_load_balance(self.best_solution)

        # 计算资源利用率
        vm_loads = self.bcbo._calculate_vm_loads(self.best_solution)
        resource_utilization = np.sum(vm_loads) / (self.best_makespan * self.N)

        print(f"\n" + "="*80)
        print("BCBO-GA三阶段混合算法运行完成!")
        print("="*80)
        print(f"最终结果:")
        print(f"  Makespan: {self.best_makespan:.2f}")
        print(f"  Total Cost: {final_cost:.2f}")
        print(f"  Load Imbalance: {final_load:.4f}")
        print(f"  Resource Utilization: {resource_utilization:.4f}")
        print(f"  Runtime: {runtime:.2f}秒")
        print("="*80)

        # 返回结果（兼容BCBO接口）
        return {
            "best_solution": self.best_solution,
            "best_fitness": self.best_makespan,  # 兼容性
            "total_cost": final_cost,
            "response_time": self.best_makespan,
            "resource_utilization": resource_utilization,
            "load_imbalance": final_load,
            "fitness_history": self.fitness_history,
            "stage_history": self.stage_history,
            "exchange_history": self.exchange_history,
            "convergence_iteration": self.iterations,
            "runtime": runtime,
            "total_time": runtime,
            "is_feasible": True,
            "algorithm_name": "BCBO-GA-Staged-Hybrid"
        }


if __name__ == "__main__":
    print("="*80)
    print("BCBO-GA分阶段混合算法测试")
    print("="*80)

    # 创建混合算法实例
    hybrid = BCBO_GA_StagedHybrid(
        M=50, N=10, n=30, iterations=100, random_seed=42,
        stage1_ratio=0.30,
        stage2_ratio=0.40,
        stage3_ratio=0.30
    )

    # 运行算法
    result = hybrid.run_complete_algorithm()

    # 输出结果
    print(f"\n性能摘要:")
    print(f"  最优Makespan: {result['response_time']:.2f}")
    print(f"  总成本: {result['total_cost']:.2f}")
    print(f"  负载不均衡度: {result['load_imbalance']:.4f}")
    print(f"  资源利用率: {result['resource_utilization']:.4f}")
    print(f"  运行时间: {result['runtime']:.2f}秒")
