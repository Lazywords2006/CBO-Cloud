#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遗传算法(GA)云调度器实现

基于遗传算法的云任务调度优化，支持多目标优化包括：
- 最大完工时间(makespan)
- 资源利用率
- 负载均衡
- 成本优化

作者：云调度算法研究团队
日期：2024年
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class GeneticAlgorithmScheduler:
    """
    遗传算法云调度器
    
    参数:
        M: 任务数量
        N: 虚拟机数量
        population_size: 种群大小
        generations: 迭代代数
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
        elite_size: 精英个体数量
    """
    
    def __init__(self, M: int, N: int, population_size: int = 50, 
                 generations: int = 100, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1, elite_size: int = 5):
        self.M = M  # 任务数量
        self.N = N  # 虚拟机数量
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # 初始化任务和虚拟机属性
        self._initialize_tasks_and_vms()
        
        # 存储进化历史
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize_tasks_and_vms(self):
        """
        初始化任务和虚拟机的属性
        """
        # 任务属性 (MIPS)
        self.task_cpu = np.random.uniform(800, 1500, self.M)
        self.task_memory = np.random.uniform(50, 300, self.M)
        self.task_priority = np.random.randint(1, 4, self.M)  # 1-3优先级
        
        # 虚拟机属性
        self.vm_cpu_capacity = np.random.uniform(1200, 2800, self.N)
        self.vm_memory_capacity = np.random.uniform(4000, 16000, self.N)
        self.vm_processing_speed = np.random.uniform(1.0, 3.0, self.N)
        self.vm_cost = np.random.uniform(0.05, 0.15, self.N)  # 每小时成本
    
    def initialize_population(self) -> List[np.ndarray]:
        """
        初始化种群
        每个个体表示任务到虚拟机的分配方案
        """
        population = []
        for _ in range(self.population_size):
            # 随机分配每个任务到虚拟机
            individual = np.random.randint(0, self.N, self.M)
            population.append(individual)
        return population
    
    def fitness_function(self, individual: np.ndarray) -> float:
        """
        云调度优化的适应度函数 - 综合考虑云环境特性
        
        Args:
            individual: 个体编码，individual[i]表示任务i分配到的虚拟机
        
        Returns:
            适应度值（越小越好）
        """
        # 1. 计算完工时间(makespan)
        vm_loads = np.zeros(self.N)
        for task_id in range(self.M):
            vm_id = individual[task_id]
            execution_time = (self.task_cpu[task_id] + self.task_memory[task_id]) / self.vm_processing_speed[vm_id]
            vm_loads[vm_id] += execution_time
        
        makespan = np.max(vm_loads)
        
        # 2. 计算资源利用率（云环境重要指标）
        total_cpu_usage = 0
        total_memory_usage = 0
        resource_violations = 0  # 资源违约次数
        
        for vm_id in range(self.N):
            assigned_tasks = np.where(individual == vm_id)[0]
            if len(assigned_tasks) > 0:
                cpu_usage = np.sum(self.task_cpu[assigned_tasks]) / self.vm_cpu_capacity[vm_id]
                memory_usage = np.sum(self.task_memory[assigned_tasks]) / self.vm_memory_capacity[vm_id]
                
                # 检查资源违约（云环境中的硬约束）
                if cpu_usage > 1.0 or memory_usage > 1.0:
                    resource_violations += 1
                
                total_cpu_usage += min(cpu_usage, 1.0)
                total_memory_usage += min(memory_usage, 1.0)
        
        avg_resource_utilization = (total_cpu_usage + total_memory_usage) / (2 * self.N)
        
        # 3. 计算负载不均衡度（云环境中的关键性能指标）- 使用变异系数CV
        active_vm_loads = vm_loads[vm_loads > 0]
        if len(active_vm_loads) > 0:
            mean_load = np.mean(active_vm_loads)
            std_load = np.std(active_vm_loads)
            load_imbalance = std_load / (mean_load + 1e-6)
        else:
            load_imbalance = 0.0
        
        # 4. 计算总成本（云环境的核心考虑因素）
        total_cost = 0
        for vm_id in range(self.N):
            if vm_loads[vm_id] > 0:
                # 云环境中的动态定价模型
                base_cost = self.vm_cost[vm_id] * vm_loads[vm_id]
                # 考虑资源利用率对成本的影响
                utilization_factor = (total_cpu_usage + total_memory_usage) / (2 * self.N)
                total_cost += base_cost * (1 + 0.1 * utilization_factor)
        
        # 5. 云环境特有的SLA违约惩罚
        sla_penalty = 0
        if makespan > np.mean(vm_loads) * 1.5:  # 如果最大完工时间超过平均值1.5倍
            sla_penalty = makespan * 0.1
        
        # 6. 能耗考虑（云环境绿色计算）
        energy_consumption = 0
        for vm_id in range(self.N):
            if vm_loads[vm_id] > 0:
                # 简化的能耗模型：基础能耗 + 负载相关能耗
                energy_consumption += 10 + vm_loads[vm_id] * 0.5
        
        # 7. 云环境优化的综合适应度（加权求和，越小越好）
        fitness = (0.25 * makespan +                              # 完工时间
                  0.20 * (1 - avg_resource_utilization) * 100 +   # 资源利用率
                  0.15 * load_imbalance +                         # 负载均衡
                  0.20 * total_cost / 100 +                       # 成本优化
                  0.10 * resource_violations * 50 +               # 资源违约惩罚
                  0.05 * sla_penalty +                            # SLA违约惩罚
                  0.05 * energy_consumption / 100)                # 能耗考虑
        
        return fitness
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_values: List[float], 
                           tournament_size: int = 3) -> np.ndarray:
        """
        锦标赛选择
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_index].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, self.M - 1)
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作 - 随机改变部分任务的分配
        """
        mutated = individual.copy()
        for i in range(self.M):
            if random.random() < self.mutation_rate:
                mutated[i] = random.randint(0, self.N - 1)
        return mutated
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行遗传算法优化
        
        Returns:
            最优解和最优适应度值
        """
        # 初始化种群
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # 计算适应度
            fitness_values = [self.fitness_function(ind) for ind in population]
            
            # 记录最优解
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_solution = population[best_idx].copy()
            
            # 记录进化历史（包含solution用于生成收敛曲线）
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': min(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'worst_fitness': max(fitness_values),
                'best_solution': self.best_solution.copy()  # 新增：记录每代的最优解
            })
            
            # 精英保留
            elite_indices = np.argsort(fitness_values)[:self.elite_size]
            elite_population = [population[i].copy() for i in elite_indices]
            
            # 生成新种群
            new_population = elite_population.copy()
            
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 保持种群大小
            population = new_population[:self.population_size]
            
            # 打印进度
            if generation % 20 == 0:
                print(f"Generation {generation}: Best Fitness = {min(fitness_values):.4f}")
        
        return self.best_solution, self.best_fitness, self.fitness_history
    
    def get_detailed_metrics(self, solution: np.ndarray) -> Dict:
        """
        获取详细的性能指标
        """
        # 计算各虚拟机负载
        vm_loads = np.zeros(self.N)
        vm_task_counts = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = solution[task_id]
            execution_time = (self.task_cpu[task_id] + self.task_memory[task_id]) / self.vm_processing_speed[vm_id]
            vm_loads[vm_id] += execution_time
            vm_task_counts[vm_id] += 1
        
        # 计算指标
        makespan = np.max(vm_loads)
        
        # 资源利用率
        total_cpu_usage = 0
        total_memory_usage = 0
        for vm_id in range(self.N):
            assigned_tasks = np.where(solution == vm_id)[0]
            if len(assigned_tasks) > 0:
                cpu_usage = np.sum(self.task_cpu[assigned_tasks]) / self.vm_cpu_capacity[vm_id]
                memory_usage = np.sum(self.task_memory[assigned_tasks]) / self.vm_memory_capacity[vm_id]
                total_cpu_usage += min(cpu_usage, 1.0)
                total_memory_usage += min(memory_usage, 1.0)
        
        avg_resource_utilization = (total_cpu_usage + total_memory_usage) / (2 * self.N)
        
        # 负载不均衡（使用变异系数CV）
        active_vm_loads = vm_loads[vm_loads > 0]
        if len(active_vm_loads) > 0:
            mean_load = np.mean(active_vm_loads)
            std_load = np.std(active_vm_loads)
            load_imbalance = std_load / (mean_load + 1e-6)  # CV = std/mean
        else:
            load_imbalance = 0.0
        
        # 总成本
        total_cost = sum(self.vm_cost[vm_id] * vm_loads[vm_id] for vm_id in range(self.N) if vm_loads[vm_id] > 0)
        
        return {
            'makespan': makespan,
            'resource_utilization': avg_resource_utilization,
            'load_imbalance': load_imbalance,
            'total_cost': total_cost,
            'vm_loads': vm_loads,
            'vm_task_counts': vm_task_counts
        }
    
    def plot_convergence(self, save_path: str = None):
        """
        绘制收敛曲线
        """
        if not self.fitness_history:
            print("No fitness history available. Run optimize() first.")
            return
        
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness = [h['avg_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('GA Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建GA调度器实例
    ga_scheduler = GeneticAlgorithmScheduler(
        M=100,  # 100个任务
        N=10,   # 10个虚拟机
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    print("开始遗传算法优化...")
    best_solution, best_fitness = ga_scheduler.optimize()
    
    print(f"\n优化完成！")
    print(f"最优适应度值: {best_fitness:.4f}")
    
    # 获取详细指标
    metrics = ga_scheduler.get_detailed_metrics(best_solution)
    print(f"\n详细性能指标:")
    print(f"完工时间(makespan): {metrics['makespan']:.2f}")
    print(f"资源利用率: {metrics['resource_utilization']:.3f}")
    print(f"负载不均衡度: {metrics['load_imbalance']:.2f}")
    print(f"总成本: {metrics['total_cost']:.2f}")
    
    # 绘制收敛曲线
    ga_scheduler.plot_convergence()
    
    print(f"\n任务分配方案:")
    for i, vm_id in enumerate(best_solution[:10]):  # 显示前10个任务的分配
        print(f"任务 {i} -> 虚拟机 {vm_id}")