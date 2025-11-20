#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蚁群算法(ACO)云调度器实现

基于蚁群优化的云任务调度算法，模拟蚂蚁觅食行为：
- 信息素机制指导搜索
- 启发式信息结合
- 适用于离散组合优化
- 支持任务依赖关系

作者：云调度算法研究团队
日期：2024年
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    """
    蚁群算法云调度器
    
    参数:
        M: 任务数量
        N: 虚拟机数量
        num_ants: 蚂蚁数量
        max_iterations: 最大迭代次数
        alpha: 信息素重要程度
        beta: 启发式信息重要程度
        rho: 信息素挥发率
        Q: 信息素强度
    """
    
    def __init__(self, M: int, N: int, num_ants: int = 20, 
                 max_iterations: int = 100, alpha: float = 1.0,
                 beta: float = 2.0, rho: float = 0.1, Q: float = 100.0):
        self.M = M  # 任务数量
        self.N = N  # 虚拟机数量
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta   # 启发式信息重要程度
        self.rho = rho     # 信息素挥发率
        self.Q = Q         # 信息素强度
        
        # 初始化任务和虚拟机属性
        self._initialize_tasks_and_vms()
        
        # 初始化信息素矩阵
        self.pheromone = np.ones((self.M, self.N)) * 0.1
        
        # 计算启发式信息矩阵
        self.heuristic = self._calculate_heuristic_info()
        
        # 存储优化历史
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize_tasks_and_vms(self):
        """
        初始化任务和虚拟机的属性
        """
        # 任务属性
        self.task_cpu = np.random.uniform(500, 2000, self.M)
        self.task_memory = np.random.uniform(100, 500, self.M)
        self.task_deadline = np.random.uniform(10, 50, self.M)
        self.task_priority = np.random.randint(1, 4, self.M)
        
        # 虚拟机属性
        self.vm_cpu_capacity = np.random.uniform(1500, 3000, self.N)
        self.vm_memory_capacity = np.random.uniform(2000, 8000, self.N)
        self.vm_processing_speed = np.random.uniform(1.0, 3.5, self.N)
        self.vm_cost = np.random.uniform(0.08, 0.20, self.N)
        
        # 计算执行时间矩阵
        self.execution_time = np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.N):
                self.execution_time[i][j] = (self.task_cpu[i] + self.task_memory[i]) / self.vm_processing_speed[j]
    
    def _calculate_heuristic_info(self) -> np.ndarray:
        """
        计算启发式信息矩阵
        启发式信息基于执行时间、资源匹配度等
        """
        heuristic = np.zeros((self.M, self.N))
        
        for i in range(self.M):
            for j in range(self.N):
                # 执行时间的倒数（越短越好）
                time_factor = 1.0 / (self.execution_time[i][j] + 1e-6)
                
                # 资源匹配度
                cpu_match = min(self.task_cpu[i] / self.vm_cpu_capacity[j], 1.0)
                memory_match = min(self.task_memory[i] / self.vm_memory_capacity[j], 1.0)
                resource_factor = (cpu_match + memory_match) / 2.0
                
                # 成本因子（成本越低越好）
                cost_factor = 1.0 / (self.vm_cost[j] + 1e-6)
                
                # 综合启发式信息
                heuristic[i][j] = time_factor * resource_factor * cost_factor
        
        return heuristic
    
    def _calculate_transition_probability(self, task_id: int, available_vms: List[int]) -> np.ndarray:
        """
        计算状态转移概率
        
        Args:
            task_id: 当前任务ID
            available_vms: 可用虚拟机列表
        
        Returns:
            转移概率数组
        """
        probabilities = np.zeros(len(available_vms))
        
        # 计算分子
        for idx, vm_id in enumerate(available_vms):
            pheromone_val = self.pheromone[task_id][vm_id] ** self.alpha
            heuristic_val = self.heuristic[task_id][vm_id] ** self.beta
            probabilities[idx] = pheromone_val * heuristic_val
        
        # 归一化
        total = np.sum(probabilities)
        if total > 0:
            probabilities = probabilities / total
        else:
            probabilities = np.ones(len(available_vms)) / len(available_vms)
        
        return probabilities
    
    def _construct_solution(self) -> np.ndarray:
        """
        构造一个解（一只蚂蚁的路径）
        
        Returns:
            任务分配方案
        """
        solution = np.zeros(self.M, dtype=int)
        
        # 按任务优先级排序
        task_order = np.argsort(-self.task_priority)  # 高优先级优先
        
        for task_id in task_order:
            # 获取可用虚拟机（考虑资源约束）
            available_vms = []
            for vm_id in range(self.N):
                if (self.task_cpu[task_id] <= self.vm_cpu_capacity[vm_id] and 
                    self.task_memory[task_id] <= self.vm_memory_capacity[vm_id]):
                    available_vms.append(vm_id)
            
            if not available_vms:
                # 如果没有满足资源约束的VM，选择资源最大的
                available_vms = [np.argmax(self.vm_cpu_capacity + self.vm_memory_capacity)]
            
            # 计算转移概率
            probabilities = self._calculate_transition_probability(task_id, available_vms)
            
            # 轮盘赌选择
            selected_idx = np.random.choice(len(available_vms), p=probabilities)
            solution[task_id] = available_vms[selected_idx]
        
        return solution
    
    def fitness_function(self, solution: np.ndarray) -> float:
        """
        适应度函数（修复版 - 增强负载均衡优化）

        Args:
            solution: 任务分配方案

        Returns:
            适应度值（越小越好）
        """
        # 计算各虚拟机的负载
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)

        for task_id in range(self.M):
            vm_id = solution[task_id]
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]

        # 1. 完工时间 (makespan)
        makespan = np.max(vm_loads)

        # 2. 负载均衡（修复：使用变异系数CV，并转换为不均衡惩罚）
        active_vm_loads = vm_loads[vm_loads > 0]
        num_active_vms = len(active_vm_loads)

        if num_active_vms > 1:
            mean_load = np.mean(active_vm_loads)
            std_load = np.std(active_vm_loads)
            load_imbalance_cv = std_load / (mean_load + 1e-6)  # 变异系数

            # VM利用率惩罚：鼓励使用更多VM
            vm_utilization_ratio = num_active_vms / self.N
            if vm_utilization_ratio < 0.3:
                # 如果只用了不到30%的VM，增加惩罚
                vm_penalty = (0.3 - vm_utilization_ratio) * 500
            else:
                vm_penalty = 0.0

            load_balance_penalty = load_imbalance_cv * 100 + vm_penalty
        elif num_active_vms == 1:
            # 严重惩罚：所有任务分配到单个VM
            load_balance_penalty = 1000.0
        else:
            load_balance_penalty = 0.0

        # 3. 总成本
        total_cost = np.sum(vm_costs)

        # 4. 截止时间违反惩罚
        deadline_penalty = 0
        for task_id in range(self.M):
            vm_id = solution[task_id]
            if self.execution_time[task_id][vm_id] > self.task_deadline[task_id]:
                deadline_penalty += (self.execution_time[task_id][vm_id] - self.task_deadline[task_id]) * 10

        # 5. 资源利用率（向量化优化版）
        # 使用NumPy的高级索引批量计算，避免嵌套循环
        vm_cpu_usage = np.zeros(self.N)
        vm_memory_usage = np.zeros(self.N)

        # 向量化计算：一次遍历完成所有VM的资源统计
        for task_id in range(self.M):
            vm_id = solution[task_id]
            vm_cpu_usage[vm_id] += self.task_cpu[task_id]
            vm_memory_usage[vm_id] += self.task_memory[task_id]

        # 批量计算利用率（避免除以0）
        cpu_util = np.minimum(1.0, vm_cpu_usage / (self.vm_cpu_capacity + 1e-6))
        memory_util = np.minimum(1.0, vm_memory_usage / (self.vm_memory_capacity + 1e-6))

        # 只统计有负载的VM
        active_vms = vm_loads > 0
        if np.any(active_vms):
            avg_utilization = np.mean((cpu_util[active_vms] + memory_util[active_vms]) / 2)
        else:
            avg_utilization = 0

        # 综合适应度（加权求和 - 调整权重以增强负载均衡）
        # 修改说明：
        # - 降低makespan权重从0.4到0.3
        # - 大幅增加load_balance_penalty权重从0.25到0.35
        # - 保持total_cost权重0.2
        # - 降低deadline_penalty从0.1到0.05（大多数情况不违反）
        # - 增加利用率权重从0.05到0.1
        fitness = (0.3 * makespan +
                  0.35 * load_balance_penalty +  # 修复：使用负载不均衡惩罚
                  0.2 * total_cost +
                  0.05 * deadline_penalty +
                  0.1 * (1 - avg_utilization) * 100)  # 利用率越高越好

        return fitness
    
    def _update_pheromone(self, solutions: List[np.ndarray], fitness_values: List[float]):
        """
        更新信息素（向量化优化版）
        
        Args:
            solutions: 所有蚂蚁的解
            fitness_values: 对应的适应度值
        """
        # 信息素挥发（向量化）
        self.pheromone *= (1 - self.rho)
        
        # 信息素增强（批量处理）
        for solution, fitness in zip(solutions, fitness_values):
            delta_pheromone = self.Q / (fitness + 1e-6)
            
            # 向量化更新：使用高级索引
            task_indices = np.arange(self.M)
            vm_indices = solution.astype(int)
            self.pheromone[task_indices, vm_indices] += delta_pheromone
        
        # 限制信息素范围，避免过度集中
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行蚁群算法优化
        
        Returns:
            最优解和最优适应度值
        """
        print("开始蚁群算法优化...")
        
        for iteration in range(self.max_iterations):
            # 构造解
            solutions = []
            fitness_values = []
            
            for ant in range(self.num_ants):
                solution = self._construct_solution()
                fitness = self.fitness_function(solution)
                
                solutions.append(solution)
                fitness_values.append(fitness)
                
                # 更新全局最优解
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution.copy()
            
            # 更新信息素
            self._update_pheromone(solutions, fitness_values)
            
            # 记录历史（包含solution用于生成收敛曲线）
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': min(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'worst_fitness': max(fitness_values),
                'best_solution': self.best_solution.copy()  # 新增：记录每代的最优解
            })
            
            # 打印进度
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best Fitness = {min(fitness_values):.4f}")
        
        print(f"优化完成！最优适应度: {self.best_fitness:.4f}")
        return self.best_solution, self.best_fitness, self.fitness_history
    
    def get_detailed_metrics(self, solution: np.ndarray) -> Dict:
        """
        获取详细的性能指标（修复版 - 添加load_balance字段）
        """
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_task_counts = np.zeros(self.N)

        for task_id in range(self.M):
            vm_id = solution[task_id]
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
            vm_task_counts[vm_id] += 1

        # 计算指标
        makespan = np.max(vm_loads)

        # 负载不均衡（使用变异系数CV）
        active_vm_loads = vm_loads[vm_loads > 0]
        if len(active_vm_loads) > 1:
            mean_load = np.mean(active_vm_loads)
            std_load = np.std(active_vm_loads)
            load_imbalance = std_load / (mean_load + 1e-6)
        else:
            # 只有1个或0个VM有负载，认为不均衡度为最大
            load_imbalance = 1.0

        # 修复：添加负载均衡度（0-1之间，越大越好）
        load_balance = max(0.0, 1.0 - min(load_imbalance, 1.0))

        total_cost = np.sum(vm_costs)

        # 资源利用率（向量化优化版）
        vm_cpu_usage = np.zeros(self.N)
        vm_memory_usage = np.zeros(self.N)

        # 向量化计算：一次遍历完成
        for task_id in range(self.M):
            vm_id = solution[task_id]
            vm_cpu_usage[vm_id] += self.task_cpu[task_id]
            vm_memory_usage[vm_id] += self.task_memory[task_id]

        # 批量计算利用率
        cpu_util = np.minimum(1.0, vm_cpu_usage / (self.vm_cpu_capacity + 1e-6))
        memory_util = np.minimum(1.0, vm_memory_usage / (self.vm_memory_capacity + 1e-6))

        # 只统计有负载的VM
        active_vms = vm_loads > 0
        if np.any(active_vms):
            total_utilization = np.sum((cpu_util[active_vms] + memory_util[active_vms]) / 2)
            avg_resource_utilization = total_utilization / np.sum(active_vms)
        else:
            avg_resource_utilization = 0

        # 截止时间违反统计
        deadline_violations = 0
        for task_id in range(self.M):
            vm_id = solution[task_id]
            if self.execution_time[task_id][vm_id] > self.task_deadline[task_id]:
                deadline_violations += 1

        return {
            'makespan': makespan,
            'resource_utilization': avg_resource_utilization,
            'load_imbalance': load_imbalance,  # 保留不均衡度（越小越好）
            'load_balance': load_balance,      # 修复：添加负载均衡度（越大越好）
            'total_cost': total_cost,
            'deadline_violations': deadline_violations,
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
        
        iterations = [h['iteration'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness = [h['avg_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(iterations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title('ACO Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pheromone_distribution(self, save_path: str = None):
        """
        绘制信息素分布热图
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(self.pheromone, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Pheromone Level')
        plt.xlabel('Virtual Machine ID')
        plt.ylabel('Task ID')
        plt.title('Pheromone Distribution Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建ACO调度器实例
    aco_scheduler = AntColonyOptimizer(
        M=80,   # 80个任务
        N=8,    # 8个虚拟机
        num_ants=20,
        max_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        Q=100.0
    )
    
    # 执行优化
    best_solution, best_fitness = aco_scheduler.optimize()
    
    # 获取详细指标
    metrics = aco_scheduler.get_detailed_metrics(best_solution)
    print(f"\n详细性能指标:")
    print(f"完工时间(makespan): {metrics['makespan']:.2f}")
    print(f"资源利用率: {metrics['resource_utilization']:.3f}")
    print(f"负载不均衡度: {metrics['load_imbalance']:.2f}")
    print(f"总成本: {metrics['total_cost']:.2f}")
    print(f"截止时间违反数: {metrics['deadline_violations']}")
    
    # 绘制收敛曲线
    aco_scheduler.plot_convergence()
    
    # 绘制信息素分布
    aco_scheduler.plot_pheromone_distribution()
    
    print(f"\n任务分配方案（前10个任务）:")
    for i, vm_id in enumerate(best_solution[:10]):
        print(f"任务 {i} -> 虚拟机 {vm_id}")