#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
布谷鸟搜索(CS)云调度器实现

基于布谷鸟搜索优化的云任务调度算法特点：
- 结合勒维飞行策略
- 搜索效率高且易实现
- 全局搜索能力强
- 适用于成本和能耗优化

作者：云调度算法研究团队
日期：2024年
"""

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class CuckooSearch:
    """
    布谷鸟搜索云调度器
    
    参数:
        M: 任务数量
        N: 虚拟机数量
        num_nests: 鸟巢数量
        max_iterations: 最大迭代次数
        pa: 发现概率
        beta: 勒维飞行参数
        step_size: 步长参数
    """
    
    def __init__(self, M: int, N: int, num_nests: int = 25, 
                 max_iterations: int = 100, pa: float = 0.25,
                 beta: float = 1.5, step_size: float = 0.01):
        self.M = M  # 任务数量
        self.N = N  # 虚拟机数量
        self.num_nests = num_nests
        self.max_iterations = max_iterations
        self.pa = pa            # 发现概率
        self.beta = beta        # 勒维飞行参数
        self.step_size = step_size
        
        # 初始化任务和虚拟机属性
        self._initialize_tasks_and_vms()
        
        # 初始化鸟巢群体
        self.nests = self._initialize_nests()
        self.fitness = [0.0] * self.num_nests
        self.best_nest = None
        self.best_fitness = float('-inf')
        
        # 存储优化历史
        self.fitness_history = []
    
    def _initialize_tasks_and_vms(self):
        """
        初始化任务和虚拟机的属性
        """
        # 任务属性
        self.task_cpu = np.random.uniform(200, 2000, self.M)
        self.task_memory = np.random.uniform(50, 600, self.M)
        self.task_storage = np.random.uniform(10, 150, self.M)
        self.task_network = np.random.uniform(5, 80, self.M)
        self.task_priority = np.random.randint(1, 4, self.M)
        self.task_deadline = np.random.uniform(5, 30, self.M)
        
        # 虚拟机属性
        self.vm_cpu_capacity = np.random.uniform(2000, 4500, self.N)
        self.vm_memory_capacity = np.random.uniform(3000, 16000, self.N)
        self.vm_storage_capacity = np.random.uniform(800, 8000, self.N)
        self.vm_network_capacity = np.random.uniform(80, 800, self.N)
        self.vm_processing_speed = np.random.uniform(1.5, 4.0, self.N)
        self.vm_cost = np.random.uniform(0.04, 0.20, self.N)
        self.vm_energy_efficiency = np.random.uniform(0.65, 0.92, self.N)
        
        # 计算执行时间和能耗矩阵
        self.execution_time = np.zeros((self.M, self.N))
        self.energy_consumption = np.zeros((self.M, self.N))
        
        for i in range(self.M):
            for j in range(self.N):
                workload = (self.task_cpu[i] + self.task_memory[i] + 
                           self.task_storage[i] + self.task_network[i])
                self.execution_time[i][j] = workload / self.vm_processing_speed[j]
                self.energy_consumption[i][j] = workload * (1 - self.vm_energy_efficiency[j])
    
    def _initialize_nests(self) -> List[np.ndarray]:
        """
        初始化鸟巢群体
        """
        nests = []
        for _ in range(self.num_nests):
            # 使用智能初始化策略
            nest = self._generate_smart_solution()
            nests.append(nest)
        return nests
    
    def _generate_smart_solution(self) -> np.ndarray:
        """
        生成智能初始解
        考虑成本效益和性能平衡
        """
        solution = np.zeros(self.M, dtype=int)
        
        for task_id in range(self.M):
            # 计算每个VM的综合评分
            vm_scores = np.zeros(self.N)
            
            for vm_id in range(self.N):
                # 性能评分
                performance_score = self.vm_processing_speed[vm_id] / np.max(self.vm_processing_speed)
                
                # 成本效益评分（成本越低越好）
                cost_efficiency = 1.0 / (self.vm_cost[vm_id] + 1e-6)
                cost_score = cost_efficiency / np.max([1.0 / (c + 1e-6) for c in self.vm_cost])
                
                # 能效评分
                energy_score = self.vm_energy_efficiency[vm_id]
                
                # 资源匹配度
                cpu_match = min(self.task_cpu[task_id] / self.vm_cpu_capacity[vm_id], 1.0)
                memory_match = min(self.task_memory[task_id] / self.vm_memory_capacity[vm_id], 1.0)
                storage_match = min(self.task_storage[task_id] / self.vm_storage_capacity[vm_id], 1.0)
                network_match = min(self.task_network[task_id] / self.vm_network_capacity[vm_id], 1.0)
                resource_score = (cpu_match + memory_match + storage_match + network_match) / 4.0
                
                # 截止时间满足度
                deadline_score = 1.0 if self.execution_time[task_id][vm_id] <= self.task_deadline[task_id] else 0.5
                
                # 综合评分（CS算法特别关注成本优化）
                vm_scores[vm_id] = (performance_score * 0.2 + cost_score * 0.35 + 
                                   energy_score * 0.2 + resource_score * 0.15 + deadline_score * 0.1)
            
            # 基于评分的概率选择
            probabilities = vm_scores / np.sum(vm_scores)
            solution[task_id] = np.random.choice(self.N, p=probabilities)
        
        return solution
    
    def calculate_fitness(self, nest: np.ndarray) -> float:
        """
        计算鸟巢的适应度
        
        Args:
            nest: 鸟巢位置（任务分配方案）
        
        Returns:
            适应度值（越大越好）
        """
        # 计算各虚拟机的负载
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_energy = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = nest[task_id]
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
            vm_energy[vm_id] += self.energy_consumption[task_id][vm_id]
        
        # 1. 完工时间优化
        makespan = np.max(vm_loads)
        makespan_score = 1.0 / (makespan + 1e-6)
        
        # 2. 成本优化（CS算法的重点）
        total_cost = np.sum(vm_costs)
        cost_score = 1.0 / (total_cost + 1e-6)
        
        # 3. 能耗优化
        total_energy = np.sum(vm_energy)
        energy_score = 1.0 / (total_energy + 1e-6)
        
        # 4. 资源利用率（向量化优化版 - 从O(M×N)降到O(M)）
        vm_cpu_usage = np.zeros(self.N)
        vm_memory_usage = np.zeros(self.N)
        vm_storage_usage = np.zeros(self.N)
        vm_network_usage = np.zeros(self.N)
        
        # 一次遍历完成所有资源统计
        for task_id in range(self.M):
            vm_id = nest[task_id]
            vm_cpu_usage[vm_id] += self.task_cpu[task_id]
            vm_memory_usage[vm_id] += self.task_memory[task_id]
            vm_storage_usage[vm_id] += self.task_storage[task_id]
            vm_network_usage[vm_id] += self.task_network[task_id]
        
        # 批量计算利用率
        cpu_util = np.minimum(1.0, vm_cpu_usage / (self.vm_cpu_capacity + 1e-6))
        memory_util = np.minimum(1.0, vm_memory_usage / (self.vm_memory_capacity + 1e-6))
        storage_util = np.minimum(1.0, vm_storage_usage / (self.vm_storage_capacity + 1e-6))
        network_util = np.minimum(1.0, vm_network_usage / (self.vm_network_capacity + 1e-6))
        
        # 只统计有任务的VM
        active_vms_mask = vm_loads > 0
        if np.any(active_vms_mask):
            total_utilization = np.sum((cpu_util[active_vms_mask] + memory_util[active_vms_mask] + 
                                       storage_util[active_vms_mask] + network_util[active_vms_mask]) / 4.0)
            utilization_score = total_utilization / np.sum(active_vms_mask)
        else:
            utilization_score = 0.0
        
        # 5. 负载均衡
        load_balance_score = 1.0 / (np.std(vm_loads) + 1e-6)
        
        # 6. 截止时间满足度
        deadline_satisfaction = 0
        for task_id in range(self.M):
            vm_id = nest[task_id]
            if self.execution_time[task_id][vm_id] <= self.task_deadline[task_id]:
                deadline_satisfaction += 1
        deadline_score = deadline_satisfaction / self.M
        
        # 7. 任务优先级满足度
        priority_score = 0
        for task_id in range(self.M):
            vm_id = nest[task_id]
            # 高优先级任务应分配到高性能VM
            if self.task_priority[task_id] == 3:  # 高优先级
                if self.vm_processing_speed[vm_id] > np.median(self.vm_processing_speed):
                    priority_score += 1
            elif self.task_priority[task_id] == 1:  # 低优先级
                if self.vm_cost[vm_id] < np.median(self.vm_cost):  # 分配到低成本VM
                    priority_score += 1
            else:  # 中等优先级
                priority_score += 0.5
        priority_score /= self.M
        
        # 综合适应度（CS算法特别强调成本和能耗优化）
        fitness = (makespan_score * 0.2 +
                  cost_score * 0.3 +
                  energy_score * 0.25 +
                  utilization_score * 0.1 +
                  load_balance_score * 0.05 +
                  deadline_score * 0.05 +
                  priority_score * 0.05)
        
        return fitness
    
    def _levy_flight(self, dimension: int) -> np.ndarray:
        """
        生成勒维飞行步长
        
        Args:
            dimension: 维度
        
        Returns:
            勒维飞行向量
        """
        # 计算sigma
        num = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma = (num / den) ** (1 / self.beta)
        
        # 生成勒维飞行
        u = np.random.normal(0, sigma, dimension)
        v = np.random.normal(0, 1, dimension)
        step = u / (np.abs(v) ** (1 / self.beta))
        
        return step
    
    def _get_cuckoo(self, nest: np.ndarray) -> np.ndarray:
        """
        通过勒维飞行生成新的布谷鸟
        
        Args:
            nest: 当前鸟巢
        
        Returns:
            新的布谷鸟位置
        """
        new_nest = nest.copy()
        
        # 生成勒维飞行步长
        levy_step = self._levy_flight(self.M)
        
        # 计算移动的任务数量
        num_moves = max(1, int(self.M * self.step_size * np.abs(np.mean(levy_step))))
        
        # 随机选择要移动的任务
        tasks_to_move = random.sample(range(self.M), min(num_moves, self.M))
        
        for task_id in tasks_to_move:
            # 基于勒维飞行的VM选择
            current_vm = nest[task_id]
            step = int(levy_step[task_id] * self.N)
            new_vm = (current_vm + step) % self.N
            
            # 确保新VM是有效的
            new_vm = max(0, min(new_vm, self.N - 1))
            new_nest[task_id] = new_vm
        
        return new_nest
    
    def _empty_nests(self, nests: List[np.ndarray], fitness: List[float]) -> List[np.ndarray]:
        """
        抛弃一部分鸟巢并生成新的
        
        Args:
            nests: 当前鸟巢群体
            fitness: 适应度列表
        
        Returns:
            更新后的鸟巢群体
        """
        new_nests = []
        
        # 计算需要抛弃的鸟巢数量
        num_abandon = int(self.pa * self.num_nests)
        
        # 按适应度排序，抛弃最差的
        sorted_indices = np.argsort(fitness)
        
        for i in range(self.num_nests):
            if i < num_abandon:  # 抛弃最差的鸟巢
                # 在两个随机鸟巢之间生成新解
                nest1_idx = random.randint(0, self.num_nests - 1)
                nest2_idx = random.randint(0, self.num_nests - 1)
                
                new_nest = np.zeros(self.M, dtype=int)
                for task_id in range(self.M):
                    if random.random() < 0.5:
                        new_nest[task_id] = nests[nest1_idx][task_id]
                    else:
                        new_nest[task_id] = nests[nest2_idx][task_id]
                    
                    # 添加随机扰动
                    if random.random() < 0.1:
                        new_nest[task_id] = random.randint(0, self.N - 1)
                
                new_nests.append(new_nest)
            else:
                new_nests.append(nests[sorted_indices[i]].copy())
        
        return new_nests
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行布谷鸟搜索优化
        
        Returns:
            最优解和最优适应度值
        """
        print("开始布谷鸟搜索优化...")
        
        # 初始化适应度
        for i in range(self.num_nests):
            self.fitness[i] = self.calculate_fitness(self.nests[i])
            
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_nest = self.nests[i].copy()
        
        for iteration in range(self.max_iterations):
            # 生成新的布谷鸟并评估
            for i in range(self.num_nests):
                # 通过勒维飞行生成新布谷鸟
                new_cuckoo = self._get_cuckoo(self.nests[i])
                new_fitness = self.calculate_fitness(new_cuckoo)
                
                # 随机选择一个鸟巢进行比较
                j = random.randint(0, self.num_nests - 1)
                
                # 如果新布谷鸟更好，则替换
                if new_fitness > self.fitness[j]:
                    self.nests[j] = new_cuckoo
                    self.fitness[j] = new_fitness
                    
                    # 更新全局最优
                    if new_fitness > self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_nest = new_cuckoo.copy()
            
            # 抛弃一部分鸟巢
            self.nests = self._empty_nests(self.nests, self.fitness)
            
            # 重新计算适应度
            for i in range(self.num_nests):
                self.fitness[i] = self.calculate_fitness(self.nests[i])
                
                if self.fitness[i] > self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_nest = self.nests[i].copy()
            
            # 记录历史
            current_best = max(self.fitness)
            current_avg = np.mean(self.fitness)
            current_worst = min(self.fitness)
            
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': current_best,
                'avg_fitness': current_avg,
                'worst_fitness': current_worst,
                'global_best_fitness': self.best_fitness,
                'best_solution': self.best_nest.copy()  # 新增：记录每代的最优解
            })
            
            # 打印进度
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best Fitness = {current_best:.4f}, Global Best = {self.best_fitness:.4f}")
        
        print(f"优化完成！全局最优适应度: {self.best_fitness:.4f}")
        return self.best_nest, self.best_fitness
    
    def get_detailed_metrics(self, solution: np.ndarray) -> Dict:
        """
        获取详细的性能指标
        """
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_energy = np.zeros(self.N)
        vm_task_counts = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = solution[task_id]
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
            vm_energy[vm_id] += self.energy_consumption[task_id][vm_id]
            vm_task_counts[vm_id] += 1
        
        # 计算指标
        makespan = np.max(vm_loads)
        # 负载不均衡（使用变异系数CV）
        active_vm_loads = vm_loads[vm_loads > 0]
        if len(active_vm_loads) > 0:
            mean_load = np.mean(active_vm_loads)
            std_load = np.std(active_vm_loads)
            load_imbalance = std_load / (mean_load + 1e-6)
        else:
            load_imbalance = 0.0
        total_cost = np.sum(vm_costs)
        total_energy = np.sum(vm_energy)
        
        # 资源利用率（向量化优化版）
        vm_cpu_usage = np.zeros(self.N)
        vm_memory_usage = np.zeros(self.N)
        vm_storage_usage = np.zeros(self.N)
        vm_network_usage = np.zeros(self.N)
        
        # 一次遍历完成
        for task_id in range(self.M):
            vm_id = solution[task_id]
            vm_cpu_usage[vm_id] += self.task_cpu[task_id]
            vm_memory_usage[vm_id] += self.task_memory[task_id]
            vm_storage_usage[vm_id] += self.task_storage[task_id]
            vm_network_usage[vm_id] += self.task_network[task_id]
        
        # 批量计算利用率
        cpu_util = np.minimum(1.0, vm_cpu_usage / (self.vm_cpu_capacity + 1e-6))
        memory_util = np.minimum(1.0, vm_memory_usage / (self.vm_memory_capacity + 1e-6))
        storage_util = np.minimum(1.0, vm_storage_usage / (self.vm_storage_capacity + 1e-6))
        network_util = np.minimum(1.0, vm_network_usage / (self.vm_network_capacity + 1e-6))
        
        # 只统计有任务的VM
        active_vms_mask = vm_loads > 0
        if np.any(active_vms_mask):
            total_utilization = np.sum((cpu_util[active_vms_mask] + memory_util[active_vms_mask] + 
                                       storage_util[active_vms_mask] + network_util[active_vms_mask]) / 4.0)
            avg_resource_utilization = total_utilization / np.sum(active_vms_mask)
        else:
            avg_resource_utilization = 0.0
        
        # 截止时间满足度
        deadline_met = sum(1 for task_id in range(self.M) 
                          if self.execution_time[task_id][solution[task_id]] <= self.task_deadline[task_id])
        deadline_satisfaction = deadline_met / self.M
        
        # 成本效益分析
        cost_per_task = total_cost / self.M
        energy_per_task = total_energy / self.M
        
        return {
            'makespan': makespan,
            'resource_utilization': avg_resource_utilization,
            'load_imbalance': load_imbalance,
            'total_cost': total_cost,
            'total_energy': total_energy,
            'deadline_satisfaction': deadline_satisfaction,
            'cost_per_task': cost_per_task,
            'energy_per_task': energy_per_task,
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
        global_best_fitness = [h['global_best_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, best_fitness, 'b-', label='Best Fitness (Current)', linewidth=2)
        plt.plot(iterations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
        plt.plot(iterations, global_best_fitness, 'g-', label='Global Best Fitness', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Cuckoo Search Algorithm Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cost_analysis(self, save_path: str = None):
        """
        绘制成本分析图
        """
        if self.best_nest is None:
            print("No solution available. Run optimize() first.")
            return
        
        vm_costs = np.zeros(self.N)
        for task_id in range(self.M):
            vm_id = self.best_nest[task_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
        
        plt.figure(figsize=(12, 6))
        vm_indices = range(self.N)
        plt.bar(vm_indices, vm_costs, alpha=0.7, color='lightcoral')
        plt.xlabel('Virtual Machine ID')
        plt.ylabel('Total Cost')
        plt.title('Cost Distribution Across Virtual Machines')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建CS调度器实例
    cs_scheduler = CuckooSearch(
        M=80,   # 80个任务
        N=10,   # 10个虚拟机
        num_nests=25,
        max_iterations=100,
        pa=0.25,
        beta=1.5,
        step_size=0.01
    )
    
    # 执行优化
    best_solution, best_fitness = cs_scheduler.optimize()
    
    # 获取详细指标
    metrics = cs_scheduler.get_detailed_metrics(best_solution)
    print(f"\n详细性能指标:")
    print(f"完工时间(makespan): {metrics['makespan']:.2f}")
    print(f"资源利用率: {metrics['resource_utilization']:.3f}")
    print(f"负载不均衡度: {metrics['load_imbalance']:.2f}")
    print(f"总成本: {metrics['total_cost']:.2f}")
    print(f"总能耗: {metrics['total_energy']:.2f}")
    print(f"截止时间满足度: {metrics['deadline_satisfaction']:.3f}")
    print(f"平均任务成本: {metrics['cost_per_task']:.3f}")
    print(f"平均任务能耗: {metrics['energy_per_task']:.3f}")
    
    # 绘制收敛曲线
    cs_scheduler.plot_convergence()
    
    # 绘制成本分析
    cs_scheduler.plot_cost_analysis()
    
    print(f"\n任务分配方案（前10个任务）:")
    for i, vm_id in enumerate(best_solution[:10]):
        cost = cs_scheduler.vm_cost[vm_id] * cs_scheduler.execution_time[i][vm_id]
        print(f"任务 {i} -> 虚拟机 {vm_id} (成本: {cost:.3f})")