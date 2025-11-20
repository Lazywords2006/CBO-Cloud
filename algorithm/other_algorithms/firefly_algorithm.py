#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
萤火虫算法(FA)云调度器实现

基于萤火虫优化的云任务调度算法特点：
- 模拟萤火虫发光吸引机制
- 光强度与适应度相关
- 支持多模态优化
- 适用于云-边缘混合调度场景

作者：云调度算法研究团队
日期：2024年
"""

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class FireflyAlgorithm:
    """
    萤火虫算法云调度器
    
    参数:
        M: 任务数量
        N: 虚拟机数量
        num_fireflies: 萤火虫数量
        max_iterations: 最大迭代次数
        alpha: 随机化参数
        beta0: 最大吸引度
        gamma: 光强吸收系数
        delta: 步长缩放因子
    """
    
    def __init__(self, M: int, N: int, num_fireflies: int = 25, 
                 max_iterations: int = 100, alpha: float = 0.2,
                 beta0: float = 1.0, gamma: float = 1.0, delta: float = 0.97):
        self.M = M  # 任务数量
        self.N = N  # 虚拟机数量
        self.num_fireflies = num_fireflies
        self.max_iterations = max_iterations
        self.alpha = alpha      # 随机化参数
        self.beta0 = beta0      # 最大吸引度
        self.gamma = gamma      # 光强吸收系数
        self.delta = delta      # 步长缩放因子
        
        # 初始化任务和虚拟机属性
        self._initialize_tasks_and_vms()
        
        # 初始化萤火虫群体
        self.fireflies = self._initialize_fireflies()
        self.light_intensity = [0.0] * self.num_fireflies
        self.best_solution = None
        self.best_intensity = float('-inf')  # 修复：初始化为负无穷大
        
        # 存储优化历史
        self.fitness_history = []
    
    def _initialize_tasks_and_vms(self):
        """
        初始化任务和虚拟机的属性
        """
        # 任务属性
        self.task_cpu = np.random.uniform(300, 2500, self.M)
        self.task_memory = np.random.uniform(50, 800, self.M)
        self.task_storage = np.random.uniform(10, 200, self.M)
        self.task_network = np.random.uniform(5, 100, self.M)
        self.task_priority = np.random.randint(1, 4, self.M)
        self.task_deadline = np.random.uniform(5, 35, self.M)
        
        # 虚拟机属性（包括云和边缘节点）
        self.vm_cpu_capacity = np.random.uniform(2500, 5000, self.N)
        self.vm_memory_capacity = np.random.uniform(4000, 20000, self.N)
        self.vm_storage_capacity = np.random.uniform(1000, 10000, self.N)
        self.vm_network_capacity = np.random.uniform(100, 1000, self.N)
        self.vm_processing_speed = np.random.uniform(1.8, 4.5, self.N)
        self.vm_cost = np.random.uniform(0.05, 0.25, self.N)
        self.vm_energy_efficiency = np.random.uniform(0.7, 0.95, self.N)
        
        # VM类型（0: 边缘节点, 1: 云节点）
        self.vm_type = np.random.choice([0, 1], self.N, p=[0.4, 0.6])
        
        # 边缘节点通常成本低但性能有限
        for i in range(self.N):
            if self.vm_type[i] == 0:  # 边缘节点
                self.vm_cost[i] *= 0.6  # 成本更低
                self.vm_processing_speed[i] *= 0.8  # 性能稍低
        
        # 计算执行时间和能耗矩阵
        self.execution_time = np.zeros((self.M, self.N))
        self.energy_consumption = np.zeros((self.M, self.N))
        
        for i in range(self.M):
            for j in range(self.N):
                workload = (self.task_cpu[i] + self.task_memory[i] + 
                           self.task_storage[i] + self.task_network[i])
                self.execution_time[i][j] = workload / self.vm_processing_speed[j]
                self.energy_consumption[i][j] = workload * (1 - self.vm_energy_efficiency[j])
    
    def _initialize_fireflies(self) -> List[np.ndarray]:
        """
        初始化萤火虫群体
        """
        fireflies = []
        for _ in range(self.num_fireflies):
            # 使用智能初始化策略
            firefly = self._generate_smart_solution()
            fireflies.append(firefly)
        return fireflies
    
    def _generate_smart_solution(self) -> np.ndarray:
        """
        生成智能初始解
        考虑任务特性和VM能力匹配
        """
        solution = np.zeros(self.M, dtype=int)
        
        for task_id in range(self.M):
            # 计算每个VM的适合度分数
            vm_scores = np.zeros(self.N)
            
            for vm_id in range(self.N):
                # 性能匹配度
                performance_score = self.vm_processing_speed[vm_id] / np.max(self.vm_processing_speed)
                
                # 资源匹配度
                cpu_match = min(self.task_cpu[task_id] / self.vm_cpu_capacity[vm_id], 1.0)
                memory_match = min(self.task_memory[task_id] / self.vm_memory_capacity[vm_id], 1.0)
                storage_match = min(self.task_storage[task_id] / self.vm_storage_capacity[vm_id], 1.0)
                network_match = min(self.task_network[task_id] / self.vm_network_capacity[vm_id], 1.0)
                resource_score = (cpu_match + memory_match + storage_match + network_match) / 4.0
                
                # 成本效益
                cost_score = 1.0 / (self.vm_cost[vm_id] + 1e-6)
                
                # 能效分数
                energy_score = self.vm_energy_efficiency[vm_id]
                
                # 任务优先级匹配（高优先级任务偏向云节点）
                priority_score = 1.0
                if self.task_priority[task_id] == 3 and self.vm_type[vm_id] == 1:  # 高优先级+云节点
                    priority_score = 1.5
                elif self.task_priority[task_id] == 1 and self.vm_type[vm_id] == 0:  # 低优先级+边缘节点
                    priority_score = 1.3
                
                vm_scores[vm_id] = (performance_score * 0.3 + resource_score * 0.25 + 
                                   cost_score * 0.2 + energy_score * 0.15 + priority_score * 0.1)
            
            # 基于分数的概率选择
            probabilities = vm_scores / np.sum(vm_scores)
            solution[task_id] = np.random.choice(self.N, p=probabilities)
        
        return solution
    
    def calculate_light_intensity(self, firefly: np.ndarray) -> float:
        """
        计算萤火虫的光强度（适应度）
        
        Args:
            firefly: 萤火虫位置（任务分配方案）
        
        Returns:
            光强度值（越大越好）
        """
        # 计算各虚拟机的负载
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_energy = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = firefly[task_id]
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
            vm_energy[vm_id] += self.energy_consumption[task_id][vm_id]
        
        # 1. 完工时间优化
        makespan = np.max(vm_loads)
        makespan_score = 1.0 / (makespan + 1e-3)  # 增大防护值
        
        # 2. 负载均衡 - 修复数值稳定性
        load_std = np.std(vm_loads)
        if load_std < 1e-6:  # 如果标准差太小，使用固定高分
            load_balance_score = 10.0  # 给予高分但不是无穷大
        else:
            load_balance_score = min(10.0, 1.0 / (load_std + 1e-3))  # 限制最大值
        
        # 3. 资源利用率
        total_utilization = 0
        active_vms = 0
        
        for vm_id in range(self.N):
            assigned_tasks = [task_id for task_id in range(self.M) if firefly[task_id] == vm_id]
            if assigned_tasks:
                active_vms += 1
                cpu_util = min(1.0, sum(self.task_cpu[task_id] for task_id in assigned_tasks) / self.vm_cpu_capacity[vm_id])
                memory_util = min(1.0, sum(self.task_memory[task_id] for task_id in assigned_tasks) / self.vm_memory_capacity[vm_id])
                storage_util = min(1.0, sum(self.task_storage[task_id] for task_id in assigned_tasks) / self.vm_storage_capacity[vm_id])
                network_util = min(1.0, sum(self.task_network[task_id] for task_id in assigned_tasks) / self.vm_network_capacity[vm_id])
                total_utilization += (cpu_util + memory_util + storage_util + network_util) / 4.0
        
        utilization_score = total_utilization / max(active_vms, 1)
        
        # 4. 成本效益 - 修复数值稳定性
        total_cost = np.sum(vm_costs)
        if total_cost < 1e-6:  # 如果成本太小，使用固定高分
            cost_score = 10.0
        else:
            cost_score = min(10.0, 1.0 / (total_cost + 1e-3))  # 限制最大值
        
        # 5. 能耗效率 - 修复数值稳定性
        total_energy = np.sum(vm_energy)
        if total_energy < 1e-6:  # 如果能耗太小，使用固定高分
            energy_score = 10.0
        else:
            energy_score = min(10.0, 1.0 / (total_energy + 1e-3))  # 限制最大值
        
        # 6. 截止时间满足度
        deadline_satisfaction = 0
        for task_id in range(self.M):
            vm_id = firefly[task_id]
            if self.execution_time[task_id][vm_id] <= self.task_deadline[task_id]:
                deadline_satisfaction += 1
        deadline_score = deadline_satisfaction / self.M
        
        # 7. 云-边缘协同效果
        edge_tasks = sum(1 for task_id in range(self.M) if self.vm_type[firefly[task_id]] == 0)
        cloud_tasks = self.M - edge_tasks
        collaboration_score = min(edge_tasks, cloud_tasks) / (self.M / 2)  # 鼓励均衡分配
        
        # 综合光强度
        intensity = (makespan_score * 0.25 +
                    load_balance_score * 0.15 +
                    utilization_score * 0.2 +
                    cost_score * 0.15 +
                    energy_score * 0.1 +
                    deadline_score * 0.1 +
                    collaboration_score * 0.05)
        
        # 数值稳定性保护
        if np.isnan(intensity) or np.isinf(intensity):
            intensity = 0.1  # 返回一个小的正值
        else:
            intensity = max(0.0, min(100.0, intensity))  # 限制在合理范围内
        
        return intensity
    
    def _calculate_distance(self, firefly1: np.ndarray, firefly2: np.ndarray) -> float:
        """
        计算两个萤火虫之间的距离
        
        Args:
            firefly1, firefly2: 两个萤火虫位置
        
        Returns:
            距离值
        """
        # 汉明距离（不同分配的任务数）
        hamming_distance = np.sum(firefly1 != firefly2)
        return hamming_distance / self.M  # 归一化
    
    def _calculate_attractiveness(self, distance: float) -> float:
        """
        计算吸引度
        
        Args:
            distance: 萤火虫间距离
        
        Returns:
            吸引度值
        """
        return self.beta0 * math.exp(-self.gamma * distance * distance)
    
    def _move_firefly(self, firefly_i: np.ndarray, firefly_j: np.ndarray, 
                     attractiveness: float) -> np.ndarray:
        """
        移动萤火虫i向萤火虫j
        
        Args:
            firefly_i: 被吸引的萤火虫
            firefly_j: 吸引的萤火虫
            attractiveness: 吸引度
        
        Returns:
            新位置
        """
        new_firefly = firefly_i.copy()
        
        # 计算移动的任务数量
        num_moves = max(1, int(self.M * attractiveness * 0.5))
        
        # 随机选择要移动的任务
        tasks_to_move = random.sample(range(self.M), min(num_moves, self.M))
        
        for task_id in tasks_to_move:
            if random.random() < attractiveness:
                # 向更亮的萤火虫学习
                new_firefly[task_id] = firefly_j[task_id]
            
            # 添加随机扰动
            if random.random() < self.alpha:
                new_firefly[task_id] = random.randint(0, self.N - 1)
        
        return new_firefly
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行萤火虫算法优化
        
        Returns:
            最优解和最优光强度值
        """
        print("开始萤火虫算法优化...")
        
        # 初始化光强度
        for i in range(self.num_fireflies):
            self.light_intensity[i] = self.calculate_light_intensity(self.fireflies[i])
            
            if self.light_intensity[i] > self.best_intensity:
                self.best_intensity = self.light_intensity[i]
                self.best_solution = self.fireflies[i].copy()
        
        for iteration in range(self.max_iterations):
            # 优化：按光强度排序，只让暗的向亮的移动
            sorted_indices = np.argsort(self.light_intensity)[::-1]  # 从亮到暗
            
            # 萤火虫移动（优化版）
            for idx in range(self.num_fireflies):
                i = sorted_indices[idx]
                
                # idx=0是最亮的，不需要移动
                if idx == 0:
                    continue
                
                # 随机选择一个比自己亮的萤火虫（前idx个）
                brighter_indices = sorted_indices[:idx]
                
                # 检查是否有更亮的萤火虫
                if len(brighter_indices) == 0:
                    continue
                
                j = np.random.choice(brighter_indices)
                
                # 计算距离和吸引度
                distance = self._calculate_distance(self.fireflies[i], self.fireflies[j])
                attractiveness = self._calculate_attractiveness(distance)
                
                # 移动萤火虫i向萤火虫j
                new_firefly = self._move_firefly(self.fireflies[i], self.fireflies[j], attractiveness)
                new_intensity = self.calculate_light_intensity(new_firefly)
                
                # 如果新位置更好，则更新
                if new_intensity > self.light_intensity[i]:
                    self.fireflies[i] = new_firefly
                    self.light_intensity[i] = new_intensity
                    
                    # 更新全局最优
                    if new_intensity > self.best_intensity:
                                self.best_intensity = new_intensity
                                self.best_solution = new_firefly.copy()
            
            # 随机移动最暗的萤火虫
            darkest_idx = np.argmin(self.light_intensity)
            if random.random() < 0.1:  # 10%概率
                self.fireflies[darkest_idx] = self._generate_smart_solution()
                self.light_intensity[darkest_idx] = self.calculate_light_intensity(self.fireflies[darkest_idx])
            
            # 更新随机化参数
            self.alpha *= self.delta
            
            # 记录历史
            current_best = max(self.light_intensity)
            current_avg = np.mean(self.light_intensity)
            current_worst = min(self.light_intensity)
            
            self.fitness_history.append({
                'iteration': iteration,
                'best_intensity': current_best,
                'avg_intensity': current_avg,
                'worst_intensity': current_worst,
                'global_best_intensity': self.best_intensity,
                'best_solution': self.best_solution.copy() if self.best_solution is not None else None  # 新增：记录每代的最优解
            })
            
            # 打印进度
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best Intensity = {current_best:.4f}, Global Best = {self.best_intensity:.4f}")
        
        print(f"优化完成！全局最优光强度: {self.best_intensity:.4f}")
        return self.best_solution, self.best_intensity
    
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
        
        # 资源利用率
        total_utilization = 0
        active_vms = 0
        
        for vm_id in range(self.N):
            assigned_tasks = [task_id for task_id in range(self.M) if solution[task_id] == vm_id]
            if assigned_tasks:
                active_vms += 1
                cpu_util = min(1.0, sum(self.task_cpu[task_id] for task_id in assigned_tasks) / self.vm_cpu_capacity[vm_id])
                memory_util = min(1.0, sum(self.task_memory[task_id] for task_id in assigned_tasks) / self.vm_memory_capacity[vm_id])
                storage_util = min(1.0, sum(self.task_storage[task_id] for task_id in assigned_tasks) / self.vm_storage_capacity[vm_id])
                network_util = min(1.0, sum(self.task_network[task_id] for task_id in assigned_tasks) / self.vm_network_capacity[vm_id])
                total_utilization += (cpu_util + memory_util + storage_util + network_util) / 4.0
        
        avg_resource_utilization = total_utilization / max(active_vms, 1)
        
        # 截止时间满足度
        deadline_met = sum(1 for task_id in range(self.M) 
                          if self.execution_time[task_id][solution[task_id]] <= self.task_deadline[task_id])
        deadline_satisfaction = deadline_met / self.M
        
        # 云-边缘分布
        edge_tasks = sum(1 for task_id in range(self.M) if self.vm_type[solution[task_id]] == 0)
        cloud_tasks = self.M - edge_tasks
        
        return {
            'makespan': makespan,
            'resource_utilization': avg_resource_utilization,
            'load_imbalance': load_imbalance,
            'total_cost': total_cost,
            'total_energy': total_energy,
            'deadline_satisfaction': deadline_satisfaction,
            'edge_tasks': edge_tasks,
            'cloud_tasks': cloud_tasks,
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
        best_intensity = [h['best_intensity'] for h in self.fitness_history]
        avg_intensity = [h['avg_intensity'] for h in self.fitness_history]
        global_best_intensity = [h['global_best_intensity'] for h in self.fitness_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, best_intensity, 'b-', label='Best Intensity (Current)', linewidth=2)
        plt.plot(iterations, avg_intensity, 'r--', label='Average Intensity', linewidth=1)
        plt.plot(iterations, global_best_intensity, 'g-', label='Global Best Intensity', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Light Intensity')
        plt.title('Firefly Algorithm Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cloud_edge_distribution(self, save_path: str = None):
        """
        绘制云-边缘任务分布
        """
        if self.best_solution is None:
            print("No solution available. Run optimize() first.")
            return
        
        edge_tasks = sum(1 for task_id in range(self.M) if self.vm_type[self.best_solution[task_id]] == 0)
        cloud_tasks = self.M - edge_tasks
        
        plt.figure(figsize=(8, 6))
        labels = ['Edge Nodes', 'Cloud Nodes']
        sizes = [edge_tasks, cloud_tasks]
        colors = ['lightcoral', 'skyblue']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Task Distribution: Cloud vs Edge Nodes')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建FA调度器实例
    fa_scheduler = FireflyAlgorithm(
        M=90,   # 90个任务
        N=12,   # 12个虚拟机（云+边缘）
        num_fireflies=25,
        max_iterations=100,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0,
        delta=0.97
    )
    
    # 执行优化
    best_solution, best_intensity = fa_scheduler.optimize()
    
    # 获取详细指标
    metrics = fa_scheduler.get_detailed_metrics(best_solution)
    print(f"\n详细性能指标:")
    print(f"完工时间(makespan): {metrics['makespan']:.2f}")
    print(f"资源利用率: {metrics['resource_utilization']:.3f}")
    print(f"负载不均衡度: {metrics['load_imbalance']:.2f}")
    print(f"总成本: {metrics['total_cost']:.2f}")
    print(f"总能耗: {metrics['total_energy']:.2f}")
    print(f"截止时间满足度: {metrics['deadline_satisfaction']:.3f}")
    print(f"边缘节点任务数: {metrics['edge_tasks']}")
    print(f"云节点任务数: {metrics['cloud_tasks']}")
    
    # 绘制收敛曲线
    fa_scheduler.plot_convergence()
    
    # 绘制云-边缘分布
    fa_scheduler.plot_cloud_edge_distribution()
    
    print(f"\n任务分配方案（前10个任务）:")
    for i, vm_id in enumerate(best_solution[:10]):
        node_type = "Edge" if fa_scheduler.vm_type[vm_id] == 0 else "Cloud"
        print(f"任务 {i} -> 虚拟机 {vm_id} ({node_type})")