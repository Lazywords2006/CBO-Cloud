#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰狼优化(GWO)云调度器实现

基于灰狼优化的云任务调度算法特点：
- 模拟灰狼群体捕猎机制
- 平衡探索和利用能力
- 适用于能源和成本敏感的调度优化
- 支持物联网任务的雾-云混合环境

作者：云调度算法研究团队
日期：2024年
"""

import numpy as np
import random
import copy
import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class GreyWolfOptimizer:
    """
    灰狼优化云调度器
    
    参数:
        M: 任务数量
        N: 虚拟机数量
        num_wolves: 狼群数量
        max_iterations: 最大迭代次数
        a_max: 收敛因子最大值
        a_min: 收敛因子最小值
    """
    
    def __init__(self, M: int, N: int, num_wolves: int = 30, 
                 max_iterations: int = 100, a_max: float = 2.0, a_min: float = 0.2):
        self.M = M  # 任务数量
        self.N = N  # 虚拟机数量
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.a_max = a_max
        self.a_min = a_min
        
        # 初始化任务和虚拟机属性
        self._initialize_tasks_and_vms()
        
        # 初始化狼群
        self.wolves = self._initialize_wolves()
        self.fitness = [0.0] * self.num_wolves
        
        # 领导狼（Alpha, Beta, Delta）
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None
        self.alpha_fitness = float('-inf')
        self.beta_fitness = float('-inf')
        self.delta_fitness = float('-inf')
        
        # 存储优化历史
        self.fitness_history = []
    
    def _initialize_tasks_and_vms(self):
        """
        初始化任务和虚拟机的属性
        包括雾节点和云节点的特性
        """
        # 任务属性（包括IoT任务特性）
        self.task_cpu = np.random.uniform(100, 1500, self.M)
        self.task_memory = np.random.uniform(20, 400, self.M)
        self.task_storage = np.random.uniform(5, 100, self.M)
        self.task_network = np.random.uniform(2, 50, self.M)
        self.task_priority = np.random.randint(1, 4, self.M)
        self.task_deadline = np.random.uniform(3, 25, self.M)
        self.task_data_size = np.random.uniform(1, 50, self.M)  # MB
        
        # 虚拟机属性（雾节点 + 云节点）
        self.vm_cpu_capacity = np.random.uniform(1500, 4000, self.N)
        self.vm_memory_capacity = np.random.uniform(2000, 12000, self.N)
        self.vm_storage_capacity = np.random.uniform(500, 6000, self.N)
        self.vm_network_capacity = np.random.uniform(50, 500, self.N)
        self.vm_processing_speed = np.random.uniform(1.2, 3.8, self.N)
        self.vm_cost = np.random.uniform(0.03, 0.18, self.N)
        self.vm_energy_efficiency = np.random.uniform(0.6, 0.9, self.N)
        
        # VM类型（0: 雾节点, 1: 云节点）
        self.vm_type = np.random.choice([0, 1], self.N, p=[0.5, 0.5])
        
        # 网络延迟（雾节点延迟低，云节点延迟高）
        self.vm_latency = np.zeros(self.N)
        
        for i in range(self.N):
            if self.vm_type[i] == 0:  # 雾节点
                self.vm_cost[i] *= 0.7  # 成本更低
                self.vm_processing_speed[i] *= 0.85  # 性能稍低
                self.vm_latency[i] = np.random.uniform(1, 10)  # 低延迟
                self.vm_energy_efficiency[i] *= 1.1  # 能效更高
            else:  # 云节点
                self.vm_latency[i] = np.random.uniform(20, 100)  # 高延迟
        
        # 计算执行时间、能耗和传输时间矩阵
        self.execution_time = np.zeros((self.M, self.N))
        self.energy_consumption = np.zeros((self.M, self.N))
        self.transmission_time = np.zeros((self.M, self.N))
        
        for i in range(self.M):
            for j in range(self.N):
                workload = (self.task_cpu[i] + self.task_memory[i] + 
                           self.task_storage[i] + self.task_network[i])
                self.execution_time[i][j] = workload / self.vm_processing_speed[j]
                self.energy_consumption[i][j] = workload * (1 - self.vm_energy_efficiency[j])
                self.transmission_time[i][j] = self.task_data_size[i] / self.vm_network_capacity[j]
    
    def _initialize_wolves(self) -> List[np.ndarray]:
        """
        初始化狼群
        """
        wolves = []
        for _ in range(self.num_wolves):
            # 使用智能初始化策略
            wolf = self._generate_smart_solution()
            wolves.append(wolf)
        return wolves
    
    def _generate_smart_solution(self) -> np.ndarray:
        """
        生成智能初始解
        考虑雾-云协同和能耗优化
        """
        solution = np.zeros(self.M, dtype=int)
        
        for task_id in range(self.M):
            # 计算每个VM的适合度分数
            vm_scores = np.zeros(self.N)
            
            for vm_id in range(self.N):
                # 性能评分
                performance_score = self.vm_processing_speed[vm_id] / np.max(self.vm_processing_speed)
                
                # 能效评分（GWO重点关注）
                energy_score = self.vm_energy_efficiency[vm_id]
                
                # 成本效益
                cost_efficiency = 1.0 / (self.vm_cost[vm_id] + 1e-6)
                cost_score = cost_efficiency / np.max([1.0 / (c + 1e-6) for c in self.vm_cost])
                
                # 延迟敏感度（IoT任务特性）
                latency_score = 1.0 / (self.vm_latency[vm_id] + 1e-6)
                latency_score = latency_score / np.max([1.0 / (l + 1e-6) for l in self.vm_latency])
                
                # 资源匹配度
                cpu_match = min(self.task_cpu[task_id] / self.vm_cpu_capacity[vm_id], 1.0)
                memory_match = min(self.task_memory[task_id] / self.vm_memory_capacity[vm_id], 1.0)
                storage_match = min(self.task_storage[task_id] / self.vm_storage_capacity[vm_id], 1.0)
                network_match = min(self.task_network[task_id] / self.vm_network_capacity[vm_id], 1.0)
                resource_score = (cpu_match + memory_match + storage_match + network_match) / 4.0
                
                # 任务类型匹配（高优先级任务偏向云节点，低延迟需求偏向雾节点）
                type_score = 1.0
                if self.task_priority[task_id] == 3 and self.vm_type[vm_id] == 1:  # 高优先级+云节点
                    type_score = 1.4
                elif self.task_deadline[task_id] < 10 and self.vm_type[vm_id] == 0:  # 紧急任务+雾节点
                    type_score = 1.3
                elif self.task_data_size[task_id] < 10 and self.vm_type[vm_id] == 0:  # 小数据+雾节点
                    type_score = 1.2
                
                # 综合评分（GWO特别关注能效和成本）
                vm_scores[vm_id] = (performance_score * 0.2 + energy_score * 0.3 + 
                                   cost_score * 0.25 + latency_score * 0.1 + 
                                   resource_score * 0.1 + type_score * 0.05)
            
            # 基于分数的概率选择
            probabilities = vm_scores / np.sum(vm_scores)
            solution[task_id] = np.random.choice(self.N, p=probabilities)
        
        return solution
    
    def calculate_fitness(self, wolf: np.ndarray) -> float:
        """
        计算狼的适应度
        
        Args:
            wolf: 狼的位置（任务分配方案）
        
        Returns:
            适应度值（越大越好）
        """
        # 计算各虚拟机的负载
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_energy = np.zeros(self.N)
        vm_transmission = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = int(wolf[task_id])  # 确保vm_id是标量值
            total_time = self.execution_time[task_id][vm_id] + self.transmission_time[task_id][vm_id]
            vm_loads[vm_id] += total_time
            vm_costs[vm_id] += self.vm_cost[vm_id] * total_time
            vm_energy[vm_id] += self.energy_consumption[task_id][vm_id]
            vm_transmission[vm_id] += self.transmission_time[task_id][vm_id]
        
        # 1. 完工时间优化
        makespan = np.max(vm_loads)
        makespan_score = 1.0 / (makespan + 1e-6)
        
        # 2. 能耗优化（GWO的重点）
        total_energy = np.sum(vm_energy)
        energy_score = 1.0 / (total_energy + 1e-6)
        
        # 3. 成本优化
        total_cost = np.sum(vm_costs)
        cost_score = 1.0 / (total_cost + 1e-6)
        
        # 4. 资源利用率
        total_utilization = 0
        active_vms = 0
        
        for vm_id in range(self.N):
            assigned_tasks = [task_id for task_id in range(self.M) if int(wolf[task_id]) == vm_id]
            if assigned_tasks:
                active_vms += 1
                cpu_util = min(1.0, sum(self.task_cpu[task_id] for task_id in assigned_tasks) / self.vm_cpu_capacity[vm_id])
                memory_util = min(1.0, sum(self.task_memory[task_id] for task_id in assigned_tasks) / self.vm_memory_capacity[vm_id])
                storage_util = min(1.0, sum(self.task_storage[task_id] for task_id in assigned_tasks) / self.vm_storage_capacity[vm_id])
                network_util = min(1.0, sum(self.task_network[task_id] for task_id in assigned_tasks) / self.vm_network_capacity[vm_id])
                total_utilization += (cpu_util + memory_util + storage_util + network_util) / 4.0
        
        utilization_score = total_utilization / max(active_vms, 1)
        
        # 5. 负载均衡
        load_balance_score = 1.0 / (np.std(vm_loads) + 1e-6)
        
        # 6. 延迟优化（IoT任务重要指标）
        total_latency = 0
        for task_id in range(self.M):
            vm_id = int(wolf[task_id])  # 确保vm_id是标量值
            total_latency += self.vm_latency[vm_id] + self.transmission_time[task_id][vm_id]
        latency_score = 1.0 / (total_latency + 1e-6)
        
        # 7. 截止时间满足度
        deadline_satisfaction = 0
        for task_id in range(self.M):
            vm_id = int(wolf[task_id])  # 确保vm_id是标量值
            total_time = self.execution_time[task_id][vm_id] + self.transmission_time[task_id][vm_id]
            if total_time <= self.task_deadline[task_id]:
                deadline_satisfaction += 1
        deadline_score = deadline_satisfaction / self.M
        
        # 8. 雾-云协同效果
        fog_tasks = sum(1 for task_id in range(self.M) if self.vm_type[int(wolf[task_id])] == 0)
        cloud_tasks = self.M - fog_tasks
        
        # 鼓励合理的雾-云分布
        collaboration_score = 1.0
        if fog_tasks > 0 and cloud_tasks > 0:
            collaboration_score = 1.2  # 奖励混合部署
        
        # 综合适应度（GWO特别强调能耗和成本优化）
        fitness = (makespan_score * 0.15 +
                  energy_score * 0.25 +
                  cost_score * 0.2 +
                  utilization_score * 0.15 +
                  load_balance_score * 0.1 +
                  latency_score * 0.1 +
                  deadline_score * 0.03 +
                  collaboration_score * 0.02)
        
        return fitness
    
    def _update_leadership(self):
        """
        更新领导狼（Alpha, Beta, Delta）
        """
        # 按适应度排序
        sorted_indices = np.argsort(self.fitness)[::-1]  # 降序排列
        
        # 更新Alpha狼（最优）
        if self.fitness[sorted_indices[0]] > self.alpha_fitness:
            self.alpha_fitness = self.fitness[sorted_indices[0]]
            self.alpha_wolf = self.wolves[sorted_indices[0]].copy()
        
        # 更新Beta狼（次优）
        if len(sorted_indices) > 1 and self.fitness[sorted_indices[1]] > self.beta_fitness:
            self.beta_fitness = self.fitness[sorted_indices[1]]
            self.beta_wolf = self.wolves[sorted_indices[1]].copy()
        
        # 更新Delta狼（第三优）
        if len(sorted_indices) > 2 and self.fitness[sorted_indices[2]] > self.delta_fitness:
            self.delta_fitness = self.fitness[sorted_indices[2]]
            self.delta_wolf = self.wolves[sorted_indices[2]].copy()
    
    def _update_wolf_position(self, wolf_idx: int, a: float):
        """
        更新狼的位置
        
        Args:
            wolf_idx: 狼的索引
            a: 收敛因子
        """
        if self.alpha_wolf is None or self.beta_wolf is None or self.delta_wolf is None:
            return
        
        # 计算A和C向量
        A1 = 2 * a * np.random.random() - a
        A2 = 2 * a * np.random.random() - a
        A3 = 2 * a * np.random.random() - a
        
        C1 = 2 * np.random.random()
        C2 = 2 * np.random.random()
        C3 = 2 * np.random.random()
        
        # 计算向Alpha、Beta、Delta的距离和新位置
        new_wolf = np.zeros(self.M, dtype=int)
        
        for task_id in range(self.M):
            # 向Alpha学习
            D_alpha = abs(C1 * self.alpha_wolf[task_id] - self.wolves[wolf_idx][task_id])
            X1 = self.alpha_wolf[task_id] - A1 * D_alpha
            
            # 向Beta学习
            D_beta = abs(C2 * self.beta_wolf[task_id] - self.wolves[wolf_idx][task_id])
            X2 = self.beta_wolf[task_id] - A2 * D_beta
            
            # 向Delta学习
            D_delta = abs(C3 * self.delta_wolf[task_id] - self.wolves[wolf_idx][task_id])
            X3 = self.delta_wolf[task_id] - A3 * D_delta
            
            # 计算新位置（取平均并转换为整数VM ID）
            new_position = (X1 + X2 + X3) / 3.0
            new_vm = int(round(new_position)) % self.N
            new_vm = max(0, min(new_vm, self.N - 1))
            
            # 添加随机扰动以增加多样性
            if random.random() < 0.1:  # 10%概率
                new_vm = random.randint(0, self.N - 1)
            
            new_wolf[task_id] = new_vm
        
        # 更新狼的位置
        new_fitness = self.calculate_fitness(new_wolf)
        if new_fitness > self.fitness[wolf_idx]:
            self.wolves[wolf_idx] = new_wolf
            self.fitness[wolf_idx] = new_fitness
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行灰狼优化
        
        Returns:
            最优解和最优适应度值
        """
        # print("开始灰狼优化...")  # 注释掉以避免编码问题
        
        # 初始化适应度
        for i in range(self.num_wolves):
            self.fitness[i] = self.calculate_fitness(self.wolves[i])
        
        # 初始化领导狼
        self._update_leadership()
        
        for iteration in range(self.max_iterations):
            # 更新收敛因子a
            a = self.a_max - iteration * (self.a_max - self.a_min) / self.max_iterations
            
            # 更新每只狼的位置
            for i in range(self.num_wolves):
                self._update_wolf_position(i, a)
            
            # 更新领导狼
            self._update_leadership()
            
            # 记录历史
            current_best = max(self.fitness)
            current_avg = np.mean(self.fitness)
            current_worst = min(self.fitness)
            
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': current_best,
                'avg_fitness': current_avg,
                'worst_fitness': current_worst,
                'alpha_fitness': self.alpha_fitness,
                'best_solution': self.alpha_wolf.copy(),  # 新增：记录每代的最优解
                'convergence_factor': a
            })
            
            # 打印进度
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best Fitness = {current_best:.4f}, Alpha Fitness = {self.alpha_fitness:.4f}, a = {a:.3f}")
        
        # print(f"优化完成！Alpha狼适应度: {self.alpha_fitness:.4f}")  # 注释掉以避免编码问题
        return self.alpha_wolf, self.alpha_fitness, self.fitness_history
    
    def get_detailed_metrics(self, solution: np.ndarray) -> Dict:
        """
        获取详细的性能指标
        """
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_energy = np.zeros(self.N)
        vm_task_counts = np.zeros(self.N)
        vm_transmission = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = int(solution[task_id])  # 确保vm_id是标量值
            total_time = self.execution_time[task_id][vm_id] + self.transmission_time[task_id][vm_id]
            vm_loads[vm_id] += total_time
            vm_costs[vm_id] += self.vm_cost[vm_id] * total_time
            vm_energy[vm_id] += self.energy_consumption[task_id][vm_id]
            vm_task_counts[vm_id] += 1
            vm_transmission[vm_id] += self.transmission_time[task_id][vm_id]
        
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
        total_transmission = np.sum(vm_transmission)
        
        # 资源利用率
        total_utilization = 0
        active_vms = 0
        
        for vm_id in range(self.N):
            assigned_tasks = [task_id for task_id in range(self.M) if int(solution[task_id]) == vm_id]
            if assigned_tasks:
                active_vms += 1
                cpu_util = min(1.0, sum(self.task_cpu[task_id] for task_id in assigned_tasks) / self.vm_cpu_capacity[vm_id])
                memory_util = min(1.0, sum(self.task_memory[task_id] for task_id in assigned_tasks) / self.vm_memory_capacity[vm_id])
                storage_util = min(1.0, sum(self.task_storage[task_id] for task_id in assigned_tasks) / self.vm_storage_capacity[vm_id])
                network_util = min(1.0, sum(self.task_network[task_id] for task_id in assigned_tasks) / self.vm_network_capacity[vm_id])
                total_utilization += (cpu_util + memory_util + storage_util + network_util) / 4.0
        
        avg_resource_utilization = total_utilization / max(active_vms, 1)
        
        # 截止时间满足度
        deadline_met = 0
        for task_id in range(self.M):
            vm_id = int(solution[task_id])  # 确保vm_id是标量值
            total_time = self.execution_time[task_id][vm_id] + self.transmission_time[task_id][vm_id]
            if total_time <= self.task_deadline[task_id]:
                deadline_met += 1
        deadline_satisfaction = deadline_met / self.M
        
        # 雾-云分布
        fog_tasks = 0
        for task_id in range(self.M):
            vm_id = int(solution[task_id])  # 确保vm_id是标量值
            if self.vm_type[vm_id] == 0:
                fog_tasks += 1
        cloud_tasks = self.M - fog_tasks
        
        # 平均延迟
        total_latency = 0
        for task_id in range(self.M):
            vm_id = int(solution[task_id])  # 确保vm_id是标量值
            total_latency += self.vm_latency[vm_id]
        avg_latency = total_latency / self.M
        
        return {
            'makespan': makespan,
            'resource_utilization': avg_resource_utilization,
            'load_imbalance': load_imbalance,
            'total_cost': total_cost,
            'total_energy': total_energy,
            'total_transmission_time': total_transmission,
            'deadline_satisfaction': deadline_satisfaction,
            'fog_tasks': fog_tasks,
            'cloud_tasks': cloud_tasks,
            'avg_latency': avg_latency,
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
        alpha_fitness = [h['alpha_fitness'] for h in self.fitness_history]
        convergence_factor = [h['convergence_factor'] for h in self.fitness_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 适应度曲线
        ax1.plot(iterations, best_fitness, 'b-', label='Best Fitness (Current)', linewidth=2)
        ax1.plot(iterations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
        ax1.plot(iterations, alpha_fitness, 'g-', label='Alpha Fitness (Global Best)', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Grey Wolf Optimizer Convergence Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 收敛因子曲线
        ax2.plot(iterations, convergence_factor, 'purple', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Convergence Factor (a)')
        ax2.set_title('Convergence Factor Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fog_cloud_distribution(self, save_path: str = None):
        """
        绘制雾-云任务分布
        """
        if self.alpha_wolf is None:
            print("No solution available. Run optimize() first.")
            return
        
        fog_tasks = sum(1 for task_id in range(self.M) if self.vm_type[self.alpha_wolf[task_id]] == 0)
        cloud_tasks = self.M - fog_tasks
        
        plt.figure(figsize=(10, 6))
        
        # 饼图
        plt.subplot(1, 2, 1)
        labels = ['Fog Nodes', 'Cloud Nodes']
        sizes = [fog_tasks, cloud_tasks]
        colors = ['lightgreen', 'lightblue']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Task Distribution: Fog vs Cloud Nodes')
        
        # 柱状图
        plt.subplot(1, 2, 2)
        categories = ['Fog Nodes', 'Cloud Nodes']
        task_counts = [fog_tasks, cloud_tasks]
        plt.bar(categories, task_counts, color=colors, alpha=0.7)
        plt.ylabel('Number of Tasks')
        plt.title('Task Count by Node Type')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建GWO调度器实例
    gwo_scheduler = GreyWolfOptimizer(
        M=75,   # 75个任务
        N=12,   # 12个虚拟机（雾+云）
        num_wolves=30,
        max_iterations=100,
        a_max=2.0,
        a_min=0.0
    )
    
    # 执行优化
    best_solution, best_fitness = gwo_scheduler.optimize()
    
    # 获取详细指标
    metrics = gwo_scheduler.get_detailed_metrics(best_solution)
    print(f"\n详细性能指标:")
    print(f"完工时间(makespan): {metrics['makespan']:.2f}")
    print(f"资源利用率: {metrics['resource_utilization']:.3f}")
    print(f"负载不均衡度: {metrics['load_imbalance']:.2f}")
    print(f"总成本: {metrics['total_cost']:.2f}")
    print(f"总能耗: {metrics['total_energy']:.2f}")
    print(f"总传输时间: {metrics['total_transmission_time']:.2f}")
    print(f"截止时间满足度: {metrics['deadline_satisfaction']:.3f}")
    print(f"雾节点任务数: {metrics['fog_tasks']}")
    print(f"云节点任务数: {metrics['cloud_tasks']}")
    print(f"平均延迟: {metrics['avg_latency']:.2f}")
    
    # 绘制收敛曲线
    gwo_scheduler.plot_convergence()
    
    # 绘制雾-云分布
    gwo_scheduler.plot_fog_cloud_distribution()
    
    print(f"\n任务分配方案（前10个任务）:")
    for i, vm_id in enumerate(best_solution[:10]):
        node_type = "Fog" if gwo_scheduler.vm_type[vm_id] == 0 else "Cloud"
        latency = gwo_scheduler.vm_latency[vm_id]
        energy = gwo_scheduler.energy_consumption[i][vm_id]
        print(f"任务 {i} -> 虚拟机 {vm_id} ({node_type}, 延迟: {latency:.1f}ms, 能耗: {energy:.2f})")