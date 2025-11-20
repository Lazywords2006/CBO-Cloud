#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子群优化(PSO)云调度器实现

基于粒子群优化的云任务调度算法特点：
- 结构简单，收敛快速
- 全局搜索能力强
- 支持自适应参数调整
- 适用于连续和离散优化问题

作者：云调度算法研究团队
日期：2024年
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class ParticleSwarmOptimizer:
    """
    粒子群优化云调度器
    
    参数:
        M: 任务数量
        N: 虚拟机数量
        num_particles: 粒子数量
        max_iterations: 最大迭代次数
        w_max: 最大惯性权重
        w_min: 最小惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
        v_max: 最大速度
    """
    
    def __init__(self, M: int, N: int, num_particles: int = 30, 
                 max_iterations: int = 100, w_max: float = 0.9,
                 w_min: float = 0.4, c1: float = 2.0, c2: float = 2.0,
                 v_max: float = 4.0):
        self.M = M  # 任务数量
        self.N = N  # 虚拟机数量
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w_max = w_max  # 最大惯性权重
        self.w_min = w_min  # 最小惯性权重
        self.c1 = c1        # 个体学习因子
        self.c2 = c2        # 社会学习因子
        self.v_max = v_max  # 最大速度
        
        # 初始化任务和虚拟机属性
        self._initialize_tasks_and_vms()
        
        # 初始化粒子群
        self.particles = self._initialize_particles()
        self.velocities = self._initialize_velocities()
        self.personal_best = copy.deepcopy(self.particles)
        self.personal_best_fitness = [float('inf')] * self.num_particles
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        # 存储优化历史
        self.fitness_history = []
    
    def _initialize_tasks_and_vms(self):
        """
        初始化任务和虚拟机的属性
        """
        # 任务属性
        self.task_cpu = np.random.uniform(600, 1800, self.M)
        self.task_memory = np.random.uniform(80, 400, self.M)
        self.task_io = np.random.uniform(10, 100, self.M)
        self.task_priority = np.random.randint(1, 4, self.M)
        
        # 虚拟机属性
        self.vm_cpu_capacity = np.random.uniform(1800, 3200, self.N)
        self.vm_memory_capacity = np.random.uniform(3000, 12000, self.N)
        self.vm_io_capacity = np.random.uniform(100, 500, self.N)
        self.vm_processing_speed = np.random.uniform(1.2, 3.8, self.N)
        self.vm_cost = np.random.uniform(0.06, 0.18, self.N)
        
        # 计算执行时间矩阵
        self.execution_time = np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.N):
                workload = self.task_cpu[i] + self.task_memory[i] + self.task_io[i]
                self.execution_time[i][j] = workload / self.vm_processing_speed[j]
    
    def _initialize_particles(self) -> List[np.ndarray]:
        """
        初始化粒子群位置
        每个粒子表示一个任务分配方案
        """
        particles = []
        for _ in range(self.num_particles):
            # 随机分配任务到虚拟机
            particle = np.random.randint(0, self.N, self.M)
            particles.append(particle)
        return particles
    
    def _initialize_velocities(self) -> List[np.ndarray]:
        """
        初始化粒子速度
        """
        velocities = []
        for _ in range(self.num_particles):
            # 初始速度为小的随机值
            velocity = np.random.uniform(-self.v_max/2, self.v_max/2, self.M)
            velocities.append(velocity)
        return velocities
    
    def fitness_function(self, particle: np.ndarray) -> float:
        """
        云调度优化的适应度函数
        
        Args:
            particle: 粒子位置（任务分配方案）
        
        Returns:
            适应度值（越小越好）
        """
        # 计算各虚拟机的负载
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = int(particle[task_id]) % self.N  # 确保在有效范围内
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
        
        # 1. 完工时间 (makespan)
        makespan = np.max(vm_loads)
        
        # 2. 云环境资源利用率（包含资源违约检查）
        total_cpu_usage = 0
        total_memory_usage = 0
        total_io_usage = 0
        resource_violations = 0
        
        for vm_id in range(self.N):
            assigned_tasks = [task_id for task_id in range(self.M) if int(particle[task_id]) % self.N == vm_id]
            if assigned_tasks:
                cpu_usage = sum(self.task_cpu[task_id] for task_id in assigned_tasks) / self.vm_cpu_capacity[vm_id]
                memory_usage = sum(self.task_memory[task_id] for task_id in assigned_tasks) / self.vm_memory_capacity[vm_id]
                io_usage = sum(self.task_io[task_id] for task_id in assigned_tasks) / self.vm_io_capacity[vm_id]
                
                # 云环境资源违约检查
                if cpu_usage > 1.0 or memory_usage > 1.0 or io_usage > 1.0:
                    resource_violations += 1
                
                total_cpu_usage += min(cpu_usage, 1.0)
                total_memory_usage += min(memory_usage, 1.0)
                total_io_usage += min(io_usage, 1.0)
        
        avg_resource_utilization = (total_cpu_usage + total_memory_usage + total_io_usage) / (3 * self.N)
        
        # 3. 负载均衡（云环境关键指标）- 使用变异系数CV
        active_vm_loads = vm_loads[vm_loads > 0]
        if len(active_vm_loads) > 0:
            mean_load = np.mean(active_vm_loads)
            std_load = np.std(active_vm_loads)
            load_imbalance = std_load / (mean_load + 1e-6)
        else:
            load_imbalance = 0.0
        
        # 4. 云环境动态成本模型
        total_cost = 0
        for vm_id in range(self.N):
            if vm_loads[vm_id] > 0:
                # 基础成本
                base_cost = vm_costs[vm_id]
                # 云环境动态定价：高负载时成本增加
                load_factor = vm_loads[vm_id] / np.mean(vm_loads) if np.mean(vm_loads) > 0 else 1
                dynamic_cost = base_cost * (1 + 0.2 * max(0, load_factor - 1))
                total_cost += dynamic_cost
        
        # 5. 云环境SLA保证（任务优先级和响应时间）
        sla_penalty = 0
        for task_id in range(self.M):
            vm_id = int(particle[task_id]) % self.N
            task_completion_time = self.execution_time[task_id][vm_id]
            
            # 高优先级任务的SLA要求更严格
            if self.task_priority[task_id] == 3:  # 高优先级
                if self.vm_processing_speed[vm_id] < np.mean(self.vm_processing_speed):
                    sla_penalty += 10.0
                if task_completion_time > np.mean(self.execution_time[task_id]) * 1.2:
                    sla_penalty += 5.0
        
        # 6. 云环境能耗优化
        energy_consumption = 0
        for vm_id in range(self.N):
            if vm_loads[vm_id] > 0:
                # 云环境能耗模型：基础功耗 + 负载相关功耗
                base_power = 50  # 基础功耗
                load_power = vm_loads[vm_id] * 2  # 负载相关功耗
                energy_consumption += base_power + load_power
        
        # 7. 云环境网络延迟考虑
        network_penalty = 0
        for task_id in range(self.M):
            vm_id = int(particle[task_id]) % self.N
            # 简化的网络延迟模型
            if vm_id > self.N // 2:  # 假设后半部分VM网络延迟较高
                network_penalty += 2.0
        
        # 8. 云环境弹性扩展惩罚
        elasticity_penalty = 0
        active_vms = sum(1 for load in vm_loads if load > 0)
        if active_vms < self.N * 0.3:  # 如果使用的VM太少，失去弹性优势
            elasticity_penalty = (self.N * 0.3 - active_vms) * 10
        
        # 云环境优化的综合适应度（加权求和，越小越好）
        fitness = (0.20 * makespan +                              # 完工时间
                  0.18 * (1 - avg_resource_utilization) * 100 +   # 资源利用率
                  0.15 * load_imbalance +                         # 负载均衡
                  0.15 * total_cost / 100 +                       # 动态成本
                  0.12 * resource_violations * 50 +               # 资源违约
                  0.08 * sla_penalty +                            # SLA违约
                  0.06 * energy_consumption / 100 +               # 能耗优化
                  0.03 * network_penalty +                        # 网络延迟
                  0.03 * elasticity_penalty)                      # 弹性扩展
        
        return fitness
    
    def _update_inertia_weight(self, iteration: int) -> float:
        """
        自适应更新惯性权重
        
        Args:
            iteration: 当前迭代次数
        
        Returns:
            当前惯性权重
        """
        # 线性递减策略
        w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations
        return w
    
    def _update_velocity(self, particle_idx: int, iteration: int):
        """
        更新粒子速度
        
        Args:
            particle_idx: 粒子索引
            iteration: 当前迭代次数
        """
        w = self._update_inertia_weight(iteration)
        
        for dim in range(self.M):
            r1 = random.random()
            r2 = random.random()
            
            # 速度更新公式
            cognitive_component = self.c1 * r1 * (self.personal_best[particle_idx][dim] - self.particles[particle_idx][dim])
            social_component = self.c2 * r2 * (self.global_best[dim] - self.particles[particle_idx][dim])
            
            self.velocities[particle_idx][dim] = (w * self.velocities[particle_idx][dim] + 
                                                 cognitive_component + social_component)
            
            # 速度限制
            self.velocities[particle_idx][dim] = np.clip(self.velocities[particle_idx][dim], 
                                                        -self.v_max, self.v_max)
    
    def _update_position(self, particle_idx: int):
        """
        更新粒子位置（修复版：去除sigmoid饱和问题）
        
        Args:
            particle_idx: 粒子索引
        """
        for dim in range(self.M):
            # 位置更新（在连续空间）
            self.particles[particle_idx][dim] += self.velocities[particle_idx][dim]
            
            # 修复：使用取模映射代替sigmoid，避免饱和
            # 这样粒子可以在整个搜索空间自由移动
            vm_id = int(round(self.particles[particle_idx][dim])) % self.N
            
            # 处理负数（Python的%对负数处理不同）
            if vm_id < 0:
                vm_id += self.N
            
            # 更新为离散VM ID（保证在[0, N-1]范围）
            self.particles[particle_idx][dim] = vm_id
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行粒子群优化
        
        Returns:
            最优解和最优适应度值
        """
        print("开始粒子群优化...")
        
        # 初始化个体最优和全局最优
        for i in range(self.num_particles):
            fitness = self.fitness_function(self.particles[i])
            self.personal_best_fitness[i] = fitness
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = self.particles[i].copy()
        
        for iteration in range(self.max_iterations):
            fitness_values = []
            
            for i in range(self.num_particles):
                # 更新速度和位置
                self._update_velocity(i, iteration)
                self._update_position(i)
                
                # 计算适应度
                fitness = self.fitness_function(self.particles[i])
                fitness_values.append(fitness)
                
                # 更新个体最优
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                
                # 更新全局最优
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()
            
            # 记录历史（包含solution用于生成收敛曲线）
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': min(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'worst_fitness': max(fitness_values),
                'global_best_fitness': self.global_best_fitness,
                'best_solution': self.global_best.copy()  # 新增：记录每代的最优解
            })
            
            # 打印进度
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best Fitness = {min(fitness_values):.4f}, Global Best = {self.global_best_fitness:.4f}")
        
        print(f"优化完成！全局最优适应度: {self.global_best_fitness:.4f}")
        return self.global_best, self.global_best_fitness, self.fitness_history
    
    def get_detailed_metrics(self, solution: np.ndarray) -> Dict:
        """
        获取详细的性能指标
        """
        vm_loads = np.zeros(self.N)
        vm_costs = np.zeros(self.N)
        vm_task_counts = np.zeros(self.N)
        
        for task_id in range(self.M):
            vm_id = int(solution[task_id]) % self.N
            vm_loads[vm_id] += self.execution_time[task_id][vm_id]
            vm_costs[vm_id] += self.vm_cost[vm_id] * self.execution_time[task_id][vm_id]
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
        
        # 资源利用率
        total_cpu_usage = 0
        total_memory_usage = 0
        total_io_usage = 0
        
        for vm_id in range(self.N):
            assigned_tasks = [task_id for task_id in range(self.M) if int(solution[task_id]) % self.N == vm_id]
            if assigned_tasks:
                cpu_usage = sum(self.task_cpu[task_id] for task_id in assigned_tasks) / self.vm_cpu_capacity[vm_id]
                memory_usage = sum(self.task_memory[task_id] for task_id in assigned_tasks) / self.vm_memory_capacity[vm_id]
                io_usage = sum(self.task_io[task_id] for task_id in assigned_tasks) / self.vm_io_capacity[vm_id]
                
                total_cpu_usage += min(cpu_usage, 1.0)
                total_memory_usage += min(memory_usage, 1.0)
                total_io_usage += min(io_usage, 1.0)
        
        avg_resource_utilization = (total_cpu_usage + total_memory_usage + total_io_usage) / (3 * self.N)
        
        # 优先级分配统计
        high_priority_on_fast_vm = 0
        high_priority_tasks = sum(1 for p in self.task_priority if p == 3)
        
        for task_id in range(self.M):
            if self.task_priority[task_id] == 3:
                vm_id = int(solution[task_id]) % self.N
                if self.vm_processing_speed[vm_id] >= np.mean(self.vm_processing_speed):
                    high_priority_on_fast_vm += 1
        
        priority_satisfaction = high_priority_on_fast_vm / max(high_priority_tasks, 1)
        
        return {
            'makespan': makespan,
            'resource_utilization': avg_resource_utilization,
            'load_imbalance': load_imbalance,
            'total_cost': total_cost,
            'priority_satisfaction': priority_satisfaction,
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
        plt.ylabel('Fitness Value')
        plt.title('PSO Convergence Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_particle_diversity(self, save_path: str = None):
        """
        绘制粒子多样性变化
        """
        if not self.fitness_history:
            print("No fitness history available. Run optimize() first.")
            return
        
        iterations = [h['iteration'] for h in self.fitness_history]
        diversity = []
        
        for h in self.fitness_history:
            # 计算适应度标准差作为多样性指标
            diversity.append(h['worst_fitness'] - h['best_fitness'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, diversity, 'purple', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Diversity')
        plt.title('Particle Swarm Diversity Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 创建PSO调度器实例
    pso_scheduler = ParticleSwarmOptimizer(
        M=120,  # 120个任务
        N=12,   # 12个虚拟机
        num_particles=30,
        max_iterations=100,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        v_max=4.0
    )
    
    # 执行优化
    best_solution, best_fitness = pso_scheduler.optimize()
    
    # 获取详细指标
    metrics = pso_scheduler.get_detailed_metrics(best_solution)
    print(f"\n详细性能指标:")
    print(f"完工时间(makespan): {metrics['makespan']:.2f}")
    print(f"资源利用率: {metrics['resource_utilization']:.3f}")
    print(f"负载不均衡度: {metrics['load_imbalance']:.2f}")
    print(f"总成本: {metrics['total_cost']:.2f}")
    print(f"优先级满意度: {metrics['priority_satisfaction']:.3f}")
    
    # 绘制收敛曲线
    pso_scheduler.plot_convergence()
    
    # 绘制粒子多样性
    pso_scheduler.plot_particle_diversity()
    
    print(f"\n任务分配方案（前10个任务）:")
    for i, vm_id in enumerate(best_solution[:10]):
        print(f"任务 {i} -> 虚拟机 {vm_id}")