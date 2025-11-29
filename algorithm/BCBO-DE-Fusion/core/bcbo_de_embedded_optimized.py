#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-DE嵌入式融合云任务调度器 - 优化版 v3.3
针对大规模场景的负载均衡和执行效率优化

主要改进:
1. 负载均衡约束强化 - 添加负载均衡惩罚和修复机制
2. 规模自适应参数优化 - 根据问题规模动态调整
3. 多样性维护增强 - 防止过早收敛
4. 局部搜索增强 - 提升精英解质量
"""

import sys
import os
import copy
from typing import Dict, List, Tuple
import numpy as np

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
fusion_dir = os.path.dirname(current_dir)
algorithm_dir = os.path.dirname(fusion_dir)
bcbo_path = os.path.join(algorithm_dir, 'BCBO')

if fusion_dir not in sys.path:
    sys.path.insert(0, fusion_dir)
if bcbo_path not in sys.path:
    sys.path.insert(0, bcbo_path)

from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from config.fusion_config import (
    PHASE_RATIOS, PHASE_SEQUENCE, FUSION_PHASES, PURE_BCBO_PHASES,
    determine_current_phase, get_phase_iterations, get_fusion_intensity
)
from config.parameters import DE_CONFIG, FUSION_CONFIG, EXPERIMENT_CONFIG
from de_operators import DEOperators
from utils.diversity_calculator import DiversityCalculator
from utils.adaptive_controller import AdaptiveFController, AdaptiveCRController
from utils.performance_monitor import PerformanceMonitor
from utils.adaptive_strategies import (
    get_adaptive_fusion_intensity,
    get_scale_adaptive_params,
    get_combined_fusion_intensity,
    ConvergenceMonitor,
    EliteProtectionConfig,
    apply_scale_adaptive_params
)


class LoadBalanceEnhancer:
    """负载均衡增强器"""

    def __init__(self, M: int, N: int, execution_time: np.ndarray):
        """
        初始化负载均衡增强器

        参数:
            M: 任务数量
            N: 虚拟机数量
            execution_time: M×N执行时间矩阵
        """
        self.M = M
        self.N = N
        self.execution_time = execution_time

        # 规模自适应惩罚权重
        self.penalty_weight = min(1.0, M / 1000.0)  # M=1000时权重=1.0

    def calculate_load_balance(self, solution: List[int]) -> float:
        """
        计算负载均衡度

        参数:
            solution: 任务分配方案

        返回:
            load_balance: 负载均衡度 (0-1,越接近1越均衡)
        """
        workloads = self._calculate_workloads(solution)
        max_load = max(workloads)
        min_load = min(workloads)

        if max_load == 0:
            return 1.0

        # 负载均衡度 = 1 - (最大负载-最小负载)/最大负载
        balance = 1.0 - (max_load - min_load) / max_load
        return balance

    def _calculate_workloads(self, solution: List[int]) -> List[float]:
        """计算每台虚拟机的工作负载"""
        workloads = [0.0] * self.N
        for task_id, vm_id in enumerate(solution):
            workloads[vm_id] += self.execution_time[task_id][vm_id]
        return workloads

    def get_balance_penalty(self, solution: List[int]) -> float:
        """
        获取负载不均衡惩罚值

        参数:
            solution: 任务分配方案

        返回:
            penalty: 惩罚值(越大表示越不均衡)
        """
        balance = self.calculate_load_balance(solution)
        # 不均衡度 = (1 - 负载均衡度)
        imbalance = 1.0 - balance
        # 应用规模自适应权重
        penalty = self.penalty_weight * imbalance * 100
        return penalty

    def repair_balance(self, solution: List[int], threshold: float = 0.8) -> List[int]:
        """
        修复负载不均衡的解

        参数:
            solution: 原始解
            threshold: 负载均衡阈值,低于此值触发修复

        返回:
            repaired_solution: 修复后的解
        """
        repaired = solution.copy()
        balance = self.calculate_load_balance(repaired)

        # 如果负载均衡度已经很好,不需要修复
        if balance >= threshold:
            return repaired

        # 迭代修复,最多10次
        max_repairs = 10
        for _ in range(max_repairs):
            workloads = self._calculate_workloads(repaired)
            max_load = max(workloads)
            min_load = min(workloads)

            # 如果达到阈值,停止修复
            if max_load == 0 or (1.0 - (max_load - min_load) / max_load) >= threshold:
                break

            # 找出最忙和最空闲的VM
            overloaded_vm = workloads.index(max_load)
            underloaded_vm = workloads.index(min_load)

            # 找出分配到过载VM的任务
            tasks_on_overloaded = [
                i for i, vm in enumerate(repaired) if vm == overloaded_vm
            ]

            if not tasks_on_overloaded:
                break

            # 选择一个任务移动到空闲VM
            # 优先选择执行时间较短的任务,减少影响
            best_task = None
            best_improvement = -float('inf')

            for task in tasks_on_overloaded:
                # 计算移动这个任务后的负载改善
                old_diff = max_load - min_load
                new_max = max_load - self.execution_time[task][overloaded_vm]
                new_min = min_load + self.execution_time[task][underloaded_vm]
                new_diff = abs(new_max - new_min)
                improvement = old_diff - new_diff

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_task = task

            if best_task is not None and best_improvement > 0:
                repaired[best_task] = underloaded_vm
            else:
                break  # 无法进一步改善

        return repaired


class DiversityMaintainer:
    """多样性维护器"""

    def __init__(self, M: int, N: int, diversity_threshold: float = 0.1):
        """
        初始化多样性维护器

        参数:
            M: 任务数量
            N: 虚拟机数量
            diversity_threshold: 多样性阈值
        """
        self.M = M
        self.N = N
        self.diversity_threshold = diversity_threshold
        self.calculator = DiversityCalculator()

    def check_and_restore(self, population: List, iteration: int,
                          check_interval: int = 10) -> List:
        """
        检查并恢复种群多样性

        参数:
            population: 当前种群
            iteration: 当前迭代次数
            check_interval: 检查间隔

        返回:
            restored_population: 恢复多样性后的种群
        """
        # 只在特定间隔检查
        if iteration % check_interval != 0:
            return population

        # 计算多样性
        diversity = self.calculator.hamming_distance_diversity(population)

        # 如果多样性足够,不需要恢复
        if diversity >= self.diversity_threshold:
            return population

        # 多样性不足,重新初始化部分种群
        restored = population.copy()
        reinit_ratio = 0.3  # 重新初始化30%
        num_reinit = max(1, int(len(population) * reinit_ratio))

        # 保留精英,重新初始化差解
        # 按适应度排序(需要传入适应度函数,这里简化处理)
        # 实际使用时需要从外部传入
        for i in range(len(population) - num_reinit, len(population)):
            restored[i] = self._random_solution()

        return restored

    def _random_solution(self) -> List[int]:
        """生成随机解"""
        return [np.random.randint(0, self.N) for _ in range(self.M)]


class LocalSearchEnhancer:
    """局部搜索增强器"""

    def __init__(self, M: int, N: int, execution_time: np.ndarray):
        """
        初始化局部搜索增强器

        参数:
            M: 任务数量
            N: 虚拟机数量
            execution_time: 执行时间矩阵
        """
        self.M = M
        self.N = N
        self.execution_time = execution_time

    def hill_climbing(self, solution: List[int], fitness_func,
                      max_attempts: int = 10) -> List[int]:
        """
        爬山法局部搜索

        参数:
            solution: 当前解
            fitness_func: 适应度函数
            max_attempts: 最大尝试次数

        返回:
            improved_solution: 改进后的解
        """
        current = solution.copy()
        current_fitness = fitness_func(current)

        for _ in range(max_attempts):
            # 随机选择一个任务
            task = np.random.randint(0, self.M)
            old_vm = current[task]

            # 尝试分配到其他VM
            new_vm = np.random.randint(0, self.N)

            # 避免无效移动
            if new_vm == old_vm:
                continue

            # 尝试移动
            current[task] = new_vm
            new_fitness = fitness_func(current)

            # 贪心接受
            if new_fitness > current_fitness:
                current_fitness = new_fitness
            else:
                # 恢复
                current[task] = old_vm

        return current


class BCBO_DE_Embedded_Optimized:
    """BCBO-DE嵌入式融合云任务调度器 - 优化版"""

    def __init__(self, M: int, N: int, n: int, iterations: int, **kwargs):
        """
        初始化融合调度器

        参数:
            M: 任务数量
            N: 虚拟机数量
            n: 种群大小
            iterations: 总迭代次数
            **kwargs: 其他可选参数
        """
        # 1. 保存基本参数
        self.M = M
        self.N = N
        self.n = n
        self.iterations = iterations
        self.random_seed = kwargs.get('random_seed', 42)
        self.verbose = kwargs.get('verbose', True)
        self.print_interval = kwargs.get('print_interval', 10)

        # 2. 创建BCBO实例
        self.bcbo = BCBO_CloudScheduler(
            M=M, N=N, n=n, iterations=iterations
        )

        # 3. 初始化DE参数 - 优化版
        self.de_config = DE_CONFIG.copy()
        self.de_config.update(kwargs.get('de_config', {}))

        # 3.5. 规模自适应调整 + 优化
        self.scale_params = get_scale_adaptive_params(M, N)
        self.de_config = apply_scale_adaptive_params(self.de_config, self.scale_params)

        # 针对大规模场景进一步优化DE参数
        if M > 1000:
            # 增加探索能力
            self.de_config['F'] = min(0.9, self.de_config['F'] * 1.3)
            # 降低交叉概率,保持多样性
            self.de_config['CR'] = max(0.6, self.de_config['CR'] * 0.85)

        # 4. 创建DE算子实例
        self.de_operators = DEOperators(
            M=M, N=N,
            F=self.de_config['F'],
            CR=self.de_config['CR']
        )

        # 5. 初始化融合配置
        self.fusion_config = FUSION_CONFIG.copy()
        self.fusion_config.update(kwargs.get('fusion_config', {}))
        self.fusion_config['elite_ratio'] = self.scale_params['elite_ratio']

        # 6. 初始化自适应控制器
        self.F_controller = AdaptiveFController(
            F_max=self.de_config.get('F_max', 0.32),
            F_min=self.de_config.get('F_min', 0.18),
            alpha=self.de_config['alpha']
        )
        self.CR_controller = AdaptiveCRController(
            CR_min=self.de_config.get('CR_min', 0.28),
            CR_max=self.de_config.get('CR_max', 0.62)
        )

        # 7. 初始化性能监控器
        self.monitor = PerformanceMonitor()

        # 8. 初始化多样性计算器
        self.diversity_calculator = DiversityCalculator()

        # 9. 初始化收敛监控器
        self.convergence_monitor = ConvergenceMonitor(patience=15)  # 增加耐心

        # 10. 初始化增强器 (新增)
        self.load_balancer = LoadBalanceEnhancer(
            M, N, self.bcbo.execution_time
        )
        self.diversity_maintainer = DiversityMaintainer(
            M, N, diversity_threshold=0.15
        )
        self.local_searcher = LocalSearchEnhancer(
            M, N, self.bcbo.execution_time
        )

        # 11. 初始化全局最优
        self.global_best_solution = None
        self.global_best_fitness = float('-inf')

        if self.verbose:
            print(f"BCBO-DE融合调度器初始化完成 (优化版v3.3):")
            print(f"  任务数M={M}, VM数N={N}, 种群大小n={n}, 迭代次数={iterations}")
            print(f"  优化DE参数: F={self.de_config['F']:.3f}, CR={self.de_config['CR']:.3f}")
            print(f"  负载均衡惩罚权重: {self.load_balancer.penalty_weight:.3f}")
            print(f"  规模自适应参数: elite_ratio={self.scale_params['elite_ratio']:.2%}")

    def run_fusion_optimization(self) -> Dict:
        """
        运行融合优化(主方法) - 优化版

        返回:
            result: 包含最优解和性能指标的字典
        """
        if self.verbose:
            print("\n" + "="*60)
            print("开始BCBO-DE融合优化 (v3.3优化版)".center(60))
            print("="*60)

        # 初始化种群
        population = self.bcbo.initialize_population()

        # 主循环
        for iteration in range(self.iterations):
            # 1. 确定当前阶段
            current_phase = determine_current_phase(iteration, self.iterations)

            # 2. 判断是否为融合阶段
            is_fusion = self._is_fusion_phase(current_phase)

            # 3. 更新收敛监控器
            self.convergence_monitor.update(self.global_best_fitness)

            # 4. 获取收敛状态调整建议
            convergence_adjustment = self.convergence_monitor.get_adaptive_adjustment()

            # 5. 计算综合融合强度
            fusion_intensity = get_combined_fusion_intensity(
                phase=current_phase,
                iteration=iteration,
                total_iterations=self.iterations,
                M=self.M,
                N=self.N,
                convergence_adjustment=convergence_adjustment['intensity_adjust']
            )

            # 6. 根据融合强度决定是否应用DE
            apply_de = is_fusion and (np.random.random() < fusion_intensity)

            # 7. 执行对应的更新策略 (优化版)
            if apply_de:
                population = self._bcbo_de_fusion_update_optimized(
                    population, current_phase, iteration
                )
                update_type = f"BCBO-DE融合优化({fusion_intensity*100:.0f}%)"
            else:
                population = self._bcbo_pure_update(
                    population, current_phase, iteration
                )
                update_type = "纯BCBO"

            # 8. 多样性维护 (新增)
            population = self.diversity_maintainer.check_and_restore(
                population, iteration, check_interval=15
            )

            # 9. 更新全局最优 + 负载均衡检查
            for individual in population:
                fitness = self._calculate_fitness_with_balance(individual)
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = copy.deepcopy(individual)

            # 10. 记录历史
            bcbo_ratio = self._get_current_bcbo_ratio(population, iteration, current_phase)
            self.monitor.record(
                iteration=iteration,
                population=population,
                best_fitness=self.global_best_fitness,
                bcbo_ratio=bcbo_ratio,
                phase=current_phase,
                fitness_func=self._calculate_fitness_with_balance,
                best_solution=copy.deepcopy(self.global_best_solution)
            )

            # 11. 打印信息
            if self.verbose and (iteration % self.print_interval == 0 or iteration == self.iterations - 1):
                diversity = self.diversity_calculator.hamming_distance_diversity(population)
                balance = self.load_balancer.calculate_load_balance(self.global_best_solution)
                conv_status = self.convergence_monitor.get_status()
                print(f"Iter {iteration:3d} | 阶段: {current_phase:15s} | "
                      f"更新: {update_type:25s} | 最优: {self.global_best_fitness:.6f} | "
                      f"负载均衡: {balance:.4f} | 多样性: {diversity:.4f} | "
                      f"停滞: {conv_status['stagnation_count']:2d}")

        # 12. 生成结果
        result = {
            'best_solution': self.global_best_solution,
            'best_fitness': self.global_best_fitness,
            'load_balance': self.load_balancer.calculate_load_balance(self.global_best_solution),
            'history': self.monitor.history,
            'summary': self.monitor.get_summary(),
            'diagnosis': self.monitor.diagnose()
        }

        if self.verbose:
            print("\n" + "="*60)
            print("优化完成".center(60))
            print("="*60)
            print(f"最优适应度: {self.global_best_fitness:.6f}")
            print(f"负载均衡度: {result['load_balance']:.6f}")
            print(f"优化诊断: {result['diagnosis']}")

        return result

    def _calculate_fitness_with_balance(self, solution: List[int]) -> float:
        """
        计算带负载均衡惩罚的适应度 (新增)

        参数:
            solution: 任务分配方案

        返回:
            adjusted_fitness: 调整后的适应度
        """
        # 基础适应度
        base_fitness = self.bcbo.comprehensive_fitness(solution)

        # 负载均衡惩罚
        balance_penalty = self.load_balancer.get_balance_penalty(solution)

        # 调整后的适应度 = 基础适应度 - 惩罚
        adjusted_fitness = base_fitness - balance_penalty

        return adjusted_fitness

    def _is_fusion_phase(self, phase: str) -> bool:
        """判断是否为融合阶段"""
        return phase in FUSION_PHASES

    def _bcbo_de_fusion_update_optimized(self, population: List, phase: str,
                                        iteration: int) -> List:
        """
        BCBO-DE融合更新 - 优化版

        新增优化:
        1. 负载均衡修复
        2. 精英局部搜索
        3. 更严格的精英保护

        参数:
            population: 当前种群
            phase: 当前阶段
            iteration: 当前迭代

        返回:
            final_population: 更新后的种群
        """
        # 步骤1: 全员执行 BCBO 更新
        bcbo_updated_pop = self._bcbo_pure_update(population, phase, iteration)

        # 步骤2: 识别精英
        elite_ratio = self.fusion_config.get('elite_ratio', 0.2)

        sorted_pop = sorted(
            bcbo_updated_pop,
            key=lambda x: self._calculate_fitness_with_balance(x),
            reverse=True
        )

        elite_count = max(1, int(len(population) * elite_ratio))
        elites = sorted_pop[:elite_count]
        others = sorted_pop[elite_count:]

        # 步骤3: 分级精英保护
        top_elite_count = max(1, elite_count // 2)
        top_elites = elites[:top_elite_count]
        mid_elites = elites[top_elite_count:]

        # 获取基础DE参数
        current_F = self.F_controller.get_F(iteration, self.iterations)
        current_diversity = self._calculate_diversity(population)
        current_CR = self.CR_controller.get_CR(current_diversity)

        # 步骤4: 对mid_elites应用DE + 负载均衡修复
        enhanced_mid_elites = []
        for target in mid_elites:
            mutant = self.de_operators.mutate(bcbo_updated_pop, target, current_F)
            trial = self.de_operators.crossover(target, mutant, current_CR)

            # 负载均衡修复 (新增)
            trial = self.load_balancer.repair_balance(trial, threshold=0.85)

            selected = self.de_operators.select(
                target, trial,
                fitness_func=self._calculate_fitness_with_balance
            )
            enhanced_mid_elites.append(selected)

        # 步骤5: 对top_elites严格保护 + 局部搜索
        protection_params = EliteProtectionConfig.get_protection_params('top', M=self.M)
        protected_F = current_F * protection_params['f_decay']
        protected_CR = current_CR * protection_params['cr_decay']
        threshold = protection_params['threshold']

        protected_top_elites = []
        for target in top_elites:
            target_fitness = self._calculate_fitness_with_balance(target)

            # DE变异
            mutant = self.de_operators.mutate(bcbo_updated_pop, target, protected_F)
            trial = self.de_operators.crossover(target, mutant, protected_CR)

            # 负载均衡修复
            trial = self.load_balancer.repair_balance(trial, threshold=0.9)

            # 局部搜索增强 (新增)
            if iteration % 5 == 0:  # 每5次迭代应用一次局部搜索
                trial = self.local_searcher.hill_climbing(
                    trial,
                    self._calculate_fitness_with_balance,
                    max_attempts=5
                )

            trial_fitness = self._calculate_fitness_with_balance(trial)

            # 严格选择
            if trial_fitness > target_fitness * (1 + threshold):
                protected_top_elites.append(trial)
            else:
                protected_top_elites.append(target)

        # 步骤6: 合并
        final_population = protected_top_elites + enhanced_mid_elites + others

        return final_population

    def _bcbo_pure_update(self, population: List, phase: str, iteration: int) -> List:
        """纯BCBO更新逻辑"""
        if phase == 'dynamic_search' or 'dynamic' in phase:
             return self.bcbo.dynamic_search_phase(population, iteration, self.iterations)
        elif phase == 'static_search' or 'static' in phase:
             return self.bcbo.static_search_phase(population, iteration)
        else:
             return population

    def _calculate_diversity(self, population: List) -> float:
        """计算种群多样性"""
        return self.diversity_calculator.hamming_distance_diversity(population)

    def _get_current_bcbo_ratio(self, population: List, iteration: int, phase: str) -> float:
        """获取当前BCBO组比例"""
        return 1.0


# 测试代码
if __name__ == '__main__':
    print("BCBO-DE优化版算法测试")
    print("=" * 60)

    # 创建优化器(中等规模测试)
    optimizer = BCBO_DE_Embedded_Optimized(
        M=100,         # 100个任务
        N=20,          # 20个VM
        n=50,          # 种群大小50
        iterations=50, # 50次迭代
        verbose=True,
        print_interval=10
    )

    # 运行优化
    result = optimizer.run_fusion_optimization()

    # 显示结果
    print("\n" + "=" * 60)
    print("测试完成".center(60))
    print("=" * 60)
    print(f"最优适应度: {result['best_fitness']:.6f}")
    print(f"负载均衡度: {result['load_balance']:.6f}")
    print(f"性能摘要: {result['summary']}")
