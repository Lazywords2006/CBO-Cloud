#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-DE嵌入式融合云任务调度器

实现BCBO六阶段与DE算法的嵌入式融合策略。
"""

import sys
import os
import copy
from typing import Dict, List, Tuple

# 添加BCBO路径
# 当前文件在 algorithm/BCBO-DE-Fusion/core/
# BCBO在 algorithm/BCBO/
# 需要上两级到algorithm,然后进入BCBO
current_dir = os.path.dirname(os.path.abspath(__file__))  # algorithm/BCBO-DE-Fusion/core
fusion_dir = os.path.dirname(current_dir)                 # algorithm/BCBO-DE-Fusion
algorithm_dir = os.path.dirname(fusion_dir)               # algorithm
bcbo_path = os.path.join(algorithm_dir, 'BCBO')          # algorithm/BCBO

if fusion_dir not in sys.path:
    sys.path.insert(0, fusion_dir)

if bcbo_path not in sys.path:
    sys.path.insert(0, bcbo_path)
from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler

# 导入项目内部模块
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
import numpy as np  # 需要使用np.random.random()判断融合强度


class BCBO_DE_Embedded:
    """BCBO-DE嵌入式融合云任务调度器"""

    def __init__(self, M: int, N: int, n: int, iterations: int, **kwargs):
        """
        初始化融合调度器

        参数:
            M: 任务数量
            N: 虚拟机数量
            n: 种群大小
            iterations: 总迭代次数
            **kwargs: 其他可选参数
                - random_seed: 随机种子
                - verbose: 是否输出详细信息
                - print_interval: 打印间隔
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

        # 3. 初始化DE参数
        self.de_config = DE_CONFIG.copy()
        self.de_config.update(kwargs.get('de_config', {}))

        # 3.5. 规模自适应调整（新增）
        self.scale_params = get_scale_adaptive_params(M, N)
        self.de_config = apply_scale_adaptive_params(self.de_config, self.scale_params)

        # 4. 创建DE算子实例 (方案C: 添加execution_time用于负载均衡计算)
        self.de_operators = DEOperators(
            M=M, N=N,
            F=self.de_config['F'],
            CR=self.de_config['CR'],
            execution_time=None  # 初始化后再设置
        )

        # 5. 初始化融合配置
        self.fusion_config = FUSION_CONFIG.copy()
        self.fusion_config.update(kwargs.get('fusion_config', {}))
        # 应用规模自适应的精英比例
        self.fusion_config['elite_ratio'] = self.scale_params['elite_ratio']

        # 6. 初始化自适应控制器(使用优化后的参数)
        self.F_controller = AdaptiveFController(
            F_max=self.de_config.get('F_max', 0.32),  # 基准值
            F_min=self.de_config.get('F_min', 0.18),  # 基准值
            alpha=self.de_config['alpha']
        )
        self.CR_controller = AdaptiveCRController(
            CR_min=self.de_config.get('CR_min', 0.28), # 基准值
            CR_max=self.de_config.get('CR_max', 0.62)  # 基准值
        )

        # 7. 初始化性能监控器
        self.monitor = PerformanceMonitor()

        # 8. 初始化多样性计算器
        self.diversity_calculator = DiversityCalculator()

        # 9. 初始化收敛监控器（新增）
        self.convergence_monitor = ConvergenceMonitor(patience=10)

        # 10. 初始化全局最优
        self.global_best_solution = None
        self.global_best_fitness = float('-inf')

        # 11. 方案D: 启用负载均衡导向适应度 (M>=1000时)
        self.use_balance_oriented_fitness = (M >= 1000)

        # 方案D适应度权重参数 (v3.5 最小干预策略)
        # 基于失败经验：回归接近原始BCBO的权重
        self.fitness_weights = {
            'alpha': 0.001,  # 成本权重 (回归原始值)
            'beta': 10,      # 负载均衡权重 (接近原始)
            'gamma': 10000   # makespan权重 (回归原始值)
        }

        if self.verbose:
            print(f"BCBO-DE融合调度器初始化完成:")
            print(f"  任务数M={M}, VM数N={N}, 种群大小n={n}, 迭代次数={iterations}")
            print(f"  融合阶段: {FUSION_PHASES}")
            print(f"  纯BCBO阶段: {PURE_BCBO_PHASES}")
            print(f"  规模自适应参数: F_scale={self.scale_params['F_scale']:.2f}, "
                  f"CR_scale={self.scale_params['CR_scale']:.2f}, "
                  f"elite_ratio={self.scale_params['elite_ratio']:.2%}, "
                  f"intensity_scale={self.scale_params['intensity_scale']:.2f}")
            if self.use_balance_oriented_fitness:
                print(f"  [方案D] 启用负载均衡导向适应度 (M={M}>=1000)")
                print(f"  [方案D] 权重: alpha={self.fitness_weights['alpha']}, "
                      f"beta={self.fitness_weights['beta']}, gamma={self.fitness_weights['gamma']}")

    def run_fusion_optimization(self) -> Dict:
        """
        运行融合优化(主方法)

        返回:
            result: 包含最优解和性能指标的字典
        """
        if self.verbose:
            print("\n" + "="*60)
            strategy_name = "方案D: 负载均衡导向适应度" if self.use_balance_oriented_fitness else "标准BCBO-DE"
            print(f"开始BCBO-DE融合优化 ({strategy_name})".center(60))
            print("="*60)

        # 初始化种群
        population = self.bcbo.initialize_population()

        # 方案C+D: 设置DE算子的execution_time (用于负载均衡计算)
        self.de_operators.execution_time = self.bcbo.execution_time

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

            # 5. 计算综合融合强度（整合三维自适应）
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

            # 7. 执行对应的更新策略
            if apply_de:
                population = self._bcbo_de_fusion_update_v2(
                    population, current_phase, iteration
                )
                update_type = f"BCBO-DE融合({fusion_intensity*100:.0f}%)"
            else:
                population = self._bcbo_pure_update(
                    population, current_phase, iteration
                )
                update_type = "纯BCBO"

            # 8. 更新全局最优
            for individual in population:
                fitness = self.bcbo.comprehensive_fitness(individual)
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = copy.deepcopy(individual)

            # 9. 记录历史
            bcbo_ratio = self._get_current_bcbo_ratio(population, iteration, current_phase)
            self.monitor.record(
                iteration=iteration,
                population=population,
                best_fitness=self.global_best_fitness,
                bcbo_ratio=bcbo_ratio,
                phase=current_phase,
                fitness_func=self.bcbo.comprehensive_fitness,
                best_solution=copy.deepcopy(self.global_best_solution)
            )

            # 10. 打印信息（增强版）
            if self.verbose and (iteration % self.print_interval == 0 or iteration == self.iterations - 1):
                diversity = self.diversity_calculator.hamming_distance_diversity(population)
                conv_status = self.convergence_monitor.get_status()
                print(f"Iter {iteration:3d} | 阶段: {current_phase:15s} | "
                      f"更新: {update_type:20s} | 最优: {self.global_best_fitness:.6f} | "
                      f"多样性: {diversity:.4f} | 停滞: {conv_status['stagnation_count']:2d} | "
                      f"强度: {fusion_intensity:.3f}")

        # 7. 生成结果
        result = {
            'best_solution': self.global_best_solution,
            'best_fitness': self.global_best_fitness,
            'history': self.monitor.history,
            'summary': self.monitor.get_summary(),
            'diagnosis': self.monitor.diagnose()
        }

        if self.verbose:
            print("\n" + "="*60)
            print("优化完成".center(60))
            print("="*60)
            print(f"最优适应度: {self.global_best_fitness:.6f}")
            print(f"优化诊断: {result['diagnosis']}")

        return result

    def _is_fusion_phase(self, phase: str) -> bool:
        """判断是否为融合阶段"""
        return phase in FUSION_PHASES

    def _bcbo_de_fusion_update(self, population: List, phase: str, iteration: int) -> List:
        """
        BCBO-DE融合更新 (串行精英增强拓扑 - Series Elite-Enhancement Topology)

        策略变更 (2025-11-19):
        原策略: 种群切分 (70% BCBO + 30% DE)，DE用于差解。
        新策略: 串行增强。
            1. 全员执行标准 BCBO 更新 (保持协作机制)。
            2. 选取 Top k% (e.g. 20%) 精英。
            3. 对精英应用 DE 算子进行局部开发 (Exploitation)。

        修复 (2025-11-24):
        问题: 在融合阶段(encircle/attack)没有执行BCBO更新,导致性能下降
        修复: 所有阶段都先执行BCBO更新,确保基础优化逻辑

        注意: 此方法保留作为备用，新版本使用 _bcbo_de_fusion_update_v2
        """
        # 步骤1: 全员执行 BCBO 更新
        # 修复: 无论什么阶段,都先执行BCBO更新,确保基础优化
        bcbo_updated_pop = self._bcbo_pure_update(population, phase, iteration)

        # 步骤2: 识别精英 (Top 20%)
        # 按适应度排序
        sorted_pop = sorted(
            bcbo_updated_pop,
            key=lambda x: self.bcbo.comprehensive_fitness(x),
            reverse=True
        )

        elite_ratio = self.fusion_config.get('elite_ratio', 0.2) # 默认 20% 精英
        elite_count = max(1, int(len(population) * elite_ratio))
        elites = sorted_pop[:elite_count]
        others = sorted_pop[elite_count:]

        # 步骤3: DE 精英增强
        # 仅对精英应用 DE，试图找到更好的解
        enhanced_elites = []

        # 获取自适应参数
        current_F = self.F_controller.get_F(iteration, self.iterations)
        # 精英开发阶段，CR 可以稍大，保留更多优良基因，或者根据多样性调整
        current_diversity = self._calculate_diversity(population)
        current_CR = self.CR_controller.get_CR(current_diversity)

        for i, target in enumerate(elites):
            # 变异: 使用 DE/best/1 或 DE/current-to-best/1
            # 这里我们用 DE/best/1，因为 target 本身就是精英之一，best 也是精英
            # 为了避免陷入局部最优，可以引入 random 扰动

            # 从整个种群中选择基向量，增加差异性
            mutant = self.de_operators.mutate(
                population, target, current_F # 注意这里用 population 作为变异池
            )

            # 交叉
            trial = self.de_operators.crossover(
                target, mutant, current_CR
            )

            # 选择 (Greedy Selection)
            # 只有当 trial 更好时才替换
            selected = self.de_operators.select(
                target, trial,
                fitness_func=self.bcbo.comprehensive_fitness
            )
            enhanced_elites.append(selected)

        # 步骤4: 合并
        # 保持原来的顺序可能不重要，但为了逻辑清晰，我们重新组合
        final_population = enhanced_elites + others

        return final_population

    def _bcbo_de_fusion_update_v2(self, population: List, phase: str, iteration: int) -> List:
        """
        BCBO-DE融合更新 v2.2（方案D: 负载均衡导向适应度）

        改进策略 (2025-11-28 方案D):
        1. 使用专用适应度函数comprehensive_fitness_de (M>=1000时)
        2. 大幅提升负载均衡权重: beta=100 (vs 原5-20)
        3. 降低makespan权重: gamma=50 (vs 原10000)
        4. 保留方案C的交叉约束作为辅助

        原有策略 (保留):
        1. 分级精英保护：
           - Top 50%精英：严格保护（需提升2%才接受DE结果）
           - 其余50%精英：正常DE增强
        2. Top精英使用更保守的DE参数（F×0.7, CR×0.8）
        3. 规模自适应精英比例

        参数:
            population: 当前种群
            phase: 当前阶段
            iteration: 当前迭代

        返回:
            final_population: 更新后的种群
        """
        # 步骤1: 全员执行 BCBO 更新
        bcbo_updated_pop = self._bcbo_pure_update(population, phase, iteration)

        # 步骤2: 识别精英（使用规模自适应比例）
        elite_ratio = self.fusion_config.get('elite_ratio', 0.2)

        # 方案D: 使用DE专用适应度排序
        sorted_pop = sorted(
            bcbo_updated_pop,
            key=lambda x: self.comprehensive_fitness_de(x),  # 方案D
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

        # 步骤4: 对mid_elites正常应用DE (方案D: 使用DE专用适应度)
        enhanced_mid_elites = []
        for target in mid_elites:
            mutant = self.de_operators.mutate(bcbo_updated_pop, target, current_F)

            # 方案C: crossover内部检查负载均衡 (优化版v2: 提升阈值0.75→0.85)
            trial = self.de_operators.crossover(
                target, mutant, current_CR,
                enable_balance_check=True,  # 启用负载均衡检查
                balance_threshold=0.85       # 提升阈值 (0.75→0.85)
            )

            # 方案D: 使用DE专用适应度选择
            selected = self.de_operators.select(
                target, trial,
                fitness_func=self.comprehensive_fitness_de  # 方案D
            )
            enhanced_mid_elites.append(selected)

        # 步骤5: 对top_elites严格保护 (方案D: 同样使用DE专用适应度)
        protection_params = EliteProtectionConfig.get_protection_params('top', M=self.M)
        protected_F = current_F * protection_params['f_decay']
        protected_CR = current_CR * protection_params['cr_decay']
        threshold = protection_params['threshold']

        protected_top_elites = []
        for target in top_elites:
            target_fitness = self.comprehensive_fitness_de(target)  # 方案D

            mutant = self.de_operators.mutate(bcbo_updated_pop, target, protected_F)

            # 方案C: crossover内部检查负载均衡 (优化版v2: 提升阈值0.80→0.90)
            trial = self.de_operators.crossover(
                target, mutant, protected_CR,
                enable_balance_check=True,  # 启用负载均衡检查
                balance_threshold=0.90      # 提升阈值 (0.80→0.90)
            )

            trial_fitness = self.comprehensive_fitness_de(trial)  # 方案D

            # 严格选择：必须提升threshold以上才接受
            if trial_fitness > target_fitness * (1 + threshold):
                protected_top_elites.append(trial)
            else:
                protected_top_elites.append(target)  # 保持原解

        # 步骤6: 合并
        final_population = protected_top_elites + enhanced_mid_elites + others

        return final_population

    def _bcbo_pure_update(self, population: List, phase: str, iteration: int) -> List:
        """纯BCBO更新逻辑 (封装以便复用)"""
        # 这里需要根据具体的子阶段来调用
        # 由于 phase 参数可能只是 'dynamic_search' 或 'static_search'
        # 我们需要推断具体的 BCBO 行为，或者直接调用 BCBO 的高级接口
        # 查看 BCBO 代码，它通常在内部根据迭代次数决定 encircle 或 attack
        
        # 为了稳健，我们直接调用 BCBO 的主流程方法，如果存在的话
        # 如果 BCBO 类没有统一的 step 方法，我们需要手动分发
        
        # 假设 phase 参数实际上是来自 determine_current_phase 的返回值
        # 它可能是 'encircle_dynamic', 'attack_dynamic' 等具体阶段
        # 如果只是 'dynamic_search'，我们需要进一步细分
        
        # 修正: 这里的 phase 参数是由 determine_current_phase 返回的
        # 在 config/fusion_config.py 中，FUSION_PHASES 可能包含 'dynamic_search'
        # 但 BCBO 内部逻辑是:
        # Iter < 0.5 * Max -> Dynamic
        #   Iter < 0.25 * Max -> Encircle Dynamic
        #   Iter > 0.25 * Max -> Attack Dynamic
        
        # 我们让 BCBO 实例自己决定具体动作
        if phase == 'dynamic_search' or 'dynamic' in phase:
             return self.bcbo.dynamic_search_phase(population, iteration, self.iterations)
        elif phase == 'static_search' or 'static' in phase:
             return self.bcbo.static_search_phase(population, iteration)
        else:
             # 默认 fallback
             return population

    def _calculate_diversity(self, population: List) -> float:
        """计算种群多样性"""
        return self.diversity_calculator.hamming_distance_diversity(population)

    def _get_current_bcbo_ratio(self, population: List, iteration: int, phase: str) -> float:
        """
        获取当前BCBO组比例(用于记录)
        在新策略下，所有人都跑 BCBO，所以是 1.0
        或者我们可以记录 'Elite Ratio'
        """
        return 1.0

    def _repair_load_balance(self, solution: List[int], threshold: float = 0.85) -> List[int]:
        """
        修复负载不均衡的解 (已弃用 - 方案C不再使用)

        方案C改用DE交叉约束而非事后修复
        此方法保留以供将来参考或对比实验

        针对大规模场景的负载均衡问题,通过迭代调整任务分配来改善负载均衡度

        参数:
            solution: 原始任务分配方案
            threshold: 负载均衡阈值,低于此值触发修复

        返回:
            repaired_solution: 修复后的解
        """
        # 方案C: 不再调用此方法
        return solution

    def _calculate_workloads(self, solution: List[int]) -> List[float]:
        """计算每台VM的工作负载"""
        workloads = [0.0] * self.N
        for task_id, vm_id in enumerate(solution):
            workloads[vm_id] += self.bcbo.execution_time[task_id][vm_id]
        return workloads

    def _calculate_load_balance(self, solution: List[int]) -> float:
        """
        计算负载均衡度

        返回:
            load_balance: 0-1之间,越接近1越均衡
        """
        workloads = self._calculate_workloads(solution)
        max_load = max(workloads)
        min_load = min(workloads)

        if max_load == 0:
            return 1.0

        # 负载均衡度 = 1 - (最大负载-最小负载)/最大负载
        balance = 1.0 - (max_load - min_load) / max_load
        return balance

    def comprehensive_fitness_de(self, solution: List[int]) -> float:
        """
        BCBO-DE专用适应度函数 (方案D)

        仅在M>=1000时使用负载均衡导向适应度，
        小规模场景仍使用BCBO原适应度函数

        参数:
            solution: 任务分配方案

        返回:
            fitness: 适应度值
        """
        # 小规模场景使用BCBO原适应度
        if not self.use_balance_oriented_fitness:
            return self.bcbo.comprehensive_fitness(solution)

        # 大规模场景使用负载均衡导向适应度
        # 计算各项指标
        try:
            # 计算makespan
            workloads = self._calculate_workloads(solution)
            makespan = max(workloads) if workloads else 0

            # 计算总成本
            total_cost = 0.0
            for task_id, vm_id in enumerate(solution):
                if task_id < len(self.bcbo.execution_time) and vm_id < self.N:
                    exec_time = self.bcbo.execution_time[task_id][vm_id]
                    cost = self.bcbo.vm_cost[vm_id] * exec_time
                    total_cost += cost

            # 计算负载均衡度
            balance = self._calculate_load_balance(solution)

            # 方案D适应度函数
            alpha = self.fitness_weights['alpha']
            beta = self.fitness_weights['beta']
            gamma = self.fitness_weights['gamma']

            fitness = gamma / (makespan + 1.0) - alpha * total_cost + beta * balance

            return fitness

        except Exception as e:
            # 异常情况返回极低适应度
            return float('-inf')



# 测试代码
if __name__ == '__main__':
    print("BCBO-DE嵌入式融合算法测试")
    print("=" * 60)

    # 创建优化器(小规模测试)
    optimizer = BCBO_DE_Embedded(
        M=10,          # 10个任务
        N=5,           # 5个VM
        n=10,          # 种群大小10
        iterations=20, # 20次迭代
        verbose=True,
        print_interval=5
    )

    # 运行优化
    result = optimizer.run_fusion_optimization()

    # 显示结果
    print("\n" + "=" * 60)
    print("测试完成".center(60))
    print("=" * 60)
    print(f"最优适应度: {result['best_fitness']:.6f}")
    print(f"最优解: {result['best_solution']}")
    print(f"性能摘要: {result['summary']}")
    print(f"优化诊断: {result['diagnosis']}")
