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

        # 4. 创建DE算子实例
        self.de_operators = DEOperators(
            M=M, N=N,
            F=self.de_config['F'],
            CR=self.de_config['CR']
        )

        # 5. 初始化融合配置
        self.fusion_config = FUSION_CONFIG.copy()
        self.fusion_config.update(kwargs.get('fusion_config', {}))

        # 6. 初始化自适应控制器(使用优化后的参数)
        self.F_controller = AdaptiveFController(
            F_max=self.de_config.get('F_max', 0.4),   # 优化: 降低自0.9
            F_min=self.de_config.get('F_min', 0.15),  # 优化: 降低自0.4
            alpha=self.de_config['alpha']
        )
        self.CR_controller = AdaptiveCRController(
            CR_min=self.de_config.get('CR_min', 0.3), # 优化: 降低自0.5
            CR_max=self.de_config.get('CR_max', 0.7)  # 优化: 降低自0.9
        )

        # 7. 初始化性能监控器
        self.monitor = PerformanceMonitor()

        # 8. 初始化多样性计算器
        self.diversity_calculator = DiversityCalculator()

        # 9. 初始化全局最优
        self.global_best_solution = None
        self.global_best_fitness = float('-inf')

        if self.verbose:
            print(f"BCBO-DE融合调度器初始化完成:")
            print(f"  任务数M={M}, VM数N={N}, 种群大小n={n}, 迭代次数={iterations}")
            print(f"  融合阶段: {FUSION_PHASES}")
            print(f"  纯BCBO阶段: {PURE_BCBO_PHASES}")

    def run_fusion_optimization(self) -> Dict:
        """
        运行融合优化(主方法)

        返回:
            result: 包含最优解和性能指标的字典
        """
        if self.verbose:
            print("\n" + "="*60)
            print("开始BCBO-DE融合优化".center(60))
            print("="*60)

        # 初始化种群
        population = self.bcbo.initialize_population()

        # 主循环
        for iteration in range(self.iterations):
            # 1. 确定当前阶段
            current_phase = determine_current_phase(iteration, self.iterations)

            # 2. 判断是否为融合阶段
            is_fusion = self._is_fusion_phase(current_phase)

            # 3. 获取融合强度(渐进式融合策略)
            fusion_intensity = get_fusion_intensity(current_phase)

            # 4. 根据融合强度决定是否应用DE
            apply_de = is_fusion and (np.random.random() < fusion_intensity)

            # 5. 执行对应的更新策略
            if apply_de:
                population = self._bcbo_de_fusion_update(
                    population, current_phase, iteration
                )
                update_type = f"BCBO-DE融合({fusion_intensity*100:.0f}%)"
            else:
                population = self._bcbo_pure_update(
                    population, current_phase, iteration
                )
                update_type = "纯BCBO"

            # 6. 更新全局最优
            for individual in population:
                fitness = self.bcbo.comprehensive_fitness(individual)
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_solution = copy.deepcopy(individual)

            # 7. 记录历史
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

            # 8. 打印信息
            if self.verbose and (iteration % self.print_interval == 0 or iteration == self.iterations - 1):
                diversity = self.diversity_calculator.hamming_distance_diversity(population)
                print(f"Iter {iteration:3d} | 阶段: {current_phase:15s} | "
                      f"更新: {update_type:20s} | 最优适应度: {self.global_best_fitness:.6f} | "
                      f"多样性: {diversity:.4f} | BCBO比例: {bcbo_ratio:.2f}")

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
