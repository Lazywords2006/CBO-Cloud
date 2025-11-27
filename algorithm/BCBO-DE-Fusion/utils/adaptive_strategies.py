#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应融合策略模块

实现四维自适应融合策略:
1. 迭代自适应融合强度（余弦衰减）
2. 规模自适应参数调整
3. 收敛状态监控与自适应
4. 增强精英保护机制
"""

import math
import numpy as np
from typing import Dict, List, Optional


# ============================================================================
# 改进1: 迭代自适应融合强度（余弦衰减策略）
# ============================================================================

def get_adaptive_fusion_intensity(
    phase: str,
    iteration: int,
    total_iterations: int
) -> float:
    """
    计算自适应融合强度（平滑余弦衰减）

    理论依据:
    - 前期（0-40%）: 适度探索，DE可参与
    - 中期（40-70%）: 逐步收敛，DE辅助搜索
    - 后期（70-100%）: 精细优化，DE最小干扰

    公式: intensity = base_intensity * cosine_decay(progress)
    其中: cosine_decay(t) = 0.5 * (1 + cos(π * t))

    参数:
        phase: 当前阶段名称
        iteration: 当前迭代次数
        total_iterations: 总迭代次数

    返回:
        final_intensity: 最终融合强度 (0.05-0.35)
    """
    # 基础强度（保守设置，避免过度破坏BCBO优秀解）
    # v3.1 调整：适当提高基础强度，避免过度衰减
    base_intensity = {
        'dynamic_search': 0.0,      # 不融合
        'static_search': 0.0,       # 不融合
        'encircle_dynamic': 0.35,   # 35%基础强度（从0.20提高）
        'encircle_static': 0.45,    # 45%基础强度（从0.28提高）
        'attack_dynamic': 0.40,     # 40%基础强度（从0.25提高）
        'attack_static': 0.38       # 38%基础强度（从0.22提高）
    }

    base = base_intensity.get(phase, 0.0)

    if base == 0.0:
        return 0.0  # 不融合阶段

    # 计算进度比例 (0.0 到 1.0)
    progress = iteration / total_iterations

    # 余弦退火衰减
    # progress=0时: decay=1.0（保持基础强度）
    # progress=0.5时: decay=0.5（减半）
    # progress=1.0时: decay=0.0（几乎不用DE）
    decay_factor = 0.5 * (1 + math.cos(math.pi * progress))

    # 最终强度 = 基础强度 × 衰减系数
    # 保持最小值0.05，避免完全不用DE（保持一定探索能力）
    final_intensity = max(0.05, base * decay_factor)

    return final_intensity


# ============================================================================
# 改进2: 规模自适应参数调整
# ============================================================================

def get_scale_adaptive_params(M: int, N: int) -> Dict[str, float]:
    """
    根据问题规模自适应调整DE参数和融合策略

    理论依据（v3.2修正版）:
    - 超小规模（M<150）: 搜索空间极小，关闭DE避免过度扰动
    - 小规模（150≤M<200）: 搜索空间小，适度使用DE
    - 中规模（200≤M<600）: 平衡探索和开发，适度使用DE（最佳区间）
    - 大规模（600≤M<1000）: 搜索空间巨大，增强DE局部搜索
    - 超大规模（M≥1000）: 极大搜索空间，大幅增强DE精细搜索

    核心发现：规模越大，越需要DE局部搜索！

    v3.2关键修正：
    - 小规模（M<150）: intensity_scale=0.0（完全关闭DE）
    - 大规模（M≥1000）: intensity_scale=0.95（大幅增强DE）

    参数:
        M: 任务数量
        N: 虚拟机数量

    返回:
        params: 包含调整参数的字典
            - F_scale: 缩放因子倍率
            - CR_scale: 交叉概率倍率
            - elite_ratio: 精英比例
            - intensity_scale: 融合强度倍率
    """
    # 计算任务密度（可选，用于更精细的调整）
    task_density = M / N

    if M < 150:
        # 超小规模：完全关闭DE融合
        # 理由：搜索空间极小，BCBO探索已足够，DE反而增加扰动
        return {
            'F_scale': 1.0,         # 保持原始F
            'CR_scale': 1.0,        # 保持原始CR
            'elite_ratio': 0.20,    # 保持20%精英（虽然不会用到）
            'intensity_scale': 0.0  # 完全关闭DE融合
        }
    elif M < 200:
        # 小规模：适度使用DE（v3.2边界）
        return {
            'F_scale': 1.0,         # 保持原始F
            'CR_scale': 1.0,        # 保持原始CR
            'elite_ratio': 0.20,    # 20%精英
            'intensity_scale': 0.3  # 30%强度
        }
    elif M < 600:
        # 中规模：适度融合（v3.1/v3.2最佳表现区间）
        return {
            'F_scale': 0.90,        # F降低10%
            'CR_scale': 0.85,       # CR降低15%
            'elite_ratio': 0.15,    # 15%精英
            'intensity_scale': 0.75 # 75%强度
        }
    elif M < 1000:
        # 大规模：增强DE使用（v3.2修正，保持）
        return {
            'F_scale': 0.85,        # F适度降低
            'CR_scale': 0.80,       # CR适度降低
            'elite_ratio': 0.12,    # 12%精英
            'intensity_scale': 0.85 # 85%强度
        }
    else:
        # 超大规模：大幅增强DE（v3.2修正，保持）
        return {
            'F_scale': 0.80,        # F降低20%
            'CR_scale': 0.75,       # CR降低25%
            'elite_ratio': 0.15,    # 15%精英
            'intensity_scale': 0.95 # 95%强度
        }


# ============================================================================
# 改进3: 收敛状态监控与自适应
# ============================================================================

class ConvergenceMonitor:
    """
    收敛监控器 - 监控算法收敛状态并提供自适应建议

    功能:
    1. 跟踪最优适应度历史
    2. 检测停滞状态（连续N代无改进）
    3. 计算改进率
    4. 提供自适应调整建议
    """

    def __init__(self, patience: int = 10):
        """
        初始化收敛监控器

        参数:
            patience: 容忍停滞次数（连续多少代无改进视为停滞）
        """
        self.patience = patience
        self.best_fitness_history = []
        self.stagnation_count = 0
        self.improvement_rate = 0.0
        self.last_best_fitness = float('-inf')

    def update(self, current_fitness: float):
        """
        更新监控状态

        参数:
            current_fitness: 当前最优适应度
        """
        self.best_fitness_history.append(current_fitness)

        # 检查是否有改进
        if current_fitness > self.last_best_fitness:
            # 有改进：计算改进率
            if self.last_best_fitness != float('-inf') and self.last_best_fitness != 0:
                improvement = (current_fitness - self.last_best_fitness) / abs(self.last_best_fitness)
                self.improvement_rate = improvement
            else:
                self.improvement_rate = 0.0

            self.stagnation_count = 0
            self.last_best_fitness = current_fitness
        else:
            # 无改进：增加停滞计数
            self.stagnation_count += 1

    def get_adaptive_adjustment(self) -> Dict[str, any]:
        """
        根据收敛状态返回调整建议

        返回:
            adjustment: 包含调整建议的字典
                - action: 'increase_exploration' | 'mild_exploration' | 'maintain' | 'decrease_exploration'
                - intensity_adjust: 调整倍率 (-0.3 到 +0.3)
                - description: 调整原因描述
        """
        if self.stagnation_count >= self.patience:
            # 严重停滞：增加探索（提升DE使用）
            return {
                'action': 'increase_exploration',
                'intensity_adjust': +0.30,  # 增加30%融合强度
                'description': f'严重停滞({self.stagnation_count}代)，增加DE探索'
            }
        elif self.stagnation_count >= self.patience // 2:
            # 轻度停滞：小幅增加探索
            return {
                'action': 'mild_exploration',
                'intensity_adjust': +0.15,  # 增加15%
                'description': f'轻度停滞({self.stagnation_count}代)，小幅增加DE'
            }
        elif self.improvement_rate > 0.01:
            # 快速改进（>1%）：保持当前策略
            return {
                'action': 'maintain',
                'intensity_adjust': 0.0,
                'description': f'快速改进({self.improvement_rate*100:.2f}%)，保持策略'
            }
        else:
            # 缓慢改进或无改进：减少DE干扰
            return {
                'action': 'decrease_exploration',
                'intensity_adjust': -0.20,  # 降低20%
                'description': '改进缓慢，减少DE干扰'
            }

    def reset(self):
        """重置监控器状态"""
        self.best_fitness_history = []
        self.stagnation_count = 0
        self.improvement_rate = 0.0
        self.last_best_fitness = float('-inf')

    def get_status(self) -> Dict[str, any]:
        """
        获取当前监控状态

        返回:
            status: 包含状态信息的字典
        """
        return {
            'stagnation_count': self.stagnation_count,
            'improvement_rate': self.improvement_rate,
            'history_length': len(self.best_fitness_history),
            'is_stagnant': self.stagnation_count >= self.patience // 2
        }


# ============================================================================
# 改进4: 增强精英保护参数
# ============================================================================

class EliteProtectionConfig:
    """精英保护配置类（v3.2增强：规模自适应阈值）"""

    # 精英分级比例
    TOP_ELITE_RATIO = 0.5       # Top 50%的精英受严格保护

    # 基础保护阈值（已弃用，改用动态阈值）
    # PROTECTION_THRESHOLD = 0.02

    # Top精英的DE参数衰减
    TOP_ELITE_F_DECAY = 0.7     # Top精英的F降低至70%
    TOP_ELITE_CR_DECAY = 0.8    # Top精英的CR降低至80%

    @staticmethod
    def get_protection_threshold(M: int) -> float:
        """
        根据问题规模获取动态保护阈值

        理论依据（v3.2新增）:
        - 小规模：严格保护（高阈值），避免破坏优秀解
        - 中规模：中等保护（中阈值），平衡探索和保护
        - 大规模：宽松保护（低阈值），允许DE充分搜索

        参数:
            M: 任务数量

        返回:
            threshold: 保护阈值（Top精英需提升threshold才接受DE结果）
        """
        if M < 150:
            # 超小规模：极严格保护（不过不会用到，因为DE已关闭）
            return 0.10  # 需提升10%
        elif M < 200:
            # 小规模：严格保护
            return 0.05  # 需提升5%
        elif M < 600:
            # 中规模：中等保护
            return 0.02  # 需提升2%
        elif M < 1000:
            # 大规模：宽松保护
            return 0.01  # 需提升1%
        else:
            # 超大规模：最宽松保护
            return 0.005 # 需提升0.5%

    @staticmethod
    def get_protection_params(elite_position: str, M: int = 500) -> Dict[str, float]:
        """
        获取不同位置精英的保护参数

        参数:
            elite_position: 'top' | 'mid'
            M: 任务数量（用于动态阈值）

        返回:
            params: 保护参数字典
        """
        if elite_position == 'top':
            return {
                'f_decay': EliteProtectionConfig.TOP_ELITE_F_DECAY,
                'cr_decay': EliteProtectionConfig.TOP_ELITE_CR_DECAY,
                'threshold': EliteProtectionConfig.get_protection_threshold(M)
            }
        else:  # 'mid'
            return {
                'f_decay': 1.0,      # 正常F
                'cr_decay': 1.0,     # 正常CR
                'threshold': 0.0     # 无阈值，正常选择
            }


# ============================================================================
# 辅助函数
# ============================================================================

def apply_scale_adaptive_params(
    base_config: Dict,
    scale_params: Dict[str, float]
) -> Dict:
    """
    应用规模自适应参数到基础配置

    参数:
        base_config: 基础DE配置字典
        scale_params: 规模自适应参数

    返回:
        adjusted_config: 调整后的配置
    """
    adjusted = base_config.copy()

    # 调整F范围
    if 'F_max' in adjusted:
        adjusted['F_max'] *= scale_params['F_scale']
    if 'F_min' in adjusted:
        adjusted['F_min'] *= scale_params['F_scale']

    # 调整CR范围
    if 'CR_max' in adjusted:
        adjusted['CR_max'] *= scale_params['CR_scale']
    if 'CR_min' in adjusted:
        adjusted['CR_min'] *= scale_params['CR_scale']

    return adjusted


def get_combined_fusion_intensity(
    phase: str,
    iteration: int,
    total_iterations: int,
    M: int,
    N: int,
    convergence_adjustment: float = 0.0
) -> float:
    """
    获取综合融合强度（整合三种自适应策略）

    参数:
        phase: 当前阶段
        iteration: 当前迭代
        total_iterations: 总迭代次数
        M: 任务数
        N: 虚拟机数
        convergence_adjustment: 收敛监控器的调整值

    返回:
        combined_intensity: 综合融合强度 (0.0-0.8)
    """
    # 1. 迭代自适应基础强度（余弦衰减）
    base_intensity = get_adaptive_fusion_intensity(phase, iteration, total_iterations)

    # 2. 规模自适应倍率
    scale_params = get_scale_adaptive_params(M, N)
    scale_factor = scale_params['intensity_scale']

    # 3. 收敛状态调整
    # convergence_adjustment 是 ConvergenceMonitor.get_adaptive_adjustment()['intensity_adjust']
    convergence_factor = 1.0 + convergence_adjustment

    # 综合计算
    combined = base_intensity * scale_factor * convergence_factor

    # 限制在合理范围内 (5%-80%)
    combined = np.clip(combined, 0.05, 0.80)

    return combined


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("自适应策略模块测试")
    print("=" * 80)

    # 测试1: 迭代自适应融合强度
    print("\n测试1: 迭代自适应融合强度（余弦衰减）")
    print("-" * 80)
    total_iters = 100
    phase = 'attack_static'

    print(f"阶段: {phase}, 总迭代: {total_iters}")
    print(f"{'迭代':>6} | {'进度%':>6} | {'融合强度':>8} | {'说明'}")
    print("-" * 60)

    for iter in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        intensity = get_adaptive_fusion_intensity(phase, iter, total_iters)
        progress = (iter / total_iters) * 100
        desc = "前期" if progress < 40 else "中期" if progress < 70 else "后期"
        print(f"{iter:6d} | {progress:5.0f}% | {intensity:8.4f} | {desc}")

    # 测试2: 规模自适应参数
    print("\n测试2: 规模自适应参数")
    print("-" * 80)
    test_scales = [100, 300, 500, 800, 1000, 1500]

    print(f"{'M':>6} | {'F缩放':>7} | {'CR缩放':>7} | {'精英%':>6} | {'强度倍率':>8} | {'策略'}")
    print("-" * 80)

    for M in test_scales:
        params = get_scale_adaptive_params(M, 20)
        strategy = "激进" if M < 200 else "保守" if M < 600 else "极保守" if M < 1000 else "最小干预"
        print(f"{M:6d} | {params['F_scale']:7.2f} | {params['CR_scale']:7.2f} | "
              f"{params['elite_ratio']*100:5.0f}% | {params['intensity_scale']:8.2f} | {strategy}")

    # 测试3: 收敛监控器
    print("\n测试3: 收敛监控器")
    print("-" * 80)

    monitor = ConvergenceMonitor(patience=10)

    # 模拟优化过程
    fitness_sequence = [
        100, 105, 110, 115, 120,  # 前5代：稳步提升
        120, 120, 120, 120, 120,  # 第6-10代：停滞
        120, 120, 120, 120, 120,  # 第11-15代：继续停滞
        125, 130, 135             # 第16-18代：突破
    ]

    print(f"{'迭代':>4} | {'适应度':>8} | {'停滞计数':>8} | {'改进率%':>8} | {'建议动作':>20} | {'强度调整'}")
    print("-" * 110)

    for i, fitness in enumerate(fitness_sequence):
        monitor.update(fitness)
        adj = monitor.get_adaptive_adjustment()
        print(f"{i:4d} | {fitness:8.2f} | {monitor.stagnation_count:8d} | "
              f"{monitor.improvement_rate*100:8.2f} | {adj['action']:>20s} | {adj['intensity_adjust']:+.2f}")

    # 测试4: 综合融合强度
    print("\n测试4: 综合融合强度（整合三维自适应）")
    print("-" * 80)

    print("场景: M=500（中规模）, attack_static阶段")
    print(f"{'迭代':>6} | {'基础强度':>8} | {'规模倍率':>8} | {'收敛调整':>8} | {'综合强度':>8}")
    print("-" * 70)

    for iter in [20, 50, 80]:
        for conv_adj in [0.0, 0.15, -0.20]:
            combined = get_combined_fusion_intensity(
                'attack_static', iter, 100, 500, 20, conv_adj
            )
            base = get_adaptive_fusion_intensity('attack_static', iter, 100)
            scale = get_scale_adaptive_params(500, 20)['intensity_scale']

            print(f"{iter:6d} | {base:8.4f} | {scale:8.2f} | {conv_adj:+8.2f} | {combined:8.4f}")

    print("\n" + "=" * 80)
    print("测试完成！")
