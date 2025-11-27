#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO六阶段配置和融合阶段定义

本模块定义了BCBO算法的六个阶段配置、融合阶段设置以及辅助函数。
"""

# BCBO六阶段比例配置
PHASE_RATIOS = {
    'dynamic_search': 0.10,    # 动态搜索 10%
    'static_search': 0.10,     # 静态搜索 10%
    'encircle_dynamic': 0.25,  # 动态包围 25%
    'encircle_static': 0.20,   # 静态包围 20%
    'attack_dynamic': 0.20,    # 动态攻击 20%
    'attack_static': 0.15      # 静态攻击 15%
}

# 阶段顺序
PHASE_SEQUENCE = [
    'dynamic_search',   # 阶段1: 动态搜索
    'static_search',    # 阶段2: 静态搜索
    'encircle_dynamic', # 阶段3: 动态包围
    'encircle_static',  # 阶段4: 静态包围
    'attack_dynamic',   # 阶段5: 动态攻击
    'attack_static'     # 阶段6: 静态攻击
]

# ============================================================================
# 融合阶段配置(优化版本)
# ============================================================================
# 核心改进: 反转融合阶段
# 原策略: 在前20%弱搜索阶段融合DE(过度探索,种群分散)
# 新策略: 在后60%强搜索阶段融合DE(DE发挥局部搜索优势)
#
# 理论依据:
# 1. DE擅长局部精细搜索,应在收敛阶段发挥作用
# 2. BCBO前期探索已强,不需要DE增强
# 3. 在attack阶段融合DE,发挥其局部搜索优势

FUSION_PHASES = [
    'encircle_dynamic',  # 25% 迭代(渐进引入DE)
    'encircle_static',   # 20% 迭代(中等强度融合)
    'attack_dynamic',    # 20% 迭代(重点融合阶段)
    'attack_static'      # 15% 迭代(极致优化阶段)
]

# 纯BCBO阶段(前20%,保持BCBO强探索能力)
PURE_BCBO_PHASES = [
    'dynamic_search',
    'static_search'
]

# 渐进式融合强度(不同阶段的DE应用概率)
# 精细调优 (2025-11-25): 平衡DE融合强度，保护负载均衡性能
# 策略: 降低后期融合强度，避免过度DE导致负载均衡劣化
PHASE_FUSION_INTENSITY = {
    'dynamic_search': 0.0,    # 不融合,保持BCBO探索
    'static_search': 0.0,     # 不融合
    'encircle_dynamic': 0.45, # 45%概率应用DE (从0.6降低，减少早期干扰)
    'encircle_static': 0.65,  # 65%概率应用DE (从0.8降低，平衡融合)
    'attack_dynamic': 0.75,   # 75%概率应用DE (从0.9降低，关键调整)
    'attack_static': 0.85     # 85%概率应用DE (从1.0降低，保留BCBO能力)
}


def get_phase_iterations(total_iterations: int) -> dict:
    """
    计算每个阶段的迭代范围

    参数:
        total_iterations: 总迭代次数

    返回:
        phase_ranges: 每个阶段的迭代范围字典
            格式: {'phase_name': (start_iter, end_iter)}
    """
    phase_ranges = {}
    current_iter = 0

    for phase_name in PHASE_SEQUENCE:
        phase_ratio = PHASE_RATIOS[phase_name]
        phase_iters = int(total_iterations * phase_ratio)
        phase_ranges[phase_name] = (current_iter, current_iter + phase_iters)
        current_iter += phase_iters

    return phase_ranges


def determine_current_phase(iteration: int, total_iterations: int) -> str:
    """
    确定当前迭代属于哪个阶段

    参数:
        iteration: 当前迭代次数
        total_iterations: 总迭代次数

    返回:
        phase_name: 当前阶段名称
    """
    phase_ranges = get_phase_iterations(total_iterations)

    for phase_name, (start, end) in phase_ranges.items():
        if start <= iteration < end:
            return phase_name

    # 如果超出范围,返回最后一个阶段
    return PHASE_SEQUENCE[-1]


def is_fusion_phase(phase: str) -> bool:
    """
    判断给定阶段是否为融合阶段

    参数:
        phase: 阶段名称

    返回:
        is_fusion: 是否为融合阶段
    """
    return phase in FUSION_PHASES


def get_fusion_intensity(phase: str) -> float:
    """
    获取当前阶段的融合强度(渐进式融合策略)

    参数:
        phase: 阶段名称

    返回:
        intensity: 融合强度(0.0-1.0),表示应用DE的概率

    说明:
        - 0.0: 不融合,完全使用BCBO
        - 0.3-0.5: 低强度融合(encircle阶段)
        - 0.7-0.9: 高强度融合(attack阶段)
    """
    return PHASE_FUSION_INTENSITY.get(phase, 0.0)
