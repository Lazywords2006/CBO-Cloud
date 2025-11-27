#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE参数、融合参数、实验参数配置

本模块定义了差分进化(DE)算法参数、BCBO-DE融合策略参数、以及实验配置参数。
"""

# ============================================================================
# DE算法参数配置(优化版本)
# ============================================================================
DE_CONFIG = {
    # 基本参数
    'F': 0.5,                      # 缩放因子 (基础值,被自适应覆盖)
    'CR': 0.8,                     # 交叉概率 (基础值,被自适应覆盖)
    'alpha': 0.1,                  # 自适应系数

    # ========== 精细调优: 进一步优化F参数范围 ==========
    # 数据分析发现: M>=600时负载均衡开始劣化
    # 调优策略: 降低F_max减少跳跃，提高F_min保持局部搜索
    # 目标: 在保持成本优势的同时改善负载均衡
    'use_adaptive_F': True,        # 使用自适应F
    'F_max': 0.32,                 # 最大缩放因子 (从0.4降至0.32, -20%)
    'F_min': 0.18,                 # 最小缩放因子 (从0.15升至0.18, +20%)

    # ========== 精细调优: 优化CR参数范围 ==========
    # 数据分析: 高CR导致大规模时过度交叉破坏好解
    # 调优策略: 降低CR_max，保持CR_min以维持多样性
    # 目标: 减少交叉强度，保护优秀个体的负载均衡特征
    'CR_min': 0.28,                # 最小交叉概率 (从0.3降至0.28, -6.7%)
    'CR_max': 0.62,                # 最大交叉概率 (从0.7降至0.62, -11.4%)

    # 算子策略
    'mutation_strategy': 'DE/rand/1',      # 变异策略
    'crossover_strategy': 'binomial',      # 交叉策略
}

# ============================================================================
# 融合策略参数配置(优化版本)
# ============================================================================
FUSION_CONFIG = {
    # ========== 精细调优: 提高BCBO比例增强负载均衡 ==========
    # 数据分析: BCBO负载均衡表现(0.9+)显著优于BCBO-DE(0.75-0.77大规模时)
    # 调优策略: 提高BCBO比例至90%，保留10% DE用于成本优化
    # 理论依据: BCBO擅长全局均衡，DE擅长局部成本优化
    'bcbo_ratio': 0.90,            # BCBO组占90%, DE组占10% (从85%提升)

    # ========== 核心改进: 关闭自适应划分 ==========
    # 原配置: use_adaptive_split=True (根据多样性动态调整)
    # 新配置: use_adaptive_split=False (固定比例,简化策略)
    # 理论: 固定比例更稳定,便于分析和复现
    'use_adaptive_split': False,   # 禁用自适应划分(改为固定比例)

    # 多样性阈值(保留但不使用)
    'diversity_threshold_low': 0.3,
    'diversity_threshold_high': 0.7,

    # 自适应比例调整(保留但不使用)
    'bcbo_ratio_low_diversity': 0.5,
    'bcbo_ratio_high_diversity': 0.8,

    # 信息交换(当前版本暂不启用)
    'enable_info_exchange': False,
    'exchange_interval': 5,
    'exchange_ratio': 0.1,
}

# ============================================================================
# 实验配置参数
# ============================================================================
EXPERIMENT_CONFIG = {
    # 问题规模
    'M': 100,                  # 任务数量
    'N': 20,                   # 虚拟机数量
    'n': 50,                   # 种群大小

    # 优化参数
    'iterations': 100,         # 迭代次数

    # 随机性控制
    'random_seed': 42,         # 随机种子(便于结果复现)

    # 实验重复次数
    'n_runs': 30,              # 独立运行次数(用于统计分析)

    # 输出控制
    'verbose': True,           # 是否输出详细信息
    'print_interval': 10,      # 打印间隔(每隔多少代打印一次)

    # 结果保存
    'save_results': True,      # 是否保存结果
    'save_interval': 20,       # 保存间隔(每隔多少代保存一次中间结果)
}

# ============================================================================
# 参数验证函数
# ============================================================================
def validate_parameters():
    """
    验证所有参数的合理性

    检查:
        - DE参数范围(F: 0-2, CR: 0-1)
        - 融合参数范围(比例: 0-1)
        - 实验参数有效性(M>0, N>0, n>0, iterations>0)

    抛出:
        AssertionError: 当参数不符合要求时
    """
    print("开始验证参数配置...")

    # ========== 验证DE参数 ==========
    assert 0 <= DE_CONFIG['F'] <= 2, f"F必须在0-2之间,当前值: {DE_CONFIG['F']}"
    assert 0 <= DE_CONFIG['CR'] <= 1, f"CR必须在0-1之间,当前值: {DE_CONFIG['CR']}"

    if DE_CONFIG['use_adaptive_F']:
        assert 0 <= DE_CONFIG['F_min'] <= DE_CONFIG['F_max'] <= 2, \
            f"F_min和F_max必须满足: 0 <= F_min <= F_max <= 2"

    # ========== 验证融合参数 ==========
    assert 0 <= FUSION_CONFIG['bcbo_ratio'] <= 1, \
        f"BCBO比例必须在0-1之间,当前值: {FUSION_CONFIG['bcbo_ratio']}"

    assert 0 <= FUSION_CONFIG['diversity_threshold_low'] <= 1, \
        f"低多样性阈值必须在0-1之间,当前值: {FUSION_CONFIG['diversity_threshold_low']}"

    assert 0 <= FUSION_CONFIG['diversity_threshold_high'] <= 1, \
        f"高多样性阈值必须在0-1之间,当前值: {FUSION_CONFIG['diversity_threshold_high']}"

    assert FUSION_CONFIG['diversity_threshold_low'] < FUSION_CONFIG['diversity_threshold_high'], \
        "低多样性阈值必须小于高多样性阈值"

    # ========== 验证实验参数 ==========
    assert EXPERIMENT_CONFIG['M'] > 0, f"任务数必须大于0,当前值: {EXPERIMENT_CONFIG['M']}"
    assert EXPERIMENT_CONFIG['N'] > 0, f"VM数必须大于0,当前值: {EXPERIMENT_CONFIG['N']}"
    assert EXPERIMENT_CONFIG['n'] > 0, f"种群大小必须大于0,当前值: {EXPERIMENT_CONFIG['n']}"
    assert EXPERIMENT_CONFIG['iterations'] > 0, \
        f"迭代次数必须大于0,当前值: {EXPERIMENT_CONFIG['iterations']}"

    print("OK - All parameters validated!")
