#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DE参数、融合参数、实验参数配置

本模块定义了差分进化(DE)算法参数、BCBO-DE融合策略参数、以及实验配置参数。
"""

# ============================================================================
# DE算法参数配置(v3.3 中等调优优化版)
# 基于调优测试结果：中等方案获得最佳性能(平均改进3.18%)
# ============================================================================
DE_CONFIG = {
    # 基本参数 (v3.5 最小干预策略)
    'F': 0.3,                      # 缩放因子 (0.7→0.3, 降低57%)
    'CR': 0.5,                     # 交叉概率 (0.9→0.5, 降低44%)
    'alpha': 0.1,                  # 自适应系数

    # ========== v3.5 最小干预: 温和DE参数 ==========
    # 调优策略: 最小化DE干预，保护BCBO核心机制
    # 目标: 仅在必要时提供轻微改进
    'use_adaptive_F': True,        # 使用自适应F
    'F_max': 0.4,                  # 最大缩放因子 (0.9→0.4, -56%)
    'F_min': 0.2,                  # 最小缩放因子 (0.5→0.2, -60%)

    # ========== v3.5 最小干预: 保守CR参数 ==========
    # 调优策略: 降低交叉强度，保留原解特征
    # 目标: 最小化对优秀解的破坏
    'CR_min': 0.3,                 # 最小交叉概率 (0.7→0.3, -57%)
    'CR_max': 0.7,                 # 最大交叉概率 (0.95→0.7, -26%)

    # 算子策略
    'mutation_strategy': 'DE/rand/1',      # 变异策略
    'crossover_strategy': 'binomial',      # 交叉策略
}

# ============================================================================
# 融合策略参数配置(v3.3 中等调优优化版)
# ============================================================================
FUSION_CONFIG = {
    # ========== v3.5 最小干预: 极低精英比例 ==========
    # 调优策略: 最小化精英干预，增强探索
    # 目标: 让DE只影响极少数个体
    'bcbo_ratio': 0.95,            # BCBO组占95% (提高)
    'elite_ratio': 0.05,           # 精英比例 (0.10→0.05, -50%)

    # ========== 核心改进: 关闭自适应划分 ==========
    'use_adaptive_split': False,   # 禁用自适应划分(v3.5保守策略)

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
