#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块
========
导出BCBO-DE融合算法的所有配置参数
"""

from .fusion_config import (
    FUSION_PHASES,
    PURE_BCBO_PHASES,
    PHASE_RATIOS,
    PHASE_SEQUENCE,
    get_phase_iterations,
    determine_current_phase
)

from .parameters import (
    DE_CONFIG,
    FUSION_CONFIG,
    EXPERIMENT_CONFIG,
    validate_parameters
)

__all__ = [
    # 融合配置
    'FUSION_PHASES',
    'PURE_BCBO_PHASES',
    'PHASE_RATIOS',
    'PHASE_SEQUENCE',
    'get_phase_iterations',
    'determine_current_phase',

    # 参数配置
    'DE_CONFIG',
    'FUSION_CONFIG',
    'EXPERIMENT_CONFIG',
    'validate_parameters'
]

__version__ = '1.0.0'
