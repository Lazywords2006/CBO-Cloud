#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-DE融合算法 - 快速验证脚本
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试所有关键模块的导入"""
    print("="*60)
    print("BCBO-DE Integration Test - Module Import Verification")
    print("="*60)

    try:
        # 测试配置模块
        print("\n[1/6] Testing config module...")
        from config.fusion_config import FUSION_PHASES, determine_current_phase
        from config.parameters import DE_CONFIG, FUSION_CONFIG, validate_parameters
        print("  OK - Config module imported")

        # 测试DE算子
        print("\n[2/6] Testing DE operators module...")
        from de_operators import DEOperators
        print("  OK - DE operators module imported")

        # 测试工具模块
        print("\n[3/6] Testing utils module...")
        from utils.diversity_calculator import DiversityCalculator
        from utils.adaptive_controller import AdaptiveFController, AdaptiveCRController
        from utils.performance_monitor import PerformanceMonitor
        print("  OK - Utils module imported")

        # 测试核心算法
        print("\n[4/6] Testing core algorithm module...")
        from core.bcbo_de_embedded import BCBO_DE_Embedded
        print("  OK - Core algorithm module imported")

        # 验证参数
        print("\n[5/6] Validating parameters...")
        validate_parameters()
        print("  OK - Parameters validated")

        # 快速实例化测试
        print("\n[6/6] Testing algorithm instantiation...")
        optimizer = BCBO_DE_Embedded(
            M=10, N=5, n=10, iterations=20, verbose=False
        )
        print("  OK - Algorithm instantiated successfully")
        print(f"     - Tasks M={optimizer.M}")
        print(f"     - VMs N={optimizer.N}")
        print(f"     - Population n={optimizer.n}")
        print(f"     - Iterations={optimizer.iterations}")

        print("\n" + "="*60)
        print("SUCCESS: All modules verified!".center(60))
        print("="*60)
        print("\nProject location:", project_root)
        print("\nYou can run:")
        print("  1. python core/bcbo_de_embedded.py")
        print("  2. python experiments/scripts/run_bcbo_de.py")
        print("  3. python tests/integration_tests/test_bcbo_de_integration.py")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
