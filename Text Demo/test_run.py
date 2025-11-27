#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 诊断运行问题
"""

import sys
import os

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

print("="*60)
print("环境诊断")
print("="*60)
print(f"Python: {sys.version}")
print(f"工作目录: {os.getcwd()}")
print(f"BASE_DIR: {BASE_DIR}")
print(f"SCRIPTS_DIR: {SCRIPTS_DIR}")
print()

# 1. 检查依赖
print("="*60)
print("检查依赖模块")
print("="*60)
required_modules = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'json': 'JSON',
}

for module, name in required_modules.items():
    try:
        __import__(module)
        print(f"[OK] {name}")
    except ImportError as e:
        print(f"[FAIL] {name}: {e}")
print()

# 2. 检查自定义模块
print("="*60)
print("检查自定义模块")
print("="*60)
try:
    from real_algorithm_integration import RealAlgorithmIntegrator
    print("[OK] RealAlgorithmIntegrator")

    # 初始化
    integrator = RealAlgorithmIntegrator()
    print(f"[OK] Integrator initialized")
    print(f"[INFO] Available algorithms: {integrator.available_algorithms}")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 3. 测试快速运行
print("="*60)
print("测试快速运行（单次BCBO，5次迭代）")
print("="*60)
try:
    # 简单参数
    test_params = {
        'M': 10,
        'N': 5,
        'n': 10,
        'iterations': 5,
        'random_seed': 42
    }

    result = integrator.run_algorithm(
        algorithm_name='BCBO',
        params=test_params
    )

    print(f"[OK] BCBO test run successful")
    print(f"     Final fitness: {result['best_fitness']:.4f}")
    print(f"     Convergence history length: {len(result.get('convergence_history', []))}")

except Exception as e:
    print(f"[FAIL] Test run failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("="*60)
print("所有检查通过！脚本应该可以正常运行")
print("="*60)
print()
print("如果 generate_data_for_charts_optimized.py 仍然无法运行，")
print("请提供完整的错误信息。")
