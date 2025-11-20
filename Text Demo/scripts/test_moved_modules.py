#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试移动后的模块
=====================================
验证 bcbo_visualization.py 和 real_algorithm_integration.py
在 scripts 目录下是否可以正常导入和运行
"""

import sys
import os

# 设置当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)

print("="*80)
print("测试移动后的模块")
print("="*80)
print(f"当前工作目录: {os.getcwd()}")
print(f"脚本所在目录: {CURRENT_DIR}")
print()

# 测试1: 导入 bcbo_visualization
print("[测试1] 导入 bcbo_visualization 模块...")
try:
    from bcbo_visualization import BCBOVisualizer
    print("[OK] bcbo_visualization 导入成功")

    # 尝试创建实例
    visualizer = BCBOVisualizer(save_dir="..")
    print("[OK] BCBOVisualizer 实例创建成功")
    print(f"  - RAW_data 目录: {visualizer.raw_data_dir}")
    print()
except Exception as e:
    print(f"[ERROR] bcbo_visualization 导入失败: {e}")
    import traceback
    traceback.print_exc()
    print()

# 测试2: 导入 real_algorithm_integration
print("[测试2] 导入 real_algorithm_integration 模块...")
try:
    from real_algorithm_integration import RealAlgorithmIntegrator
    print("[OK] real_algorithm_integration 导入成功")

    # 尝试创建实例
    integrator = RealAlgorithmIntegrator()
    print("[OK] RealAlgorithmIntegrator 实例创建成功")
    print(f"  - 可用算法数量: {len(integrator.available_algorithms)}")
    print(f"  - 可用算法列表: {', '.join(integrator.available_algorithms)}")
    print()
except Exception as e:
    print(f"[ERROR] real_algorithm_integration 导入失败: {e}")
    import traceback
    traceback.print_exc()
    print()

# 测试3: 检查路径设置
print("[测试3] 检查关键路径...")
parent_dir = os.path.join(CURRENT_DIR, '..')
raw_data_dir = os.path.join(parent_dir, 'RAW_data')
charts_dir = os.path.join(parent_dir, 'charts')
tables_dir = os.path.join(parent_dir, 'tables')

print(f"父目录: {os.path.abspath(parent_dir)}")
print(f"  存在: {os.path.exists(parent_dir)}")
print(f"RAW_data目录: {os.path.abspath(raw_data_dir)}")
print(f"  存在: {os.path.exists(raw_data_dir)}")
print(f"charts目录: {os.path.abspath(charts_dir)}")
print(f"  存在: {os.path.exists(charts_dir)}")
print(f"tables目录: {os.path.abspath(tables_dir)}")
print(f"  存在: {os.path.exists(tables_dir)}")
print()

# 测试4: 检查算法模块路径
print("[测试4] 检查算法模块路径...")
project_root = os.path.join(CURRENT_DIR, '..', '..')
python_dir = os.path.join(project_root, '程序', 'python')
bcbo_dir = os.path.join(python_dir, 'BCBO')
algo_dir = os.path.join(python_dir, 'algorithms')

print(f"项目根目录: {os.path.abspath(project_root)}")
print(f"  存在: {os.path.exists(project_root)}")
print(f"Python目录: {os.path.abspath(python_dir)}")
print(f"  存在: {os.path.exists(python_dir)}")
print(f"BCBO目录: {os.path.abspath(bcbo_dir)}")
print(f"  存在: {os.path.exists(bcbo_dir)}")
print(f"算法目录: {os.path.abspath(algo_dir)}")
print(f"  存在: {os.path.exists(algo_dir)}")
print()

print("="*80)
print("测试完成!")
print("="*80)
print()
print("说明:")
print("- 如果算法模块路径不存在，需要根据实际项目结构调整")
print("- 修改 real_algorithm_integration.py 中的路径配置")
print("- 确保所有必需的算法模块都在正确的位置")
print()
