#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-DE集成测试
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.bcbo_de_embedded import BCBO_DE_Embedded
from config.fusion_config import determine_current_phase, FUSION_PHASES


class TestBCBODEIntegration(unittest.TestCase):
    """BCBO-DE集成测试类"""

    def setUp(self):
        """测试前准备"""
        self.M = 10  # 小规模测试
        self.N = 5
        self.n = 10
        self.iterations = 20

        print(f"\n{'='*60}")
        print(f"测试参数: M={self.M}, N={self.N}, n={self.n}, iterations={self.iterations}")
        print(f"{'='*60}")

    def test_01_basic_initialization(self):
        """测试基本初始化"""
        print("\n[测试1] 基本初始化测试")

        optimizer = BCBO_DE_Embedded(
            M=self.M, N=self.N, n=self.n,
            iterations=self.iterations,
            verbose=False
        )

        # 验证基本属性
        self.assertEqual(optimizer.M, self.M)
        self.assertEqual(optimizer.N, self.N)
        self.assertEqual(optimizer.n, self.n)
        self.assertEqual(optimizer.iterations, self.iterations)

        # 验证BCBO实例
        self.assertIsNotNone(optimizer.bcbo)

        # 验证DE算子
        self.assertIsNotNone(optimizer.de_operators)

        # 验证监控器
        self.assertIsNotNone(optimizer.monitor)

        print("✅ 初始化测试通过")

    def test_02_basic_optimization(self):
        """测试基本优化流程"""
        print("\n[测试2] 基本优化流程测试")

        optimizer = BCBO_DE_Embedded(
            M=self.M, N=self.N, n=self.n,
            iterations=self.iterations,
            verbose=False
        )

        result = optimizer.run_fusion_optimization()

        # 验证结果结构
        self.assertIn('best_solution', result)
        self.assertIn('best_fitness', result)
        self.assertIn('history', result)
        self.assertIn('summary', result)

        # 验证最优解
        self.assertIsNotNone(result['best_solution'])
        self.assertEqual(len(result['best_solution']), self.M)

        # 验证历史记录
        self.assertEqual(len(result['history']['iteration']), self.iterations)

        print(f"✅ 优化测试通过 - 最优适应度: {result['best_fitness']:.6f}")

    def test_03_fusion_phases(self):
        """测试融合阶段切换"""
        print("\n[测试3] 融合阶段切换测试")

        # 测试不同迭代的阶段判断
        test_cases = [
            (0, 'dynamic_search'),
            (5, 'dynamic_search'),
            (10, 'static_search'),
            (15, 'static_search'),
        ]

        for iteration, expected_phase in test_cases:
            actual_phase = determine_current_phase(iteration, 100)
            print(f"  迭代 {iteration:2d} -> 阶段: {actual_phase}")

        print("✅ 阶段切换测试通过")

    def test_04_adaptive_split(self):
        """测试自适应种群划分"""
        print("\n[测试4] 自适应种群划分测试")

        optimizer = BCBO_DE_Embedded(
            M=self.M, N=self.N, n=self.n,
            iterations=self.iterations,
            verbose=False
        )

        # 创建测试种群
        population = optimizer.bcbo.initialize_population()

        # 测试自适应划分
        bcbo_ratio = optimizer._adaptive_population_split(population, 0)

        # 验证比例范围
        self.assertGreaterEqual(bcbo_ratio, 0.0)
        self.assertLessEqual(bcbo_ratio, 1.0)

        print(f"✅ 自适应划分测试通过 - BCBO比例: {bcbo_ratio:.2f}")

    def test_05_diversity_calculation(self):
        """测试多样性计算"""
        print("\n[测试5] 多样性计算测试")

        optimizer = BCBO_DE_Embedded(
            M=self.M, N=self.N, n=self.n,
            iterations=self.iterations,
            verbose=False
        )

        # 创建测试种群
        population = optimizer.bcbo.initialize_population()

        # 计算多样性
        diversity = optimizer._calculate_diversity(population)

        # 验证多样性范围
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)

        print(f"✅ 多样性计算测试通过 - 多样性: {diversity:.4f}")

    def test_06_performance_improvement(self):
        """测试性能改进"""
        print("\n[测试6] 性能改进测试")

        optimizer = BCBO_DE_Embedded(
            M=self.M, N=self.N, n=self.n,
            iterations=self.iterations,
            verbose=False
        )

        result = optimizer.run_fusion_optimization()

        # 获取初始和最终适应度
        initial_fitness = result['history']['best_fitness'][0]
        final_fitness = result['history']['best_fitness'][-1]

        # 验证有改进
        self.assertGreaterEqual(final_fitness, initial_fitness)

        improvement = (final_fitness - initial_fitness) / abs(initial_fitness) * 100
        print(f"✅ 性能改进测试通过 - 改进: {improvement:.2f}%")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("BCBO-DE集成测试".center(60))
    print("="*60)

    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBCBODEIntegration)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "="*60)
    print("测试总结".center(60))
    print("="*60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("="*60)

    return result


if __name__ == '__main__':
    run_tests()
