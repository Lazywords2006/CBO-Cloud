#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO与MBCBO算法深度对比分析
分析两种算法在不同维度的性能差异
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
import os
import sys

# 添加算法路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
algorithm_path = os.path.join(parent_dir, 'algorithm')
sys.path.insert(0, algorithm_path)

from BCBO.bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from MBCBO.mbcbo_cloud_scheduler import MBCBO_CloudScheduler


class BCBOvsMBCBOAnalyzer:
    """BCBO与MBCBO对比分析器"""

    def __init__(self):
        self.chart_data = {}
        self.mbcbo_data = None
        self.test_results = []
        self.load_all_data()

    def load_all_data(self):
        """加载所有测试数据"""
        # 加载图表数据
        data_dir = os.path.join(current_dir, 'Text Demo', 'RAW_data')
        if not os.path.exists(data_dir):
            data_dir = os.path.join(current_dir, 'RAW_data')

        for i in range(1, 5):
            try:
                file_path = os.path.join(data_dir, f'chart_set_{i}_bcbo_comparison.json')
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.chart_data[f'chart_{i}'] = json.load(f)
                    print(f"[OK] 加载图表集{i}数据")
            except Exception as e:
                print(f"[SKIP] 图表集{i}数据不存在: {e}")

        # 加载MBCBO测试数据
        try:
            mbcbo_file = os.path.join(current_dir, 'mbcbo_test_results.json')
            with open(mbcbo_file, 'r', encoding='utf-8') as f:
                self.mbcbo_data = json.load(f)
                print("[OK] 加载MBCBO测试数据")
        except Exception as e:
            print(f"[SKIP] MBCBO测试数据不存在: {e}")

    def run_comparison_tests(self, scenarios=None):
        """运行BCBO与MBCBO对比测试"""
        print("\n" + "="*70)
        print("执行BCBO vs MBCBO对比测试".center(70))
        print("="*70)

        if scenarios is None:
            scenarios = [
                {'name': '小规模', 'M': 100, 'N': 20, 'n': 50, 'iterations': 50},
                {'name': '中规模', 'M': 200, 'N': 30, 'n': 80, 'iterations': 80},
            ]

        import time

        for scenario in scenarios:
            print(f"\n测试场景: {scenario['name']} (M={scenario['M']}, N={scenario['N']})")
            print("-" * 50)

            # 设置种子保证公平对比
            np.random.seed(42)

            # 测试BCBO
            print("  运行BCBO算法...")
            bcbo = BCBO_CloudScheduler(
                M=scenario['M'],
                N=scenario['N'],
                n=scenario['n'],
                iterations=scenario['iterations']
            )
            bcbo_start = time.time()
            bcbo_result = bcbo.run_complete_algorithm()
            bcbo_time = time.time() - bcbo_start
            bcbo_fitness = bcbo_result['best_fitness']

            # 重置种子
            np.random.seed(42)

            # 测试MBCBO
            print("  运行MBCBO算法...")
            mbcbo = MBCBO_CloudScheduler(
                M=scenario['M'],
                N=scenario['N'],
                n=scenario['n'],
                iterations=scenario['iterations'],
                verbose=False
            )
            mbcbo_start = time.time()
            mbcbo_result = mbcbo.optimize()
            mbcbo_time = time.time() - mbcbo_start
            mbcbo_fitness = mbcbo_result['best_fitness']

            # 计算改进率
            improvement = ((mbcbo_fitness - bcbo_fitness) / abs(bcbo_fitness)) * 100

            # 保存结果
            result = {
                'scenario': scenario['name'],
                'M': scenario['M'],
                'N': scenario['N'],
                'bcbo_fitness': float(bcbo_fitness),
                'mbcbo_fitness': float(mbcbo_fitness),
                'improvement': float(improvement),
                'bcbo_time': float(bcbo_time),
                'mbcbo_time': float(mbcbo_time),
                'time_ratio': float(mbcbo_time / bcbo_time),
                'mbcbo_strategies': mbcbo_result.get('strategy_contributions', {})
            }
            self.test_results.append(result)

            # 显示结果
            print(f"  BCBO适应度:  {bcbo_fitness:.4f} (耗时: {bcbo_time:.2f}s)")
            print(f"  MBCBO适应度: {mbcbo_fitness:.4f} (耗时: {mbcbo_time:.2f}s)")
            print(f"  性能改进: {improvement:+.2f}%")
            print(f"  时间效率: {mbcbo_time/bcbo_time:.2f}x")

            if improvement > 0.1:
                print(f"  结果: MBCBO更优 [+]")
            elif improvement < -0.1:
                print(f"  结果: BCBO更优 [-]")
            else:
                print(f"  结果: 性能相当 [=]")

        # 保存测试结果
        output_file = os.path.join(current_dir, 'bcbo_vs_mbcbo_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] 测试结果已保存到 bcbo_vs_mbcbo_results.json")

    def analyze_convergence(self):
        """分析收敛特性"""
        print("\n" + "="*70)
        print("收敛特性分析".center(70))
        print("="*70)

        # 使用图表集1数据分析BCBO收敛
        if 'chart_1' in self.chart_data:
            data = self.chart_data['chart_1']
            bcbo_results = data['algorithms']['BCBO']['results']

            print("\nBCBO收敛特性 (M=100, N=20):")
            print("-"*50)

            convergence_points = [5, 10, 20, 50, 100]
            print(f"{'迭代':<10} {'适应度':<15} {'提升':<10}")
            print("-"*50)

            prev_val = bcbo_results[0]['best_fitness']
            for point in convergence_points:
                curr_val = next(r['best_fitness'] for r in bcbo_results if r['iteration'] == point)
                improvement = curr_val - prev_val
                print(f"{point:<10} {curr_val:<15.4f} {improvement:+.4f}")
                prev_val = curr_val

            # 计算收敛速度
            early_improvement = bcbo_results[4]['best_fitness'] - bcbo_results[0]['best_fitness']
            late_improvement = bcbo_results[-1]['best_fitness'] - bcbo_results[-5]['best_fitness']

            print(f"\n前期收敛速度 (0-25迭代): {early_improvement:.4f}")
            print(f"后期收敛速度 (75-100迭代): {late_improvement:.4f}")
            print(f"收敛速度下降: {((late_improvement - early_improvement) / early_improvement * 100):.1f}%")

    def analyze_performance_comparison(self):
        """分析性能对比"""
        print("\n" + "="*70)
        print("BCBO vs MBCBO 性能对比分析".center(70))
        print("="*70)

        if not self.test_results:
            print("[警告] 无测试结果，请先运行对比测试")
            return

        print("\n性能对比汇总:")
        print("-"*70)
        print(f"{'场景':<15} {'BCBO':<15} {'MBCBO':<15} {'改进率':<12} {'结论':<10}")
        print("-"*70)

        wins = {'MBCBO': 0, 'BCBO': 0, 'Tie': 0}
        improvements = []

        for result in self.test_results:
            scenario = result['scenario']
            bcbo_fit = result['bcbo_fitness']
            mbcbo_fit = result['mbcbo_fitness']
            improvement = result['improvement']
            improvements.append(improvement)

            if improvement > 0.1:
                conclusion = "MBCBO胜"
                wins['MBCBO'] += 1
            elif improvement < -0.1:
                conclusion = "BCBO胜"
                wins['BCBO'] += 1
            else:
                conclusion = "平局"
                wins['Tie'] += 1

            print(f"{scenario:<15} {bcbo_fit:<15.4f} {mbcbo_fit:<15.4f} {improvement:+.2f}% {conclusion:<10}")

        # 统计分析
        print("\n统计分析:")
        print("-"*50)
        print(f"  MBCBO获胜: {wins['MBCBO']} 次")
        print(f"  BCBO获胜:  {wins['BCBO']} 次")
        print(f"  平局:      {wins['Tie']} 次")
        print(f"  平均改进率: {np.mean(improvements):+.2f}%")
        print(f"  最大改进率: {max(improvements):+.2f}%")
        print(f"  最小改进率: {min(improvements):+.2f}%")

        # 时间效率分析
        if self.test_results[0].get('bcbo_time'):
            print("\n时间效率分析:")
            print("-"*50)
            for result in self.test_results:
                time_ratio = (result['mbcbo_time'] - result['bcbo_time']) / result['bcbo_time'] * 100
                print(f"  {result['scenario']}: MBCBO用时 {time_ratio:+.1f}% {'更多' if time_ratio > 0 else '更少'}")

    def analyze_mbcbo_strategies(self):
        """分析MBCBO策略贡献"""
        print("\n" + "="*70)
        print("MBCBO策略贡献度分析".center(70))
        print("="*70)

        if not self.test_results or not self.test_results[0].get('mbcbo_strategies'):
            print("[警告] 无策略数据")
            return

        for result in self.test_results:
            print(f"\n{result['scenario']}场景策略贡献:")
            print("-"*40)

            strategies = result.get('mbcbo_strategies', {})
            if strategies:
                # 排序策略
                sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)

                for strategy, contrib in sorted_strategies:
                    bar_length = int(contrib / 5)  # 简单的条形图
                    bar = '█' * bar_length
                    print(f"  {strategy:<12}: {contrib:>8.2f} {bar}")

    def analyze_algorithm_characteristics(self):
        """分析算法特性"""
        print("\n" + "="*70)
        print("算法特性对比".center(70))
        print("="*70)

        characteristics = {
            'BCBO': {
                '复杂度': 'O(n × M × iter)',
                '参数数量': '少 (4-5个)',
                '收敛速度': '快',
                '稳定性': '高',
                '实现难度': '简单',
                '内存占用': '低',
                '并行化': '容易',
                '理论基础': '秃鹰-郊狼行为'
            },
            'MBCBO': {
                '复杂度': 'O(4 × n × M × iter)',
                '参数数量': '中等 (8-10个)',
                '收敛速度': '较快',
                '稳定性': '较高',
                '实现难度': '中等',
                '内存占用': '中等',
                '并行化': '天然并行',
                '理论基础': '多策略协同进化'
            }
        }

        print("\n算法特性对比表:")
        print("-"*60)
        print(f"{'特性':<15} {'BCBO':<20} {'MBCBO':<20}")
        print("-"*60)

        for char in characteristics['BCBO'].keys():
            bcbo_val = characteristics['BCBO'][char]
            mbcbo_val = characteristics['MBCBO'][char]
            print(f"{char:<15} {bcbo_val:<20} {mbcbo_val:<20}")

    def generate_final_recommendation(self):
        """生成最终推荐"""
        print("\n" + "="*70)
        print("最终结论与推荐".center(70))
        print("="*70)

        if not self.test_results:
            print("[警告] 请先运行对比测试")
            return

        # 计算总体性能
        avg_improvement = np.mean([r['improvement'] for r in self.test_results])

        print("\n性能评估:")
        print("-"*50)

        if avg_improvement > 1.0:
            print(f"MBCBO平均改进率: {avg_improvement:+.2f}%")
            print("结论: MBCBO显著优于BCBO")
            winner = "MBCBO"
        elif avg_improvement > 0:
            print(f"MBCBO平均改进率: {avg_improvement:+.2f}%")
            print("结论: MBCBO略优于BCBO")
            winner = "MBCBO"
        elif avg_improvement > -1.0:
            print(f"MBCBO平均改进率: {avg_improvement:+.2f}%")
            print("结论: 两种算法性能相当")
            winner = "Both"
        else:
            print(f"MBCBO平均改进率: {avg_improvement:+.2f}%")
            print("结论: BCBO优于MBCBO")
            winner = "BCBO"

        print("\n应用场景推荐:")
        print("-"*50)

        if winner == "MBCBO":
            print("【期刊论文发表】")
            print("  推荐算法: MBCBO")
            print("  理由:")
            print("    1. 理论创新性强，多策略协同框架新颖")
            print("    2. 性能优于基准算法")
            print("    3. 具有良好的可扩展性")
            print("    4. 适合作为研究贡献")

            print("\n【工程实践应用】")
            print("  推荐算法: BCBO 或 MBCBO")
            print("  理由:")
            print("    - 小规模问题: BCBO (简单高效)")
            print("    - 大规模问题: MBCBO (性能更好)")
            print("    - 资源受限: BCBO (开销小)")
            print("    - 性能优先: MBCBO (效果好)")

        elif winner == "BCBO":
            print("【所有场景】")
            print("  推荐算法: BCBO")
            print("  理由:")
            print("    1. 算法简单，易于实现和维护")
            print("    2. 性能稳定可靠")
            print("    3. 计算开销小")
            print("    4. 适用范围广")

        else:
            print("【期刊论文发表】")
            print("  推荐算法: MBCBO")
            print("  理由: 创新性强，虽性能相当但理论贡献大")

            print("\n【工程实践应用】")
            print("  推荐算法: BCBO")
            print("  理由: 实现简单，性能相当的情况下选择简单方案")

        print("\n研究方向建议:")
        print("-"*50)
        print("1. 继续优化MBCBO的参数配置")
        print("2. 探索更多的策略组合")
        print("3. 研究自适应参数调整机制")
        print("4. 扩展到多目标优化问题")
        print("5. 开发并行化实现提升效率")

    def plot_comparison_charts(self):
        """生成对比图表"""
        if not self.test_results:
            print("\n[警告] 无数据用于绘图")
            return

        print("\n生成对比图表...")

        # 准备数据
        scenarios = [r['scenario'] for r in self.test_results]
        bcbo_fitness = [r['bcbo_fitness'] for r in self.test_results]
        mbcbo_fitness = [r['mbcbo_fitness'] for r in self.test_results]
        improvements = [r['improvement'] for r in self.test_results]

        # 创建对比数据DataFrame
        df = pd.DataFrame({
            '场景': scenarios,
            'BCBO': bcbo_fitness,
            'MBCBO': mbcbo_fitness,
            '改进率(%)': improvements
        })

        print("\n对比数据表:")
        print(df.to_string(index=False))

        # 保存数据
        df.to_csv('bcbo_vs_mbcbo_comparison.csv', index=False, encoding='utf-8')
        print("\n[OK] 对比数据已保存到 bcbo_vs_mbcbo_comparison.csv")

        return df

    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("BCBO vs MBCBO 完整对比分析".center(80))
        print("="*80)

        # 1. 运行对比测试
        self.run_comparison_tests()

        # 2. 分析收敛特性
        self.analyze_convergence()

        # 3. 分析性能对比
        self.analyze_performance_comparison()

        # 4. 分析MBCBO策略贡献
        self.analyze_mbcbo_strategies()

        # 5. 分析算法特性
        self.analyze_algorithm_characteristics()

        # 6. 生成最终推荐
        self.generate_final_recommendation()

        # 7. 生成图表数据
        self.plot_comparison_charts()

        print("\n" + "="*80)
        print("分析完成！".center(80))
        print("="*80)


# 主程序
if __name__ == "__main__":
    analyzer = BCBOvsMBCBOAnalyzer()
    analyzer.run_full_analysis()