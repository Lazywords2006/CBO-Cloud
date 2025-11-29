#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析RAW_data中的数据，判断MBCBO是否优于其他算法
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List
import os


class MBCBODataAnalyzer:
    """分析RAW_data中的MBCBO性能"""

    def __init__(self):
        self.data_path = "Text Demo/RAW_data/"
        self.all_data = {}
        self.load_all_data()

    def load_all_data(self):
        """加载所有chart_set数据"""
        for i in range(1, 5):
            file_path = os.path.join(self.data_path, f"chart_set_{i}_bcbo_comparison.json")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.all_data[f'chart_{i}'] = json.load(f)
                    print(f"[OK] 成功加载 chart_set_{i}")
            except Exception as e:
                print(f"[ERROR] 加载 chart_set_{i} 失败: {e}")

    def analyze_chart_set_1(self):
        """分析Chart Set 1: 迭代次数对比"""
        print("\n" + "="*70)
        print("Chart Set 1: 迭代次数对比分析 (M=100, N=20)".center(70))
        print("="*70)

        if 'chart_1' not in self.all_data:
            print("Chart Set 1 数据不存在")
            return None

        data = self.all_data['chart_1']
        algorithms = data['algorithms']

        # 收集所有算法的最终适应度
        results = {}
        for algo_name, algo_data in algorithms.items():
            if 'results' in algo_data:
                # 获取最后一个迭代的结果
                final_fitness = algo_data['results'][-1]['best_fitness']
                results[algo_name] = final_fitness

        # 排序并显示
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print("\n算法性能排名（适应度越高越好）:")
        print("-"*50)
        print(f"{'排名':<5} {'算法名称':<15} {'最终适应度':<15} {'与第一名差距':<15}")
        print("-"*50)

        best_fitness = sorted_results[0][1]
        mbcbo_rank = 0
        mbcbo_fitness = 0

        for rank, (algo, fitness) in enumerate(sorted_results, 1):
            gap = ((fitness - best_fitness) / best_fitness * 100)
            print(f"{rank:<5} {algo:<15} {fitness:<15.4f} {gap:+.2f}%")

            if 'MBCBO' in algo:
                mbcbo_rank = rank
                mbcbo_fitness = fitness

        # 分析MBCBO的表现
        print("\nMBCBO性能分析:")
        print("-"*50)

        if mbcbo_rank > 0:
            print(f"  MBCBO排名: 第{mbcbo_rank}名 (共{len(sorted_results)}个算法)")
            print(f"  MBCBO适应度: {mbcbo_fitness:.4f}")

            # 与BCBO对比
            if 'BCBO' in results:
                bcbo_fitness = results['BCBO']
                improvement = ((mbcbo_fitness - bcbo_fitness) / bcbo_fitness * 100)
                print(f"  BCBO适应度: {bcbo_fitness:.4f}")
                print(f"  相对BCBO改进: {improvement:+.2f}%")

                if improvement > 0:
                    print("  结论: MBCBO优于BCBO ✓")
                else:
                    print("  结论: MBCBO劣于BCBO ✗")
        else:
            print("  未找到MBCBO数据")

        return results

    def analyze_chart_set_2(self):
        """分析Chart Set 2: 任务规模对比"""
        print("\n" + "="*70)
        print("Chart Set 2: 任务规模对比分析".center(70))
        print("="*70)

        if 'chart_2' not in self.all_data:
            print("Chart Set 2 数据不存在")
            return None

        data = self.all_data['chart_2']
        algorithms = data['algorithms']

        # 分析不同任务规模下的表现
        print("\n不同任务规模下的算法表现:")
        print("-"*60)
        print(f"{'算法':<15} {'M=100':<12} {'M=500':<12} {'M=1000':<12}")
        print("-"*60)

        for algo_name, algo_data in algorithms.items():
            if 'results' in algo_data:
                results_dict = {}
                for result in algo_data['results']:
                    m_value = result.get('M', 0)
                    fitness = result.get('best_fitness', 0)
                    results_dict[m_value] = fitness

                m100 = results_dict.get(100, 0)
                m500 = results_dict.get(500, 0)
                m1000 = results_dict.get(1000, 0)

                print(f"{algo_name:<15} {m100:<12.4f} {m500:<12.4f} {m1000:<12.4f}")

    def analyze_chart_set_3(self):
        """分析Chart Set 3: 任务规模对比"""
        print("\n" + "="*70)
        print("Chart Set 3: 任务规模对比分析".center(70))
        print("="*70)

        if 'chart_3' not in self.all_data:
            print("Chart Set 3 数据不存在")
            return None

        data = self.all_data['chart_3']

        # 检查数据结构
        if 'test_scenarios' in data:
            # 处理多场景测试数据
            scenarios = data['test_scenarios']

            print("\n不同任务规模下的算法表现:")
            print("-"*70)
            print(f"{'算法':<15} {'小规模(50)':<15} {'中规模(100)':<15} {'大规模(200)':<15}")
            print("-"*70)

            # 收集各算法在不同规模下的表现
            algo_performance = {}

            for scenario_name, scenario_data in scenarios.items():
                task_count = scenario_data.get('task_count', 0)

                if 'algorithms' in scenario_data:
                    for algo_name, algo_results in scenario_data['algorithms'].items():
                        if algo_name not in algo_performance:
                            algo_performance[algo_name] = {}

                        if isinstance(algo_results, dict) and 'fitness' in algo_results:
                            algo_performance[algo_name][task_count] = algo_results['fitness']

            # 显示结果
            for algo_name, performances in algo_performance.items():
                small = performances.get(50, 0)
                medium = performances.get(100, 0)
                large = performances.get(200, 0)
                print(f"{algo_name:<15} {small:<15.4f} {medium:<15.4f} {large:<15.4f}")

        elif 'algorithms' in data:
            # 处理单一结构数据
            algorithms = data['algorithms']

            print("\n算法性能对比:")
            print("-"*50)

            for algo_name, algo_data in algorithms.items():
                if 'results' in algo_data and len(algo_data['results']) > 0:
                    # 获取最后的结果
                    final_result = algo_data['results'][-1]
                    fitness = final_result.get('best_fitness', 0)
                    print(f"{algo_name:<20}: {fitness:.4f}")

    def analyze_chart_set_4(self):
        """分析Chart Set 4: VM数量对比"""
        print("\n" + "="*70)
        print("Chart Set 4: VM数量对比分析 (M=100)".center(70))
        print("="*70)

        if 'chart_4' not in self.all_data:
            print("Chart Set 4 数据不存在")
            return None

        data = self.all_data['chart_4']
        algorithms = data['algorithms']

        print("\n不同VM数量下的算法表现:")
        print("-"*60)
        print(f"{'算法':<15} {'N=10':<12} {'N=20':<12} {'N=30':<12} {'N=40':<12}")
        print("-"*60)

        for algo_name, algo_data in algorithms.items():
            if 'results' in algo_data:
                vm_results = {}
                for result in algo_data['results']:
                    vm_count = result['vm_count']
                    fitness = result['best_fitness']
                    vm_results[vm_count] = fitness

                n10 = vm_results.get(10, 0)
                n20 = vm_results.get(20, 0)
                n30 = vm_results.get(30, 0)
                n40 = vm_results.get(40, 0)

                print(f"{algo_name:<15} {n10:<12.4f} {n20:<12.4f} {n30:<12.4f} {n40:<12.4f}")

    def comprehensive_analysis(self):
        """综合分析所有数据"""
        print("\n" + "="*80)
        print("MBCBO综合性能分析报告".center(80))
        print("="*80)

        # 收集所有比较数据
        mbcbo_wins = 0
        bcbo_wins = 0
        total_comparisons = 0
        improvements = []

        # 分析每个chart set
        for chart_name, chart_data in self.all_data.items():
            if 'algorithms' not in chart_data:
                continue

            algorithms = chart_data['algorithms']

            # 查找MBCBO和BCBO
            mbcbo_data = None
            bcbo_data = None

            for algo_name in algorithms:
                if 'MBCBO' in algo_name:
                    mbcbo_data = algorithms[algo_name]
                if algo_name == 'BCBO':
                    bcbo_data = algorithms[algo_name]

            # 如果都存在，进行比较
            if mbcbo_data and bcbo_data:
                if 'results' in mbcbo_data and 'results' in bcbo_data:
                    # 比较每个数据点
                    for i in range(min(len(mbcbo_data['results']), len(bcbo_data['results']))):
                        mbcbo_fitness = mbcbo_data['results'][i].get('best_fitness', 0)
                        bcbo_fitness = bcbo_data['results'][i].get('best_fitness', 0)

                        if bcbo_fitness != 0:
                            improvement = ((mbcbo_fitness - bcbo_fitness) / abs(bcbo_fitness)) * 100
                            improvements.append(improvement)
                            total_comparisons += 1

                            if improvement > 0.1:
                                mbcbo_wins += 1
                            elif improvement < -0.1:
                                bcbo_wins += 1

        # 统计分析
        print("\n1. 总体性能统计:")
        print("-"*50)

        if total_comparisons > 0:
            print(f"  总比较次数: {total_comparisons}")
            print(f"  MBCBO获胜: {mbcbo_wins} ({mbcbo_wins/total_comparisons*100:.1f}%)")
            print(f"  BCBO获胜: {bcbo_wins} ({bcbo_wins/total_comparisons*100:.1f}%)")
            print(f"  平局: {total_comparisons - mbcbo_wins - bcbo_wins}")

            if improvements:
                avg_improvement = np.mean(improvements)
                std_improvement = np.std(improvements)
                max_improvement = max(improvements)
                min_improvement = min(improvements)

                print(f"\n2. 改进率分析:")
                print("-"*50)
                print(f"  平均改进率: {avg_improvement:+.2f}%")
                print(f"  标准差: {std_improvement:.2f}%")
                print(f"  最大改进: {max_improvement:+.2f}%")
                print(f"  最小改进: {min_improvement:+.2f}%")
                print(f"  正改进占比: {sum(1 for x in improvements if x > 0)/len(improvements)*100:.1f}%")

                # 判断MBCBO是否更好
                print(f"\n3. 最终判断:")
                print("-"*50)

                if avg_improvement > 1.0:
                    print(f"  【结论】MBCBO显著优于BCBO")
                    print(f"  - 平均性能提升 {avg_improvement:.2f}%")
                    print(f"  - 获胜率 {mbcbo_wins/total_comparisons*100:.1f}%")
                    print(f"  - 推荐: 适合期刊发表")
                    is_better = True
                elif avg_improvement > 0:
                    print(f"  【结论】MBCBO略优于BCBO")
                    print(f"  - 平均性能提升 {avg_improvement:.2f}%")
                    print(f"  - 需要进一步优化")
                    print(f"  - 推荐: 可以发表，强调创新性")
                    is_better = True
                else:
                    print(f"  【结论】MBCBO不如BCBO")
                    print(f"  - 平均性能下降 {abs(avg_improvement):.2f}%")
                    print(f"  - 需要重新设计或优化")
                    print(f"  - 推荐: 暂不适合发表")
                    is_better = False

                return is_better, avg_improvement
        else:
            print("  未找到可比较的MBCBO和BCBO数据")
            return None, 0

    def run_full_analysis(self):
        """运行完整分析"""
        print("\n开始分析RAW_data中的数据...")
        print("="*80)

        # 分析各个chart set
        self.analyze_chart_set_1()
        self.analyze_chart_set_2()
        self.analyze_chart_set_3()
        self.analyze_chart_set_4()

        # 综合分析
        is_better, avg_improvement = self.comprehensive_analysis()

        # 生成最终报告
        print("\n" + "="*80)
        print("最终分析报告".center(80))
        print("="*80)

        if is_better is not None:
            if is_better:
                print("\n✓ MBCBO算法表现更好！")
                print(f"  平均改进: {avg_improvement:+.2f}%")
                print("\n建议:")
                print("  1. MBCBO适合用于期刊论文发表")
                print("  2. 重点强调多策略协同的创新性")
                print("  3. 可以进一步优化参数以提升性能")
            else:
                print("\n✗ MBCBO算法需要改进")
                print(f"  平均差距: {avg_improvement:+.2f}%")
                print("\n建议:")
                print("  1. 调整策略权重分配")
                print("  2. 优化信息交换机制")
                print("  3. 考虑添加自适应参数调整")
        else:
            print("\n⚠ 数据不完整，无法得出明确结论")
            print("  建议重新生成完整的测试数据")

        return is_better, avg_improvement


if __name__ == "__main__":
    analyzer = MBCBODataAnalyzer()
    is_better, improvement = analyzer.run_full_analysis()

    # 保存分析结果
    result = {
        "is_mbcbo_better": is_better,
        "average_improvement": improvement,
        "conclusion": "MBCBO更优" if is_better else "需要改进"
    }

    with open('mbcbo_analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 分析结果已保存到 mbcbo_analysis_result.json")