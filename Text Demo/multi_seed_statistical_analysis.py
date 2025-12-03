#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种子验证数据统计分析工具
========================================
对三个种子（42, 43, 44）的Chart Set 1数据进行统计分析

功能：
1. 计算每个迭代的均值和标准差（BCBO vs BCBO-GA）
2. 执行配对t检验验证统计显著性
3. 生成统计分析报告
4. 为图表生成提供数据（均值±标准差）

Author: Multi-seed Validation Team
Date: 2025-12-02
"""

import os
import sys
import json
import numpy as np
from scipy import stats
from datetime import datetime

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(BASE_DIR, 'multi_seed_validation')


class MultiSeedStatisticalAnalyzer:
    """多种子数据统计分析器"""

    def __init__(self, seeds=[42, 43, 44], chart_set='chart_set_1'):
        """
        初始化统计分析器

        参数:
            seeds: 随机种子列表
            chart_set: 图表集名称
        """
        self.seeds = seeds
        self.chart_set = chart_set
        self.data = {}  # {seed: {BCBO: [...], BCBO-GA: [...]}}
        self.statistics = {}  # {iteration: {metric: {mean, std, ...}}}

    def load_data(self):
        """加载所有种子的数据"""
        print(f"\n[INFO] 加载数据: {self.chart_set}, seeds={self.seeds}")

        for seed in self.seeds:
            filepath = os.path.join(VALIDATION_DIR, f'{self.chart_set}_seed_{seed}.json')

            if not os.path.exists(filepath):
                print(f"[ERROR] 文件不存在: {filepath}")
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.data[seed] = {
                    'BCBO': data['algorithms']['BCBO']['results'],
                    'BCBO-GA': data['algorithms']['BCBO-GA']['results']
                }

                print(f"  [OK] Seed {seed}: {len(self.data[seed]['BCBO'])} iterations")

            except Exception as e:
                print(f"[ERROR] 读取seed {seed}失败: {e}")

        if len(self.data) != len(self.seeds):
            print(f"[WARNING] 只加载了 {len(self.data)}/{len(self.seeds)} 个种子的数据")

        return len(self.data) > 0

    def compute_statistics(self):
        """计算每个迭代的统计指标（均值、标准差）"""
        print(f"\n[INFO] 计算统计指标...")

        if not self.data:
            print(f"[ERROR] 没有数据可分析")
            return False

        # 获取迭代数（假设所有种子迭代数相同）
        first_seed = self.seeds[0]
        num_iterations = len(self.data[first_seed]['BCBO'])

        metrics = ['total_cost', 'execution_time', 'load_balance', 'best_fitness']

        for iteration_idx in range(num_iterations):
            iter_num = iteration_idx + 1
            self.statistics[iter_num] = {}

            for algorithm in ['BCBO', 'BCBO-GA']:
                self.statistics[iter_num][algorithm] = {}

                for metric in metrics:
                    # 收集所有种子在该迭代的该指标值
                    values = []
                    for seed in self.seeds:
                        if seed in self.data:
                            try:
                                value = self.data[seed][algorithm][iteration_idx][metric]
                                values.append(value)
                            except (IndexError, KeyError):
                                continue

                    if len(values) > 0:
                        self.statistics[iter_num][algorithm][metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values, ddof=1)),  # 样本标准差
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'values': values  # 保留原始值用于t检验
                        }

        print(f"  [OK] 计算了 {num_iterations} 个迭代的统计指标")
        return True

    def paired_t_test(self, iteration_num=100):
        """
        执行配对t检验（BCBO vs BCBO-GA）

        参数:
            iteration_num: 要检验的迭代编号（默认最终迭代100）

        返回:
            统计检验结果字典
        """
        print(f"\n[INFO] 执行配对t检验 (Iteration {iteration_num})...")

        if iteration_num not in self.statistics:
            print(f"[ERROR] 迭代 {iteration_num} 无统计数据")
            return None

        results = {}
        metrics = ['total_cost', 'execution_time', 'load_balance', 'best_fitness']

        for metric in metrics:
            try:
                bcbo_values = self.statistics[iteration_num]['BCBO'][metric]['values']
                bcbo_ga_values = self.statistics[iteration_num]['BCBO-GA'][metric]['values']

                # 配对t检验
                t_stat, p_value = stats.ttest_rel(bcbo_values, bcbo_ga_values)

                # 计算改进率
                bcbo_mean = np.mean(bcbo_values)
                bcbo_ga_mean = np.mean(bcbo_ga_values)

                if metric == 'load_balance':
                    # 负载均衡越高越好
                    improvement = (bcbo_ga_mean - bcbo_mean) / bcbo_mean * 100
                else:
                    # 成本、时间越低越好
                    improvement = (bcbo_mean - bcbo_ga_mean) / bcbo_mean * 100

                results[metric] = {
                    'bcbo_mean': bcbo_mean,
                    'bcbo_ga_mean': bcbo_ga_mean,
                    'improvement_%': improvement,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                significance_mark = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
                print(f"  {metric}: {improvement:+.2f}%, p={p_value:.4f} {significance_mark}")

            except Exception as e:
                print(f"  [ERROR] {metric} 检验失败: {e}")

        return results

    def comprehensive_improvement_rate(self, iteration_num=100):
        """
        计算综合改进率（weighted metric）

        公式: time_imp * 0.5 + balance_imp * 0.3 + cost_imp * 0.2
        (所有指标改进率为正值表示改进,加权求和)
        """
        print(f"\n[INFO] 计算综合改进率 (Iteration {iteration_num})...")

        if iteration_num not in self.statistics:
            print(f"[ERROR] 迭代 {iteration_num} 无统计数据")
            return None

        try:
            # 获取各指标的均值
            bcbo_cost = self.statistics[iteration_num]['BCBO']['total_cost']['mean']
            bcbo_time = self.statistics[iteration_num]['BCBO']['execution_time']['mean']
            bcbo_balance = self.statistics[iteration_num]['BCBO']['load_balance']['mean']

            bcbo_ga_cost = self.statistics[iteration_num]['BCBO-GA']['total_cost']['mean']
            bcbo_ga_time = self.statistics[iteration_num]['BCBO-GA']['execution_time']['mean']
            bcbo_ga_balance = self.statistics[iteration_num]['BCBO-GA']['load_balance']['mean']

            # 计算改进率
            cost_imp = (bcbo_cost - bcbo_ga_cost) / bcbo_cost * 100
            time_imp = (bcbo_time - bcbo_ga_time) / bcbo_time * 100
            balance_imp = (bcbo_ga_balance - bcbo_balance) / bcbo_balance * 100

            # 综合改进率（正值表示改进，加权求和）
            comprehensive = time_imp * 0.5 + balance_imp * 0.3 + cost_imp * 0.2

            print(f"  Cost improvement: {cost_imp:+.2f}%")
            print(f"  Time improvement: {time_imp:+.2f}%")
            print(f"  Balance improvement: {balance_imp:+.2f}%")
            print(f"  Comprehensive: {comprehensive:+.2f}%")

            return {
                'cost_improvement_%': cost_imp,
                'time_improvement_%': time_imp,
                'balance_improvement_%': balance_imp,
                'comprehensive_%': comprehensive
            }

        except Exception as e:
            print(f"[ERROR] 计算综合改进率失败: {e}")
            return None

    def generate_report(self, output_path=None):
        """
        生成统计分析报告

        参数:
            output_path: 报告输出路径（默认为multi_seed_validation/目录）
        """
        if output_path is None:
            output_path = os.path.join(VALIDATION_DIR, 'statistical_analysis_report.txt')

        print(f"\n[INFO] 生成统计分析报告...")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("多种子验证统计分析报告\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"图表集: {self.chart_set}\n")
            f.write(f"随机种子: {self.seeds}\n")
            f.write(f"样本数量: {len(self.seeds)}\n")
            f.write("="*80 + "\n\n")

            # 1. 最终迭代统计
            f.write("1. 最终迭代（Iteration 100）统计摘要\n")
            f.write("-"*80 + "\n")

            final_iter = 100
            if final_iter in self.statistics:
                for algorithm in ['BCBO', 'BCBO-GA']:
                    f.write(f"\n{algorithm}:\n")
                    for metric in ['total_cost', 'execution_time', 'load_balance']:
                        stats_data = self.statistics[final_iter][algorithm][metric]
                        f.write(f"  {metric}: {stats_data['mean']:.2f} ± {stats_data['std']:.2f}\n")
                        f.write(f"    Range: [{stats_data['min']:.2f}, {stats_data['max']:.2f}]\n")

            # 2. 配对t检验结果
            f.write("\n\n2. 配对t检验结果（Iteration 100）\n")
            f.write("-"*80 + "\n")

            t_test_results = self.paired_t_test(iteration_num=100)
            if t_test_results:
                f.write(f"\n{'指标':<20} {'改进率':>10} {'p值':>10} {'显著性':>10}\n")
                f.write("-"*80 + "\n")
                for metric, result in t_test_results.items():
                    significance = '***' if result['p_value'] < 0.001 else \
                                   '**' if result['p_value'] < 0.01 else \
                                   '*' if result['p_value'] < 0.05 else 'n.s.'
                    f.write(f"{metric:<20} {result['improvement_%']:>+9.2f}% {result['p_value']:>10.4f} {significance:>10}\n")

                f.write("\n注: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\n")

            # 3. 综合改进率
            f.write("\n\n3. 综合改进率（Iteration 100）\n")
            f.write("-"*80 + "\n")

            comp_result = self.comprehensive_improvement_rate(iteration_num=100)
            if comp_result:
                f.write(f"  成本改进率: {comp_result['cost_improvement_%']:+.2f}%\n")
                f.write(f"  时间改进率: {comp_result['time_improvement_%']:+.2f}%\n")
                f.write(f"  负载均衡改进率: {comp_result['balance_improvement_%']:+.2f}%\n")
                f.write(f"\n  综合改进率: {comp_result['comprehensive_%']:+.2f}%\n")
                f.write(f"  (公式: time*0.5 + balance*0.3 + cost*0.2)\n")

            # 4. 结论
            f.write("\n\n4. 统计验证结论\n")
            f.write("-"*80 + "\n")

            if t_test_results:
                significant_count = sum(1 for r in t_test_results.values() if r['significant'])
                total_metrics = len(t_test_results)

                f.write(f"显著性检验通过率: {significant_count}/{total_metrics} ({significant_count/total_metrics*100:.1f}%)\n")

                # 修正逻辑：必须同时满足"显著性"和"正向改进"
                is_statistically_significant = significant_count >= 2  # 至少2/4指标显著
                is_positive_improvement = comp_result and comp_result['comprehensive_%'] > 0

                if is_statistically_significant and is_positive_improvement:
                    f.write(f"\n[结论] BCBO-GA在统计学上显著优于BCBO ✓\n")
                    f.write(f"       综合改进率为 {comp_result['comprehensive_%']:+.2f}%\n")
                    f.write(f"       {significant_count}/{total_metrics} 指标达到显著性水平 (p<0.05)\n")
                elif is_positive_improvement and not is_statistically_significant:
                    f.write(f"\n[结论] BCBO-GA数值上优于BCBO，但统计学上不显著 ⚠\n")
                    f.write(f"       综合改进率为 {comp_result['comprehensive_%']:+.2f}%\n")
                    f.write(f"       显著性检验: {significant_count}/{total_metrics} 指标显著 (需≥2)\n")
                    f.write(f"       建议: 增加样本量（更多种子）以提高统计功效\n")
                else:
                    f.write(f"\n[结论] BCBO-GA性能未达预期 ✗\n")
                    f.write(f"       综合改进率为 {comp_result['comprehensive_%']:+.2f}% (负向)\n")
                    f.write(f"       需要进一步分析和算法优化\n")

            f.write("\n" + "="*80 + "\n")

        print(f"  [OK] 报告已保存: {output_path}")
        return output_path

    def save_statistics_json(self, output_path=None):
        """
        保存统计数据为JSON格式（供图表生成使用）

        参数:
            output_path: 输出路径
        """
        if output_path is None:
            output_path = os.path.join(VALIDATION_DIR, 'multi_seed_statistics.json')

        print(f"\n[INFO] 保存统计数据...")

        # 准备输出数据（去除原始values，只保留统计量）
        output_data = {
            'chart_set': self.chart_set,
            'seeds': self.seeds,
            'num_seeds': len(self.seeds),
            'timestamp': datetime.now().isoformat(),
            'statistics': {}
        }

        for iter_num, iter_data in self.statistics.items():
            output_data['statistics'][str(iter_num)] = {}

            for algorithm in ['BCBO', 'BCBO-GA']:
                output_data['statistics'][str(iter_num)][algorithm] = {}

                for metric, stats_data in iter_data[algorithm].items():
                    output_data['statistics'][str(iter_num)][algorithm][metric] = {
                        'mean': stats_data['mean'],
                        'std': stats_data['std'],
                        'min': stats_data['min'],
                        'max': stats_data['max']
                    }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"  [OK] 统计数据已保存: {output_path}")
            return output_path

        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")
            return None


def auto_detect_seeds(chart_set='chart_set_1'):
    """自动检测multi_seed_validation目录中所有可用的种子"""
    import glob
    import re

    pattern = os.path.join(VALIDATION_DIR, f'{chart_set}_seed_*.json')
    files = glob.glob(pattern)

    seeds = []
    for filepath in files:
        match = re.search(r'seed_(\d+)\.json$', filepath)
        if match:
            seed = int(match.group(1))
            seeds.append(seed)

    seeds.sort()
    return seeds


def main():
    """主函数"""
    print("="*80)
    print("多种子验证数据统计分析")
    print("="*80)

    # 自动检测所有可用种子
    seeds = auto_detect_seeds(chart_set='chart_set_1')

    if not seeds:
        print("\n[ERROR] 未检测到任何种子数据文件")
        print(f"[INFO] 请确保文件位于: {VALIDATION_DIR}")
        return 1

    print(f"\n[INFO] 检测到 {len(seeds)} 个种子: {seeds}")

    # 创建分析器
    analyzer = MultiSeedStatisticalAnalyzer(
        seeds=seeds,
        chart_set='chart_set_1'
    )

    # 加载数据
    if not analyzer.load_data():
        print("\n[ERROR] 数据加载失败，退出")
        return 1

    # 计算统计指标
    if not analyzer.compute_statistics():
        print("\n[ERROR] 统计计算失败，退出")
        return 1

    # 执行t检验
    t_test_results = analyzer.paired_t_test(iteration_num=100)

    # 计算综合改进率
    comp_improvement = analyzer.comprehensive_improvement_rate(iteration_num=100)

    # 生成报告
    report_path = analyzer.generate_report()

    # 保存统计数据JSON
    stats_json_path = analyzer.save_statistics_json()

    print("\n" + "="*80)
    print("[SUCCESS] 统计分析完成！")
    print("="*80)
    print(f"报告路径: {report_path}")
    print(f"数据路径: {stats_json_path}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
