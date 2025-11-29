#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成BCBO、BCBO-DE和MBCBO数据的脚本
========================================
用于生成和对比BCBO、BCBO-DE与MBCBO三个算法的性能数据
为期刊发表提供完整的算法对比数据
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
import time

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    from real_algorithm_integration import RealAlgorithmIntegrator
    print("[OK] 算法集成模块导入成功")
except ImportError as e:
    print(f"[ERROR] 无法导入算法集成模块: {e}")
    sys.exit(1)


# NumPy类型JSON序列化编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# 四组图表的配置（方案2：增加种群规模）
CHART_CONFIGS = {
    'chart_set_1': {
        'name': '图表集1 - 迭代次数 vs 性能指标',
        'type': 'iterations',
        'variable_param': 'iterations',
        'values': list(range(5, 101, 5)),
        'fixed_params': {
            'M': 100,
            'N': 20,
            'n': 50  # 小规模保持50
        },
        'runs_per_point': 1
    },
    'chart_set_2': {
        'name': '图表集2 - 任务规模 vs 成本',
        'type': 'task_scale',
        'variable_param': 'M',
        'values': list(range(100, 1001, 100)),
        'fixed_params': {
            'iterations': 80,
            'N': 20,
            'n': 100  # 方案2: 中等规模增加到100
        },
        'runs_per_point': 1
    },
    'chart_set_3': {
        'name': '图表集3 - 迭代次数 vs 性能指标 (大规模1000任务)',
        'type': 'iterations_2',
        'variable_param': 'iterations',
        'values': list(range(5, 101, 5)),
        'fixed_params': {
            'M': 1000,
            'N': 20,
            'n': 150  # 方案2: 大规模增加到150
        },
        'runs_per_point': 1
    },
    'chart_set_4': {
        'name': '图表集4 - 任务规模 vs 成本 (大规模)',
        'type': 'large_scale',
        'variable_param': 'M',
        'values': list(range(1000, 5001, 1000)),
        'fixed_params': {
            'iterations': 50,
            'N': 20,
            'n': 200  # 方案2: 超大规模增加到200
        },
        'runs_per_point': 1
    }
}

# 包含BCBO、BCBO-DE和MBCBO三个算法
ALGORITHMS = ['BCBO', 'BCBO-DE', 'MBCBO']


class BCBOComparisonGenerator:
    """BCBO、BCBO-DE与MBCBO对比数据生成器"""

    def __init__(self):
        self.integrator = RealAlgorithmIntegrator()
        self.output_dir = os.path.join(BASE_DIR, 'Text Demo', 'RAW_data')
        os.makedirs(self.output_dir, exist_ok=True)

    def run_algorithm_single_point(self, algorithm, params, run_num=1):
        """
        运行单个算法在单个数据点上

        参数:
            algorithm: 算法名称
            params: 参数字典 {'M': 100, 'N': 20, 'n': 50, 'iterations': 80}
            run_num: 运行次数编号

        返回:
            result: 算法运行结果
        """
        try:
            # 使用固定seed确保BCBO和BCBO-DE比较相同的问题实例（v3.2方法）
            params_with_seed = params.copy()
            params_with_seed['random_seed'] = 42  # 固定seed，确保公平比较

            result = self.integrator.run_algorithm(
                algorithm_name=algorithm,
                params=params_with_seed
            )
            return result
        except Exception as e:
            print(f"    [ERROR] {algorithm} 运行失败: {e}")
            return None

    def generate_chart_set(self, chart_set_name):
        """
        生成指定图表集的BCBO和BCBO-DE数据

        参数:
            chart_set_name: 图表集名称 (chart_set_1, chart_set_2, etc.)
        """
        if chart_set_name not in CHART_CONFIGS:
            print(f"[ERROR] 未知的图表集: {chart_set_name}")
            return False

        config = CHART_CONFIGS[chart_set_name]
        print("\n" + "="*80)
        print(f"开始生成: {config['name']}")
        print("="*80)
        print(f"配置: {config['fixed_params']}")
        print(f"变量参数: {config['variable_param']} = {config['values']}")
        print(f"算法: {', '.join(ALGORITHMS)}")
        print(f"每点运行次数: {config['runs_per_point']}")
        print("-"*80)

        start_time = time.time()

        # 存储所有算法的结果
        all_results = {}

        # 对每个算法运行
        for algorithm in ALGORITHMS:
            print(f"\n[算法] {algorithm}")
            algorithm_results = []

            # 根据图表类型决定是否使用收敛历史
            use_convergence = config['type'] in ['iterations', 'iterations_2']

            if use_convergence:
                # 使用收敛历史模式（一次运行，记录所有迭代）
                print(f"  [INFO] 使用收敛历史模式")

                max_iterations = config['values'][-1]  # 最大迭代次数
                params = {
                    'M': config['fixed_params'].get('M', 100),
                    'N': config['fixed_params'].get('N', 20),
                    'n': config['fixed_params'].get('n', 50),
                    'iterations': max_iterations
                }

                result = self.run_algorithm_single_point(algorithm, params, run_num=1)

                if result and 'convergence_history' in result:
                    # 从收敛历史中提取数据点
                    history = result['convergence_history']
                    for target_iter in config['values']:
                        if target_iter <= len(history):
                            point_data = history[target_iter - 1]
                            point_data['iteration'] = target_iter
                            algorithm_results.append(point_data)
                        else:
                            print(f"    [WARN] 迭代{target_iter}超出范围")

                    print(f"  [OK] 提取了 {len(algorithm_results)} 个数据点")
                else:
                    print(f"  [ERROR] 无法获取收敛历史")

            else:
                # 常规模式（每个数据点单独运行）
                for value in config['values']:
                    params = config['fixed_params'].copy()
                    params[config['variable_param']] = value

                    print(f"  {config['variable_param']}={value}: ", end='', flush=True)

                    result = self.run_algorithm_single_point(algorithm, params, run_num=1)

                    if result:
                        result_point = {
                            config['variable_param']: value,
                            'total_cost': result.get('total_cost', 0),
                            'execution_time': result.get('execution_time', 0),
                            'load_balance': result.get('load_balance', 0),
                            'price_efficiency': result.get('price_efficiency', 0),
                            'algorithm': algorithm,
                            'best_fitness': result.get('best_fitness', 0)
                        }
                        algorithm_results.append(result_point)
                        print(f"time={result['execution_time']:.2f}, cost={result['total_cost']:.2f}")
                    else:
                        print(f"[FAILED]")

            all_results[algorithm] = {
                'algorithm': algorithm,
                'results': algorithm_results
            }

            print(f"  [OK] {algorithm} 完成，共 {len(algorithm_results)} 个数据点")

        # 保存结果
        output_data = {
            'chart_set': chart_set_name,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'algorithms': all_results
        }

        output_file = os.path.join(self.output_dir, f'{chart_set_name}_bcbo_comparison.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        print(f"\n[SUCCESS] {config['name']} 数据生成完成!")
        print(f"[TIME] 总耗时: {elapsed_time:.2f} 秒")
        print(f"[SAVED] 文件已保存: {output_file}")

        return True

    def generate_all(self):
        """生成所有四个图表集的数据"""
        print("\n" + "="*80)
        print("开始生成所有图表集的BCBO vs BCBO-DE对比数据")
        print("="*80)

        for chart_set_name in ['chart_set_1', 'chart_set_2', 'chart_set_3', 'chart_set_4']:
            success = self.generate_chart_set(chart_set_name)
            if not success:
                print(f"[ERROR] {chart_set_name} 生成失败")
                return False

        print("\n" + "="*80)
        print("所有数据生成完成!")
        print("="*80)
        return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='生成BCBO vs BCBO-DE对比数据')
    parser.add_argument('--chart-set', type=int, choices=[1, 2, 3, 4],
                       help='指定生成哪个图表集 (1-4)')
    parser.add_argument('--all', action='store_true',
                       help='生成所有四个图表集')

    args = parser.parse_args()

    generator = BCBOComparisonGenerator()

    if args.all:
        generator.generate_all()
    elif args.chart_set:
        chart_set_name = f'chart_set_{args.chart_set}'
        generator.generate_chart_set(chart_set_name)
    else:
        print("请指定 --chart-set <1-4> 或 --all")
        parser.print_help()


if __name__ == "__main__":
    main()
