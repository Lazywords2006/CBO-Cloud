#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO vs BCBO-GA 性能对比数据生成工具
========================================
专门用于生成BCBO和BCBO-GA的对比测试数据

特性：
1. 只对比BCBO和BCBO-GA两个算法
2. 生成四个图表集的完整数据
3. 固定随机种子确保公平对比
4. 自动数据验证和质量检查
5. 支持单独或批量生成

BCBO-GA版本：
- 版本：v2.0 自适应参数版 (2025-11-30)
- 特性：GA智能交叉、2-opt局部搜索、自适应变异、温和负载均衡、**自适应参数机制**
- 核心改进：根据问题规模M自动调整交叉率、变异率、精英规模、局部搜索概率等参数
- 目标：在保持时间效率的同时，显著提升性能指标（预期综合改进率从-0.43%提升至-0.08%）

使用方法：
```bash
# 生成所有图表集数据
python generate_bcbo_bcbo_ga_comparison.py --all

# 生成指定图表集
python generate_bcbo_bcbo_ga_comparison.py --chart-set 1

# 只测试BCBO
python generate_bcbo_bcbo_ga_comparison.py --chart-set 1 --algorithm BCBO
```
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
import time
import traceback
import random
import shutil
import tempfile

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

# 将scripts目录添加到路径
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
    print(f"[OK] 添加脚本路径: {SCRIPTS_DIR}")

# 检查real_algorithm_integration.py是否存在
integration_file = os.path.join(SCRIPTS_DIR, 'real_algorithm_integration.py')
print(f"[INFO] 检查集成文件: {integration_file}")
print(f"[INFO] 集成文件存在: {os.path.exists(integration_file)}")

try:
    from real_algorithm_integration import RealAlgorithmIntegrator
    INTEGRATOR_AVAILABLE = True
    print("[OK] 算法集成模块导入成功")
    print("[INFO] BCBO-GA使用v2.0 自适应参数版本 (adaptive_params=True)")
except ImportError as e:
    print(f"[ERROR] 导入算法集成模块失败: {e}")
    INTEGRATOR_AVAILABLE = False
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


# 四组图表的配置
CHART_CONFIGS = {
    'chart_set_1': {
        'name': '图表集1 - 迭代次数 vs 性能指标 (小规模)',
        'type': 'iterations',
        'variable_param': 'iterations',
        'values': list(range(5, 101, 5)),
        'fixed_params': {
            'M': 100,
            'N': 20,
            'n': 50
        },
        'runs_per_point': 1
    },
    'chart_set_2': {
        'name': '图表集2 - 任务规模 vs 成本 (中等规模)',
        'type': 'task_scale',
        'variable_param': 'M',
        'values': list(range(100, 1001, 100)),
        'fixed_params': {
            'iterations': 80,
            'N': 20,
            'n': 100
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
            'n': 150
        },
        'runs_per_point': 1
    },
    'chart_set_4': {
        'name': '图表集4 - 任务规模 vs 成本 (超大规模)',
        'type': 'large_scale',
        'variable_param': 'M',
        'values': list(range(1000, 5001, 1000)),
        'fixed_params': {
            'iterations': 50,
            'N': 20,
            'n': 200
        },
        'runs_per_point': 1
    }
}

# 只对比BCBO和BCBO-GA
ALGORITHMS = ['BCBO', 'BCBO-GA']


class BCBOComparisonGenerator:
    """BCBO vs BCBO-GA 对比数据生成器"""

    def __init__(self, random_seed=42, use_temp_dir=False, merge_results=True):
        """
        初始化数据生成器

        参数:
            random_seed: 随机种子
            use_temp_dir: 是否使用临时目录（程序结束后自动清理）
            merge_results: 是否将同一图表集的所有算法结果合并到一个文件
        """
        self.random_seed = random_seed
        self.use_temp_dir = use_temp_dir
        self.merge_results = merge_results
        self.set_random_seeds()
        self.experiment_log = []

        # 输出目录
        if use_temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix='bcbo_comparison_')
            self.output_dir = self.temp_dir
            print(f"[INFO] 使用临时目录: {self.output_dir}")
        else:
            self.output_dir = os.path.join(BASE_DIR, 'BCBO_vs_BCBO_GA_Data')
            os.makedirs(self.output_dir, exist_ok=True)
            self.temp_dir = None
            print(f"[INFO] 输出目录: {self.output_dir}")

        # 用于存储合并结果
        self.merged_data = {}

        # 初始化算法集成器
        self.integrator = RealAlgorithmIntegrator()

        # 验证算法可用性
        self._verify_algorithms()

    def set_random_seeds(self):
        """设置所有随机种子确保可重复性"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)

    def _verify_algorithms(self):
        """验证BCBO和BCBO-GA的可用性"""
        print(f"\n[INFO] 验证算法可用性...")
        available_algorithms = []

        for algo in ALGORITHMS:
            try:
                # 尝试创建算法实例
                test_result = self.integrator.run_algorithm(algo, {
                    'M': 10, 'N': 3, 'n': 5, 'iterations': 5, 'random_seed': 42
                })

                if test_result and 'best_solution' in test_result:
                    available_algorithms.append(algo)
                    print(f"  [OK] {algo} 可用")
                else:
                    print(f"  [FAIL] {algo} 返回结果无效")

            except Exception as e:
                print(f"  [FAIL] {algo} 不可用: {e}")

        if len(available_algorithms) < 2:
            print(f"\n[ERROR] 需要BCBO和BCBO-GA都可用才能进行对比")
            sys.exit(1)

        print(f"\n[SUCCESS] {len(available_algorithms)}/2 算法可用")

    def calculate_metrics_from_solution(self, solution, algorithm_name, config):
        """
        从任务分配方案计算详细的性能指标

        参数:
            solution: 任务分配数组 solution[i] = vm_id
            algorithm_name: 算法名称
            config: 配置参数

        返回:
            包含total_cost, execution_time, load_balance, price_efficiency的字典
        """
        M = config.get('M', 50)
        N = config.get('N', 10)

        # 优先使用算法实例的get_detailed_metrics方法
        try:
            algorithm_instance = self.integrator.algorithms.get(algorithm_name)

            if algorithm_instance and hasattr(algorithm_instance, 'get_detailed_metrics'):
                metrics = algorithm_instance.get_detailed_metrics(solution)
                return {
                    'total_cost': metrics.get('total_cost', 0),
                    'execution_time': metrics.get('makespan', 0),
                    'load_balance': 1.0 - min(metrics.get('load_imbalance', 0), 1.0),
                    'price_efficiency': 1.0 / (metrics.get('total_cost', 1) + 1e-6)
                }
        except Exception as e:
            pass  # 静默失败，使用备用方法

        # 备用方法：直接计算
        try:
            algorithm_instance = self.integrator.algorithms.get(algorithm_name)

            if not algorithm_instance:
                return {'total_cost': 0, 'execution_time': 0, 'load_balance': 0, 'price_efficiency': 0}

            # 计算VM负载
            vm_loads = np.zeros(N)
            vm_costs = np.zeros(N)

            for task_id in range(M):
                vm_id = int(solution[task_id]) % N
                if hasattr(algorithm_instance, 'execution_time'):
                    exec_time = algorithm_instance.execution_time[task_id][vm_id]
                else:
                    exec_time = 1.0
                vm_loads[vm_id] += exec_time

                if hasattr(algorithm_instance, 'vm_cost'):
                    cost = algorithm_instance.vm_cost[vm_id] * exec_time
                else:
                    cost = exec_time
                vm_costs[vm_id] += cost

            # 计算指标
            makespan = np.max(vm_loads) if len(vm_loads) > 0 else 0
            total_cost = np.sum(vm_costs)

            # Load balance（只考虑有任务的VM）
            active_vm_loads = vm_loads[vm_loads > 0]
            if len(active_vm_loads) > 1:
                mean_load = np.mean(active_vm_loads)
                std_load = np.std(active_vm_loads)
                load_imbalance = std_load / (mean_load + 1e-6)
                load_balance = max(0.0, 1.0 - min(load_imbalance, 1.0))
            else:
                load_balance = 0.0

            price_efficiency = 1.0 / (total_cost + 1e-6)

            return {
                'total_cost': float(total_cost),
                'execution_time': float(makespan),
                'load_balance': float(load_balance),
                'price_efficiency': float(price_efficiency)
            }
        except Exception as e:
            print(f"\n    [ERROR] 指标计算失败: {e}")
            return {'total_cost': 0, 'execution_time': 0, 'load_balance': 0, 'price_efficiency': 0}

    def generate_convergence_data_from_history(self, algorithm_name, config):
        """
        从convergence_history生成收敛曲线数据

        此方法运行算法固定次数迭代，然后从convergence_history中提取每一代的数据
        """
        fixed_params = config['fixed_params'].copy()
        max_iterations = 100
        fixed_params['iterations'] = max_iterations
        runs_per_point = config['runs_per_point']

        print(f"  [INFO] 运行 {algorithm_name} {max_iterations} 次迭代, {runs_per_point} 次...")

        all_histories = []
        all_solutions_by_iteration = {}

        for run in range(runs_per_point):
            try:
                # 使用固定seed确保所有算法比较相同的问题实例
                run_seed = 42 + run

                np.random.seed(run_seed)
                random.seed(run_seed)

                # 将random_seed添加到params中
                params_with_seed = fixed_params.copy()
                params_with_seed['random_seed'] = run_seed

                # 运行算法
                result = self.integrator.run_algorithm(algorithm_name, params_with_seed)

                # 尝试提取convergence_history
                history = None
                if isinstance(result, dict):
                    history = result.get('convergence_history') or result.get('fitness_history')

                # 如果没有history，尝试从算法实例获取
                if not history:
                    algorithm_instance = self.integrator.algorithms.get(algorithm_name)
                    if algorithm_instance and hasattr(algorithm_instance, 'fitness_history'):
                        history = algorithm_instance.fitness_history

                if history and len(history) > 0:
                    all_histories.append(history)

                    # 收集每个迭代的solution
                    for iter_idx, record in enumerate(history):
                        if iter_idx not in all_solutions_by_iteration:
                            all_solutions_by_iteration[iter_idx] = []

                        if isinstance(record, dict):
                            solution = record.get('best_solution')
                            if solution is not None:
                                all_solutions_by_iteration[iter_idx].append({
                                    'solution': solution,
                                    'fitness': record.get('best_fitness') or record.get('global_best_fitness') or 0,
                                    'run_id': run
                                })

                    print(f"    Run {run+1}/{runs_per_point}: {len(history)} 迭代记录", end='\r')

            except Exception as e:
                print(f"\n    [ERROR] Run {run+1} 失败: {e}")
                traceback.print_exc()
                continue

        if not all_histories:
            print(f"\n  [ERROR] 没有有效的历史数据")
            return []

        print(f"\n  [INFO] 收集了 {len(all_histories)} 次有效运行，处理中...")

        # 对每个迭代，使用所有运行的solution计算metrics并平均
        algorithm_results = []
        max_iterations_found = max(all_solutions_by_iteration.keys()) + 1 if all_solutions_by_iteration else 0

        for iteration_idx in range(max_iterations_found):
            if iteration_idx not in all_solutions_by_iteration:
                continue

            solutions_data = all_solutions_by_iteration[iteration_idx]
            if not solutions_data:
                continue

            # 为每个solution分别计算metrics，然后平均
            all_metrics = []
            all_fitness = []

            for sol_data in solutions_data:
                try:
                    solution = sol_data['solution']
                    fitness = sol_data['fitness']

                    # 计算该solution的metrics
                    metrics = self.calculate_metrics_from_solution(
                        solution, algorithm_name, fixed_params
                    )

                    # 验证metrics有效性
                    if all(np.isfinite(v) for v in metrics.values()):
                        all_metrics.append(metrics)
                        all_fitness.append(fitness)

                except Exception as e:
                    continue

            # 如果有有效的metrics，计算平均值
            if all_metrics:
                avg_total_cost = np.mean([m['total_cost'] for m in all_metrics])
                avg_execution_time = np.mean([m['execution_time'] for m in all_metrics])
                avg_load_balance = np.mean([m['load_balance'] for m in all_metrics])
                avg_price_efficiency = np.mean([m['price_efficiency'] for m in all_metrics])
                avg_fitness = np.mean(all_fitness)

                data_point = {
                    'iteration': iteration_idx + 1,
                    'total_cost': float(avg_total_cost),
                    'execution_time': float(avg_execution_time),
                    'load_balance': float(avg_load_balance),
                    'price_efficiency': float(avg_price_efficiency),
                    'algorithm': algorithm_name,
                    'best_fitness': float(avg_fitness)
                }
                algorithm_results.append(data_point)

        print(f"  [OK] 生成了 {len(algorithm_results)} 个收敛数据点")
        return algorithm_results

    def generate_data_for_chart_set(self, chart_set_name, specific_algorithms=None):
        """为特定图表集生成数据"""
        if chart_set_name not in CHART_CONFIGS:
            print(f"[ERROR] 未知图表集: {chart_set_name}")
            return False

        config = CHART_CONFIGS[chart_set_name]
        algorithms = specific_algorithms if specific_algorithms else ALGORITHMS

        print(f"\n[START] 生成 {config['name']} 的数据")
        print(f"[INFO] 参数配置: {config['fixed_params']}")
        print(f"[INFO] 可变参数: {config['variable_param']}")
        print(f"[INFO] 算法: {', '.join(algorithms)}")
        print(f"[INFO] 每个数据点运行次数: {config['runs_per_point']}")
        print("-" * 60)

        start_time = time.time()

        # 判断是否使用收敛方法
        use_convergence_method = config['variable_param'] == 'iterations'

        if use_convergence_method:
            print(f"[INFO] 使用收敛历史方法 (生成100个数据点)")

        for algorithm in algorithms:
            print(f"\n[ALGORITHM] 处理: {algorithm}")

            if use_convergence_method:
                # 使用收敛历史方法
                algorithm_results = self.generate_convergence_data_from_history(algorithm, config)
            else:
                # 使用原方法：变化参数值（用于task_scale类型）
                algorithm_results = []
                for param_value in config['values']:
                    # 构建实验参数
                    experiment_params = config['fixed_params'].copy()
                    experiment_params[config['variable_param']] = param_value

                    # 多次运行获取统计结果
                    run_results = []
                    for run in range(config['runs_per_point']):
                        try:
                            # 使用固定seed确保所有算法比较相同的问题实例
                            run_seed = 42 + param_value + run

                            np.random.seed(run_seed)
                            random.seed(run_seed)

                            # 将random_seed添加到params中
                            params_with_seed = experiment_params.copy()
                            params_with_seed['random_seed'] = run_seed

                            # 运行算法
                            result = self.integrator.run_algorithm(algorithm, params_with_seed)

                            if result and self.validate_result(result):
                                run_results.append(result)
                            else:
                                print(f"\n[WARNING] 算法 {algorithm} 失败, params: {experiment_params}")

                        except Exception as e:
                            print(f"\n[ERROR] 运行错误: {e}")
                            continue

                    # 计算该参数值的平均结果
                    if run_results:
                        avg_result = self.calculate_average_results(run_results)
                        avg_result[config['variable_param']] = param_value
                        avg_result['algorithm'] = algorithm
                        avg_result['runs'] = len(run_results)
                        algorithm_results.append(avg_result)

                        print(f"  {config['variable_param']}={param_value}: "
                              f"cost={avg_result['total_cost']:.2f}, "
                              f"time={avg_result['execution_time']:.2f}")

            # 保存该算法的结果
            if algorithm_results:
                self.save_results(algorithm, chart_set_name, algorithm_results)
                print(f"\n[OK] 算法 {algorithm} 完成，生成 {len(algorithm_results)} 个数据点")

        total_time = time.time() - start_time
        print(f"\n[SUCCESS] {config['name']} 数据生成完成!")
        print(f"[TIME] 总耗时: {total_time:.2f} 秒")

        # 保存合并结果（如果启用）
        if self.merge_results:
            self.save_merged_results(chart_set_name)

        return True

    def validate_result(self, result):
        """验证结果的有效性"""
        required_fields = ['total_cost', 'execution_time', 'load_balance', 'price_efficiency']

        for field in required_fields:
            if field not in result or result[field] is None:
                return False
            if not np.isfinite(result[field]):
                return False

        return True

    def calculate_average_results(self, results):
        """计算多次运行的平均结果"""
        if not results:
            return {}

        avg_result = {}
        numeric_fields = ['total_cost', 'execution_time', 'load_balance', 'price_efficiency',
                         'convergence_iteration', 'success_rate']

        for field in numeric_fields:
            values = [r.get(field, 0) for r in results if field in r and r[field] is not None]
            if values:
                avg_result[field] = np.mean(values)
                avg_result[f"{field}_std"] = np.std(values)
            else:
                avg_result[field] = 0
                avg_result[f"{field}_std"] = 0

        # 保留非数字字段
        for field in ['algorithm', 'parameters']:
            if field in results[0]:
                avg_result[field] = results[0][field]

        return avg_result

    def save_results(self, algorithm, chart_set_name, results):
        """保存结果到JSON文件"""
        result_data = {
            'algorithm': algorithm,
            'results': results
        }

        if self.merge_results:
            # 合并模式：将结果添加到合并数据中
            if chart_set_name not in self.merged_data:
                self.merged_data[chart_set_name] = {
                    'chart_set': chart_set_name,
                    'config': CHART_CONFIGS[chart_set_name],
                    'timestamp': datetime.now().isoformat(),
                    'algorithms': {}
                }

            self.merged_data[chart_set_name]['algorithms'][algorithm] = result_data
            print(f"  [CACHED] 数据已缓存: {algorithm}")
        else:
            # 独立文件模式
            filename = f"{chart_set_name}_{algorithm}_results.json"
            filepath = os.path.join(self.output_dir, filename)

            data = {
                'algorithm': algorithm,
                'chart_set': chart_set_name,
                'config': CHART_CONFIGS[chart_set_name],
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'total_data_points': len(results)
            }

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
                print(f"  [SAVED] 数据已保存: {filename}")
            except Exception as e:
                print(f"[ERROR] 保存失败: {e}")

    def save_merged_results(self, chart_set_name=None):
        """
        保存合并的结果到文件

        参数:
            chart_set_name: 指定要保存的图表集，如果为None则保存所有
        """
        if not self.merge_results or not self.merged_data:
            return

        chart_sets_to_save = [chart_set_name] if chart_set_name else list(self.merged_data.keys())

        for cs_name in chart_sets_to_save:
            if cs_name not in self.merged_data:
                continue

            filename = f"{cs_name}_bcbo_comparison.json"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.merged_data[cs_name], f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

                algo_count = len(self.merged_data[cs_name]['algorithms'])
                print(f"\n[SAVED] 合并文件已保存: {filename} (包含{algo_count}个算法)")

            except Exception as e:
                print(f"[ERROR] 保存合并文件失败: {e}")

    def cleanup(self):
        """清理临时文件和目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"\n[CLEANUP] 已删除临时目录: {self.temp_dir}")
            except Exception as e:
                print(f"[WARNING] 无法删除临时目录: {e}")

    def __del__(self):
        """析构函数，自动清理"""
        if hasattr(self, 'use_temp_dir') and self.use_temp_dir:
            self.cleanup()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BCBO vs BCBO-GA 性能对比数据生成工具')
    parser.add_argument('--chart-set', type=int, choices=[1, 2, 3, 4],
                       help='指定要生成数据的图表集编号')
    parser.add_argument('--algorithm', type=str, choices=['BCBO', 'BCBO-GA'],
                       help='指定特定算法')
    parser.add_argument('--all', action='store_true',
                       help='生成所有图表集的数据')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--merge-results', action='store_true', default=True,
                       help='合并同一图表集的所有算法结果到一个文件 (默认: 启用)')
    parser.add_argument('--no-merge', action='store_true',
                       help='禁用结果合并，每个算法单独保存文件')
    parser.add_argument('--use-temp-dir', action='store_true',
                       help='使用临时目录（程序结束后自动清理）')

    args = parser.parse_args()

    # 处理merge_results参数
    merge_results = args.merge_results and not args.no_merge

    print("="*80)
    print("BCBO vs BCBO-GA 性能对比数据生成工具")
    print("="*80)
    print(f"随机种子: {args.seed}")
    print(f"结果合并: {'启用' if merge_results else '禁用'}")
    print(f"临时目录: {'启用' if args.use_temp_dir else '禁用'}")
    print("="*80)

    generator = BCBOComparisonGenerator(
        random_seed=args.seed,
        use_temp_dir=args.use_temp_dir,
        merge_results=merge_results
    )

    try:
        if args.chart_set:
            chart_set_name = f"chart_set_{args.chart_set}"
            algorithms = [args.algorithm] if args.algorithm else None

            print(f"[TARGET] 生成数据: chart set {args.chart_set}")
            if args.algorithm:
                print(f"[ALGORITHM] 指定算法: {args.algorithm}")

            success = generator.generate_data_for_chart_set(chart_set_name, algorithms)

            sys.exit(0 if success else 1)

        elif args.all:
            print("[START] 生成所有图表集的数据")

            for chart_set_num in range(1, 5):
                chart_set_name = f"chart_set_{chart_set_num}"
                success = generator.generate_data_for_chart_set(chart_set_name)

                if chart_set_num < 4:
                    print(f"\n[WAIT] 等待3秒后继续...")
                    time.sleep(3)

            print("\n[SUCCESS] 所有数据生成完成!")

        else:
            # 交互式模式
            while True:
                print("\n请选择操作:")
                print("1. 生成图表集1的数据 (迭代次数 vs 性能指标 - 小规模)")
                print("2. 生成图表集2的数据 (任务规模 vs 成本 - 中等规模)")
                print("3. 生成图表集3的数据 (迭代次数 vs 性能指标 - 大规模)")
                print("4. 生成图表集4的数据 (任务规模 vs 成本 - 超大规模)")
                print("5. 生成所有图表集的数据")
                print("0. 退出")
                print("-" * 60)

                try:
                    choice = input("请输入选项 (0-5): ").strip()
                    choice = int(choice)

                    if choice == 0:
                        print("[EXIT] 退出程序")
                        break
                    elif choice in [1, 2, 3, 4]:
                        chart_set_name = f"chart_set_{choice}"
                        generator.generate_data_for_chart_set(chart_set_name)
                    elif choice == 5:
                        for chart_set_num in range(1, 5):
                            chart_set_name = f"chart_set_{chart_set_num}"
                            generator.generate_data_for_chart_set(chart_set_name)

                            if chart_set_num < 4:
                                print(f"\n[WAIT] 等待3秒后继续...")
                                time.sleep(3)

                        print("\n[SUCCESS] 所有数据生成完成!")
                    else:
                        print("[ERROR] 无效选项，请重新选择")

                except (ValueError, KeyboardInterrupt):
                    print("\n[EXIT] 退出程序")
                    break

    finally:
        # 确保清理
        if args.use_temp_dir:
            print("\n[INFO] 程序即将退出，临时文件将被清理...")


if __name__ == "__main__":
    main()
