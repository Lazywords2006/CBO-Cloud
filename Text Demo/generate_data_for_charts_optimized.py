#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的图表数据生成脚本
========================================
为四个图表集分别生成所需的数据，支持单独或批量生成

优化点：
1. 改进算法调用确保数据一致性
2. 增强错误处理和验证
3. 提供算法可用性检查
4. 优化收敛数据提取

BCBO-GA版本同步 (2025-12-01 v2.3最终版):
========================================
本脚本通过 real_algorithm_integration.py 自动使用BCBO-GA v2.3负载均衡增强版：
- 版本：v2.3 负载均衡增强版（最终版）
- 特性：三段式混合自适应参数、GA智能交叉、2-opt局部搜索、温和负载均衡
- 性能：综合+0.04%，超大规模(M=1000-5000)+0.09%（超越-0.50%目标）
- 核心优化：针对超大规模提高参数下限，增强负载均衡优化能力

v2.3关键改进：
- 交叉率(M=5000): 0.60 → 0.651 (+8.5%)
- 变异率(M=5000): 0.06 → 0.0695 (+15.8%)
- 局部搜索强度: max_iters 40→50, reassign_num 12→15 (+25%)
- 负载均衡阈值: 1.4 → 1.45 (触发更多修复)

使用说明：
- 通过 RealAlgorithmIntegrator 调用所有算法确保一致性
- BCBO-GA自动使用v2.3自适应参数机制（adaptive_params=True）
- 所有算法共享相同的问题实例(execution_time矩阵)确保公平对比
- 固定随机种子(seed=42)确保可重现性
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
    print(f"[OK] Added scripts path to sys.path: {SCRIPTS_DIR}")

# 检查real_algorithm_integration.py是否存在
integration_file = os.path.join(SCRIPTS_DIR, 'real_algorithm_integration.py')
print(f"[INFO] Checking integration file: {integration_file}")
print(f"[INFO] Integration file exists: {os.path.exists(integration_file)}")

try:
    from real_algorithm_integration import RealAlgorithmIntegrator
    INTEGRATOR_AVAILABLE = True
    print("[OK] Real algorithm integration module imported successfully")
    print("[INFO] BCBO-GA使用v1.0 GA增强版本 (2025-11-30)")
except ImportError as e:
    print(f"[ERROR] Failed to import real algorithm integration module: {e}")
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
        'name': '图表集1 - 迭代次数 vs 性能指标',
        'type': 'iterations',
        'variable_param': 'iterations',
        'values': list(range(5, 101, 5)),
        'fixed_params': {
            'M': 100,
            'N': 20,
            'n': 50
        },
        'runs_per_point': 1  # 测试模式：单次运行，快速验证N=20配置
    },
    'chart_set_2': {
        'name': '图表集2 - 任务规模 vs 成本',
        'type': 'task_scale',
        'variable_param': 'M',
        'values': list(range(100, 1001, 100)),
        'fixed_params': {
            'iterations': 80,
            'N': 20,  # 统一为20，解决高维空间DE失效问题
            'n': 100  # 方案2: 从50增加到100，适应中等规模问题
        },
        'runs_per_point': 1  # 测试模式：单次运行，快速验证N=20配置
    },
    'chart_set_3': {
        'name': '图表集3 - 迭代次数 vs 性能指标 (大规模1000任务)',
        'type': 'iterations_2',
        'variable_param': 'iterations',
        'values': list(range(5, 101, 5)),
        'fixed_params': {
            'M': 1000,
            'N': 20,  # 统一为20，解决高维空间DE失效问题
            'n': 150  # 方案2: 从50增加到150，适应大规模问题
        },
        'runs_per_point': 1  # 测试模式：单次运行，快速验证N=20配置
    },
    'chart_set_4': {
        'name': '图表集4 - 任务规模 vs 成本 (大规模)',
        'type': 'large_scale',
        'variable_param': 'M',
        'values': list(range(1000, 5001, 1000)),
        'fixed_params': {
            'iterations': 50,
            'N': 20,  # 统一为20，解决高维空间DE失效问题
            'n': 200  # 方案2: 从50增加到200，适应超大规模问题
        },
        'runs_per_point': 1  # 测试模式：单次运行，快速验证N=20配置
    }
}

# 所有算法列表
ALGORITHMS = ['BCBO', 'GA', 'PSO', 'ACO', 'FA', 'CS', 'GWO', 'BCBO-GA']


class OptimizedDataGenerator:
    """优化的数据生成器"""

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
            self.temp_dir = tempfile.mkdtemp(prefix='chart_data_')
            self.output_dir = self.temp_dir
            print(f"[INFO] 使用临时目录: {self.output_dir}")
        else:
            self.output_dir = os.path.join(BASE_DIR, 'RAW_data')
            os.makedirs(self.output_dir, exist_ok=True)
            self.temp_dir = None

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
        """验证所有算法的可用性"""
        global ALGORITHMS  # 必须在使用ALGORITHMS之前声明

        print(f"\n[INFO] 验证算法可用性...")
        available_algorithms = []
        unavailable_algorithms = []

        for algo in ALGORITHMS:
            try:
                # 尝试创建算法实例
                test_result = self.integrator.run_algorithm(algo, {
                    'M': 10, 'N': 3, 'n': 5, 'iterations': 5
                })

                if test_result and 'best_solution' in test_result:
                    available_algorithms.append(algo)
                    print(f"  [OK] {algo} 可用")
                else:
                    unavailable_algorithms.append(algo)
                    print(f"  [WARN] {algo} 返回结果无效")

            except Exception as e:
                unavailable_algorithms.append(algo)
                print(f"  [ERROR] {algo} 不可用: {e}")

        print(f"\n[SUMMARY] 可用算法: {len(available_algorithms)}/{len(ALGORITHMS)}")
        print(f"  可用: {', '.join(available_algorithms)}")
        if unavailable_algorithms:
            print(f"  不可用: {', '.join(unavailable_algorithms)}")

        # 更新算法列表
        ALGORITHMS = available_algorithms

    def calculate_metrics_from_solution(self, solution, algorithm_name, config):
        """
        从任务分配方案计算详细的性能指标（改进版）

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

            # 如果是BCBO-GA,使用其父类BCBO的方法
            # 注意: BCBO-GA继承自BCBO_CloudScheduler,直接使用即可
            # if algorithm_name == 'BCBO-GA' and hasattr(algorithm_instance, 'bcbo'):
            #     algorithm_instance = algorithm_instance.bcbo

            if algorithm_instance and hasattr(algorithm_instance, 'get_detailed_metrics'):
                metrics = algorithm_instance.get_detailed_metrics(solution)
                return {
                    'total_cost': metrics.get('total_cost', 0),
                    'execution_time': metrics.get('makespan', 0),
                    'load_balance': 1.0 - min(metrics.get('load_imbalance', 0), 1.0),
                    'price_efficiency': 1.0 / (metrics.get('total_cost', 1) + 1e-6)
                }
        except Exception as e:
            print(f"\n    [DEBUG] get_detailed_metrics失败 ({algorithm_name}): {e}")
            pass  # 静默失败，使用备用方法

        # 备用方法：直接计算
        try:
            algorithm_instance = self.integrator.algorithms.get(algorithm_name)

            # 如果是BCBO-GA,使用其父类BCBO的方法
            # 注意: BCBO-GA继承自BCBO_CloudScheduler,直接使用即可
            # if algorithm_name == 'BCBO-GA' and hasattr(algorithm_instance, 'bcbo'):
            #     algorithm_instance = algorithm_instance.bcbo

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
            print(f"\n    [ERROR] Metrics calculation failed: {e}")
            return {'total_cost': 0, 'execution_time': 0, 'load_balance': 0, 'price_efficiency': 0}

    # REMOVED: smooth_and_interpolate_data 函数已删除
    # 原因：违反学术诚信，人为平滑和插值数据
    # 修改日期：2025-11-19
    # 修改人：根据预审稿报告要求删除

    def validate_generated_data(self, chart_set_name, algorithm_results):
        """
        验证生成的数据是否合理（新增：数据质量检查）

        参数:
            chart_set_name: 图表集名称
            algorithm_results: 算法结果字典 {algorithm_name: [data_points]}

        返回:
            (is_valid, warnings) - 验证结果和警告列表
        """
        warnings = []
        errors = []

        print(f"\n[INFO] 验证 {chart_set_name} 的数据质量...")

        for algo_name, data_points in algorithm_results.items():
            if not data_points:
                errors.append(f"[ERROR] {algo_name} 没有数据点")
                continue

            # 检查1: Load Balance 范围检查
            for i, point in enumerate(data_points):
                lb = point.get('load_balance', 0)

                # 负载均衡应该在[0, 1]范围内
                if lb < 0 or lb > 1:
                    warnings.append(f"[WARN] {algo_name} 第{i+1}个点负载均衡超出范围: {lb:.4f}")

                # ACO以外的算法，负载均衡不应该太低
                if lb < 0.1 and algo_name not in ['ACO']:
                    warnings.append(f"[WARN] {algo_name} 第{i+1}个点负载均衡异常低: {lb:.4f}")

            # 检查2: 成本和时间为正数
            for i, point in enumerate(data_points):
                if point.get('total_cost', 0) <= 0:
                    errors.append(f"[ERROR] {algo_name} 第{i+1}个点总成本为非正数: {point.get('total_cost')}")
                if point.get('execution_time', 0) <= 0:
                    errors.append(f"[ERROR] {algo_name} 第{i+1}个点执行时间为非正数: {point.get('execution_time')}")

            # 检查3: 数据一致性（同一算法，成本不应该突然大幅波动）
            if len(data_points) > 1:
                for i in range(1, len(data_points)):
                    prev_cost = data_points[i-1].get('total_cost', 1)
                    curr_cost = data_points[i].get('total_cost', 1)

                    # 如果成本突然增加超过50%或减少超过50%（可能有问题）
                    change_ratio = abs(curr_cost - prev_cost) / prev_cost
                    if change_ratio > 0.5:
                        warnings.append(f"[WARN] {algo_name} 第{i}→{i+1}点成本变化过大: {change_ratio*100:.1f}%")

        # 检查4: BCBO-GA vs BCBO 对比分析
        if 'BCBO' in algorithm_results and 'BCBO-GA' in algorithm_results:
            bcbo_data = algorithm_results['BCBO']
            bcbo_ga_data = algorithm_results['BCBO-GA']

            if bcbo_data and bcbo_ga_data:
                # 比较最后一个数据点
                bcbo_final = bcbo_data[-1]
                bcbo_ga_final = bcbo_ga_data[-1]

                bcbo_cost = bcbo_final.get('total_cost', 1)
                bcbo_ga_cost = bcbo_ga_final.get('total_cost', 1)

                improvement = (bcbo_cost - bcbo_ga_cost) / bcbo_cost * 100

                if improvement < -5:  # BCBO-GA成本高于BCBO超过5%
                    warnings.append(f"[WARN] BCBO-GA成本显著高于BCBO: {improvement:.2f}%")
                elif improvement > 1:  # BCBO-GA有明显改进
                    print(f"  [INFO] ✓ BCBO-GA成本优于BCBO: {improvement:.2f}%")

                # 负载均衡对比
                bcbo_lb = bcbo_final.get('load_balance', 0)
                bcbo_ga_lb = bcbo_ga_final.get('load_balance', 0)
                lb_improvement = (bcbo_ga_lb - bcbo_lb) / (bcbo_lb + 1e-6) * 100

                if lb_improvement > 3:
                    print(f"  [INFO] ✓ BCBO-GA负载均衡优于BCBO: {lb_improvement:.2f}%")

        # 检查5: ACO负载均衡修复验证
        if 'ACO' in algorithm_results:
            aco_data = algorithm_results['ACO']
            if aco_data:
                aco_final_lb = aco_data[-1].get('load_balance', 0)

                if aco_final_lb < 0.1:
                    warnings.append(f"[WARN] ACO负载均衡仍然很低: {aco_final_lb:.4f} (修复可能未生效)")
                elif aco_final_lb > 0.3:
                    print(f"  [INFO] ✓ ACO负载均衡修复成功: {aco_final_lb:.4f}")
                else:
                    print(f"  [INFO] ACO负载均衡: {aco_final_lb:.4f} (有改善但仍偏低)")

        # 汇总验证结果
        print(f"\n[VALIDATION] 验证完成:")
        print(f"  - 错误数: {len(errors)}")
        print(f"  - 警告数: {len(warnings)}")

        if errors:
            print(f"\n[ERRORS]:")
            for error in errors[:10]:  # 最多显示10个错误
                print(f"  {error}")

        if warnings:
            print(f"\n[WARNINGS]:")
            for warning in warnings[:10]:  # 最多显示10个警告
                print(f"  {warning}")

        is_valid = len(errors) == 0

        return is_valid, warnings + errors

    def generate_convergence_data_from_history(self, algorithm_name, config):
        """
        从convergence_history生成收敛曲线数据（改进版）

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
                # v3.2方法：使用固定seed确保所有算法比较相同的问题实例
                # 每次运行使用相同的基础seed，保证BCBO和BCBO-GA解决同一个问题
                run_seed = 42 + run  # 简单的确定性种子：42, 43, 44, ...

                np.random.seed(run_seed)
                random.seed(run_seed)

                print(f"\n    [DEBUG] Run {run+1} seed: {run_seed}")

                # 将random_seed添加到params中（v3.2方法）
                params_with_seed = fixed_params.copy()
                params_with_seed['random_seed'] = run_seed

                # 运行算法（改进：捕获convergence_history）
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

                    # 诊断收敛情况
                    if len(history) > 20:
                        unique_count = 0
                        for i in range(1, min(len(history), 100)):
                            if isinstance(history[i], dict) and isinstance(history[i-1], dict):
                                curr_fit = history[i].get('best_fitness') or history[i].get('global_best_fitness') or 0
                                prev_fit = history[i-1].get('best_fitness') or history[i-1].get('global_best_fitness') or 0

                                if abs(curr_fit - prev_fit) > abs(prev_fit) * 1e-5:
                                    unique_count += 1

                        convergence_rate = unique_count / min(len(history), 100) * 100

                        if unique_count == 0:
                            print(f"\n    [WARN] Run {run+1}: 算法未优化（所有迭代fitness相同）")
                        elif unique_count < 5:
                            print(f"\n    [WARN] Run {run+1}: 收敛过快（只有 {unique_count} 次变化）")

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
                    if iteration_idx < 5:
                        print(f"\n    [DEBUG] Iteration {iteration_idx+1}, 处理错误: {e}")
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

        # REMOVED: 数据平滑逻辑已删除，保留真实算法表现
        # 原因：违反学术诚信，人为修改实验数据
        # 修改日期：2025-11-19

        print(f"  [OK] 生成了 {len(algorithm_results)} 个收敛数据点（真实数据，未平滑）")
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
            print(f"[INFO] 使用收敛历史方法 (生成100个平滑数据点)")

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
                            # v3.2方法：使用固定seed确保所有算法比较相同的问题实例
                            # 对于相同的param_value，所有算法使用相同的seed
                            run_seed = 42 + param_value + run  # 基于参数值和运行次数的确定性种子

                            np.random.seed(run_seed)
                            random.seed(run_seed)

                            # 将random_seed添加到params中（v3.2方法）
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

    def validate_data(self, algorithm, chart_set_name, results):
        """验证生成的数据是否合理（增强版）"""
        warnings = []
        errors = []

        if not results or len(results) == 0:
            errors.append("结果为空")
            return False, warnings, errors

        # 计算合理范围（基于所有数据点）
        all_costs = [point.get('total_cost', 0) for point in results if point.get('total_cost', 0) > 0]
        all_times = [point.get('execution_time', 0) for point in results if point.get('execution_time', 0) > 0]

        if len(all_costs) > 0:
            avg_cost = np.mean(all_costs)
            std_cost = np.std(all_costs)
        else:
            avg_cost = 0
            std_cost = 0

        for i, point in enumerate(results):
            iteration = point.get('iteration') or point.get('task_count', i+1)

            # 检查Total Cost是否过低（新增：绝对值检查）
            total_cost = point.get('total_cost', 0)

            # 对于100任务20VM的配置，total_cost应该在合理范围内
            config = CHART_CONFIGS.get(chart_set_name, {})
            M = config.get('fixed_params', {}).get('M', 100)
            N = config.get('fixed_params', {}).get('N', 20)

            # 合理成本范围估算：每个任务至少消耗一定成本
            min_reasonable_cost = M * 0.5  # 每任务至少0.5成本
            max_reasonable_cost = M * N * 2  # 每任务最多N*2成本

            if total_cost < min_reasonable_cost and i > 5:
                errors.append(f"迭代{iteration}: Total Cost过低 ({total_cost:.2f} < {min_reasonable_cost:.2f})")
            elif total_cost > max_reasonable_cost:
                warnings.append(f"迭代{iteration}: Total Cost过高 ({total_cost:.2f} > {max_reasonable_cost:.2f})")

            # 检查Load Balance是否为0
            load_balance = point.get('load_balance', 0)
            if load_balance == 0 and i > 0:
                warnings.append(f"迭代{iteration}: Load Balance为0")

            # 检查execution_time是否过低
            execution_time = point.get('execution_time', 0)
            min_reasonable_time = M * 0.01  # 每任务至少0.01时间单位
            if execution_time < min_reasonable_time and i > 5:
                errors.append(f"迭代{iteration}: Execution Time过低 ({execution_time:.2f} < {min_reasonable_time:.2f})")

            # 检查相邻迭代间的突变
            if i > 0:
                prev_cost = results[i-1].get('total_cost', 0)
                if prev_cost > 0:
                    change_rate = abs(total_cost - prev_cost) / prev_cost
                    if change_rate > 0.3:
                        warnings.append(f"迭代{iteration}: Total Cost突变 {change_rate*100:.1f}%")

                    if prev_cost > min_reasonable_cost and total_cost < min_reasonable_cost:
                        errors.append(f"迭代{iteration}: Total Cost从{prev_cost:.4f}突降至{total_cost:.6f}")

        # 检查收敛性（新增）
        if len(results) >= 20:
            early_20_pct = int(len(results) * 0.2)
            late_20_pct_start = int(len(results) * 0.8)

            early_costs = [results[i]['total_cost'] for i in range(early_20_pct)]
            late_costs = [results[i]['total_cost'] for i in range(late_20_pct_start, len(results))]

            if len(early_costs) > 0 and len(late_costs) > 0:
                early_avg = np.mean(early_costs)
                late_avg = np.mean(late_costs)

                if abs(early_avg - late_avg) / early_avg < 0.001:
                    warnings.append(f"收敛曲线几乎平坦（变化<0.1%），可能未优化")

        is_valid = len(errors) == 0
        return is_valid, warnings, errors

    def save_results(self, algorithm, chart_set_name, results):
        """保存结果到JSON文件（包含验证）"""
        # 验证数据
        is_valid, warnings, errors = self.validate_data(algorithm, chart_set_name, results)

        result_data = {
            'algorithm': algorithm,
            'results': results,
            'validation': {
                'is_valid': is_valid,
                'warnings': warnings,
                'errors': errors
            }
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

            # 显示验证结果
            if is_valid:
                print(f"  [CACHED] 数据已缓存: {algorithm} [OK]")
            else:
                print(f"  [CACHED] 数据已缓存: {algorithm} [WARN] (有{len(errors)}个错误)")
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
                'total_data_points': len(results),
                'validation': {
                    'is_valid': is_valid,
                    'warnings': warnings,
                    'errors': errors
                }
            }

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

                # 显示验证结果
                if is_valid:
                    print(f"  [SAVED] 数据已保存: {filename} [OK]")
                else:
                    print(f"  [SAVED] 数据已保存: {filename} [WARN] (有{len(errors)}个错误)")

            except Exception as e:
                print(f"[ERROR] 保存失败: {e}")

        # 显示警告和错误
        if warnings:
            print(f"  [WARN] {len(warnings)}个警告")
            for w in warnings[:3]:
                print(f"    - {w}")

        if errors:
            print(f"  [ERROR] {len(errors)}个错误")
            for e in errors[:3]:
                print(f"    - {e}")

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

            filename = f"{cs_name}_merged_results.json"
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

    parser = argparse.ArgumentParser(description='优化的图表数据生成工具')
    parser.add_argument('--chart-set', type=int, choices=[1, 2, 3, 4],
                       help='指定要生成数据的图表集编号')
    parser.add_argument('--algorithm', type=str,
                       help='指定特定算法 (BCBO, GA, PSO, ACO, FA, CS, GWO)')
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
    print("优化的图表数据生成工具")
    print("="*80)
    print(f"随机种子: {args.seed}")
    print(f"结果合并: {'启用' if merge_results else '禁用'}")
    print(f"临时目录: {'启用' if args.use_temp_dir else '禁用'}")
    print("="*80)

    generator = OptimizedDataGenerator(
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

            # 最终保存所有合并结果
            if merge_results and not args.chart_set:
                generator.save_merged_results()

            sys.exit(0 if success else 1)

        elif args.all:
            print("[START] 生成所有图表集的数据")

            for chart_set_num in range(1, 5):
                chart_set_name = f"chart_set_{chart_set_num}"
                success = generator.generate_data_for_chart_set(chart_set_name)

                if chart_set_num < 4:
                    print(f"\n[WAIT] 等待5秒后继续...")
                    time.sleep(5)

            # 最终保存所有合并结果
            if merge_results:
                generator.save_merged_results()

            print("\n[SUCCESS] 所有数据生成完成!")

        else:
            # 交互式模式
            while True:
                print("\n请选择操作:")
                print("1. 生成图表集1的数据 (迭代次数 vs 性能指标)")
                print("2. 生成图表集2的数据 (任务规模 vs 成本)")
                print("3. 生成图表集3的数据 (迭代次数 vs 性能指标 - 大规模)")
                print("4. 生成图表集4的数据 (任务规模 vs 成本 - 大规模)")
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
                                print(f"\n[WAIT] 等待5秒后继续...")
                                time.sleep(5)

                        # 最终保存所有合并结果
                        if merge_results:
                            generator.save_merged_results()

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
            # 析构函数会自动清理


if __name__ == "__main__":
    main()
