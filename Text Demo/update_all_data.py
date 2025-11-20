#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新RAW_data中所有4组数据的BCBO和BCBO-DE部分
保持其他算法数据不变
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

from real_algorithm_integration import RealAlgorithmIntegrator

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


def update_bcbo_data_for_chart_set(chart_set_name, config, integrator):
    """
    更新指定chart_set的BCBO和BCBO-DE数据

    参数:
        chart_set_name: 图表集名称 (如'chart_set_1')
        config: 配置字典
        integrator: RealAlgorithmIntegrator实例
    """
    print(f"\n{'='*80}")
    print(f"正在更新 {chart_set_name}: {config['name']}")
    print(f"{'='*80}")

    # 读取现有数据
    raw_data_file = os.path.join(BASE_DIR, 'RAW_data', f'{chart_set_name}_merged_results.json')

    if not os.path.exists(raw_data_file):
        print(f"[错误] 找不到数据文件: {raw_data_file}")
        return False

    with open(raw_data_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    print(f"[信息] 已加载现有数据,包含 {len(existing_data['algorithms'])} 个算法")

    # 只更新BCBO和BCBO-DE
    algorithms_to_update = ['BCBO', 'BCBO-DE']

    for algo_name in algorithms_to_update:
        if algo_name not in integrator.available_algorithms:
            print(f"[警告] {algo_name} 不可用,跳过")
            continue

        print(f"\n[{algo_name}] 开始重新生成数据...")

        # 根据chart_set类型确定参数
        if config['type'] in ['iterations', 'iterations_2']:
            # 迭代次数变化实验
            updated_results = []

            for iter_value in config['values']:
                print(f"  处理迭代次数 {iter_value}...", end='', flush=True)

                params = {
                    'M': config['fixed_params']['M'],
                    'N': config['fixed_params']['N'],
                    'n': config['fixed_params']['n'],
                    'iterations': iter_value,
                    'random_seed': 42
                }

                result = integrator.run_algorithm(algo_name, params)

                if result:
                    # 提取收敛历史中的每个迭代点
                    convergence_history = result.get('convergence_history', [])

                    for record in convergence_history:
                        if isinstance(record, dict):
                            updated_results.append({
                                'iteration': record.get('iteration', 0),
                                'total_cost': result.get('total_cost', 0),
                                'execution_time': result.get('execution_time', 0),
                                'load_balance': result.get('load_balance', 0),
                                'price_efficiency': result.get('price_efficiency', 0),
                                'algorithm': algo_name,
                                'best_fitness': record.get('best_fitness', 0)
                            })

                    print(f" OK (fitness={result['best_fitness']:.2f})")
                else:
                    print(f" Failed")

        elif config['type'] == 'task_scale':
            # 任务规模变化实验
            updated_results = []

            for M_value in config['values']:
                print(f"  处理任务数 {M_value}...", end='', flush=True)

                # 运行多次取平均
                runs_per_point = config.get('runs_per_point', 5)
                run_results = []

                for run_idx in range(runs_per_point):
                    params = {
                        'M': M_value,
                        'N': config['fixed_params']['N'],
                        'n': config['fixed_params']['n'],
                        'iterations': config['fixed_params']['iterations'],
                        'random_seed': 42 + run_idx
                    }

                    result = integrator.run_algorithm(algo_name, params)
                    if result:
                        run_results.append(result)

                # 计算统计量
                if run_results:
                    avg_result = {
                        'total_cost': np.mean([r['total_cost'] for r in run_results]),
                        'total_cost_std': np.std([r['total_cost'] for r in run_results]),
                        'execution_time': np.mean([r['execution_time'] for r in run_results]),
                        'execution_time_std': np.std([r['execution_time'] for r in run_results]),
                        'load_balance': np.mean([r['load_balance'] for r in run_results]),
                        'load_balance_std': np.std([r['load_balance'] for r in run_results]),
                        'price_efficiency': np.mean([r['price_efficiency'] for r in run_results]),
                        'price_efficiency_std': np.std([r['price_efficiency'] for r in run_results]),
                        'convergence_iteration': np.mean([r.get('convergence_iteration', 0) for r in run_results]),
                        'convergence_iteration_std': np.std([r.get('convergence_iteration', 0) for r in run_results]),
                        'success_rate': np.mean([r.get('success_rate', 1.0) for r in run_results]),
                        'success_rate_std': np.std([r.get('success_rate', 1.0) for r in run_results]),
                        'algorithm': algo_name,
                        'M': M_value,
                        'runs': runs_per_point
                    }
                    updated_results.append(avg_result)
                    print(f" OK (avg_fitness={np.mean([r['best_fitness'] for r in run_results]):.2f})")
                else:
                    print(f" Failed")

        else:
            print(f"[警告] 未知的图表类型: {config['type']}")
            continue

        # 更新到现有数据中
        existing_data['algorithms'][algo_name] = {
            'algorithm': algo_name,
            'results': updated_results
        }

        print(f"[{algo_name}] 数据更新完成,共 {len(updated_results)} 个数据点")

    # 添加更新时间戳
    existing_data['updated_timestamp'] = datetime.now().isoformat()
    existing_data['update_note'] = '修复: best_fitness现在记录comprehensive_fitness而非makespan'

    # 保存更新后的数据
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n[成功] 已保存到: {raw_data_file}")
    return True


def export_to_csv_and_excel(chart_set_name):
    """从JSON导出CSV和Excel格式"""
    # 读取JSON数据
    raw_data_file = os.path.join(BASE_DIR, 'RAW_data', f'{chart_set_name}_merged_results.json')

    with open(raw_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 合并所有算法的结果
    all_rows = []
    for algo_name, algo_data in data['algorithms'].items():
        for result in algo_data['results']:
            all_rows.append(result)

    if not all_rows:
        print(f"[警告] {chart_set_name} 没有数据可导出")
        return

    # 创建DataFrame
    df = pd.DataFrame(all_rows)

    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存CSV
    csv_file = os.path.join(BASE_DIR, 'publication_charts',
                           f'{chart_set_name}_comparison_data_{timestamp}.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  [导出] CSV: {os.path.basename(csv_file)}")

    # 保存Excel
    excel_file = csv_file.replace('.csv', '.xlsx')
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"  [导出] Excel: {os.path.basename(excel_file)}")


def main():
    """主函数"""
    print("="*80)
    print("更新RAW_data中BCBO和BCBO-DE的数据".center(80))
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 初始化集成器
    integrator = RealAlgorithmIntegrator()

    # 四组图表配置
    chart_configs = {
        'chart_set_1': {
            'name': '图表集1 - 迭代次数 vs 性能指标',
            'type': 'iterations',
            'variable_param': 'iterations',
            'values': list(range(5, 101, 5)),
            'fixed_params': {'M': 100, 'N': 20, 'n': 50},
            'runs_per_point': 1
        },
        'chart_set_2': {
            'name': '图表集2 - 任务规模 vs 成本',
            'type': 'task_scale',
            'variable_param': 'M',
            'values': list(range(100, 1001, 100)),
            'fixed_params': {'iterations': 80, 'N': 50, 'n': 50},
            'runs_per_point': 5
        },
        'chart_set_3': {
            'name': '图表集3 - 迭代次数 vs 性能指标 (大规模1000任务)',
            'type': 'iterations_2',
            'variable_param': 'iterations',
            'values': list(range(5, 101, 5)),
            'fixed_params': {'M': 1000, 'N': 50, 'n': 50},
            'runs_per_point': 1
        },
        'chart_set_4': {
            'name': '图表集4 - 大规模 vs 成本 (超大规模)',
            'type': 'task_scale',
            'variable_param': 'M',
            'values': [1000, 2000, 3000, 4000, 5000],
            'fixed_params': {'iterations': 150, 'N': 100, 'n': 80},
            'runs_per_point': 5
        }
    }

    # 更新所有4个chart sets
    success_count = 0
    for chart_set_name, config in chart_configs.items():
        success = update_bcbo_data_for_chart_set(chart_set_name, config, integrator)

        if success:
            # 导出CSV和Excel
            print(f"\n[导出] 正在导出 {chart_set_name} 数据...")
            export_to_csv_and_excel(chart_set_name)
            success_count += 1

        print()  # 空行分隔

    print("="*80)
    print(f"数据更新完成! 成功: {success_count}/4".center(80))
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n修复说明:")
    print("  - best_fitness 现在正确记录 comprehensive_fitness (越大越好)")
    print("  - execution_time 记录 makespan (越小越好)")
    print("  - 只更新了BCBO和BCBO-DE,其他算法数据保持不变")

    print("\n生成的文件:")
    print("  - RAW_data/*.json (4个JSON文件已更新)")
    print("  - publication_charts/*_YYYYMMDD_HHMMSS.csv (新生成)")
    print("  - publication_charts/*_YYYYMMDD_HHMMSS.xlsx (新生成)")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 用户取消操作")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
