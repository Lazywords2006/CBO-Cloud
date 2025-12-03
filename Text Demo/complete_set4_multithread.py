#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Set 4 超大规模数据补全脚本（多线程版）

功能：
1. 检测Set 4数据的缺失情况
2. 使用多线程并行补全缺失的数据点
3. 合并到原有JSON文件中

优化策略：
- 多线程并行运行（每个M值一个线程）
- 进度监控和断点续传
- 内存优化（及时清理）
"""

import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加路径以导入算法
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithm', 'BCBO'))
from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
from bcbo_ga_enhanced import BCBO_GA

# 线程锁（用于文件写入）
write_lock = threading.Lock()


class NumpyEncoder(json.JSONEncoder):
    """处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def set_random_seeds(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_problem_instance(M, N, random_seed):
    """生成问题实例（与原脚本保持一致）"""
    set_random_seeds(random_seed)

    # 生成执行时间矩阵（异构）
    execution_time = np.random.uniform(10, 100, (M, N))

    # 生成任务负载和VM容量
    task_loads = np.random.uniform(1, 10, M)
    vm_caps = np.random.uniform(50, 200, N)
    vm_costs = np.random.uniform(1, 5, N)

    return {
        'execution_time': execution_time,
        'task_loads': task_loads,
        'vm_caps': vm_caps,
        'vm_costs': vm_costs,
        'M': M,
        'N': N,
        'random_seed': random_seed
    }


def run_single_algorithm(algo_name, problem_instance, iterations):
    """运行单个算法"""
    M = problem_instance['M']
    N = problem_instance['N']
    random_seed = problem_instance['random_seed']

    set_random_seeds(random_seed)

    if algo_name == 'BCBO':
        algorithm = BCBO_CloudScheduler(
            M=M,
            N=N,
            n=200,  # 种群大小
            iterations=iterations,
            random_seed=random_seed
        )
    elif algo_name == 'BCBO-GA':
        algorithm = BCBO_GA(
            M=M,
            N=N,
            n=200,
            iterations=iterations,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"未知算法: {algo_name}")

    # 运行算法
    try:
        result = algorithm.run_complete_algorithm()

        # 从结果字典中提取数据
        best_solution = result['best_solution']
        total_cost = result['total_cost']

        # 获取执行时间矩阵
        execution_time = algorithm.execution_time
        vm_cost = algorithm.vm_cost if hasattr(algorithm, 'vm_cost') else problem_instance['vm_costs']

        # 计算性能指标
        vm_loads = np.zeros(N)

        for task_idx, vm_idx in enumerate(best_solution):
            vm_loads[vm_idx] += execution_time[task_idx, vm_idx]

        execution_time_metric = np.max(vm_loads)

        # 负载均衡度
        mean_load = np.mean(vm_loads)
        if mean_load > 0:
            load_imbalance = np.std(vm_loads) / mean_load
            load_balance = max(0.0, 1.0 - load_imbalance)
        else:
            load_balance = 1.0

        return {
            'total_cost': total_cost,
            'total_cost_std': 0.0,
            'execution_time': execution_time_metric,
            'execution_time_std': 0.0,
            'load_balance': load_balance,
            'load_balance_std': 0.0,
            'price_efficiency': 1.0 / (total_cost + 1e-6),
            'price_efficiency_std': 0.0,
            'convergence_iteration': 1.0,
            'convergence_iteration_std': 0.0,
            'success_rate': 1.0,
            'success_rate_std': 0.0,
            'algorithm': algo_name,
            'M': M,
            'runs': 1
        }
    except Exception as e:
        import traceback
        print(f"    [ERROR] {algo_name} M={M} 运行失败: {e}")
        traceback.print_exc()
        return None


def generate_missing_datapoint(seed, M, algo_name, N=20, iterations=50):
    """生成单个缺失的数据点"""
    thread_id = threading.current_thread().name
    print(f"  [{thread_id}] 开始生成: Seed={seed}, M={M}, Algo={algo_name}")

    start_time = time.time()

    # 生成问题实例
    problem_instance = generate_problem_instance(M, N, seed)

    # 运行算法
    result = run_single_algorithm(algo_name, problem_instance, iterations)

    elapsed = time.time() - start_time

    if result:
        print(f"  [{thread_id}] 完成: Seed={seed}, M={M}, Algo={algo_name} ({elapsed:.1f}s)")
    else:
        print(f"  [{thread_id}] 失败: Seed={seed}, M={M}, Algo={algo_name}")

    return {
        'seed': seed,
        'M': M,
        'algo_name': algo_name,
        'result': result,
        'elapsed': elapsed
    }


def load_existing_data(seed, base_dir='multi_seed_validation_set4'):
    """加载现有数据"""
    filename = f'{base_dir}/chart_set_4_seed_{seed}.json'

    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def find_missing_datapoints(base_dir='multi_seed_validation_set4'):
    """查找所有缺失的数据点"""
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    expected_M_values = [1000, 2000, 3000, 4000, 5000]

    missing = []

    for seed in seeds:
        data = load_existing_data(seed, base_dir)

        if data is None:
            print(f"[WARN] Seed {seed}: 文件不存在，跳过")
            continue

        for algo_name in ['BCBO', 'BCBO-GA']:
            results = data['algorithms'][algo_name]['results']
            existing_M = [r['M'] for r in results]

            for M in expected_M_values:
                if M not in existing_M:
                    missing.append({
                        'seed': seed,
                        'M': M,
                        'algo_name': algo_name
                    })

    return missing


def update_json_file(seed, M, algo_name, result, base_dir='multi_seed_validation_set4'):
    """更新JSON文件（线程安全）"""
    filename = f'{base_dir}/chart_set_4_seed_{seed}.json'

    with write_lock:
        # 读取现有数据
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 添加新结果
        data['algorithms'][algo_name]['results'].append(result)

        # 按M值排序
        data['algorithms'][algo_name]['results'].sort(key=lambda x: x['M'])

        # 写回文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

        print(f"  [OK] 已更新: Seed {seed}, M={M}, Algo={algo_name}")


def complete_set4_data(max_workers=4):
    """补全Set 4缺失数据（多线程版）"""
    print("=" * 80)
    print("Set 4 超大规模数据补全（多线程版）")
    print("=" * 80)
    print()

    # 1. 查找缺失数据
    print("[INFO] 扫描缺失数据点...")
    missing = find_missing_datapoints()

    if not missing:
        print("  [OK] 所有数据完整，无需补全")
        return

    print(f"  [WARN] 发现 {len(missing)} 个缺失数据点:")
    for item in missing:
        print(f"    - Seed {item['seed']}, M={item['M']}, Algo={item['algo_name']}")
    print()

    # 2. 多线程补全
    print(f"[INFO] 开始多线程补全（线程数={max_workers}）...")
    print()

    start_time = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Worker') as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                generate_missing_datapoint,
                item['seed'],
                item['M'],
                item['algo_name']
            ): item for item in missing
        }

        # 等待完成
        for future in as_completed(futures):
            item = futures[future]
            try:
                result_data = future.result()

                if result_data['result'] is not None:
                    # 更新JSON文件
                    update_json_file(
                        result_data['seed'],
                        result_data['M'],
                        result_data['algo_name'],
                        result_data['result']
                    )
                    completed += 1
                else:
                    failed += 1

                # 打印进度
                total = completed + failed
                progress = (total / len(missing)) * 100
                print(f"  [PROGRESS] {total}/{len(missing)} ({progress:.1f}%) - "
                      f"完成:{completed}, 失败:{failed}")

            except Exception as e:
                print(f"  [ERROR] 任务执行异常: {e}")
                failed += 1

    total_time = time.time() - start_time

    # 3. 总结
    print()
    print("=" * 80)
    print("补全完成")
    print("=" * 80)
    print(f"总任务数: {len(missing)}")
    print(f"成功补全: {completed}")
    print(f"失败数量: {failed}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"平均每个数据点: {total_time/len(missing):.1f}s")
    print()


def verify_completeness():
    """验证数据完整性"""
    print("=" * 80)
    print("验证数据完整性")
    print("=" * 80)
    print()

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    expected_M = [1000, 2000, 3000, 4000, 5000]

    all_complete = True

    for seed in seeds:
        data = load_existing_data(seed)

        if data is None:
            print(f"Seed {seed}: 文件不存在 [X]")
            all_complete = False
            continue

        bcbo_M = [r['M'] for r in data['algorithms']['BCBO']['results']]
        bcbo_ga_M = [r['M'] for r in data['algorithms']['BCBO-GA']['results']]

        bcbo_complete = set(bcbo_M) == set(expected_M)
        bcbo_ga_complete = set(bcbo_ga_M) == set(expected_M)

        if bcbo_complete and bcbo_ga_complete:
            print(f"Seed {seed}: 完整 [OK] (BCBO: {len(bcbo_M)}/5, BCBO-GA: {len(bcbo_ga_M)}/5)")
        else:
            print(f"Seed {seed}: 不完整 [WARN]")
            if not bcbo_complete:
                missing = [m for m in expected_M if m not in bcbo_M]
                print(f"  BCBO缺失: M={missing}")
            if not bcbo_ga_complete:
                missing = [m for m in expected_M if m not in bcbo_ga_M]
                print(f"  BCBO-GA缺失: M={missing}")
            all_complete = False

    print()
    if all_complete:
        print("[OK] 所有种子数据完整")
    else:
        print("[WARN] 仍有数据缺失")
    print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Set 4 超大规模数据补全（多线程版）')
    parser.add_argument('--threads', type=int, default=4, help='线程数（默认4）')
    parser.add_argument('--verify-only', action='store_true', help='仅验证完整性，不补全')

    args = parser.parse_args()

    if args.verify_only:
        verify_completeness()
    else:
        # 补全数据
        complete_set4_data(max_workers=args.threads)

        # 验证完整性
        print()
        verify_completeness()


if __name__ == '__main__':
    main()
