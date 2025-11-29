#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实算法集成模块（完整版）- Scripts版本
=====================================
为数据生成工具提供统一的算法调用接口

支持的算法:
- BCBO: Coyote-Bald Eagle Co-optimization
- GA: Genetic Algorithm
- PSO: Particle Swarm Optimization
- ACO: Ant Colony Optimization
- FA: Firefly Algorithm
- CS: Cuckoo Search
- GWO: Grey Wolf Optimizer
- BCBO-GA: BCBO-GA Staged Hybrid
"""

import sys
import os
import numpy as np
import random
from typing import Dict, Optional

# 添加路径以导入算法
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(CURRENT_DIR, '..')
sys.path.insert(0, PARENT_DIR)

# 添加算法目录到路径 - 修改为新的路径结构
# 从 Text Demo/scripts/ 向上找到算法目录
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..', '..')  # 到达 "混合算法优化方案"
ALGORITHM_ROOT = os.path.join(PROJECT_ROOT, 'algorithm')

BCBO_DIR = os.path.join(ALGORITHM_ROOT, 'BCBO')
BCBO_DE_DIR = os.path.join(ALGORITHM_ROOT, 'BCBO-DE-Fusion')
MBCBO_DIR = os.path.join(ALGORITHM_ROOT, 'MBCBO')  # 添加MBCBO目录
OTHER_ALGO_DIR = os.path.join(ALGORITHM_ROOT, 'other_algorithms')

# 添加到系统路径
for path in [BCBO_DIR, BCBO_DE_DIR, MBCBO_DIR, OTHER_ALGO_DIR]:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and abs_path not in sys.path:
        sys.path.insert(0, abs_path)
        print(f"[INFO] 添加路径: {abs_path}")

# 导入所有算法（需要根据实际可用的算法模块进行调整）
try:
    from bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler
except ImportError:
    print("[WARNING] 无法导入 BCBO_CloudScheduler，该算法将不可用")
    BCBO_CloudScheduler = None

try:
    from genetic_algorithm_scheduler import GeneticAlgorithmScheduler
except ImportError:
    print("[WARNING] 无法导入 GeneticAlgorithmScheduler，该算法将不可用")
    GeneticAlgorithmScheduler = None

try:
    from particle_swarm_optimizer import ParticleSwarmOptimizer
except ImportError:
    print("[WARNING] 无法导入 ParticleSwarmOptimizer，该算法将不可用")
    ParticleSwarmOptimizer = None

try:
    from ant_colony_optimizer import AntColonyOptimizer
except ImportError:
    print("[WARNING] 无法导入 AntColonyOptimizer，该算法将不可用")
    AntColonyOptimizer = None

try:
    from firefly_algorithm import FireflyAlgorithm
except ImportError:
    print("[WARNING] 无法导入 FireflyAlgorithm，该算法将不可用")
    FireflyAlgorithm = None

try:
    from cuckoo_search import CuckooSearch
except ImportError:
    print("[WARNING] 无法导入 CuckooSearch，该算法将不可用")
    CuckooSearch = None

try:
    from grey_wolf_optimizer import GreyWolfOptimizer
except ImportError:
    print("[WARNING] 无法导入 GreyWolfOptimizer，该算法将不可用")
    GreyWolfOptimizer = None

try:
    # 导入BCBO-DE-Fusion算法
    # BCBO-DE-Fusion的core目录已经在上面添加到路径中
    bcbo_de_core_path = os.path.join(BCBO_DE_DIR, 'core')
    if os.path.exists(bcbo_de_core_path) and bcbo_de_core_path not in sys.path:
        sys.path.insert(0, bcbo_de_core_path)
        print(f"[INFO] 添加路径: {bcbo_de_core_path}")

    # 强制使用原版(已添加负载均衡修复)
    from bcbo_de_embedded import BCBO_DE_Embedded
    print("[INFO] 使用BCBO-DE原版 (v3.2 + 负载均衡修复)")
except ImportError as e:
    print(f"[WARNING] 无法导入 BCBO_DE_Embedded，该算法将不可用: {e}")
    BCBO_DE_Embedded = None

try:
    # 导入MBCBO算法
    from mbcbo_cloud_scheduler import MBCBO_CloudScheduler
    print("[INFO] 成功导入 MBCBO_CloudScheduler")
except ImportError as e:
    print(f"[WARNING] 无法导入 MBCBO_CloudScheduler，该算法将不可用: {e}")
    MBCBO_CloudScheduler = None


class RealAlgorithmIntegrator:
    """
    真实算法集成器

    提供统一的接口来调用不同的优化算法
    """

    def __init__(self):
        """初始化集成器"""
        self.algorithms = {}
        self.available_algorithms = []

        # ===== BUG修复 (2025-11-27): 添加问题实例缓存 =====
        # 问题：每个算法实例独立生成execution_time矩阵，导致对比无效
        # 修复：缓存第一个算法的问题实例，所有算法共享相同实例
        self.problem_instance = None  # 缓存问题实例 (execution_time, task_loads, vm_caps等)
        self.problem_instance_seed = None  # 记录问题实例对应的随机种子

        # 检查哪些算法可用
        if BCBO_CloudScheduler is not None:
            self.available_algorithms.append('BCBO')
        if GeneticAlgorithmScheduler is not None:
            self.available_algorithms.append('GA')
        if ParticleSwarmOptimizer is not None:
            self.available_algorithms.append('PSO')
        if AntColonyOptimizer is not None:
            self.available_algorithms.append('ACO')
        if FireflyAlgorithm is not None:
            self.available_algorithms.append('FA')
        if CuckooSearch is not None:
            self.available_algorithms.append('CS')
        if GreyWolfOptimizer is not None:
            self.available_algorithms.append('GWO')
        if BCBO_DE_Embedded is not None:
            self.available_algorithms.append('BCBO-DE')
        if MBCBO_CloudScheduler is not None:
            self.available_algorithms.append('MBCBO')

        print(f"[INFO] 算法集成器初始化完成")
        print(f"[INFO] 可用算法: {', '.join(self.available_algorithms)}")

    def run_algorithm(self, algorithm_name: str, params: Dict) -> Optional[Dict]:
        """
        运行指定的算法

        参数:
            algorithm_name: 算法名称 (BCBO, GA, PSO, ACO, FA, CS, GWO, BCBO-DE)
            params: 算法参数字典,必须包含:
                - M: 任务数量
                - N: 虚拟机数量
                - n: 种群大小
                - iterations: 迭代次数

        返回:
            包含算法结果的字典,格式:
            {
                'best_solution': 最优解数组,
                'best_fitness': 最优适应度,
                'total_cost': 总成本,
                'execution_time': 执行时间（makespan）,
                'load_balance': 负载均衡度,
                'price_efficiency': 价格效率,
                'convergence_history': 收敛历史,
                'success_rate': 成功率
            }
        """
        # 验证参数
        required_params = ['M', 'N', 'n', 'iterations']
        for param in required_params:
            if param not in params:
                raise ValueError(f"缺少必需参数: {param}")

        M = params['M']
        N = params['N']
        n = params['n']
        iterations = params['iterations']
        random_seed = params.get('random_seed', None)

        # ===== BUG修复 (2025-11-27): 管理共享问题实例 =====
        # 如果random_seed改变或M/N改变，需要重新生成问题实例
        need_new_instance = (
            self.problem_instance is None or
            self.problem_instance_seed != random_seed or
            self.problem_instance.get('M') != M or
            self.problem_instance.get('N') != N
        )

        if need_new_instance:
            print(f"[INFO] 生成新的问题实例 (M={M}, N={N}, seed={random_seed})")
            self._generate_problem_instance(M, N, random_seed)

        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        try:
            # 根据算法名称选择不同的算法
            if algorithm_name == 'BCBO':
                return self._run_bcbo(M, N, n, iterations, random_seed)
            elif algorithm_name == 'GA':
                return self._run_ga(M, N, n, iterations, random_seed)
            elif algorithm_name == 'PSO':
                return self._run_pso(M, N, n, iterations, random_seed)
            elif algorithm_name == 'ACO':
                return self._run_aco(M, N, n, iterations, random_seed)
            elif algorithm_name == 'FA':
                return self._run_fa(M, N, n, iterations, random_seed)
            elif algorithm_name == 'CS':
                return self._run_cs(M, N, n, iterations, random_seed)
            elif algorithm_name == 'GWO':
                return self._run_gwo(M, N, n, iterations, random_seed)
            elif algorithm_name == 'BCBO-DE':
                return self._run_bcbo_de(M, N, n, iterations, random_seed)
            elif algorithm_name == 'MBCBO':
                return self._run_mbcbo(M, N, n, iterations, random_seed)
            else:
                raise ValueError(f"不支持的算法: {algorithm_name}")

        except Exception as e:
            print(f"[ERROR] 运行算法 {algorithm_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_problem_instance(self, M: int, N: int, random_seed: Optional[int]):
        """
        生成共享的问题实例

        ===== BUG修复 (2025-11-27) =====
        所有算法必须使用相同的execution_time矩阵，否则对比无效

        参数:
            M: 任务数量
            N: 虚拟机数量
            random_seed: 随机种子
        """
        # 设置随机种子以保证可重复性
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # 生成任务和VM属性（与BCBO_CloudScheduler相同的分布）
        task_loads = np.random.randint(50, 200, M)
        vm_caps = np.random.randint(10, 30, N)

        # 生成多维属性
        task_cpu = task_loads.astype(float)
        task_memory = task_loads.astype(float) * 0.5
        task_storage = np.random.uniform(5, 50, M)
        task_network = np.random.uniform(2, 20, M)
        task_priority = np.random.randint(1, 4, M)
        task_deadline = np.random.uniform(10, 50, M)
        task_data_size = np.random.uniform(1, 30, M)

        vm_cpu_capacity = vm_caps.astype(float)
        vm_memory_capacity = vm_caps.astype(float) * 2.0
        vm_storage_capacity = np.random.uniform(500, 3000, N)
        vm_network_capacity = np.random.uniform(50, 300, N)
        vm_processing_speed = np.random.uniform(1.2, 3.5, N)
        vm_cost = np.random.uniform(0.05, 0.15, N)
        vm_energy_efficiency = np.random.uniform(0.6, 0.9, N)

        # 计算执行时间矩阵
        execution_time = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                workload = (task_cpu[i] + task_memory[i] +
                           task_storage[i] + task_network[i])
                execution_time[i][j] = workload / vm_processing_speed[j]

        # 保存问题实例
        self.problem_instance = {
            'M': M,
            'N': N,
            'task_loads': task_loads,
            'vm_caps': vm_caps,
            'task_cpu': task_cpu,
            'task_memory': task_memory,
            'task_storage': task_storage,
            'task_network': task_network,
            'task_priority': task_priority,
            'task_deadline': task_deadline,
            'task_data_size': task_data_size,
            'vm_cpu_capacity': vm_cpu_capacity,
            'vm_memory_capacity': vm_memory_capacity,
            'vm_storage_capacity': vm_storage_capacity,
            'vm_network_capacity': vm_network_capacity,
            'vm_processing_speed': vm_processing_speed,
            'vm_cost': vm_cost,
            'vm_energy_efficiency': vm_energy_efficiency,
            'execution_time': execution_time
        }
        self.problem_instance_seed = random_seed

        print(f"[DEBUG] 问题实例已生成: execution_time shape={execution_time.shape}")

    def _process_convergence_history(self, history):
        """
        处理收敛历史数据

        确保返回标准格式的列表
        """
        if not history:
            return []

        processed = []
        for i, record in enumerate(history):
            if isinstance(record, dict):
                # 已经是字典格式
                processed.append({
                    'iteration': record.get('iteration', i + 1),
                    'best_fitness': float(record.get('best_fitness', 0)),
                    'best_solution': record.get('best_solution'),
                    'global_best_fitness': float(record.get('global_best_fitness', record.get('best_fitness', 0)))
                })
            elif isinstance(record, (int, float)):
                # 仅适应度值
                processed.append({
                    'iteration': i + 1,
                    'best_fitness': float(record),
                    'best_solution': None,
                    'global_best_fitness': float(record)
                })

        return processed

    def _calculate_convergence_iteration(self, history, threshold=0.01):
        """
        计算收敛迭代次数

        返回第一次达到接近最优值的迭代次数
        """
        # 处理历史记录为空或非列表的情况
        if not history:
            return 0

        if isinstance(history, dict):
            # 如果history是dict,可能是错误传递,返回0
            return 0

        if len(history) < 2:
            return len(history)

        # 获取最终适应度
        if isinstance(history[-1], dict):
            final_fitness = history[-1].get('best_fitness', 0)
        else:
            final_fitness = float(history[-1])

        # 查找第一次接近最优值的位置
        for i, record in enumerate(history):
            if isinstance(record, dict):
                current_fitness = record.get('best_fitness', 0)
            else:
                current_fitness = float(record)

            if abs(current_fitness - final_fitness) / (abs(final_fitness) + 1e-10) < threshold:
                return i + 1

        return len(history)

    # ==================== 各算法的运行方法 ====================

    def _run_bcbo(self, M, N, n, iterations, random_seed):
        """运行BCBO算法"""
        if BCBO_CloudScheduler is None:
            raise RuntimeError("BCBO算法不可用")

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        algo = BCBO_CloudScheduler(M, N, n, iterations, random_seed)

        # 强制使用缓存的问题实例
        if self.problem_instance is not None:
            algo.task_loads = self.problem_instance['task_loads']
            algo.vm_caps = self.problem_instance['vm_caps']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.task_storage = self.problem_instance['task_storage']
            algo.task_network = self.problem_instance['task_network']
            algo.task_priority = self.problem_instance['task_priority']
            algo.task_deadline = self.problem_instance['task_deadline']
            algo.task_data_size = self.problem_instance['task_data_size']
            algo.vm_cpu_capacity = self.problem_instance['vm_cpu_capacity']
            algo.vm_memory_capacity = self.problem_instance['vm_memory_capacity']
            algo.vm_storage_capacity = self.problem_instance['vm_storage_capacity']
            algo.vm_network_capacity = self.problem_instance['vm_network_capacity']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            algo.vm_energy_efficiency = self.problem_instance['vm_energy_efficiency']
            algo.execution_time = self.problem_instance['execution_time']
            print(f"[DEBUG] BCBO使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.run_complete_algorithm()

        self.algorithms['BCBO'] = algo

        return self._format_result('BCBO', result, algo)

    def _run_ga(self, M, N, n, iterations, random_seed):
        """运行遗传算法"""
        if GeneticAlgorithmScheduler is None:
            raise RuntimeError("GA算法不可用")

        algo = GeneticAlgorithmScheduler(M, N, population_size=n, generations=iterations)

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        if self.problem_instance is not None:
            # GA没有完整的任务/VM属性，只需要覆盖核心计算属性
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.vm_cpu_capacity = self.problem_instance['vm_cpu_capacity']
            algo.vm_memory_capacity = self.problem_instance['vm_memory_capacity']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            print(f"[DEBUG] GA使用共享问题实例")

        result = algo.optimize()

        self.algorithms['GA'] = algo

        return self._format_result('GA', result, algo)

    def _run_pso(self, M, N, n, iterations, random_seed):
        """运行粒子群优化算法"""
        if ParticleSwarmOptimizer is None:
            raise RuntimeError("PSO算法不可用")

        algo = ParticleSwarmOptimizer(M, N, num_particles=n, max_iterations=iterations)

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        if self.problem_instance is not None:
            algo.execution_time = self.problem_instance['execution_time']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            print(f"[DEBUG] PSO使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.optimize()

        self.algorithms['PSO'] = algo

        return self._format_result('PSO', result, algo)

    def _run_aco(self, M, N, n, iterations, random_seed):
        """运行蚁群优化算法"""
        if AntColonyOptimizer is None:
            raise RuntimeError("ACO算法不可用")

        algo = AntColonyOptimizer(M, N, num_ants=n, max_iterations=iterations)

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        if self.problem_instance is not None:
            algo.execution_time = self.problem_instance['execution_time']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            print(f"[DEBUG] ACO使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.optimize()

        self.algorithms['ACO'] = algo

        return self._format_result('ACO', result, algo)

    def _run_fa(self, M, N, n, iterations, random_seed):
        """运行萤火虫算法"""
        if FireflyAlgorithm is None:
            raise RuntimeError("FA算法不可用")

        algo = FireflyAlgorithm(M, N, num_fireflies=n, max_iterations=iterations)

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        if self.problem_instance is not None:
            algo.execution_time = self.problem_instance['execution_time']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.task_storage = self.problem_instance['task_storage']
            algo.task_network = self.problem_instance['task_network']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            print(f"[DEBUG] FA使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.optimize()

        self.algorithms['FA'] = algo

        return self._format_result('FA', result, algo)

    def _run_cs(self, M, N, n, iterations, random_seed):
        """运行布谷鸟搜索算法"""
        if CuckooSearch is None:
            raise RuntimeError("CS算法不可用")

        algo = CuckooSearch(M, N, num_nests=n, max_iterations=iterations)

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        if self.problem_instance is not None:
            algo.execution_time = self.problem_instance['execution_time']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.task_storage = self.problem_instance['task_storage']
            algo.task_network = self.problem_instance['task_network']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            print(f"[DEBUG] CS使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.optimize()

        self.algorithms['CS'] = algo

        return self._format_result('CS', result, algo)

    def _run_gwo(self, M, N, n, iterations, random_seed):
        """运行灰狼优化算法"""
        if GreyWolfOptimizer is None:
            raise RuntimeError("GWO算法不可用")

        algo = GreyWolfOptimizer(M, N, num_wolves=n, max_iterations=iterations)

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        if self.problem_instance is not None:
            algo.execution_time = self.problem_instance['execution_time']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.task_storage = self.problem_instance['task_storage']
            algo.task_network = self.problem_instance['task_network']
            algo.task_data_size = self.problem_instance['task_data_size']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            print(f"[DEBUG] GWO使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.optimize()

        self.algorithms['GWO'] = algo

        return self._format_result('GWO', result, algo)

    def _run_bcbo_de(self, M, N, n, iterations, random_seed):
        """运行BCBO-DE嵌入式融合算法（v3.2 + 负载均衡修复版）"""
        if BCBO_DE_Embedded is None:
            raise RuntimeError("BCBO-DE算法不可用")

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        algo = BCBO_DE_Embedded(
            M=M, N=N, n=n, iterations=iterations,
            random_seed=random_seed if random_seed is not None else 42,
            verbose=False  # 关闭详细输出以加快速度
        )

        # 强制使用缓存的问题实例（通过内部的bcbo实例）
        if self.problem_instance is not None:
            bcbo_instance = algo.bcbo
            bcbo_instance.task_loads = self.problem_instance['task_loads']
            bcbo_instance.vm_caps = self.problem_instance['vm_caps']
            bcbo_instance.task_cpu = self.problem_instance['task_cpu']
            bcbo_instance.task_memory = self.problem_instance['task_memory']
            bcbo_instance.task_storage = self.problem_instance['task_storage']
            bcbo_instance.task_network = self.problem_instance['task_network']
            bcbo_instance.task_priority = self.problem_instance['task_priority']
            bcbo_instance.task_deadline = self.problem_instance['task_deadline']
            bcbo_instance.task_data_size = self.problem_instance['task_data_size']
            bcbo_instance.vm_cpu_capacity = self.problem_instance['vm_cpu_capacity']
            bcbo_instance.vm_memory_capacity = self.problem_instance['vm_memory_capacity']
            bcbo_instance.vm_storage_capacity = self.problem_instance['vm_storage_capacity']
            bcbo_instance.vm_network_capacity = self.problem_instance['vm_network_capacity']
            bcbo_instance.vm_processing_speed = self.problem_instance['vm_processing_speed']
            bcbo_instance.vm_cost = self.problem_instance['vm_cost']
            bcbo_instance.vm_energy_efficiency = self.problem_instance['vm_energy_efficiency']
            bcbo_instance.execution_time = self.problem_instance['execution_time']

            print(f"[DEBUG] BCBO-DE使用共享问题实例 (v3.2+负载修复): execution_time shape={bcbo_instance.execution_time.shape}")

        result = algo.run_fusion_optimization()

        self.algorithms['BCBO-DE'] = algo

        return self._format_result('BCBO-DE', result, algo)

    def _run_mbcbo(self, M, N, n, iterations, random_seed):
        """运行MBCBO多策略协同算法"""
        if MBCBO_CloudScheduler is None:
            raise RuntimeError("MBCBO算法不可用")

        # ===== BUG修复 (2025-11-27): 使用共享问题实例 =====
        algo = MBCBO_CloudScheduler(M, N, n, iterations, verbose=False)

        # 强制使用缓存的问题实例
        if self.problem_instance is not None:
            algo.task_loads = self.problem_instance['task_loads']
            algo.vm_caps = self.problem_instance['vm_caps']
            algo.task_cpu = self.problem_instance['task_cpu']
            algo.task_memory = self.problem_instance['task_memory']
            algo.task_storage = self.problem_instance['task_storage']
            algo.task_network = self.problem_instance['task_network']
            algo.task_priority = self.problem_instance['task_priority']
            algo.task_deadline = self.problem_instance['task_deadline']
            algo.task_data_size = self.problem_instance['task_data_size']
            algo.vm_cpu_capacity = self.problem_instance['vm_cpu_capacity']
            algo.vm_memory_capacity = self.problem_instance['vm_memory_capacity']
            algo.vm_storage_capacity = self.problem_instance['vm_storage_capacity']
            algo.vm_network_capacity = self.problem_instance['vm_network_capacity']
            algo.vm_processing_speed = self.problem_instance['vm_processing_speed']
            algo.vm_cost = self.problem_instance['vm_cost']
            algo.vm_energy_efficiency = self.problem_instance['vm_energy_efficiency']
            algo.execution_time = self.problem_instance['execution_time']
            print(f"[DEBUG] MBCBO使用共享问题实例: execution_time shape={algo.execution_time.shape}")

        result = algo.optimize()

        self.algorithms['MBCBO'] = algo

        return self._format_result('MBCBO', result, algo)

    def _format_result(self, algorithm_name, result, algo_instance):
        """
        格式化算法结果为统一格式

        参数:
            algorithm_name: 算法名称
            result: 算法原始结果（可能是字典或元组）
            algo_instance: 算法实例

        返回:
            标准格式的结果字典
        """
        # 处理不同的返回格式
        if isinstance(result, tuple):
            # 处理不同长度的元组返回值
            if len(result) == 2:
                # (best_solution, best_fitness)
                best_solution, best_fitness = result
                result_dict = {
                    'best_solution': best_solution,
                    'best_fitness': best_fitness
                }
            elif len(result) == 3:
                # (best_solution, best_fitness, fitness_history)
                best_solution, best_fitness, fitness_history = result
                result_dict = {
                    'best_solution': best_solution,
                    'best_fitness': best_fitness,
                    'fitness_history': fitness_history
                }
            else:
                raise ValueError(f"不支持的元组长度: {len(result)}")
        elif isinstance(result, dict):
            # BCBO返回字典
            result_dict = result
        else:
            raise ValueError(f"不支持的结果类型: {type(result)}")

        # 提取基本信息
        best_solution = result_dict.get('best_solution')
        best_fitness = result_dict.get('best_fitness', 0)

        # 从算法实例提取收敛历史
        convergence_history = []
        if hasattr(algo_instance, 'fitness_history'):
            convergence_history = algo_instance.fitness_history
        elif hasattr(algo_instance, 'convergence_history'):
            convergence_history = algo_instance.convergence_history
        elif hasattr(algo_instance, 'monitor') and hasattr(algo_instance.monitor, 'history'):
            # BCBO-DE使用monitor.history (字典格式)
            monitor_history = algo_instance.monitor.history
            if isinstance(monitor_history, dict) and 'iteration' in monitor_history:
                # 将字典格式转换为列表格式
                iterations = monitor_history.get('iteration', [])
                best_fitness_list = monitor_history.get('best_fitness', [])
                best_solution_list = monitor_history.get('best_solution', [])

                for i in range(len(iterations)):
                    # 获取当前迭代的最优解（如果有记录）
                    current_best_solution = best_solution
                    if i < len(best_solution_list) and best_solution_list[i] is not None:
                        current_best_solution = best_solution_list[i]

                    convergence_history.append({
                        'iteration': iterations[i] + 1,
                        'best_fitness': best_fitness_list[i] if i < len(best_fitness_list) else 0,
                        'best_solution': current_best_solution,  # 使用历史记录中的最优解
                        'global_best_fitness': best_fitness_list[i] if i < len(best_fitness_list) else 0
                    })
            else:
                convergence_history = monitor_history
        elif 'history' in result_dict:
            # BCBO-DE返回的history
            history_data = result_dict['history']
            if isinstance(history_data, dict) and 'iteration' in history_data:
                # 同样处理result中的字典格式history
                iterations = history_data.get('iteration', [])
                best_fitness_list = history_data.get('best_fitness', [])
                for i in range(len(iterations)):
                    convergence_history.append({
                        'iteration': iterations[i] + 1,
                        'best_fitness': best_fitness_list[i] if i < len(best_fitness_list) else 0,
                        'best_solution': best_solution,
                        'global_best_fitness': best_fitness_list[i] if i < len(best_fitness_list) else 0
                    })
            else:
                convergence_history = history_data
        elif 'fitness_history' in result_dict:
            convergence_history = result_dict['fitness_history']

        # 计算或提取各项指标 - 使用统一的计算方法
        total_cost = 0
        makespan = 0
        load_imbalance = 0

        # 优先使用算法实例的公共接口方法
        if hasattr(algo_instance, 'get_detailed_metrics'):
            # 如果算法有get_detailed_metrics方法，直接使用
            try:
                metrics = algo_instance.get_detailed_metrics(best_solution)
                total_cost = metrics.get('total_cost', 0)
                makespan = metrics.get('makespan', 0)
                load_imbalance = metrics.get('load_imbalance', 0)
            except Exception as e:
                print(f"[DEBUG] get_detailed_metrics调用失败: {e}")

        # 如果上面没有成功，尝试BCBO-DE的内部bcbo实例
        if total_cost == 0 and hasattr(algo_instance, 'bcbo'):
            try:
                bcbo_instance = algo_instance.bcbo
                if hasattr(bcbo_instance, 'get_detailed_metrics'):
                    metrics = bcbo_instance.get_detailed_metrics(best_solution)
                    total_cost = metrics.get('total_cost', 0)
                    makespan = metrics.get('makespan', 0)
                    load_imbalance = metrics.get('load_imbalance', 0)
            except Exception as e:
                print(f"[DEBUG] BCBO实例get_detailed_metrics调用失败: {e}")

        # 如果还是没有成功，从result_dict中提取
        if total_cost == 0:
            total_cost = result_dict.get('total_cost', 0)
            makespan = result_dict.get('response_time',
                                       result_dict.get('makespan',
                                       result_dict.get('execution_time', 0)))
            load_imbalance = result_dict.get('load_imbalance', 0)

        # 构建标准结果
        formatted_result = {
            'algorithm': algorithm_name,
            'best_solution': best_solution,
            'best_fitness': float(best_fitness),
            'total_cost': float(total_cost),
            'execution_time': float(makespan),
            'load_balance': float(1.0 - min(load_imbalance, 1.0)),
            'price_efficiency': float(1.0 / (total_cost + 1e-6)),
            'convergence_history': self._process_convergence_history(convergence_history),
            'convergence_iteration': self._calculate_convergence_iteration(convergence_history),
            'success_rate': 1.0,
            'runtime': result_dict.get('runtime', result_dict.get('total_time', 0))
        }

        return formatted_result


# 简单测试
if __name__ == "__main__":
    print("="*60)
    print("真实算法集成模块测试 (Scripts版本)")
    print("="*60)

    integrator = RealAlgorithmIntegrator()

    # 测试参数（使用较小规模以加快测试）
    test_params = {
        'M': 20,
        'N': 5,
        'n': 10,
        'iterations': 20,
        'random_seed': 42
    }

    print(f"\n[TEST] 测试参数: {test_params}")
    print(f"[INFO] 可用算法数量: {len(integrator.available_algorithms)}")

    # 只测试可用的算法
    for algo_name in integrator.available_algorithms:
        print(f"\n{'='*60}")
        print(f"[TEST] 测试算法: {algo_name}")
        print(f"{'='*60}")

        try:
            result = integrator.run_algorithm(algo_name, test_params)

            if result:
                print(f"[OK] {algo_name} 运行成功")
                print(f"  - 最优适应度: {result['best_fitness']:.6f}")
                print(f"  - 总成本: {result['total_cost']:.2f}")
                print(f"  - 执行时间: {result['execution_time']:.2f}")
                print(f"  - 负载均衡: {result['load_balance']:.4f}")
                print(f"  - 收敛历史长度: {len(result['convergence_history'])}")
            else:
                print(f"[ERROR] {algo_name} 运行失败")

        except Exception as e:
            print(f"[ERROR] {algo_name} 测试出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
