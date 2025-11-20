#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO算法可视化模块
=====================================
用于生成BCBO算法的各种可视化图表和表格

主要功能:
- 创建算法收敛曲线图
- 生成阶段性能表格
- 绘制虚拟机负载分布图
- 生成完整的可视化报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import glob
from typing import Dict, Any, Tuple, List, Optional
import sys

# 添加父目录到路径以导入其他模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(BASE_DIR, '..')
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


class BCBOVisualizer:
    def __init__(self, save_dir: str = ".", config_mode: Optional[str] = None):
        self.save_dir = save_dir
        self.config_mode = config_mode
        self.config_mode_info = {
            "模式1": "描述1",
            "模式2": "描述2"
        }
        # 使用文件系统安全的时间戳格式（避免Windows文件名中的冒号导致错误）
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确认输出目录存在
        os.makedirs(os.path.join(self.save_dir, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "tables"), exist_ok=True)
        # 真实数据目录（使用相对路径，更加灵活）
        # 从 scripts 目录向上查找 RAW_data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_dirs = [
            os.path.join(script_dir, "..", "RAW_data"),  # bcbo_ga_staged_hybrid/RAW_data（主要）
            os.path.join(script_dir, "..", "..", "数据生成工具", "RAW_data"),  # 数据生成工具/RAW_data
            os.path.join(os.getcwd(), "RAW_data")  # 当前目录/RAW_data
        ]
        self.raw_data_dir = next((d for d in possible_dirs if os.path.isdir(d)), None)

    def _smooth(self, y: Any, window: int = 5) -> np.ndarray:
        """
        轻量级移动平均平滑（仅用于绘图，不修改导出的数据）
        - window: 平滑窗口大小（奇数更佳）。当序列很短时自动降级为原序列。
        """
        try:
            arr = np.asarray(y, dtype=float)
            if arr.ndim != 1 or arr.size <= 2 or window <= 1:
                return arr
            w = int(max(1, window))
            if w % 2 == 0:
                w += 1
            s = pd.Series(arr)
            return s.rolling(window=w, min_periods=1, center=True).mean().to_numpy()
        except Exception:
            return np.asarray(y, dtype=float)

    def create_algorithm_summary_table(self, results: Dict[str, Any],
                                     algorithm_params: Dict[str, Any]) -> pd.DataFrame:
        """
        创建算法摘要表格
        Create algorithm summary table

        Args:
            results (Dict[str, Any]): 算法运行结果
            algorithm_params (Dict[str, Any]): 算法参数

        Returns:
            pd.DataFrame: 算法摘要表格
        """
        basic_info = {
            '运行信息': ['任务数量 (M)', '虚拟机数量 (N)', '种群大小 (n)', '总迭代次数', '权重系数 (w1,w2,w3)', '二值化阈值',
                      '最优适应度值', '响应时间', '资源利用率', '总成本', '约束满足', '约束违反度',
                      '总运行时间', '收敛迭代次数', '是否可行解', '算法版本'],
            '参数名称': ['任务数量 (M)', '虚拟机数量 (N)', '种群大小 (n)', '总迭代次数', '权重系数 (w1,w2,w3)', '二值化阈值',
                      '最优适应度值', '响应时间', '资源利用率', '总成本', '约束满足', '约束违反度',
                      '总运行时间', '收敛迭代次数', '是否可行解', '算法版本'],
            '数值': [
                algorithm_params.get('M', 'N/A'),
                algorithm_params.get('N', 'N/A'),
                algorithm_params.get('n', 'N/A'),
                algorithm_params.get('iterations', 'N/A'),
                f"({algorithm_params.get('w1', 0.4)}, {algorithm_params.get('w2', 0.3)}, {algorithm_params.get('w3', 0.3)})",
                algorithm_params.get('random_threshold', 0.5),
                f"{results.get('best_fitness', 0):.6f}",
                f"{results.get('response_time', 0):.4f}",
                f"{results.get('resource_utilization', 0):.4f}",
                f"{results.get('total_cost', 0):.4f}",
                '是' if results.get('is_feasible', False) else '否',
                f"{results.get('violation_degree', 0):.6f}",
                f"{results.get('total_time', 0):.2f}秒",
                results.get('convergence_iteration', 0),
                '是' if results.get('is_feasible', False) else '否',
                'BCBO v1.0'
            ],
            '说明': [
                '云环境中的任务总数',
                '可用虚拟机总数',
                '优化算法的种群规模',
                '算法运行的总迭代次数',
                '响应时间、资源利用率、成本的权重',
                'sigmoid二值化函数的阈值参数',
                '多目标优化的综合适应度值（越小越好）',
                '所有任务的总响应时间',
                '虚拟机的平均资源利用率',
                '云资源使用的总成本',
                '解是否满足所有约束条件',
                '约束违反的程度（0表示完全满足）',
                '算法从开始到结束的总时间',
                '算法收敛所需的迭代次数',
                '最终解是否为可行解',
                '使用的BCBO算法版本'
            ]
        }

        df = pd.DataFrame(basic_info)
        return df

    def create_phase_performance_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        创建各阶段性能表格
        Create phase performance table

        Args:
            results (Dict[str, Any]): 算法运行结果

        Returns:
            pd.DataFrame: 阶段性能表格
        """
        phase_times = results.get('phase_times', {})

        # 六个阶段的信息
        phases_info = {
            '阶段编号': ['第一阶段', '第二阶段', '第三阶段', '第四阶段', '第五阶段', '第六阶段'],
            '阶段名称': ['动态搜索阶段', '静态搜索阶段', '动态包围阶段', '静态包围阶段', '动态攻击阶段', '静态攻击阶段'],
            '英文名称': ['Dynamic Searching', 'Static Searching', 'Encircling Dynamic',
                       'Encircling Static', 'Attacking Dynamic', 'Attacking Static'],
            '执行时间(秒)': [
                phase_times.get('动态搜索阶段', 0),
                phase_times.get('静态搜索阶段', 0),
                phase_times.get('动态包围阶段', 0),
                phase_times.get('静态包围阶段', 0),
                phase_times.get('动态攻击阶段', 0),
                phase_times.get('静态攻击阶段', 0)
            ],
            '主要功能': [
                '全局探索，初始化Coyote-Badger-Prey三元组',
                '局部优化,静态位置更新',
                '动态包围猎物，收缩搜索空间',
                '静态包围策略，精确定位',
                '动态攻击猎物，精英解优化',
                '静态攻击策略，最终局部搜索'
            ]
        }

        df = pd.DataFrame(phases_info)

        # 计算时间占比
        total_time = sum(phase_times.values()) if phase_times else 1
        df['时间占比(%)'] = (df['执行时间(秒)'] / total_time * 100).round(2)

        return df

    def create_vm_load_table(self, results: Dict[str, Any],
                           task_cpu: np.ndarray, task_memory: np.ndarray,
                           vm_cpu_capacity: np.ndarray, vm_memory_capacity: np.ndarray) -> pd.DataFrame:
        """
        创建虚拟机负载表格
        Create VM load table

        Args:
            results (Dict[str, Any]): 算法运行结果
            task_cpu (np.ndarray): 任务CPU需求
            task_memory (np.ndarray): 任务内存需求
            vm_cpu_capacity (np.ndarray): 虚拟机CPU容量
            vm_memory_capacity (np.ndarray): 虚拟机内存容量

        Returns:
            pd.DataFrame: 虚拟机负载表格
        """
        best_matrix = results.get('best_matrix')
        if best_matrix is None:
            return pd.DataFrame()

        M, N = best_matrix.shape

        vm_info = []
        for j in range(N):
            assigned_tasks = np.where(best_matrix[:, j] == 1)[0]

            if len(assigned_tasks) > 0:
                cpu_load = np.sum(task_cpu[assigned_tasks])
                memory_load = np.sum(task_memory[assigned_tasks])
                cpu_util = cpu_load / vm_cpu_capacity[j] * 100
                memory_util = memory_load / vm_memory_capacity[j] * 100
            else:
                cpu_load = 0
                memory_load = 0
                cpu_util = 0
                memory_util = 0

            vm_info.append({
                '虚拟机ID': f'VM{j}',
                '分配任务数': len(assigned_tasks),
                '分配任务ID': ', '.join([f'T{i}' for i in assigned_tasks]) if len(assigned_tasks) > 0 else '无',
                'CPU容量': f"{vm_cpu_capacity[j]:.2f}",
                'CPU负载': f"{cpu_load:.2f}",
                'CPU利用率(%)': f"{cpu_util:.1f}",
                '内存容量(GB)': f"{vm_memory_capacity[j]:.2f}",
                '内存负载(GB)': f"{memory_load:.2f}",
                '内存利用率(%)': f"{memory_util:.1f}",
                '负载状态': '高负载' if max(cpu_util, memory_util) > 80 else '中负载' if max(cpu_util, memory_util) > 50 else '低负载'
            })

        return pd.DataFrame(vm_info)

    def save_tables_to_excel(self, tables: Dict[str, pd.DataFrame], filename: str = None):
        """
        将所有表格保存到Excel文件
        Save all tables to Excel file

        Args:
            tables (Dict[str, pd.DataFrame]): 表格字典
            filename (str): 文件名
        """
        if filename is None:
            mode_suffix = f"_{self.config_mode}" if self.config_mode else ""
            filename = f"BCBO_算法结果表格{mode_suffix}_{self.timestamp}.xlsx"

        filepath = os.path.join(self.save_dir, "tables", filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 添加配置模式信息工作表
            if self.config_mode:
                config_info = pd.DataFrame({
                    '配置项': ['配置模式', '模式描述', '生成时间'],
                    '值': [
                        self.config_mode,
                        self.config_mode_info.get(self.config_mode, "自定义配置"),
                        self.timestamp
                    ]
                })
                config_info.to_excel(writer, sheet_name='配置信息', index=False)

        print(f"表格已保存到: {filepath}")
        return filepath

    # 其他方法继续...（由于篇幅限制，这里省略了其他方法）
    # 包括: create_convergence_chart, create_phase_time_chart, create_vm_load_chart 等
    # 如需完整代码，可以继续添加


# 简单测试
if __name__ == "__main__":
    print("BCBO可视化模块已加载")
    print(f"当前工作目录: {os.getcwd()}")
