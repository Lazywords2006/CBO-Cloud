
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCBO-DE 调优测试脚本 - moderate配置
"""

import sys
import os
import json
import time

# 添加路径
sys.path.append('algorithm/BCBO-DE-Fusion')
sys.path.append('algorithm/BCBO')

from algorithm.BCBO_DE_Fusion.core.bcbo_de_embedded import BCBO_DE_Embedded
from algorithm.BCBO.bcbo_cloud_scheduler_fixed import BCBO_CloudScheduler

def test_configuration():
    """测试moderate配置"""

    # 测试参数
    test_cases = [
        {"M": 100, "N": 20},   # 小规模
        {"M": 500, "N": 50},   # 中规模
        {"M": 1000, "N": 100}, # 大规模
    ]

    results = []

    for test_case in test_cases:
        M, N = test_case["M"], test_case["N"]
        print(f"\n测试 M={M}, N={N}")
        print("-" * 40)

        # 运行BCBO-DE
        bcbo_de = BCBO_DE_Embedded(
            M=M, N=N, n=50, iterations=100,
            verbose=False
        )
        bcbo_de_result = bcbo_de.run_fusion_optimization()

        # 运行原始BCBO作为对比
        bcbo = BCBO_CloudScheduler(M=M, N=N, n=50, iterations=100)
        bcbo_result = bcbo.run_optimization()

        # 记录结果
        improvement = ((bcbo_de_result['best_fitness'] - bcbo_result['best_fitness'])
                      / abs(bcbo_result['best_fitness']) * 100)

        results.append({
            'M': M,
            'N': N,
            'bcbo_fitness': bcbo_result['best_fitness'],
            'bcbo_de_fitness': bcbo_de_result['best_fitness'],
            'improvement': improvement
        })

        print(f"BCBO: {bcbo_result['best_fitness']:.4f}")
        print(f"BCBO-DE: {bcbo_de_result['best_fitness']:.4f}")
        print(f"改进率: {improvement:.2f}%")

    # 保存结果
    with open(f'test_results_moderate.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到 test_results_moderate.json")

    # 打印总结
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    print(f"\n平均改进率: {avg_improvement:.2f}%")

    if avg_improvement > 0:
        print("✅ 配置有效，BCBO-DE优于BCBO")
    else:
        print("❌ 配置需要进一步调整")

if __name__ == "__main__":
    test_configuration()
