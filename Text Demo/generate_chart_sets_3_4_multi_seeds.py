#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart Sets 3 和 4 多种子批量生成脚本
"""

import os
import sys
import subprocess
import time
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_single_seed(seed, chart_set):
    """生成单个种子的数据"""
    validation_dir = os.path.join(BASE_DIR, f'multi_seed_validation_set{chart_set}')
    os.makedirs(validation_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"[INFO] 生成 Chart Set {chart_set}, Seed {seed}")
    print(f"[TIME] {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        cmd = [sys.executable, 'generate_bcbo_bcbo_ga_comparison.py',
               '--chart-set', str(chart_set), '--seed', str(seed)]

        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=1200)

        if result.returncode != 0:
            print(f"[ERROR] Seed {seed} 失败")
            return False

        source_file = os.path.join(BASE_DIR, 'BCBO_vs_BCBO_GA_Data', f'chart_set_{chart_set}_bcbo_comparison.json')
        target_file = os.path.join(validation_dir, f'chart_set_{chart_set}_seed_{seed}.json')

        import shutil
        shutil.copy2(source_file, target_file)

        elapsed = time.time() - start_time
        print(f"[OK] Seed {seed} 完成！耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")

        return True

    except Exception as e:
        print(f"[ERROR] Seed {seed} 失败: {e}")
        return False

def main():
    print("="*80)
    print("Chart Sets 3 & 4 多种子生成")
    print("="*80)

    # 生成任务配置：(chart_set, seeds)
    tasks = [
        (3, [43, 44, 45, 46, 47, 48, 49, 50, 51]),  # Chart Set 3: seeds 43-51
        (4, [42, 43, 44, 45, 46, 47, 48, 49, 50, 51])  # Chart Set 4: seeds 42-51
    ]

    total_start = time.time()
    results = {}

    for chart_set, seeds in tasks:
        print(f"\n\n{'#'*80}")
        print(f"# 开始生成 Chart Set {chart_set}")
        print(f"{'#'*80}")

        success_count = 0
        for i, seed in enumerate(seeds, 1):
            print(f"\n[PROGRESS] Chart Set {chart_set}: {i}/{len(seeds)}")
            if generate_single_seed(seed, chart_set):
                success_count += 1

        results[chart_set] = {
            'total': len(seeds),
            'success': success_count,
            'failed': len(seeds) - success_count
        }

    total_elapsed = time.time() - total_start

    # 最终报告
    print("\n\n" + "="*80)
    print("批量生成完成")
    print("="*80)
    for chart_set, stats in results.items():
        print(f"Chart Set {chart_set}: {stats['success']}/{stats['total']} 成功")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
