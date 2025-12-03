#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart Set 2 多种子批量生成脚本
自动生成seeds 44-51的Chart Set 2数据
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(BASE_DIR, 'multi_seed_validation_set2')

def generate_single_seed(seed, chart_set=2):
    """生成单个种子的数据"""
    print(f"\n{'='*80}")
    print(f"[INFO] 开始生成 Chart Set {chart_set}, Seed {seed}")
    print(f"[INFO] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # 运行数据生成脚本
        cmd = [
            sys.executable,
            'generate_bcbo_bcbo_ga_comparison.py',
            '--chart-set', str(chart_set),
            '--seed', str(seed)
        ]

        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=1200  # 20分钟超时
        )

        if result.returncode != 0:
            print(f"[ERROR] Seed {seed} 生成失败:")
            print(result.stderr)
            return False

        # 复制结果文件
        source_file = os.path.join(BASE_DIR, 'BCBO_vs_BCBO_GA_Data', f'chart_set_{chart_set}_bcbo_comparison.json')
        target_file = os.path.join(VALIDATION_DIR, f'chart_set_{chart_set}_seed_{seed}.json')

        if not os.path.exists(source_file):
            print(f"[ERROR] 源文件不存在: {source_file}")
            return False

        # 复制文件
        import shutil
        shutil.copy2(source_file, target_file)

        elapsed_time = time.time() - start_time
        print(f"\n[OK] Seed {seed} 生成成功！")
        print(f"[TIME] 耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
        print(f"[SAVED] {target_file}")

        return True

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Seed {seed} 生成超时（超过20分钟）")
        return False
    except Exception as e:
        print(f"[ERROR] Seed {seed} 生成失败: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("Chart Set 2 多种子批量生成")
    print("="*80)
    print(f"目标种子: 44-51 (共8个)")
    print(f"输出目录: {VALIDATION_DIR}")
    print("="*80)

    # 确保输出目录存在
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    # 要生成的种子列表
    seeds = list(range(44, 52))  # 44-51

    total_start_time = time.time()
    success_count = 0
    failed_seeds = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n\n{'#'*80}")
        print(f"# 进度: {i}/{len(seeds)} - Seed {seed}")
        print(f"{'#'*80}")

        success = generate_single_seed(seed)

        if success:
            success_count += 1
        else:
            failed_seeds.append(seed)

        # 显示进度
        print(f"\n[PROGRESS] 已完成: {success_count}/{len(seeds)}")
        if failed_seeds:
            print(f"[WARNING] 失败种子: {failed_seeds}")

    total_elapsed = time.time() - total_start_time

    # 最终报告
    print("\n\n" + "="*80)
    print("批量生成完成")
    print("="*80)
    print(f"成功: {success_count}/{len(seeds)}")
    print(f"失败: {len(failed_seeds)}/{len(seeds)}")
    if failed_seeds:
        print(f"失败种子: {failed_seeds}")
    print(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"平均耗时: {total_elapsed/len(seeds):.2f} 秒/种子")
    print("="*80)

    return 0 if len(failed_seeds) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
