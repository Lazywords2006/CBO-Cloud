#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart Set 4 多种子批量生成脚本（Seeds 42-51）
================================================
专门用于生成Chart Set 4（超大规模场景）的多种子验证数据

配置:
- Chart Set 4: M=1000-5000, N=20（任务规模扫描）
- Seeds: 42-51（10个种子）
- 目的: 验证算法稳定性和统计显著性
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# 设置UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_single_seed(seed):
    """生成单个种子的Chart Set 4数据"""
    validation_dir = os.path.join(BASE_DIR, 'multi_seed_validation_set4')
    os.makedirs(validation_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"[INFO] 生成 Chart Set 4, Seed {seed}")
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # 调用生成脚本
        cmd = [sys.executable, 'generate_bcbo_bcbo_ga_comparison.py',
               '--chart-set', '4', '--seed', str(seed)]

        print(f"[CMD] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"[ERROR] Seed {seed} 失败")
            print(f"[STDOUT] {result.stdout}")
            print(f"[STDERR] {result.stderr}")
            return False

        # 复制生成的文件到验证目录
        source_file = os.path.join(BASE_DIR, 'BCBO_vs_BCBO_GA_Data', 'chart_set_4_bcbo_comparison.json')
        target_file = os.path.join(validation_dir, f'chart_set_4_seed_{seed}.json')

        if not os.path.exists(source_file):
            print(f"[ERROR] 源文件不存在: {source_file}")
            return False

        import shutil
        shutil.copy2(source_file, target_file)

        # 验证文件内容
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            bcbo_count = len(data.get('algorithms', {}).get('BCBO', {}).get('results', []))
            bcbo_ga_count = len(data.get('algorithms', {}).get('BCBO-GA', {}).get('results', []))

        elapsed = time.time() - start_time
        print(f"[OK] Seed {seed} 完成！")
        print(f"  - BCBO数据点: {bcbo_count}")
        print(f"  - BCBO-GA数据点: {bcbo_ga_count}")
        print(f"  - 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"  - 输出文件: {target_file}")

        return True

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Seed {seed} 超时（>1小时）")
        return False
    except Exception as e:
        print(f"[ERROR] Seed {seed} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("Chart Set 4 多种子批量生成")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置:")
    print(f"  - Chart Set: 4 (超大规模，M=1000-5000)")
    print(f"  - Seeds: 42-51 (10个种子)")
    print(f"  - 输出目录: multi_seed_validation_set4/")
    print("="*80)

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    total_start = time.time()

    success_count = 0
    failed_seeds = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'#'*80}")
        print(f"# 进度: {i}/{len(seeds)} ({i*100/len(seeds):.1f}%)")
        print(f"# 当前种子: {seed}")
        print(f"{'#'*80}")

        if generate_single_seed(seed):
            success_count += 1
        else:
            failed_seeds.append(seed)

        # 显示中间统计
        if i < len(seeds):
            remaining = len(seeds) - i
            avg_time = (time.time() - total_start) / i
            estimated_remaining = remaining * avg_time
            print(f"\n[STATS] 已完成: {i}/{len(seeds)}, 成功: {success_count}, 失败: {len(failed_seeds)}")
            print(f"[ETA] 预计剩余时间: {estimated_remaining/60:.1f} 分钟")

    total_elapsed = time.time() - total_start

    # 最终报告
    print("\n\n" + "="*80)
    print("Chart Set 4 多种子生成完成")
    print("="*80)
    print(f"总任务数: {len(seeds)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_seeds)}")
    if failed_seeds:
        print(f"失败种子: {failed_seeds}")
    print(f"成功率: {success_count*100/len(seeds):.1f}%")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print(f"平均每个种子: {total_elapsed/len(seeds)/60:.1f} 分钟")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 生成汇总报告
    validation_dir = os.path.join(BASE_DIR, 'multi_seed_validation_set4')
    summary_file = os.path.join(validation_dir, 'generation_summary.json')

    summary = {
        'chart_set': 4,
        'seeds': seeds,
        'success_count': success_count,
        'failed_seeds': failed_seeds,
        'total_time_seconds': total_elapsed,
        'avg_time_per_seed_seconds': total_elapsed / len(seeds),
        'generation_timestamp': datetime.now().isoformat()
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 汇总报告已保存: {summary_file}")

    # 返回状态码
    return 0 if success_count == len(seeds) else 1

if __name__ == "__main__":
    sys.exit(main())
