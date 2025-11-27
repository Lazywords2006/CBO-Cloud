#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行所有分析工具
=================
自动运行统计分析、箱线图、置信区间图表、能耗分析
(不包括消融实验和参数敏感性分析,因为它们需要长时间运行)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

CURRENT_DIR = Path(__file__).parent
SCRIPTS_DIR = CURRENT_DIR / 'scripts'

def run_script(script_name: str, description: str):
    """运行脚本"""
    print(f"\n{'='*80}")
    print(f"{description}".center(80))
    print(f"{'='*80}")

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        print(f"❌ 错误: 脚本不存在 {script_path}")
        return False

    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(CURRENT_DIR),
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print(f"\n✅ {description} 完成!")
            return True
        else:
            print(f"\n❌ {description} 失败! 返回码: {result.returncode}")
            return False

    except Exception as e:
        print(f"\n❌ 运行 {script_name} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("BCBO-DE分析工具套件 - 一键运行".center(80))
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查数据是否存在
    raw_data_dir = CURRENT_DIR / 'RAW_data'
    if not raw_data_dir.exists():
        print("❌ 错误: RAW_data目录不存在!")
        print("请先运行 update_all_data.py 生成实验数据")
        return

    # 检查数据文件
    data_files = list(raw_data_dir.glob("chart_set_*_merged_results.json"))
    if len(data_files) == 0:
        print("❌ 错误: 没有找到数据文件!")
        print("请先运行 update_all_data.py 生成实验数据")
        return

    print(f"✓ 找到 {len(data_files)} 个数据文件")
    print()

    # 运行快速分析工具
    tasks = [
        ("statistical_analysis.py", "1. 统计显著性检验分析"),
        ("generate_boxplots.py", "2. 箱线图生成"),
        ("generate_charts_with_ci.py", "3. 带置信区间的收敛曲线"),
        ("energy_analysis_enhanced.py", "4. 能耗分析增强")
    ]

    results = {}
    for script_name, description in tasks:
        success = run_script(script_name, description)
        results[description] = success

    # 打印总结
    print("\n" + "="*80)
    print("运行总结".center(80))
    print("="*80)

    for description, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{description:40s} {status}")

    print()
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 打印输出位置
    print("\n📁 输出文件位置:")
    print(f"  - 统计分析: {CURRENT_DIR / 'statistical_analysis'}")
    print(f"  - 箱线图: {CURRENT_DIR / 'boxplots'}")
    print(f"  - 置信区间图表: {CURRENT_DIR / 'publication_charts_with_ci'}")
    print(f"  - 能耗分析: {CURRENT_DIR / 'energy_analysis'}")

    print("\n⚠️  注意:")
    print("  消融实验和参数敏感性分析需要单独运行 (耗时较长):")
    print("  - python scripts/ablation_study.py")
    print("  - python scripts/parameter_sensitivity.py")

    success_count = sum(results.values())
    total_count = len(results)

    if success_count == total_count:
        print("\n🎉 所有分析任务完成!")
    else:
        print(f"\n⚠️  {total_count - success_count}/{total_count} 个任务失败,请检查错误信息")

if __name__ == "__main__":
    main()
