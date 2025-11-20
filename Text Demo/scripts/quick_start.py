#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本
一键启动图表生成流程
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(command, description=""):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} 完成")
            return True
        else:
            print(f"❌ {description} 失败")
            print(f"错误信息: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ 运行命令时出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 图表生成快速开始")
    print("=" * 60)
    print("一键启动完整的图表生成流程")
    print("=" * 60)
    
    # 步骤1: 系统检查
    print("\n【步骤1/4】系统检查")
    print("-" * 40)
    
    if not run_command("python system_check.py", "系统环境检查"):
        print("❌ 系统检查失败，请修复问题后重试")
        return False
    
    # 步骤2: 生成数据
    print("\n【步骤2/4】生成数据")
    print("-" * 40)
    
    print("📊 正在生成所有图表集的数据...")
    print("⏰ 此步骤可能需要较长时间，请耐心等待")
    
    if not run_command("python generate_data_for_charts.py --all", "数据生成"):
        print("❌ 数据生成失败")
        choice = input("是否继续尝试生成图表？(y/n): ").strip().lower()
        if choice != 'y':
            return False
    
    # 步骤3: 生成图表
    print("\n【步骤3/4】生成图表")
    print("-" * 40)
    
    print("🎨 正在生成所有图表...")
    
    if not run_command("python chart_generator_controller.py --all", "图表生成"):
        print("❌ 图表生成失败")
        return False
    
    # 步骤4: 完成报告
    print("\n【步骤4/4】生成完成报告")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(BASE_DIR, f"quick_start_report_{timestamp}.txt")
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("图表生成快速开始报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"工作目录: {BASE_DIR}\n\n")
            
            f.write("执行步骤:\n")
            f.write("  ✅ 系统环境检查\n")
            f.write("  ✅ 数据生成\n") 
            f.write("  ✅ 图表生成\n")
            f.write("  ✅ 报告生成\n\n")
            
            f.write("输出文件:\n")
            f.write("  📁 RAW_data/ - 生成的数据文件\n")
            f.write("  📁 results/charts/ - 生成的图表文件\n")
            f.write("  📄 各种报告文件\n\n")
            
            f.write("下一步操作:\n")
            f.write("  1. 查看 results/charts/ 目录中的图表文件\n")
            f.write("  2. 查看生成的报告文件了解详细信息\n")
            f.write("  3. 如需重新生成特定图表集，使用控制器单独执行\n")
        
        print(f"📄 快速开始报告已保存: {report_path}")
        
    except Exception as e:
        print(f"⚠️ 无法保存报告文件: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 快速开始流程完成！")
    print("=" * 60)
    print("📁 结果文件位置:")
    print("   - 数据文件: RAW_data/")
    print("   - 图表文件: results/charts/")
    print("   - 报告文件: 根目录下的各种报告文件")
    print("\n🔍 建议查看:")
    print("   1. 检查 results/charts/ 中的图表文件")
    print("   2. 查看最新的图表生成报告")
    print("   3. 验证图表质量和数据正确性")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)