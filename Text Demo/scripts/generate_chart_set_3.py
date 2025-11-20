#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表集3生成脚本 - 迭代次数 vs 性能指标 (第二实验集)
专门生成第三组图表：不同参数设置下的迭代次数与性能指标关系分析
"""

import sys
import os
from datetime import datetime
import traceback

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加路径（确保可以导入 BCBO 包）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PYTHON_DIR = os.path.join(BASE_DIR, '..', '..', '程序', 'python')
PROJECT_PYTHON_DIR = os.path.abspath(PROJECT_PYTHON_DIR)

if PROJECT_PYTHON_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PYTHON_DIR)

def generate_chart_set_3():
    """生成第三组图表：迭代次数 vs 性能指标 (第二实验集)"""
    try:
        print("🔄 开始生成图表集3 - 迭代次数 vs 性能指标 (第二实验集)")
        print("=" * 70)
        
        # 检查数据可用性
        raw_data_dir = os.path.join(BASE_DIR, '..', 'RAW_data')
        raw_data_dir = os.path.abspath(raw_data_dir)
        if not os.path.exists(raw_data_dir):
            print(f"❌ RAW_data目录不存在: {raw_data_dir}")
            return False
        
        # 检查图表集3的数据文件
        json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json') and 'chart_set_3' in f]
        if len(json_files) < 4:  # 至少需要4个算法的数据
            print(f"❌ 图表集3数据不完整，只有 {len(json_files)} 个文件")
            return False
        
        print(f"✅ 检测到 {len(json_files)} 个数据文件")
        
        # 导入可视化模块
        try:
            from bcbo_visualization import BCBOVisualizer
            print("✅ BCBOVisualizer导入成功")
        except ImportError as e:
            print(f"❌ BCBOVisualizer导入失败: {e}")
            return False
        
        # 设置输出目录
        results_dir = os.path.join(BASE_DIR, '..', 'results')
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'charts'), exist_ok=True)
        
        # 初始化可视化器
        visualizer = BCBOVisualizer(save_dir=results_dir, config_mode="paper")
        print("✅ BCBOVisualizer初始化成功")
        
        # 生成图表
        print("\n📊 生成图表集3 - 迭代次数 vs 性能指标 (第二实验集)...")
        chart_path = visualizer.create_figure11_large_iteration_analysis()
        
        if chart_path:
            print(f"✅ 图表集3生成成功: {chart_path}")
            
            # 生成完成报告（保存到tables文件夹）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tables_dir = os.path.join(results_dir, 'tables')
            os.makedirs(tables_dir, exist_ok=True)
            report_path = os.path.join(tables_dir, f"chart_set_3_report_{timestamp}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("图表集3生成报告\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"图表类型: 迭代次数 vs 性能指标 (第二实验集)\n")
                f.write(f"数据文件数量: {len(json_files)}\n")
                f.write(f"生成文件: {chart_path}\n")
                f.write(f"使用的算法: BCBO, GA, PSO, ACO, FA, CS, GWO\n")
                f.write(f"实验参数: M=80任务, N=12虚拟机, n=40种群大小\n")
            
            print(f"📄 报告已保存: {report_path}")
            return True
        else:
            print("❌ 图表生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 图表集3生成工具")
    print("专门用于生成：迭代次数 vs 性能指标 (第二实验集) 图表")
    print("=" * 70)
    
    success = generate_chart_set_3()
    
    if success:
        print("\n✅ 图表集3生成完成！")
    else:
        print("\n❌ 图表集3生成失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()