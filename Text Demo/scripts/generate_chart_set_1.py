#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表集1生成脚本 - 迭代次数 vs 性能指标
专门生成第一组图表：迭代次数与性能指标的关系分析
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

print(f"[INFO] Checking Python path: {PROJECT_PYTHON_DIR}")
print(f"[INFO] Path exists: {os.path.exists(PROJECT_PYTHON_DIR)}")

if PROJECT_PYTHON_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PYTHON_DIR)
    print(f"[OK] Added path to sys.path: {PROJECT_PYTHON_DIR}")

# 检查BCBO目录
BCBO_DIR = os.path.join(PROJECT_PYTHON_DIR, 'BCBO')
print(f"[INFO] Checking BCBO directory: {BCBO_DIR}")
print(f"[INFO] BCBO directory exists: {os.path.exists(BCBO_DIR)}")

# 检查可视化文件
visualization_file = os.path.join(BCBO_DIR, 'bcbo_visualization.py')
print(f"[INFO] Checking visualization file: {visualization_file}")
print(f"[INFO] Visualization file exists: {os.path.exists(visualization_file)}")

def generate_chart_set_1():
    """生成第一组图表：迭代次数 vs 性能指标"""
    try:
        print("[START] Generating chart set 1 - Iterations vs Performance Metrics")
        print("=" * 60)
        
        # 检查数据可用性（scripts/子目录需要向上一级）
        raw_data_dir = os.path.join(BASE_DIR, '..', 'RAW_data')
        raw_data_dir = os.path.abspath(raw_data_dir)
        if not os.path.exists(raw_data_dir):
            print(f"[ERROR] RAW_data directory not found: {raw_data_dir}")
            return False
        
        # 检查图表集1的数据文件
        json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json') and 'chart_set_1' in f]
        if len(json_files) < 4:  # 至少需要4个算法的数据
            print(f"[ERROR] Chart set 1 data incomplete, only {len(json_files)} files")
            return False
        
        print(f"[OK] Found {len(json_files)} data files")
        
        # 导入可视化模块
        try:
            from bcbo_visualization import BCBOVisualizer
            print("[OK] BCBOVisualizer imported successfully")
        except ImportError as e:
            print(f"[ERROR] BCBOVisualizer import failed: {e}")
            return False
        
        # 设置输出目录（scripts/子目录需要向上一级）
        results_dir = os.path.join(BASE_DIR, '..', 'results')
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'charts'), exist_ok=True)
        
        # 初始化可视化器
        visualizer = BCBOVisualizer(save_dir=results_dir, config_mode="paper")
        print("[OK] BCBOVisualizer initialized successfully")
        
        # 生成图表
        print("\n[START] Generating chart set 1 - Iterations vs Performance...")
        chart_path = visualizer.create_figure9_iteration_analysis()
        
        if chart_path:
            print(f"[SUCCESS] Chart set 1 generated successfully: {chart_path}")
            
            # 生成完成报告（保存到tables文件夹）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tables_dir = os.path.join(results_dir, 'tables')
            os.makedirs(tables_dir, exist_ok=True)
            report_path = os.path.join(tables_dir, f"chart_set_1_report_{timestamp}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("图表集1生成报告\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"图表类型: 迭代次数 vs 性能指标\n")
                f.write(f"数据文件数量: {len(json_files)}\n")
                f.write(f"生成文件: {chart_path}\n")
                f.write(f"使用的算法: BCBO, GA, PSO, ACO, FA, CS, GWO\n")
            
            print(f"[SAVED] Report saved: {report_path}")
            return True
        else:
            print("[ERROR] Chart generation failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error during generation: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("Chart Set 1 Generation Tool")
    print("专门用于生成：迭代次数 vs 性能指标图表")
    print("=" * 60)
    
    success = generate_chart_set_1()
    
    if success:
        print("\n[SUCCESS] Chart set 1 generation completed!")
    else:
        print("\n[ERROR] Chart set 1 generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()