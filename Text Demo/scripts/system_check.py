#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统检查脚本
验证图表生成系统的完整性和可用性
"""

import sys
import os
import importlib
from datetime import datetime

# 设置环境编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PYTHON_DIR = os.path.join(BASE_DIR, '..', '程序', 'python')
PROJECT_PYTHON_DIR = os.path.abspath(PROJECT_PYTHON_DIR)

if PROJECT_PYTHON_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PYTHON_DIR)

def check_python_environment():
    """检查Python环境"""
    print("🔍 检查Python环境...")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"工作目录: {BASE_DIR}")
    print(f"项目Python路径: {PROJECT_PYTHON_DIR}")
    
    return True

def check_directory_structure():
    """检查目录结构"""
    print("\n📁 检查目录结构...")
    
    required_dirs = [
        'RAW_data',
        'results',
        'results/charts',
        'results/tables'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(BASE_DIR, dir_path)
        if os.path.exists(full_path):
            print(f"✅ {dir_path}/ 目录存在")
        else:
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"✅ 创建目录: {dir_path}/")
            except Exception as e:
                print(f"❌ 无法创建目录 {dir_path}: {e}")
                return False
    
    return True

def check_module_availability():
    """检查模块可用性"""
    print("\n📦 检查模块可用性...")
    
    # 检查基础模块
    base_modules = ['numpy', 'json', 'datetime', 'time', 'traceback', 'random']
    for module_name in base_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} 模块可用")
        except ImportError:
            print(f"❌ {module_name} 模块不可用")
            return False
    
    # 检查可选模块
    optional_modules = ['scipy', 'sklearn']
    for module_name in optional_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} 模块可用")
        except ImportError:
            print(f"⚠️ {module_name} 模块不可用（可选）")
    
    # 检查项目特定模块
    project_modules = [
        ('real_algorithm_integration', 'RealAlgorithmIntegrator'),
        ('bcbo_visualization', 'BCBOVisualizer')
    ]
    
    for module_path, class_name in project_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                print(f"✅ {module_path}.{class_name} 可用")
            else:
                print(f"⚠️ {module_path} 可用，但 {class_name} 不可用")
        except ImportError as e:
            print(f"❌ {module_path} 模块导入失败: {e}")
            return False
    
    return True

def check_script_files():
    """检查脚本文件"""
    print("\n📝 检查脚本文件...")
    
    script_files = [
        'generate_chart_set_1.py',
        'generate_chart_set_2.py', 
        'generate_chart_set_3.py',
        'generate_chart_set_4.py',
        'chart_generator_controller.py',
        'generate_data_for_charts.py',
        'system_check.py',
        'README.md'
    ]
    
    all_exist = True
    for script_file in script_files:
        file_path = os.path.join(BASE_DIR, script_file)
        if os.path.exists(file_path):
            print(f"✅ {script_file} 存在")
        else:
            print(f"❌ {script_file} 不存在")
            all_exist = False
    
    return all_exist

def check_data_directory():
    """检查数据目录"""
    print("\n💾 检查数据目录...")
    
    raw_data_dir = os.path.join(BASE_DIR, 'RAW_data')
    
    if os.path.exists(raw_data_dir):
        json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json')]
        print(f"✅ RAW_data 目录存在，包含 {len(json_files)} 个JSON文件")
        
        # 按图表集统计
        chart_sets = {
            'chart_set_1': 0,
            'chart_set_2': 0, 
            'chart_set_3': 0,
            'chart_set_4': 0
        }
        
        for filename in json_files:
            for chart_set in chart_sets:
                if chart_set in filename:
                    chart_sets[chart_set] += 1
                    break
        
        print("\n各图表集数据文件统计:")
        for chart_set, count in chart_sets.items():
            status = "✅" if count >= 4 else "❌" if count == 0 else "⚠️"
            print(f"  {status} {chart_set}: {count} 个文件")
    else:
        print("⚠️ RAW_data 目录不存在（首次运行正常）")
    
    return True

def generate_system_report():
    """生成系统检查报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(BASE_DIR, f"system_check_report_{timestamp}.txt")
    
    checks = {
        "Python环境": check_python_environment(),
        "目录结构": check_directory_structure(),
        "模块可用性": check_module_availability(),
        "脚本文件": check_script_files(),
        "数据目录": check_data_directory()
    }
    
    print("\n" + "=" * 60)
    print("📊 系统检查摘要")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    print(f"\n总体状态: {'✅ 系统正常' if all_passed else '❌ 需要修复'}")
    
    # 保存报告
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("图表生成系统检查报告\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"工作目录: {BASE_DIR}\n\n")
            
            f.write("检查结果:\n")
            for check_name, result in checks.items():
                status = "通过" if result else "失败"
                f.write(f"  {check_name}: {status}\n")
            
            f.write(f"\n总体状态: {'正常' if all_passed else '需要修复'}\n")
            
            if all_passed:
                f.write("\n建议:\n")
                f.write("  1. 运行 'python chart_generator_controller.py' 开始生成图表\n")
                f.write("  2. 或运行 'python generate_data_for_charts.py --all' 先生成数据\n")
        
        print(f"\n📄 检查报告已保存: {report_path}")
    except Exception as e:
        print(f"\n❌ 无法保存检查报告: {e}")
    
    return all_passed

def main():
    """主函数"""
    print("🔧 图表生成系统检查工具")
    print("=" * 60)
    print("检查系统完整性和可用性")
    print("=" * 60)
    
    success = generate_system_report()
    
    if success:
        print("\n🎉 系统检查完成！系统已准备就绪。")
    else:
        print("\n⚠️  系统检查发现问题，请根据提示进行修复。")
        sys.exit(1)

if __name__ == "__main__":
    main()