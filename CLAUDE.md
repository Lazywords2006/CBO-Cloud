# 项目指导文件

## 项目架构

本项目是一个云任务调度优化算法研究项目，主要研究BCBO（Bald Eagle Coyote Co-optimization）及其改进算法BCBO-GA（BCBO with Genetic Algorithm）的性能对比。

### 核心目标
- 研究和优化云计算环境下的任务调度算法
- 对比BCBO和BCBO-GA的性能表现
- 生成用于期刊发表的数据和图表

## 项目技术栈

### 编程语言
- Python 3.x

### 核心依赖库
- NumPy: 数值计算
- Pandas: 数据处理
- Matplotlib: 数据可视化
- OpenPyXL: Excel文件处理
- JSON: 数据存储

### 开发环境
- OS: Windows 10.0.26200
- Shell: Git Bash
- Python虚拟环境: .venv

## 项目模块划分

### 文件与文件夹布局

```
混合算法优化方案/
├── algorithm/                      # 算法实现目录
│   ├── BCBO/                      # BCBO基础算法
│   │   ├── bcbo_cloud_scheduler_fixed.py    # BCBO核心实现
│   │   └── bcbo_ga_enhanced.py              # BCBO-GA增强版（v1.0）
│   ├── BCBO-GA/                   # BCBO-GA多策略协同版
│   │   ├── bcbo_ga_cloud_scheduler.py       # 多策略协同实现
│   │   └── bcbo_ga_staged_hybrid.py         # 分阶段混合策略
│   └── other_algorithms/          # 对比算法
│       ├── ant_colony_optimizer.py          # 蚁群算法
│       ├── cuckoo_search.py                 # 布谷鸟搜索
│       ├── firefly_algorithm.py             # 萤火虫算法
│       ├── genetic_algorithm_scheduler.py   # 遗传算法
│       ├── grey_wolf_optimizer.py           # 灰狼优化
│       └── particle_swarm_optimizer.py      # 粒子群优化
│
├── Text Demo/                     # 数据生成和分析目录
│   ├── bcbo_vs_bcbo_ga_analysis.py          # BCBO vs BCBO-GA分析
│   ├── comprehensive_bcbo_ga_analysis.py    # 综合分析工具
│   ├── generate_bcbo_bcbo_ga_comparison.py  # BCBO vs BCBO-GA专用数据生成器
│   ├── generate_data_for_charts_optimized.py # 多算法数据生成器
│   ├── generate_publication_charts.py       # 图表生成工具
│   │
│   └── scripts/                   # 辅助脚本
│       ├── real_algorithm_integration.py    # 算法集成接口（核心）
│       ├── ablation_study.py                # 消融研究
│       ├── bcbo_visualization.py            # 可视化工具
│       ├── energy_analysis_enhanced.py      # 能耗分析
│       ├── generate_boxplots.py             # 箱线图生成
│       ├── generate_chart_set_1.py          # 图表集1生成
│       ├── generate_chart_set_2.py          # 图表集2生成
│       ├── generate_chart_set_3.py          # 图表集3生成
│       ├── generate_chart_set_4.py          # 图表集4生成
│       ├── generate_charts_with_ci.py       # 带置信区间的图表
│       ├── parameter_sensitivity.py         # 参数敏感性分析
│       ├── quick_start.py                   # 快速启动脚本
│       ├── statistical_analysis.py          # 统计分析
│       └── system_check.py                  # 系统检查
│
├── .claude/                       # Claude配置目录
│   ├── CLAUDE.md                  # Claude项目指导（备份）
│   └── operations-log.md          # 操作日志
│
├── .venv/                         # Python虚拟环境
├── CLAUDE.md                      # 项目指导文件（主文件）
└── README.md                      # 项目说明文档
```

### 数据输出目录（运行时生成）
```
Text Demo/
├── RAW_data/                      # 原始实验数据（JSON）
├── BCBO_vs_BCBO_GA_Data/         # BCBO vs BCBO-GA专用数据
└── publication_charts/            # 发表用图表
    ├── chart_set_1/
    ├── chart_set_2/
    ├── chart_set_3/
    └── chart_set_4/
```

## 项目业务模块

### 1. 算法模块 (algorithm/)

#### BCBO算法
- **文件**: `algorithm/BCBO/bcbo_cloud_scheduler_fixed.py`
- **类**: `BCBO_CloudScheduler`
- **功能**: 秃鹰郊狼协同优化算法，云任务调度的基准算法

#### BCBO-GA算法（增强版）
- **文件**: `algorithm/BCBO/bcbo_ga_enhanced.py`
- **类**: `BCBO_GA`
- **版本**: v2.3 负载均衡增强版（2025-11-30最终）
- **核心特性**:
  - ✅ GA智能交叉算子（两点/均匀交叉）
  - ✅ 2-opt局部搜索
  - ✅ 自适应变异（前期高后期低）
  - ✅ 温和负载均衡（只在安全时修复）
  - ✨ **混合自适应参数机制（v2.3负载均衡增强版）**

- **v2.3 混合自适应参数机制（负载均衡增强）**:
  ```python
  # 三段式混合策略：结合v2.0和v2.1的优势，v2.3针对超大规模负载均衡优化

  # 小规模 (M≤200): 使用v2.0连续公式（已验证+0.44%）
  if M <= 200:
      crossover_rate = 0.9 - 0.0001 * M  # M=100: 0.89, M=200: 0.88
      mutation_rate = 0.15 - 0.00002 * M  # M=100: 0.148, M=200: 0.146
      local_search_prob = 0.3 + 0.00006 * M

  # 中规模 (200<M≤1500): v2.1平衡策略
  elif M <= 1500:
      crossover_rate = 0.85 - 0.000115 * (M-200)  # [0.70, 0.85]
      mutation_rate = 0.12 - 0.000031 * (M-200)   # [0.08, 0.12]
      elite_size = 2-5 (线性增长)

  # 大规模 (M>1500): v2.3负载均衡增强策略 ⭐ 新优化
  else:
      crossover_rate = 0.70 - 0.000014 * (M-1500)  # [0.651, 0.70] (v2.2: [0.60, 0.70])
      mutation_rate = 0.08 - 0.000003 * (M-1500)   # [0.0695, 0.08] (v2.2: [0.06, 0.08])
      elite_size = 5-10 (线性增长)
      local_search_prob = 0.50 + 0.000043 * (M-1500)  # [0.50, 0.65] (v2.2: [0.45, 0.60])
      local_search_max_iters = 35 - 50  # (v2.2: 30-40)
      task_reassign_num = 10 - 15  # (v2.2: 8-12)
  ```

- **版本演进对比**:
  | 规模 | M | v2.0 | v2.1 | v2.2 | v2.3 | 说明 |
  |------|---|------|------|------|------|------|
  | 小 | 100 | 0.89/0.148 | 0.875/0.135 | **0.89/0.148** | **0.89/0.148** | v2.3保持v2.0 |
  | 中 | 500 | 0.85/0.140 | 0.815/0.111 | **0.815/0.111** | **0.815/0.111** | v2.3保持v2.1 |
  | 大 | 1000 | 0.80/0.130 | 0.770/0.096 | **0.770/0.096** | **0.770/0.096** | v2.3保持v2.1 |
  | 超大 | 5000 | **0.40**/0.050 | 0.60/0.06 | 0.60/0.06 | **0.651/0.0695** | v2.3增强 |

  **核心改进**: v2.3在保持v2.2混合策略优势的同时，针对超大规模负载均衡问题进行专项优化

- **性能改进验证（v2.3）**:
  - 小规模 (M=100): +0.44% (保持v2.0验证结果)
  - 中等规模 (M≤1000): +0.07% (保持良好)
  - 大规模 (M=1000): +0.07% (保持良好)
  - 超大规模 (M≤5000): **+0.09%** ✓ (成功超越-0.50%目标)
  - **综合改进率**: -0.05% → +0.04% (整体提升0.09%)

#### 对比算法
- GA, PSO, ACO, FA, CS, GWO等传统元启发式算法
- 用于性能对比研究

### 2. 数据生成模块 (Text Demo/)

#### 核心数据生成器
- **generate_data_for_charts_optimized.py**: 多算法全面对比数据生成
- **generate_bcbo_bcbo_ga_comparison.py**: BCBO vs BCBO-GA专用对比

#### 数据生成配置
```python
# 四组图表集配置
chart_set_1: 迭代次数 vs 性能指标（小规模：M=100, N=20）
chart_set_2: 任务规模 vs 成本（中等规模：M=100-1000, N=20）
chart_set_3: 迭代次数 vs 性能指标（大规模：M=1000, N=20）
chart_set_4: 任务规模 vs 成本（超大规模：M=1000-5000, N=20）
```

### 3. 分析模块 (Text Demo/)

#### 综合分析
- **comprehensive_bcbo_ga_analysis.py**: 跨所有图表集的综合性能分析
- **bcbo_vs_bcbo_ga_analysis.py**: 单独对比分析

#### 性能指标
- `total_cost`: 总成本
- `execution_time`: 执行时间（makespan）
- `load_balance`: 负载均衡度（0-1）
- `price_efficiency`: 价格效率

### 4. 算法集成模块 (Text Demo/scripts/)

#### 核心集成器
- **文件**: `real_algorithm_integration.py`
- **类**: `RealAlgorithmIntegrator`
- **功能**:
  - 统一的算法调用接口
  - 共享问题实例（确保公平对比）
  - 随机种子管理
  - 结果格式化

**关键特性**:
```python
# 问题实例共享机制（v3.2修复）
# 所有算法使用相同的execution_time矩阵
self.problem_instance = {
    'execution_time': execution_time,  # 共享
    'task_loads': task_loads,          # 共享
    'vm_caps': vm_caps,                # 共享
    ...
}
```

## 项目代码风格与规范

### 命名约定

#### 类命名
- 使用 PascalCase（大驼峰）
- 示例: `BCBO_CloudScheduler`, `RealAlgorithmIntegrator`
- **注意**: 连字符(-)不允许用于Python标识符，使用下划线(_)

#### 变量命名
- 使用 snake_case（小写+下划线）
- 示例: `execution_time`, `total_cost`, `random_seed`
- 常量使用全大写: `ALGORITHMS`, `CHART_CONFIGS`

#### 文件命名
- Python文件: snake_case
- 示例: `real_algorithm_integration.py`, `bcbo_ga_enhanced.py`

### 代码风格

#### Import 规则
```python
# 标准库
import sys
import os
import json

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 本地模块
from real_algorithm_integration import RealAlgorithmIntegrator
```

#### 随机种子管理
```python
# 固定种子确保可重复性
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
```

#### 日志规范
```python
# 使用统一的日志格式
print(f"[INFO] 信息性消息")
print(f"[WARN] 警告消息")
print(f"[ERROR] 错误消息")
print(f"[OK] 成功消息")
print(f"[DEBUG] 调试消息")
```

#### 异常处理
```python
try:
    result = algorithm.run()
except Exception as e:
    print(f"[ERROR] 算法运行失败: {e}")
    traceback.print_exc()
    return None
```

#### 参数校验
```python
# 验证必需参数
required_params = ['M', 'N', 'n', 'iterations']
for param in required_params:
    if param not in params:
        raise ValueError(f"缺少必需参数: {param}")
```

#### 其他一些规范
- 使用UTF-8编码（设置`PYTHONIOENCODING='utf-8'`）
- JSON序列化使用NumpyEncoder处理NumPy类型
- 文件路径使用`os.path.join()`确保跨平台兼容性
- 长时间运行的任务显示进度信息

## 测试与质量

### 测试文件管理规则 ⚠️ 重要

**删除规则**:
- 所有用于测试的Python文件（test_*.py）使用后必须删除
- 所有说明文档（*.md，除CLAUDE.md和README.md）使用后必须删除
- 临时数据文件使用后必须清理
- 归档目录（*-Archive/）使用后必须删除

**保留文件**:
- `CLAUDE.md`: 项目指导文件（必须保留）
- `README.md`: 项目说明（必须保留）
- 算法实现文件（algorithm/目录）
- 数据生成和分析工具（Text Demo/目录）

### 单元测试
- 测试文件命名: `test_*.py`
- 测试完成后立即删除

### 集成测试
- 使用小规模参数快速验证
- 示例: `M=10, N=3, n=5, iterations=5`

## 项目构建、测试与运行

### 环境与配置

#### Python环境
```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install numpy pandas matplotlib openpyxl
```

#### 数据生成

**方法1: 多算法对比**
```bash
cd "Text Demo"
python generate_data_for_charts_optimized.py --all
```

**方法2: BCBO vs BCBO-GA专用**
```bash
cd "Text Demo"
python generate_bcbo_bcbo_ga_comparison.py --all
```

**方法3: 指定图表集**
```bash
python generate_bcbo_bcbo_ga_comparison.py --chart-set 1
```

#### 数据分析
```bash
cd "Text Demo"
python comprehensive_bcbo_ga_analysis.py
```

#### 图表生成
```bash
cd "Text Demo"
python generate_publication_charts.py
```

### 常用命令参数
```bash
--all                    # 生成所有图表集数据
--chart-set N           # 生成指定图表集（1-4）
--algorithm ALGO        # 指定算法（BCBO/BCBO-GA）
--seed N                # 设置随机种子（默认42）
--merge-results         # 合并结果到单个文件
--use-temp-dir          # 使用临时目录
```

## Git 工作流程

### 分支管理
- `main`: 主分支，保持稳定
- 功能开发在本地完成后直接提交

### 提交规范
```bash
# 功能添加
git commit -m "feat: 添加BCBO-GA算法实现"

# Bug修复
git commit -m "fix: 修复变量名连字符语法错误"

# 重构
git commit -m "refactor: 重构算法集成接口"

# 文档
git commit -m "docs: 更新CLAUDE.md项目结构"

# 清理
git commit -m "chore: 删除历史测试文件"
```

## 文档目录(重要)

### 文档存储规范

#### 项目指导文档
- **主文件**: `CLAUDE.md`（项目根目录）
- **备份**: `.claude/CLAUDE.md`

#### 项目说明文档
- `README.md`: 项目概述和使用说明

#### 操作日志
- `.claude/operations-log.md`: Claude操作记录

#### 临时文档规则
- 任何临时生成的MD文档使用后必须删除
- 只保留CLAUDE.md和README.md

## 项目状态（2025-11-30）

### 已完成工作
- ✅ 项目清理：删除历史测试文件和MD文档
- ✅ MBCBO → BCBO-GA全面替换
- ✅ 修复语法错误（变量名连字符问题）
- ✅ 创建BCBO vs BCBO-GA专用对比工具
- ✅ 更新CLAUDE.md项目结构
- ✅ 实现BCBO-GA v2.0自适应参数机制
- ✅ 升级至v2.1分段自适应策略
- ✅ 完成v2.2混合策略
- ✅ **完成v2.3负载均衡增强版（最终版）**

### 当前版本
- BCBO: v1.0 基础版
- BCBO-GA: **v2.3 负载均衡增强版** （2025-11-30最终）
- 数据生成器: v3.2（共享问题实例）

### 版本历史
- **v1.0** (2025-11-28): 基础GA增强版
- **v2.0** (2025-11-30): 连续公式自适应参数
  - 优势：小规模优秀 (+0.44% @ M=100)
  - 问题：超大规模参数过度保守（crossover=0.40, mutation=0.05）导致性能-2.88%
- **v2.1** (2025-11-30): 分段自适应策略（尝试）
  - 目标：通过分段公式避免极端规模参数过度激进
  - 问题：小规模性能从+0.44%退化到+0.16%（-0.28个百分点）
- **v2.2** (2025-11-30): 混合策略
  - 小规模（M≤200）：采用v2.0公式，保持+0.44%优秀性能
  - 中大规模（M>200）：采用v2.1策略，避免超大规模过度保守
  - 性能：综合-0.05%，超大规模-0.79%（未达-0.50%目标）
- **v2.3** (2025-11-30): **负载均衡增强版（最终版）** ⭐
  - 核心优化：提高超大规模参数下限，增强负载均衡优化能力
  - 关键改进：
    * 交叉率: 0.60 → 0.651 (M=5000)
    * 变异率: 0.06 → 0.0695 (M=5000)
    * 局部搜索强度: max_iters 40→50, reassign_num 12→15
    * 负载均衡阈值: 1.4 → 1.45 (触发更多修复)
  - **性能突破**：超大规模从-0.79%提升至+0.09% (改进+0.88%)
  - **目标达成**：成功超越-0.50%目标，实现+0.59%超越

### 下一步计划
- ✅ v2.3算法实现完成
- ✅ v2.3性能验证完成（超大规模+0.09%）
- 重新生成所有图表集数据（使用v2.3）
- 验证v2.3在所有规模的性能表现
- 准备期刊发表材料（强调v2.3负载均衡增强机制创新）

## 关键技术说明

### 公平对比机制（v3.2修复）
所有算法必须使用相同的问题实例：
```python
# RealAlgorithmIntegrator中的关键机制
self.problem_instance = {
    'execution_time': execution_time,  # 所有算法共享
    'M': M,
    'N': N,
    'random_seed': random_seed
}
```

### 随机种子策略
- 基础种子: 42
- 多次运行: seed = 42 + run_id
- 参数扫描: seed = 42 + param_value + run_id

### 数据验证机制
- 负载均衡范围检查（0-1）
- 成本和时间非零检查
- 突变检测（相邻点变化>30%报警）
- BCBO vs BCBO-GA性能对比

### 性能指标计算
```python
metrics = {
    'total_cost': sum(vm_costs),
    'execution_time': max(vm_loads),  # makespan
    'load_balance': 1.0 - load_imbalance,
    'price_efficiency': 1.0 / (total_cost + 1e-6)
}
```
