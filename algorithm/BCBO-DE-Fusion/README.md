# BCBO-DE融合算法项目

## 项目简介

本项目实现了**BCBO-DE嵌入式融合算法**,将Coyote-Bald Eagle协同优化算法(BCBO)与差分进化算法(DE)进行深度融合,用于云任务调度优化。

## 项目结构

```
BCBO-DE-Fusion/
├── README.md                              # 本文件
├── README_BCBO_DE_Embedded.md             # 详细使用文档
├── 算法融合分析与实施计划.md               # 算法设计文档
│
├── config/                                # 配置模块
│   ├── __init__.py
│   ├── fusion_config.py                   # BCBO六阶段配置
│   └── parameters.py                      # DE/融合/实验参数
│
├── core/                                  # 核心算法模块
│   ├── __init__.py
│   └── bcbo_de_embedded.py                # ⭐核心融合算法实现
│
├── de_operators/                          # DE算子库
│   ├── __init__.py
│   ├── mutation_strategies.py             # 变异策略
│   ├── crossover_strategies.py            # 交叉策略
│   ├── selection.py                       # 选择策略
│   └── de_operators_wrapper.py            # 统一接口封装
│
├── utils/                                 # 工具模块
│   ├── __init__.py
│   ├── diversity_calculator.py            # 多样性计算器
│   ├── adaptive_controller.py             # 自适应参数控制
│   └── performance_monitor.py             # 性能监控器
│
├── experiments/                           # 实验模块
│   └── scripts/
│       ├── __init__.py
│       └── run_bcbo_de.py                 # 实验运行脚本
│
└── tests/                                 # 测试模块
    ├── __init__.py
    └── integration_tests/
        ├── __init__.py
        └── test_bcbo_de_integration.py    # 集成测试
```

## 核心创新点

1. **嵌入式融合**: 在BCBO的特定阶段嵌入DE算法,而非简单的算法切换
2. **自适应种群划分**: 根据种群多样性动态调整BCBO/DE比例(70%/30%)
3. **六阶段协同**: 融合阶段(前20%) + 纯BCBO阶段(后80%)的协同优化
4. **HDE自适应参数**: F线性衰减 + CR基于多样性自适应

## 快速开始

### 方法1: Python API

```python
import sys
sys.path.insert(0, 'D:/论文/更新部分/代码/混合算法优化方案/algorithm/BCBO-DE-Fusion')

from core.bcbo_de_embedded import BCBO_DE_Embedded

# 创建优化器
optimizer = BCBO_DE_Embedded(
    M=100,         # 任务数
    N=20,          # VM数
    n=50,          # 种群大小
    iterations=100 # 迭代次数
)

# 运行优化
result = optimizer.run_fusion_optimization()

# 查看结果
print(f"最优适应度: {result['best_fitness']:.6f}")
print(f"优化诊断: {result['diagnosis']}")
```

### 方法2: 命令行运行

```bash
cd D:\论文\更新部分\代码\混合算法优化方案\algorithm\BCBO-DE-Fusion\experiments\scripts
python run_bcbo_de.py --M 100 --N 20 --iterations 100
```

### 方法3: 运行测试

```bash
cd D:\论文\更新部分\代码\混合算法优化方案\algorithm\BCBO-DE-Fusion
python tests/integration_tests/test_bcbo_de_integration.py
```

## 算法架构

### BCBO六阶段划分

| 阶段 | 名称 | 比例 | 类型 |
|------|------|------|------|
| 1 | dynamic_search | 10% | 融合阶段 |
| 2 | static_search | 10% | 融合阶段 |
| 3 | encircle_dynamic | 25% | 纯BCBO |
| 4 | encircle_static | 20% | 纯BCBO |
| 5 | attack_dynamic | 20% | 纯BCBO |
| 6 | attack_static | 15% | 纯BCBO |

### 融合策略

在融合阶段(dynamic_search, static_search):
- 种群按适应度排序
- 前70%使用BCBO搜索算子
- 后30%使用DE算子(变异→交叉→选择)
- 比例可根据多样性自适应调整

## 配置参数

### DE参数 (`config/parameters.py`)

```python
DE_CONFIG = {
    'F': 0.5,                  # 缩放因子
    'CR': 0.8,                 # 交叉概率
    'use_adaptive_F': True,    # 启用HDE自适应F
}
```

### 融合参数

```python
FUSION_CONFIG = {
    'bcbo_ratio': 0.7,                    # BCBO组70%
    'use_adaptive_split': True,           # 启用自适应划分
    'diversity_threshold_low': 0.3,       # 低多样性阈值
    'diversity_threshold_high': 0.7,      # 高多样性阈值
}
```

## 依赖关系

本项目依赖:
- **BCBO模块**: `../BCBO/bcbo_cloud_scheduler_fixed.py`
- Python 3.8+
- NumPy, SciPy, Matplotlib

## 文档说明

- **README_BCBO_DE_Embedded.md**: 详细的使用文档和API参考
- **算法融合分析与实施计划.md**: 算法设计原理和实验方案

## 更新日志

### v1.0.0 (2025-01-15)
- ✅ 完成BCBO-DE嵌入式融合算法核心实现
- ✅ 实现自适应种群划分机制
- ✅ 实现HDE自适应参数控制
- ✅ 完成集成测试框架
- ✅ 整合到algorithm统一目录

## 许可证

MIT License

---

**项目路径**: `D:\论文\更新部分\代码\混合算法优化方案\algorithm\BCBO-DE-Fusion`
**最后更新**: 2025-01-15
