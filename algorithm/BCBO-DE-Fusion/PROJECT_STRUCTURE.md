# BCBO-DE-Fusion 项目结构

## 目录说明

```
BCBO-DE-Fusion/
├── README.md                              # 项目主文档
├── 算法融合分析与实施计划.md               # 算法设计文档
├── test_setup.py                          # 环境验证脚本
├── .gitignore                             # Git忽略配置
│
├── config/                                # 配置模块
│   ├── __init__.py                        # 导出配置接口
│   ├── fusion_config.py                   # BCBO六阶段配置
│   └── parameters.py                      # DE/融合/实验参数
│
├── core/                                  # 核心算法模块
│   ├── __init__.py                        # 导出核心类
│   └── bcbo_de_embedded.py                # ⭐BCBO-DE融合算法实现
│
├── de_operators/                          # DE算子库
│   ├── __init__.py                        # 导出DE算子
│   ├── mutation_strategies.py             # 变异策略(DE/rand/1等)
│   ├── crossover_strategies.py            # 交叉策略(二项式、指数)
│   ├── selection.py                       # 选择策略(贪婪、锦标赛)
│   └── de_operators_wrapper.py            # 统一封装接口
│
├── utils/                                 # 工具模块
│   ├── __init__.py                        # 导出工具类
│   ├── diversity_calculator.py            # 多样性计算器
│   ├── adaptive_controller.py             # 自适应参数控制
│   └── performance_monitor.py             # 性能监控器
│
├── experiments/                           # 实验模块
│   ├── __init__.py
│   └── scripts/
│       ├── __init__.py
│       └── run_bcbo_de.py                 # 实验运行脚本
│
└── tests/                                 # 测试模块
    ├── __init__.py
    └── integration_tests/
        ├── __init__.py
        └── test_bcbo_de_integration.py    # 集成测试脚本
```

## 模块说明

### 1. config/ - 配置模块
- **fusion_config.py**: BCBO六阶段划分和融合阶段定义
- **parameters.py**: DE参数、融合策略参数、实验配置

### 2. core/ - 核心算法
- **bcbo_de_embedded.py**: BCBO-DE嵌入式融合算法的主要实现

### 3. de_operators/ - DE算子库
- **mutation_strategies.py**: 实现多种变异策略
- **crossover_strategies.py**: 实现交叉策略
- **selection.py**: 实现选择策略
- **de_operators_wrapper.py**: 提供统一的DEOperators类接口

### 4. utils/ - 工具模块
- **diversity_calculator.py**: 计算种群多样性(汉明距离、信息熵)
- **adaptive_controller.py**: HDE自适应参数控制(F、CR)
- **performance_monitor.py**: 记录和分析性能指标

### 5. experiments/ - 实验模块
- **run_bcbo_de.py**: 命令行实验脚本,支持参数配置

### 6. tests/ - 测试模块
- **test_bcbo_de_integration.py**: 集成测试,验证算法功能

## 文件统计

- Python源文件: 18个
- 配置文件: 1个(.gitignore)
- 文档文件: 3个(README.md, 算法设计文档, 本文档)

## 依赖关系

```
core.bcbo_de_embedded
    ├── config.fusion_config
    ├── config.parameters
    ├── de_operators (DEOperators)
    ├── utils.diversity_calculator
    ├── utils.adaptive_controller
    ├── utils.performance_monitor
    └── BCBO (外部依赖: ../BCBO/)
```

## 使用入口

1. **验证安装**: `python test_setup.py`
2. **运行实验**: `python experiments/scripts/run_bcbo_de.py`
3. **运行测试**: `python tests/integration_tests/test_bcbo_de_integration.py`
4. **Python API**: `from core.bcbo_de_embedded import BCBO_DE_Embedded`

---

**最后更新**: 2025-01-15
**项目位置**: `D:\论文\更新部分\代码\混合算法优化方案\algorithm\BCBO-DE-Fusion`
