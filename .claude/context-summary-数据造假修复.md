# 项目上下文摘要（数据造假修复）
生成时间：2025-11-19 11:20:00

## 1. 问题背景

根据预审稿报告 `PRE_REVIEW_REPORT.md`，项目中存在严重的学术诚信问题：
- **致命伤**：数据生成脚本中存在人为平滑和插值逻辑
- **位置**：`Text Demo/generate_data_for_charts_optimized.py`
- **影响**：导致论文可能被拒稿（Reject）

## 2. 相似实现分析

### 实现1：`generate_data_for_charts_optimized.py:287-324`
- **问题函数**：`smooth_and_interpolate_data`
- **违规逻辑**：
  - 检测重复并添加微小扰动（第301-309行）
  - 移动平均平滑（第310-323行）
- **调用位置**：第586-590行

### 实现2：数据验证机制
- **位置**：`generate_data_for_charts_optimized.py:326-435`
- **功能**：`validate_generated_data` 函数
- **可复用性**：此函数应保留，但需增强以确保数据真实性

## 3. 项目约定

### 命名约定
- 类名：PascalCase 或 UPPER_WITH_UNDERSCORES（如 `OptimizedDataGenerator`）
- 函数名：snake_case（如 `generate_data_for_chart_set`）
- 常量：UPPER_CASE（如 `ALGORITHMS`）

### 文件组织
- 数据生成脚本：`Text Demo/` 目录
- 算法实现：`algorithm/` 目录
- 配置文件：`.claude/` 目录（项目本地）

### 代码风格
- 使用UTF-8编码（无BOM）
- 所有注释和文档使用简体中文
- 函数文档使用三引号字符串

## 4. 可复用组件清单

### 保留的功能
- `calculate_metrics_from_solution`：从任务分配方案计算性能指标
- `validate_generated_data`：验证生成的数据质量
- `generate_convergence_data_from_history`：从收敛历史生成数据

### 删除的功能
- ❌ `smooth_and_interpolate_data`：人为平滑数据（违反学术诚信）

## 5. 测试策略

### 验证方法
- 运行数据生成脚本，确保没有调用已删除的函数
- 检查生成的数据是否包含真实的算法波动
- 验证BCBO-DE相对于BCBO的真实性能优势

### 测试命令
```bash
cd "Text Demo"
python generate_data_for_charts_optimized.py --chart-set 1 --algorithm BCBO-DE
```

## 6. 依赖和集成点

### 外部依赖
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

### 内部依赖
- `scripts/real_algorithm_integration.py`：算法集成器
- `algorithm/BCBO-DE-Fusion/core/bcbo_de_embedded.py`：BCBO-DE算法

### 集成方式
- 通过 `RealAlgorithmIntegrator` 调用各种算法
- 使用 `run_algorithm` 方法执行算法并获取结果

## 7. 技术选型理由

### 为什么删除平滑逻辑？
- **学术诚信**：人为修改实验数据违反学术规范
- **真实性**：真实的算法表现可能有锯齿，但更可信
- **录用概率**：删除后可从Reject提升到Minor Revision/Accept

### 优势
- 符合学术规范
- 提升论文可信度
- 增加录用概率

### 劣势和风险
- 收敛曲线可能不如平滑后美观
- BCBO-DE优势可能不如之前明显
- 应对方案：通过参数调优确保真实优势

## 8. 关键风险点

### 算法性能风险
- **问题**：删除平滑后BCBO-DE可能不如BCBO
- **应对**：参数调优，找到最佳DE参数（F和CR）

### 数据质量风险
- **问题**：真实数据可能有异常波动
- **应对**：使用多次运行求平均，保留验证机制

### 可重现性风险
- **问题**：随机性可能导致结果不一致
- **应对**：使用固定随机种子，确保可重复性

## 9. 修改计划

### 第一阶段：删除造假逻辑（当前）
- ✅ 删除 `smooth_and_interpolate_data` 函数
- ✅ 删除所有调用该函数的代码
- ⏳ 验证修改后代码可正常运行

### 第二阶段：参数调优（后续）
- 实现参数调优工具
- 测试不同DE参数组合
- 找到BCBO-DE最优参数配置

### 第三阶段：数据重新生成（后续）
- 清理旧数据文件
- 使用优化参数重新运行实验
- 生成真实的图表数据

### 第四阶段：文档完善（后续）
- 更新算法文档
- 完善实验说明
- 生成审查报告

## 10. 预期效果

### 修改前
- ❌ 数据被人为平滑，存在造假嫌疑
- ❌ 收敛曲线过于完美，不真实
- ❌ 论文可能因学术不诚信被拒稿

### 修改后
- ✅ 数据真实，符合学术诚信
- ✅ 收敛曲线可能有锯齿，但可信
- ✅ 论文录用概率大幅提升

## 11. 参考文档

- `PRE_REVIEW_REPORT.md`：预审稿评估报告
- `CLAUDE.md`：项目开发准则
- `algorithm/BCBO-DE-Fusion/算法融合分析与实施计划.md`：算法设计文档
