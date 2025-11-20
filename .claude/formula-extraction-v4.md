# Word文档数学公式提取报告
**文档**: BCBO-DE融合算法理论基础与数学模型_v4_格式修正.docx
**提取时间**: 2025-11-19
**公式总数**: 51

---

## 1. 公式统计

| 公式类型 | 数量 |
|----------|------|
| BCBO算法 | 4 |
| DE算法 | 2 |
| 任务执行时间 | 3 |
| 其他 | 30 |
| 成本计算 | 2 |
| 融合策略 | 2 |
| 负载均衡 | 5 |
| 资源利用率 | 1 |
| 适应度函数 | 2 |

---

## 2. 公式详细列表

### BCBO算法 (4个)

#### BCBO算法 - 公式 1
**内容**:
```
1. ×*嵌入式反转融合策略**：不同于传统在算法弱搜索阶段进行融合的策略，本文将DE算子嵌入BCBO的强搜索阶段（后60%迭代），充分发挥DE的局部精细搜索优势。
```

#### BCBO算法 - 公式 2
**内容**:
```
3. ×*自适应种群划分机制**：基于种群多样性动态调整BCBO与DE的种群比例，在低多样性时增强DE探索，在高多样性时增强BCBO收敛。
```

#### BCBO算法 - 公式 3
**编号**: 5.3

**内容**:
```
1. 按适应度降序排序种群
2. splitₚoint = floor(n × ρbcbo)
3. BCBO组 = 前splitₚoint个个体（高适应度）
4. DE组 = 剩余个体（低适应度）
    (5.3)
```

#### BCBO算法 - 公式 4
**编号**: 5.5

**内容**:
```
ρbcbo = {
  0.50,  if D < 0.3  (低多样性→增强DE探索)
  0.80,  if D > 0.7  (高多样性→增强BCBO收敛)
  0.70,  otherwise   (默认平衡)
}    (5.5)
```

### DE算法 (2个)

#### DE算法 - 公式 1
**编号**: 5.2

**内容**:
```
applyDE = (rand() < Ifusion(phase))    (5.2)
```

#### DE算法 - 公式 2
**内容**:
```
- DE算子执行：O((1-ρ) × n × M)
```

### 任务执行时间 (3个)

#### 任务执行时间 - 公式 1
**编号**: 2.1

**内容**:
```
ETᵢ,ⱼ = (CPUᵢ + Memᵢ + Storageᵢ + Networkᵢ) ÷ MIPSⱼ    (2.1)
```

#### 任务执行时间 - 公式 2
**编号**: 2.3

**内容**:
```
Loadⱼ = Σᵢ:xᵢ=j ETᵢ,ⱼ    (2.3)
```

#### 任务执行时间 - 公式 3
**内容**:
```
[1] M. E. Abd El-aziz, S. Xiong, K. P. N. Jayasena, L. Li. Task scheduling ∈ cloud computing based on hybrid moth search algorithm and differential evolution. Knowledge-Based Systems, 2019, 169: 39-52.
```

### 其他 (30个)

#### 其他 - 公式 1
**内容**:
```
4. ×*多维综合适应度函数**：设计了包含makespan、负载均衡、资源利用率、总成本、能耗、SLA违约和资源违约等7个维度的综合优化目标。
```

#### 其他 - 公式 2
**内容**:
```
云任务调度问题可以形式化定义为：给定M个任务集合T = {t₁, t₂, ..., tₘ}和N个虚拟机集合VM = {vm₁, vm₂, ..., vmₙ}，寻找一个任务到虚拟机的分配方案X = [x₁, x₂, ..., xₘ]，其中xᵢ ∈ {0, 1, ..., N-1}表示第i个任务分配到的虚拟机索引，使得综合优化目标（以makespan为主）最小化。
```

#### 其他 - 公式 3
**编号**: 2.5

**内容**:
```
U = (Uᶜᵖᵘ + Uᵐᵉᵐ) ÷ 2    (2.5)
```

#### 其他 - 公式 4
**编号**: 3.1

**内容**:
```
Phase(t) = {
  dynamicₛearch,   if t < 0.10T
  staticₛearch,    if 0.10T ≤ t < 0.20T
  encircledynamic, if 0.20T ≤ t < 0.45T
  encircleₛtatic,  if 0.45T ≤ t < 0.65T
  attackdynamic,   if 0.65T ≤ t < 0.85T
  attackₛtatic,    if t ≥ 0.85T
}    (3.1)
```

#### 其他 - 公式 5
**编号**: 3.2

**内容**:
```
r = 3.0 × (1 - progress)    (3.2)
```

#### 其他 - 公式 6
**编号**: 3.3

**内容**:
```
θ = rand(0, 2π)
rₛpiral = r × exp(β × θ)
xₙew = xbest + rₛpiral × cos(θ) + rₛpiral × sin(θ)
    (3.3)
```

#### 其他 - 公式 7
**内容**:
```
其中β为螺旋形状参数（自适应），progress = t/T为迭代进度。
```

#### 其他 - 公式 8
**编号**: 3.4

**内容**:
```
策略选择 = {
  负载均衡优化,  概率 0.5
  邻域搜索,      概率 0.5
}    (3.4)
```

#### 其他 - 公式 9
**编号**: 3.5

**内容**:
```
rcoyote = 1.5 × (1 - progress)
rbadger = 1.0 × (1 - progress²)
    (3.5)
```

#### 其他 - 公式 10
**编号**: 3.6

**内容**:
```
direction = xₜarget - xcurrent
step = floor(direction × rcoyote)
xₙew = (xcurrent + step) mod N
    (3.6)
```

#### 其他 - 公式 11
**编号**: 3.7

**内容**:
```
xₙew = {
  xₜarget,                概率 rbadger × 1.2
  中值(xcurrent, xₜarget), 否则随机选择
}    (3.7)
```

#### 其他 - 公式 12
**编号**: 3.8

**内容**:
```
对于选定的任务 i:
  对于 offset ∈ {-1, +1}:
    testᵥm = (currentᵥm + offset) mod N
    if f(testₛolution) > f(currentₛolution):
      接受新分配
    (3.8)
```

#### 其他 - 公式 13
**编号**: 3.9

**内容**:
```
EliteSize = max(2, floor(0.2 × n))    (3.9)
```

#### 其他 - 公式 14
**编号**: 3.10

**内容**:
```
1. 按任务大小排序
2. 将任务分为 numₛegments (2-3) 段
3. 每段以50%概率交换父代基因
    (3.10)
```

#### 其他 - 公式 15
**编号**: 3.11

**内容**:
```
2-opt: 随机交换两个任务的VM分配，保留更优解
智能变异: 将最忙VM上的任务迁移到最闲VM（30%概率）
    (3.11)
```

#### 其他 - 公式 16
**编号**: 4.1

**内容**:
```
Vᵢ = Xᵣ₁ + F × (Xᵣ₂ - Xᵣ₃)    (4.1)
```

#### 其他 - 公式 17
**内容**:
```
其中r₁ ≠ r₂ ≠ r₃ ≠ i为随机选择的不同个体索引，F为缩放因子。
```

#### 其他 - 公式 18
**编号**: 4.2

**内容**:
```
vᵢ,ⱼ = clip(round(Vᵢ,ⱼ), 0, N-1)    (4.2)
```

#### 其他 - 公式 19
**编号**: 4.3

**内容**:
```
Uᵢ,ⱼ = {
  Vᵢ,ⱼ,  if rand() < CR 或 j = jᵣand
  Xᵢ,ⱼ,  otherwise
}    (4.3)
```

#### 其他 - 公式 20
**编号**: 4.4

**内容**:
```
Xᵢᵗ⁺¹ = {
  Uᵢ,  if f(Uᵢ) > f(Xᵢᵗ)
  Xᵢᵗ, otherwise
}    (4.4)
```

#### 其他 - 公式 21
**内容**:
```
参数设置：Fₘax = 0.4，Fₘin = 0.15，实现前期强探索、后期精细开发。
```

#### 其他 - 公式 22
**编号**: 4.6

**内容**:
```
CR(D) = {
  0.7,  if D < 0.3  (低多样性→高CR→更多变异)
  0.3,  if D > 0.7  (高多样性→低CR→保持结构)
  0.5,  otherwise
}    (4.6)
```

#### 其他 - 公式 23
**编号**: 5.4

**内容**:
```
Dₚopulation = (1 ÷ n(n-1)) × Σᵢ Σⱼ>ᵢ (dH(Xᵢ, Xⱼ) ÷ M)
dH(Xᵢ, Xⱼ) = Σₖ 𝟙(xᵢ,ₖ ≠ xⱼ,ₖ)
    (5.4)
```

#### 其他 - 公式 24
**编号**: 6.1

**内容**:
```
O(n × M) per iteration    (6.1)
```

#### 其他 - 公式 25
**内容**:
```
变异：O(n × M)，交叉：O(n × M)，选择：O(n × M)
```

#### 其他 - 公式 26
**编号**: 6.2

**内容**:
```
Tₜotal = O(n × M × T)    (6.2)
```

#### 其他 - 公式 27
**内容**:
```
- 多样性计算：O(n² × M)
```

#### 其他 - 公式 28
**编号**: 6.3

**内容**:
```
S = O(n × M) + O(T)  ÷÷ 种群存储 + 历史记录    (6.3)
```

#### 其他 - 公式 29
**内容**:
```
[2] Qin et al. ERTH scheduler: enhanced red-tailed hawk algorithm for multi-cost optimization ∈ cloud task scheduling. Artificial Intelligence Review, 2024.
```

#### 其他 - 公式 30
**内容**:
```
[10] Crepinsek, M., Liu, S.H., Mernik, M. Exploration and exploitation ∈ evolutionary algorithms: A survey. ACM Computing Surveys, 2013.
```

### 成本计算 (2个)

#### 成本计算 - 公式 1
**编号**: 2.7

**内容**:
```
Cost = Σⱼ UCⱼ × Loadⱼ × (1 + 0.1 × Loadⱼ/Makespan)    (2.7)
```

#### 成本计算 - 公式 2
**编号**: 2.10

**内容**:
```
f(X) = 10000/(Makespan+1) + wₗb×gₗb + wᵤ×U - wc×Cost - wₑ×Energy - wₛla×SLA - wᵣv×RV    (2.10)
```

### 融合策略 (2个)

#### 融合策略 - 公式 1
**内容**:
```
2. ×*渐进式融合强度控制**：采用阶段递增的融合强度（30%→50%→70%→90%），避免突然融合导致的性能冲击，实现平滑过渡。
```

#### 融合策略 - 公式 2
**编号**: 5.1

**内容**:
```
Ifusion(phase) = {
  0.0,  if phase ∈ {dynamicₛearch, staticₛearch}
  0.3,  if phase = encircledynamic
  0.5,  if phase = encircleₛtatic
  0.7,  if phase = attackdynamic
  0.9,  if phase = attackₛtatic
}    (5.1)
```

### 负载均衡 (5个)

#### 负载均衡 - 公式 1
**编号**: 2.2

**内容**:
```
Makespan = max{Loadⱼ}, j = 1, 2, ..., N    (2.2)
```

#### 负载均衡 - 公式 2
**编号**: 2.4

**内容**:
```
LI = σ(ActiveLoads) ÷ μ(ActiveLoads)    (2.4)
```

#### 负载均衡 - 公式 3
**内容**:
```
其中ActiveLoads表示负载大于0的虚拟机集合，σ为标准差，μ为均值。
```

#### 负载均衡 - 公式 4
**编号**: 2.8

**内容**:
```
Energy = Σⱼ (BaseEnergy + LoadEnergy) ÷ EEⱼ    (2.8)
```

#### 负载均衡 - 公式 5
**编号**: 2.9

**内容**:
```
SLAPenalty = 0.1 × Makespan,  if Makespan > 1.5 × MeanLoad    (2.9)
```

### 资源利用率 (1个)

#### 资源利用率 - 公式 1
**编号**: 2.6

**内容**:
```
Uᶜᵖᵘ = Σⱼ CPUⱼᵘˢᵉᵈ ÷ Σⱼ CPUⱼᶜᵃᵖ    (2.6)
```

### 适应度函数 (2个)

#### 适应度函数 - 公式 1
**编号**: 4.5

**内容**:
```
F(t) = Fₘin + (Fₘax - Fₘin) × (T - t) ÷ T    (4.5)
```

#### 适应度函数 - 公式 2
**内容**:
```
Algorithm: BCBO-DE Embedded Fusion
Input: M, N, n, T, 融合参数
Output: bestₛolution

1. population ← initializeₚopulation()
2. for t = 0 to T-1 do
3.   phase ← determineₚhase(t, T)
4.   isfusion ← phase ∈ FUSIONPHASES
5.   intensity ← getfusionᵢntensity(phase)
6.
7.   if isfusion ∧ rand() < intensity then
8.     ÷÷ BCBO-DE融合更新
9.     sortedₚop ← sortbyfitness(population)
10.    bcboᵣatio ← adaptiveₛplit(diversity(population))
11.    splitₚoint ← floor(n × bcboᵣatio)
12.
13.    bcbogroup ← sortedₚop[0:splitₚoint]
14.    degroup ← sortedₚop[splitₚoint:n]
15.
16.    bcboᵤpdated ← bcboₚhaseᵤpdate(bcbogroup, phase)
17.    deᵤpdated ← deₒperatorsᵤpdate(degroup)
18.
19.    population ← bcboᵤpdated ∪ deᵤpdated
20.  else
21.    ÷÷ 纯BCBO更新
22.    population ← bcboₚhaseᵤpdate(population, phase)
23.  end if
24.
25.  updateglobalbest(population)
26.  recordhistory(t, population, bestfitness)
27. end for
28. return bestₛolution
```
