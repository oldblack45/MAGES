# 囚徒困境策略识别实验

## 实验概述

这个实验旨在测试认知Agent是否能够通过观察对手的行为模式，准确识别出对手采用的策略。实验使用经典的囚徒困境博弈作为测试环境。

## 实验设计

### 测试策略
实验测试认知Agent对以下策略的识别能力：

1. **一报还一报 (Tit-for-Tat)**
   - 第一轮合作
   - 之后模仿对手上轮的行为

2. **总是合作 (Always-Cooperate)**
   - 无论对手如何行动，始终选择合作

3. **总是背叛 (Always-Defect)**
   - 无论对手如何行动，始终选择背叛

4. **随机策略 (Random)**
   - 50%概率合作，50%概率背叛

### 认知框架
认知Agent使用以下技术来识别策略：

1. **假设推理系统**
   - 多步预测推演
   - 决策满意度评估

2. **侧写建模系统**
   - Action-Reaction-Strategy-Experience 四元组
   - 动态权重调整

3. **策略假设评估**
   - 基于历史行为模式的策略匹配
   - 置信度计算

## 文件结构

```
PrisonerDilemma/
├── entity/
│   ├── prisoner_dilemma_world.py      # 游戏环境和规则
│   ├── cognitive_prisoner_agent.py    # 认知Agent实现
│   └── tit_for_tat_agent.py          # 各种策略Agent
├── strategy_identification_experiment.py  # 实验框架
└── README.md                          # 说明文档
```

## 运行实验

### 快速开始
```python
from simulation.examples.PrisonerDilemma.strategy_identification_experiment import run_quick_test

# 运行快速测试（单策略，20轮，1场游戏）
result = run_quick_test()
```

### 完整实验
```python
from simulation.examples.PrisonerDilemma.strategy_identification_experiment import run_full_test

# 运行完整测试（所有策略，30轮，2场/策略）
results = run_full_test()
```

### 自定义实验
```python
from simulation.examples.PrisonerDilemma.strategy_identification_experiment import StrategyIdentificationExperiment

experiment = StrategyIdentificationExperiment("custom_experiment")

# 自定义参数运行实验
results = experiment.run_full_experiment(
    rounds_per_game=50,     # 每场游戏轮数
    games_per_strategy=5    # 每种策略测试场次
)
```

## 评估指标

### 识别准确率
- **策略级准确率**: 每种策略的识别正确率
- **总体准确率**: 所有策略的平均识别正确率

### 置信度分析
- **平均置信度**: Agent对策略判断的平均置信度
- **置信度分布**: 不同置信度水平的分布情况

### 学习效率
- **收敛轮数**: Agent需要多少轮才能准确识别策略
- **稳定性**: 识别结果的一致性

## 预期结果

理想情况下，认知Agent应能：

1. **高准确率识别简单策略**
   - Always-Cooperate: >90%
   - Always-Defect: >90%

2. **准确识别复杂策略**
   - Tit-for-Tat: >80%
   - Random: >60%

3. **快速收敛**
   - 10轮内识别简单策略
   - 20轮内识别复杂策略

## 技术特点

### 认知建模优势
1. **多维度分析**: 结合行为模式、历史数据和假设推理
2. **自适应学习**: 根据新观察动态调整策略假设
3. **不确定性处理**: 使用置信度量化识别的可靠性

### 实验设计优势
1. **对照实验**: 使用已知策略作为ground truth
2. **重复实验**: 多场游戏验证识别稳定性
3. **量化评估**: 客观的准确率和置信度指标

## 应用价值

这个实验框架可以用于：

1. **AI策略识别能力评估**
2. **博弈论研究工具**
3. **认知建模系统测试**
4. **多智能体系统研究**

## 扩展可能

1. **更多策略类型**: 添加更复杂的策略进行测试
2. **动态策略**: 测试识别策略变化的能力
3. **多轮竞赛**: 模拟锦标赛环境
4. **噪声环境**: 在有干扰的环境中测试鲁棒性