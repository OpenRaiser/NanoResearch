# NanoResearch 竞争力分析与实验设计方案

## 一、竞争力判断

### 1.1 挑战

领域发展极快。DeepScientist 已上 ICLR 2026，AI-Researcher 拿了 NeurIPS 2025 Spotlight。如果 NanoResearch 单纯定位成"又一个端到端科研系统"，很难有竞争力——审稿人会直接问"比 AI Scientist 好在哪？"

### 1.2 机会

MLR-Bench 的核心发现为我们打开了窗口：当前系统 80% 存在结果编造，AI Scientist 42% 代码执行失败。行业痛点不是"能不能生成论文"，而是"生成的论文是否可信"。

NanoResearch 的独特优势恰好命中这个痛点：

| 优势 | 说明 |
|------|------|
| 真实 GPU 训练 | SLURM 集群执行，不是 toy experiment |
| Writing grounding system | 强制引用真实结果数字，防止 LLM 编造 |
| Contract validation | metrics 传递链格式校验，确保数据完整 |
| Debug loop | 自动修复代码错误，提升执行成功率 |
| 多模型协作 | DeepSeek+Claude+GPT+Gemini 各取所长 |

### 1.3 建议定位

不要定位成"更好的 AI Scientist"，而是：

> Grounded Autonomous Research: 解决 AI 科研系统中实验结果可信度问题的端到端框架

核心 claim：现有系统生成的论文大量编造实验结果，我们通过 grounded execution pipeline 实现真实实验 + 真实结果引用，显著提升生成论文的科学可信度。

---

## 二、实验设计方案

### Exp 1: 大规模可靠性测试（主实验）

选 50 个 topic，覆盖 5 个领域（NLP/CV/RL/Tabular/AI4Science），每个领域 10 个。

记录指标：

| 指标 | 含义 |
|------|------|
| Pipeline 完成率 | 多少 topic 走完全部 9 个阶段 |
| 各阶段成功率 | IDEATION/PLANNING/.../REVIEW 每阶段通过率 |
| 代码执行成功率 | SLURM job 成功完成比例 |
| 结果引用准确率 | paper 中数字 vs 训练 log 实际数字的一致性 |
| 结果编造率 | paper 中是否包含训练 log 中不存在的结果 |
| 平均成本 | API 费用 + GPU 小时 |
| 平均耗时 | 从 topic 到完整论文的端到端时间 |

目的：直接回应 MLR-Bench 的 80% 编造发现。如果做到 0% 编造 + 高执行成功率，就是很强的 selling point。

### Exp 2: Baseline 对比

同一批 50 个 topic，对比以下系统：

| 系统 | 说明 |
|------|------|
| AI Scientist v1 | 开源直接跑，最直接的竞品 |
| NanoResearch (ours) | 完整 pipeline |
| NanoResearch w/o grounding | 去掉 writing grounding，观察编造率变化 |
| 纯 LLM 写论文 | 只给 topic 让 Claude/GPT 直接写，无实验 |

评估维度沿用社区标准（类似 MLR-Judge）：

| 维度 | 分值范围 | 说明 |
|------|---------|------|
| Clarity | 1-10 | 论文表述清晰度 |
| Novelty | 1-10 | 研究想法新颖性 |
| Soundness | 1-10 | 方法和实验的科学严谨性 |
| Significance | 1-10 | 研究的影响力和重要性 |
| Overall | 1-10 | 综合评分 |
| Faithfulness | 0/1 | 结果引用忠实度（额外维度，我们的核心指标） |

用 3 个 LLM 做 judge（GPT-4o + Claude + Gemini），取平均。

### Exp 3: Pipeline Ablation

验证各模块设计的必要性：

| 变体 | 去掉什么 | 预期影响 |
|------|---------|---------|
| w/o literature search | IDEATION 不搜文献，直接用 LLM 知识 | novelty 下降 |
| w/o debug loop | EXECUTION 代码出错不自动修复 | 执行成功率下降 |
| w/o grounding | WRITING 不强制引用真实结果 | 编造率上升 |
| w/o multi-model | 全部用单一 LLM（如只用 GPT-4o） | 质量/成本 trade-off |
| w/o contract validation | 去掉 metrics 传递链的格式校验 | 结果准确率下降 |

每个变体跑 20 个 topic，对比完成率、结果准确率、论文质量分。

### Exp 4: 人工评审

从 50 篇生成论文中随机抽 15 篇，混入 5 篇真实论文（arXiv 近期同领域），邀请 5 位 ML 研究者做盲审：

| 评审内容 | 说明 |
|---------|------|
| 论文打分 | Novelty / Soundness / Clarity / Overall（1-10） |
| AI 判别 | 判断每篇论文是否为 AI 生成（Turing test） |
| 结果可信度 | 判断实验结果是否看起来真实可信 |
| 自由评论 | 开放式文字反馈 |

实验量不大（20 篇×5 人 = 100 份评审）但说服力很强，尤其 Turing test 和结果可信度部分。

### Exp 5: Failure Analysis

对 50 个 topic 中失败的 case 做系统分类：

| 分析维度 | 内容 |
|---------|------|
| 失败阶段分布 | 哪些阶段最容易失败？占比多少？ |
| 代码 bug 类型 | 语法/import/OOM/数据路径/逻辑错误等分布 |
| Debug 修复能力 | debug loop 修复成功率 vs 轮次曲线 |
| 领域难度差异 | 哪类 topic 最难？为什么？ |
| 成本-质量关系 | 成本 vs 论文质量分的散点图/拟合 |

输出几张分析图表。审稿人很看重对系统局限性的诚实分析。

### Exp 6: 迭代改进（Optional）

选 10 个 topic，跑 3 轮迭代：

第一轮结果 → 自动分析不足 → 调整方法/超参 → 第二轮 → 再分析 → 第三轮

展示论文质量分随迭代轮次的提升曲线。时间不够可以砍掉。

---

## 三、工作量估算

| 实验 | GPU 需求 | 时间估算 | 优先级 |
|------|---------|---------|--------|
| Exp 1: 50 topic 批量测试 | 50×(2-8 GPU-hours) = 100-400 GPU-hours | 1-2 周(并行) | P0 |
| Exp 2: Baseline 对比 | AI Scientist 同样 50 topic + 2 个 ablation 变体 | 1-2 周 | P0 |
| Exp 3: Pipeline ablation | 5 变体×20 topic = 100 runs | 1-2 周 | P0 |
| Exp 4: 人工评审 | 无 GPU，需协调 5 位评审者 | 1 周收集 | P1 |
| Exp 5: Failure analysis | 无额外 GPU，分析 Exp 1 结果 | 2-3 天 | P1 |
| Exp 6: 迭代改进 | 10×3 轮 = 30 runs | 1 周 | P2 |

总计约 3-4 周实验时间，500-1000 GPU-hours。

---

## 四、论文结构建议

| Section | 内容来源 |
|---------|---------|
| Introduction | 定位 grounded research，引用 MLR-Bench 80% 编造数据作为 motivation |
| Related Work | 竞品表格（competitor_analysis.md） |
| Method | 9 阶段 pipeline + grounding system + contract validation |
| Exp 1: Reliability | 50 topic 大规模测试，完成率/执行率/编造率 |
| Exp 2: Comparison | vs AI Scientist / 纯 LLM / w/o grounding |
| Exp 3: Ablation | 各模块贡献分析 |
| Exp 4: Human Eval | 盲审结果 + Turing test |
| Exp 5: Analysis | Failure analysis + 成本分析 |
| Conclusion | 强调可信度贡献，承认局限，future work |

---

## 五、执行优先序

1. 写批量测试脚本，能自动跑 50 个 topic 并收集所有指标
2. 同时部署 AI Scientist v1 作为 baseline
3. 并行跑 Exp 1 + Exp 2 + Exp 3（共享 topic 集合可复用）
4. 跑完后做 Exp 5 failure analysis
5. 写初稿
6. 安排 Exp 4 人工评审
7. 根据评审结果修改论文，视时间决定是否补 Exp 6
