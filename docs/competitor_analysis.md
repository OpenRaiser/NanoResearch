# Automated Research Systems: Competitor & Benchmark Analysis

## 一、核心竞品系统

| 系统 | 机构 | 发表venue | Pipeline阶段 | 测试规模 | LLM backbone | 是否真实执行代码 | 代码成功率 | 评估方式 | 人工评审 | 核心指标 | 单篇成本 | 关键发现/亮点 | 局限性 |
|------|------|-----------|-------------|---------|-------------|----------------|-----------|---------|---------|---------|---------|-------------|--------|
| AI Scientist v1 | Sakana AI | arXiv 2024.08 | Idea→Code→Run→Write→Review | 3个模板域(NanoGPT/Diffusion/Grokking), 每模板~50 ideas | Claude 3.5/GPT-4o/Llama-3.1-405B/DeepSeek | 是(小规模实验) | ~58%(42%因代码错误失败) | LLM reviewer(GPT-4o, 5轮self-reflection×5 ensemble) | 否 | 1-10分reviewer评分; AUC=0.65对标人类审稿 | <$15/paper | 首个端到端自动科研系统; 成本极低 | 仅小数据集; 文献引用中位数仅5篇; 无法处理复杂实验 |
| MLR-Copilot | 未知 | arXiv 2024.08 | IdeaAgent→ExperimentAgent→Code Execution | 5个ML任务 | RL-tuned LLM | 是(HuggingFace原型代码) | 未报告 | 任务指标(如Pearson r) + 专家评估 | 是(领域专家) | task-specific metrics | 未报告 | RL优化idea生成; 代码检索 | 仅5个任务; 规模小 |
| AIDE | Weco AI | arXiv 2025.02 | Code-space tree search | Kaggle竞赛 + MLE-Bench + RE-Bench | 未详述 | 是 | 未报告 | Kaggle排名/Benchmark得分 | 间接(vs Kaggle参赛者) | 超越51.38%人类Kaggle选手; 50%竞赛排名高于中位数 | 未报告 | 实际Kaggle竞赛验证; 非论文生成而是ML工程 | 不生成论文; 聚焦ML竞赛 |
| Curie | U Michigan | arXiv 2025.02 | Architect Agent + Technician Agent(多agent) | 46个问题, 4个CS领域 | 未详述 | 是 | 未报告 | 4维评分(Design/Setup/Alignment/Correctness) | Ground truth对比 | 结论正确性3.4x提升(vs最强baseline) | 未报告 | 多agent架构; 实验严谨性模块 | 仅46题; 不生成完整论文 |
| Google AI Co-Scientist | Google | arXiv 2025.02 | Generate-Debate-Evolve(多agent) | 15个生物医学目标, 3个领域 | Gemini 2.0 | 否(提假设, 人类验证) | N/A | 自动Elo rating + 领域专家评审 | 是(专家+湿实验验证) | 专家评novelty高于人类; AML药物候选在细胞系验证; AMR结果复现数年实验 | 未报告 | 湿实验验证; 真实药物发现 | 仅生物医学; 不执行代码; 需人类完成实验 |
| AI Scientist v2 | Sakana AI | ICLR 2025 Workshop | Agentic tree search + 去模板化 | 多个ML领域 | Claude 3.5 Sonnet/GPT-4o | 是 | 未报告 | LLM reviewer + 真实peer review | 是(ICLR Workshop) | 3篇提交Workshop, 1篇被接收(score 6.33) | 未报告 | 首篇AI全自动生成论文通过同行评审 | Workshop级别非主会; 实验规模仍有限 |
| ResearchAgent | Microsoft Research | NAACL 2025 | Problem→Method→Experiment(迭代+ReviewingAgent反馈) | 300篇core papers(2024.05后) | GPT-4/Claude等 | 否(仅生成idea) | N/A | 5维评分(1-5) + human-model Spearman | 是(人工标注) | human-model Spearman ρ=0.83; 75%+ pairwise优于ablation | 未报告 | 学术知识图谱增强; 高human-model一致性 | 仅idea生成, 不执行实验 |
| AI-Researcher | HKU | NeurIPS 2025 Spotlight | Literature Review→Hypothesis→Algorithm→Experiment→Paper | 22篇benchmark论文, 2个难度级别, 4个AI领域 | GPT-4o/Claude-3.5/Gemini-1.5 | 是 | 93.8%完成率(Level 2) | 5个LLM做judge, 7分制(-3~+3)对比人类论文 | 间接(对比已发表论文) | Novelty/Comprehensiveness/Theory/Analysis多维评分; 100%实现完成率(L2) | 未报告 | NeurIPS Spotlight; 高完成率; Scientist-Bench | 仅22篇规模; 主要在算法改进类任务 |
| Agent Laboratory | 未知 | EMNLP 2025 Findings | Literature Review→Experimentation→Report Writing(PhD/Postdoc agents) | 多个ML topic | 未详述 | 是 | 未报告 | 人类研究者评估报告质量 | 是 | 支持不同程度human involvement | 未报告 | 可调节人类参与程度 | 规模和指标报告不详 |
| AlphaEvolve | Google DeepMind | arXiv 2025.06 | Gemini驱动的进化代码搜索 | 多个算法发现问题 | Gemini | 是 | 未报告 | vs已知数学结果 | 间接 | 发现新算法并改进已知算法 | 未报告 | DeepMind出品; 进化搜索 | 仅算法发现; 不生成论文 |
| AlphaResearch | 未知 | arXiv 2025.11 | Dual环境(执行验证+模拟peer review) | 8个算法问题(AlphaResearchComp) | 未详述 | 是 | 未报告 | vs人类+vs baseline系统 | 间接(vs published best) | 2/8 win rate vs人类; circle packing(n=26,32)最优 | 未报告 | 算法发现; 奖励模型训练自ICLR reviews | 仅数学/算法问题 |
| DeepScientist | 未知 | ICLR 2026 | Bayesian Opt + Hypothesize-Verify-Analyze循环 + Findings Memory | 3个前沿任务, 16×H800跑1个月 | 未详述 | 是(大规模GPU) | 5000+ ideas生成, 1100验证, 21产出创新 | 对比人类SOTA | 间接(vs published SOTA) | 超越人类SOTA: Agent Failure +183.7%, LLM Inference +1.9%, AI Detection +7.9% | 极高(月级GPU) | 首个在前沿任务超越人类的系统 | 仅3个任务; 成本极高; 任务需可自动验证 |

## 二、评估Benchmark

| Benchmark | 机构 | 发表venue | 任务数量 | 覆盖领域 | 评估维度 | 评估方法 | 人工验证 | 当前最佳系统表现 | 是否开源 | 关键发现 |
|-----------|------|-----------|---------|---------|---------|---------|---------|----------------|---------|---------|
| MLE-Bench | OpenAI | 2024 | 多个Kaggle竞赛 | ML工程(各类Kaggle任务) | Kaggle排名百分位 | 自动提交+评分 | 间接(vs Kaggle选手) | AIDE: 超越51.38%人类 | 是 | ML工程能力评估; 非论文评估 |
| RE-Bench | METR | 2024 | 多个研究工程任务 | ML研究工程 | 任务得分 | 自动评估 | 是(人类研究者做baseline) | AIDE: 与顶尖人类研究者竞争力相当 | 是 | 研究工程综合能力 |
| LitQA | FutureHouse | 2024 | ~250个问题 | 生物学文献问答 | 准确率 | 自动评估 | 是(领域专家) | Robin: ~90%(人类专家67%) | 未知 | 文献理解能力; 非端到端科研 |
| ScienceAgentBench | OSU NLP | ICLR 2025 | 102个任务 | 4个学科(Bioinformatics, Computational Chemistry, GIS, Psychology/CogNeuro) | Success Rate(SR) + CodeBERTScore(CBS) + API Cost | 自包含Python程序执行验证 | 是(9位领域专家, 多轮) | 最佳agent仅32.4%成功率(+专家知识34.3%) | [GitHub](https://github.com/OSU-NLP-Group/ScienceAgentBench) | 跨学科任务极具挑战; 提供去污染策略 |
| Si et al. Idea Eval | Stanford | ICLR 2025 | NLP prompting(单领域) | NLP | Novelty/Feasibility/Excitement/Effectiveness | 100+位NLP研究者盲审 | 是(大规模人工) | LLM ideas显著更novel(p<0.05), feasibility略弱 | 未知 | 仅评估idea生成, 非完整pipeline |
| Scientist-Bench | HKU | NeurIPS 2025(随AI-Researcher) | 22篇论文 | 4个AI领域, 2个创新难度级别 | 7分制(-3~+3): Novelty/Comprehensiveness/Theory/Analysis | 5个LLM做judge, 对比人类论文ground truth | 间接(对比已发表论文) | AI-Researcher: 93.8%完成率(L2) | [GitHub](https://github.com/HKUDS/AI-Researcher) | 两级难度设计好; 但规模偏小 |
| MLR-Bench | 未知 | NeurIPS 2025 D&B | 201个任务 | 9个ML方向(LLMs/VLMs, AI4Science, ML Theory, Trustworthy AI, CV, ML Systems, Multimodality, RL, Emerging) | 5维(Clarity/Novelty/Soundness/Significance/Overall) | MLR-Judge(Gemini+Claude双LLM, 9维评分) | 是(10位领域专家) | 所有agent均低于6.0接收线 | [GitHub](https://github.com/chchenhui/mlrbench) | ~80%存在结果编造; 当前系统远未达到接收水准 |
| AlphaResearchComp | 未知 | arXiv 2025(随AlphaResearch) | 8个问题 | 算法发现(circle packing, Littlewood polynomials, MSTD sets等) | vs人类最优解 + vs baseline系统 | 执行验证 + 模拟peer review(ICLR 2017-2024 reviews训练reward model) | 间接 | AlphaResearch: 2/8超越人类 | 未知 | 纯算法/数学; 可自动验证 |

## 三、NanoResearch 对标分析

| 维度 | 行业现状 | NanoResearch 当前状态 | 差距/优势 | 建议行动 |
|------|---------|---------------------|----------|---------|
| 测试规模 | MLR-Bench 201任务; AI Scientist ~150 runs | 单次测试中 | 需大幅扩展 | 批量跑50-100个topic |
| 真实代码执行 | AI Scientist 58%成功; MLR-Bench发现80%编造 | SLURM集群真实GPU训练 | 核心优势 | 统计成功率, 强调0%编造 |
| 评估标准化 | MLR-Judge 9维; Scientist-Bench 7分制 | 无标准化评估 | 需建立 | 采用MLR-Judge或自建评估体系 |
| 人工评审 | AI Scientist v2过Workshop审; Stanford 100+人盲审 | 无 | 必须补充 | 邀请3-5位研究者评审10+篇 |
| Baseline对比 | 各系统互相对比 | 无 | 必须补充 | 至少对比AI Scientist v1(开源) |
| Pipeline完整度 | 多数系统缺环节(AIDE无论文, ResearchAgent无实验) | 9阶段完整pipeline + 文献搜索 + Figure生成 | 核心优势 | 突出完整性 |
| 多模型协作 | 多数单一LLM | DeepSeek+Claude+GPT+Gemini多模型 | 优势 | 做model ablation证明价值 |
| 成本 | AI Scientist <$15/paper | 未统计 | 需统计 | 记录API+GPU成本 |
| 领域覆盖 | ScienceAgentBench 4学科; MLR-Bench 9方向 | ML为主 | 可扩展 | 覆盖NLP/CV/RL/tabular/AI4Sci |
