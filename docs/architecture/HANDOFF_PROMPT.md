# NanoResearch 架构改进开发 — 接手提示词

> 把下面的内容复制粘贴给新的 Claude 对话即可。

---

你好，我需要你帮我按照现有的架构改进计划，逐步实施 NanoResearch 项目的代码改进。

## 项目基本信息

- **项目路径**: `C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`
- **代码量**: ~30,000 行 Python
- **Pipeline**: IDEATION → PLANNING → SETUP → CODING → EXECUTION → ANALYSIS → FIGURE_GEN → WRITING → REVIEW（9 阶段）
- **LaTeX 编译器**: `D:/anaconda/Scripts/tectonic`
- **Git remote**: `https://github.com/OpenRaiser/NanoStudy-dev.git`
- **Config 路径**: `~/.nanobot/config.json`
- **Workspace**: `~/.nanobot/workspace/research/{session_id}/`

## 架构改进计划

项目里有一份完整的架构改进计划，已拆分成 10 个文件，在 `docs/architecture/` 下：

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `00-overview.md` | 总览、模块评分、10 阶段实施顺序、风险评估、开发检查清单 | 必读 |
| `01-analysis-rewrite.md` | Section 2: ANALYSIS 模块重写——统计检验、训练动态、消融分析、对比矩阵 | P0 |
| `02-review-citations.md` | Sections 3-4: 多模型审稿 + 引用事实核查 | P0 |
| `03-context-engine.md` | Section 5: ContextEngine + ResearchMemory + CrossRunMemory | P0 |
| `04-file-splitting-latex.md` | Sections 6-7: 大文件拆分方案 + 共享 LaTeX 修复器 | P1 |
| `05-prompts-search-coding.md` | Sections 8-10: Prompt 外部化、IDEATION 搜索自评、CODING 质量门 | P1-P2 |
| `06-infrastructure.md` | Sections 11-15: 成本追踪、常量集中、DAG 调度、进度流、结构化日志 | P2-P3 |
| `07-quality-validation.md` | Sections 16-18, 20: 论文质量基准、Blueprint 验证、Bug 修复、勘误 | P3 |
| `08-robustness.md` | Sections 21-26: 优雅关闭、临时文件清理、bare except、并发安全、检查点、SLURM | P1-P2 |
| `09-testing-consistency.md` | Sections 27-28 + 附录: 测试计划、一致性检查、最终目录结构 | — |

## 实施顺序（严格遵守）

这些文件之间有依赖关系，**不能随意跳着做**。必须按以下 10 个 Phase 顺序推进：

```
Phase 1  → 07 (Section 18 bug fixes) .............. 零风险点修复，热身
Phase 2  → 06 (Section 12 constants.py) + 04 (Section 7 shared LaTeX fixer)
Phase 3  → 01 (ANALYSIS 模块重写) .................. 新建 agents/analysis/ 包
Phase 4  → 03 (Context Engine + Memory) ............ 新建 nanoresearch/memory/ 包
Phase 5  → 02 (多模型审稿 + 引用核查) .............. 改 review.py
Phase 6  → 04 (Section 6 文件拆分) ................. 拆 execution.py/writing.py/experiment.py
Phase 7  → 05 (Section 8 Prompt 外部化) ............ 从拆分后的文件提取 prompt
Phase 8  → 05 (Sections 9-10 IDEATION + CODING 改进)
Phase 9  → 06 (Sections 11,14,15) + 07 (Sections 16-17) .. 基础设施
Phase 10 → 06 (Section 13 DAG 调度) ................ 最后做，opt-in
```

依赖关系解释：
- Phase 2 的 constants.py 和 LaTeX fixer 是基础设施，后续几乎所有 Phase 都依赖它
- Phase 6 的文件拆分必须等 Phase 3-5 的功能改动稳定后再做（否则 merge conflict）
- Phase 7 的 prompt 提取要在文件拆分之后做（从拆分后的小文件中提取）
- Phase 10 的 DAG 调度风险最高，放最后

## 开发规范

1. **先读代码再改代码**：每次修改前，先用 Read 工具读取要改的源文件，理解现有实现再动手。
2. **文档中的代码示例有已知问题**：之前的审查发现了约 12 个潜在 bug（变量名对不上、正则逻辑反转、返回值错误等）。不要盲目复制粘贴文档中的代码，用你自己的判断确保正确性。
3. **增量提交**：每完成一个独立改动就 git commit，不要批量提交。
4. **向后兼容**：大部分改动设计为 additive（增量添加）。`00-overview.md` 里有风险评估表，改动前先对照。
5. **测试**：每个 Phase 完成后跑 `python -m pytest tests/` 确保没有回归。

## 项目关键设计模式（必须了解）

### Figure Preservation
- WRITING 阶段把 figure 嵌入到 section.content 里（inline），不是放在模板的 `figures` 循环里
- PaperSkeleton 的 `figures=[]` 是 by design
- REVIEW 的 `_apply_revisions()` 必须保留 `\begin{figure}...\end{figure}` 块
- 两层防御：(1) prompt 告诉 LLM 保留，(2) 代码兜底重新插入被丢掉的 figure

### LaTeX 2-Level Fix Strategy
- Level 1: 确定性修复（Unicode、缺包、前文垃圾、环境不匹配）——不调用 LLM
- Level 2: LLM search-replace 修复——输出 JSON `[{"old":"...","new":"..."}]`，用 str.replace() 应用
- 修复前备份 tex 文件，全部失败则恢复备份

### Experiment Agent 双模式
- **Pipeline 模式**: 结构化流程（文件生成 → dry run → quick eval → 迭代）
- **ReAct 模式**: LLM 通过 tool calls 驱动（read/write/run/search/grep）

### Per-section Context Builder（已实现）
- `_build_core_context()` 生成共享上下文，5 个 section-specific builder 各取所需
- 每个 section 收到 ~8-15K context（而非之前的 ~30-40K 全量）

### Contribution Contract（已实现）
- Introduction 生成后提取 `\item` 声明，分类为 method/component/empirical
- 注入到后续 section 的写作 prompt 中确保前后一致

## 历史修复记录

项目经过 210+ 次修复迭代，主要批次：
- Batch 1-7: 核心 pipeline（JSON 截断、S2 限速、LaTeX 修复、figure 去重、review 合并）
- Batch 8-12: 排版、引用质量、PDF/Web/SymPy、反 AI 写作风格
- Batch 13-18: Deep agent、pipeline bug、S2 batch API、OpenAlex 集成、实验成功
- Batch 19-23: 实验鲁棒性、ReAct 模式、容器化执行
- Batch 24-28: BibTeX 转义、agent 全面审计、natbib 正则、路径安全、per-section context

## 已知遗留问题
- S2 API key 还没申请（Semantic Scholar）
- review.py `_apply_revisions` subsection 级别歧义（LOW）
- writing.py 消融表最优值加粗（Feature）

## 开始

请先读 `docs/architecture/00-overview.md`，了解全貌后从 Phase 1 开始：读 `07-quality-validation.md` 中 Section 18 的 bug fixes，这些是零风险的点修复，作为热身最合适。每完成一个 fix 就告诉我做了什么改动。
