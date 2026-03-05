# NanoResearch Agent Architecture

NanoResearch 的核心是 5 个串行执行的 Agent，每个 Agent 负责科研流水线中的一个阶段。所有 Agent 继承自统一的基类，通过编排器（Orchestrator）协调执行，支持断点恢复和自动重试。

---

## 系统架构

```
                    ┌─────────────────────────────────────────────┐
                    │           PipelineOrchestrator              │
                    │  调度 · 重试 · 检查点 · 状态机 · Resume     │
                    └──────────────────┬──────────────────────────┘
                                       │
        ┌──────────┬──────────┬────────┴───────┬──────────┐
        ▼          ▼          ▼                ▼          ▼
   ┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
   │Ideation │ │Planning│ │Experiment│ │Figure Gen│ │ Writing │
   │  Agent  │ │ Agent  │ │  Agent   │ │  Agent   │ │  Agent  │
   └────┬────┘ └───┬────┘ └────┬─────┘ └────┬─────┘ └────┬────┘
        │          │           │             │            │
        ▼          ▼           ▼             ▼            ▼
    ideation   experiment   code/        figures/     paper.tex
    _output.json blueprint.json *.py     *.png *.pdf   paper.pdf
```

### 数据流

```
Topic (str)
  │
  ▼
IdeationAgent ──→ ideation_output (papers, gaps, hypotheses)
  │
  ▼
PlanningAgent ──→ experiment_blueprint (datasets, baselines, metrics)
  │
  ▼
ExperimentAgent ─→ code/ (12+ runnable Python files)
  │
  ▼
FigureAgent ────→ figures/ (3 figures: architecture + results + ablation)
  │
  ▼
WritingAgent ───→ drafts/paper.tex + paper.pdf
```

---

## 基类：BaseResearchAgent

**文件**: `nanoresearch/agents/base.py`

所有 Agent 的公共基类，提供：

| 能力 | 方法 | 说明 |
|------|------|------|
| LLM 文本生成 | `generate(system, user, json_mode)` | 调用当前阶段配置的 LLM |
| JSON 结构化输出 | `generate_json(system, user)` | 生成 + 自动解析 JSON，修复 LaTeX 转义 |
| 日志 | `log(msg)` | 带阶段前缀的日志 |
| 工作区 IO | `self.workspace` | 文件读写、artifact 注册 |
| 模型调度 | `self._dispatcher` | 多模型路由 (ModelDispatcher) |

### LaTeX 转义修复

LLM 输出 JSON 时常包含 `\cite{}`, `\textbf{}` 等 LaTeX 命令。`\c`, `\t` 等是非法 JSON 转义序列，会导致 `json.loads()` 失败。基类内置的 `_fix_json_escapes()` 自动检测并双转义这些字符，确保 JSON 解析成功。

---

## Agent 1: IdeationAgent — 文献调研 + 假设生成

**文件**: `nanoresearch/agents/ideation.py`
**阶段**: `IDEATION`
**使用模型**: DeepSeek-V3.2 (temperature=0.5)

### 职责

从一个研究主题出发，自动检索相关文献、分析研究空白、生成并选择最优假设。

### 流程 (2 次 LLM + N 次 API)

```
输入: topic (str)
  │
  ├─ LLM Call 1: 生成 5-8 条搜索查询
  │    └─ JSON: {"queries": ["query1", ...]}
  │
  ├─ 并行 API 调用 (每条查询):
  │    ├─ arXiv API (Atom XML) → 10 篇/查询
  │    └─ Semantic Scholar API → 10 篇/查询
  │    └─ 去重 → ≤30 篇论文
  │
  ├─ LLM Call 2: 分析论文集合
  │    ├─ survey_summary: 300-500 词领域综述
  │    ├─ gaps: 3-5 个研究空白 (GAP-001, 严重度)
  │    ├─ hypotheses: 2-4 个假设 (新颖性/可行性评估)
  │    └─ selected_hypothesis: 最优假设 + 理由
  │
  └─ 输出: papers/ideation_output.json
```

### 关键设计

- **双源检索**: arXiv + Semantic Scholar 互补，arXiv 覆盖预印本，S2 覆盖引用数据
- **去重**: 按论文标题去重，避免重复
- **限速保护**: 每个数据源最多查询 5 次，防止 API rate limit

---

## Agent 2: PlanningAgent — 实验方案设计

**文件**: `nanoresearch/agents/planning.py`
**阶段**: `PLANNING`
**使用模型**: DeepSeek-V3.2 (temperature=0.2)

### 职责

基于 IdeationAgent 选出的假设，设计完整的实验方案。

### 流程 (1 次 LLM)

```
输入: ideation_output
  │
  ├─ 提取: 选中假设 + 研究空白 + 相关论文
  │
  ├─ LLM Call: 设计实验蓝图
  │    ├─ datasets: 数据集 + 预处理说明
  │    ├─ baselines: 基线方法 + 预期表现
  │    ├─ proposed_method: 方法名 + 关键组件 + 架构
  │    ├─ metrics: 评估指标 (primary 标记)
  │    ├─ ablation_groups: 消融实验设计
  │    └─ compute_requirements: GPU 需求
  │
  ├─ Pydantic 验证 (ExperimentBlueprint)
  │
  └─ 输出: plans/experiment_blueprint.json
```

### 关键设计

- **低温度 (0.2)**: 实验设计需要精确和一致性
- **Pydantic 强约束**: 所有输出经过 `ExperimentBlueprint` 模型验证

---

## Agent 3: ExperimentAgent — 代码项目生成

**文件**: `nanoresearch/agents/experiment.py`
**阶段**: `EXPERIMENT`
**使用模型**: DeepSeek-V3.2 (temperature=0.1, 用于 code_gen)

### 职责

根据实验蓝图，生成完整可运行的 ML 代码项目 (12+ 文件)。

### 流程 — 两阶段生成 (1 + N 次 LLM)

```
输入: experiment_blueprint
  │
  ├─ Phase 1 — 项目规划 (1 次 LLM):
  │    └─ JSON {
  │         files: [{path, description, interfaces, depends_on}],
  │         interface_contract: "所有文件的接口签名",
  │         dependencies: ["torch>=2.0", ...]
  │       }
  │
  ├─ Phase 2 — 逐文件生成 (每文件 1 次 LLM):
  │    每次传入: 文件描述 + 完整接口契约 + 实验蓝图
  │    → 确保跨文件接口一致
  │
  └─ 输出: code/
       ├── main.py
       ├── requirements.txt
       ├── config/default.yaml
       ├── src/{model,dataset,trainer,evaluate,utils}.py
       └── scripts/{train.sh, run_ablation.sh}
```

### 关键设计

- **接口契约模式**: Phase 1 先生成所有文件的 class/function 签名。Phase 2 每个文件都看到完整契约，确保跨文件调用一致（如 `model.py` 中的类名与 `trainer.py` 中的导入匹配）
- **极低温度 (0.1)**: 代码生成需要确定性，减少随机变体

---

## Agent 4: FigureAgent — AI 配图生成

**文件**: `nanoresearch/agents/figure_gen.py`
**阶段**: `FIGURE_GEN`
**使用模型**: 混合 — GPT-5.2 (prompt 生成) + Gemini Flash (AI 绘图) + Codex (代码图表)

### 职责

为论文生成 3 张图：架构图 + 实验结果图 + 消融实验图。

### 流程 — 混合策略

```
输入: experiment_blueprint
  │
  ├─ Fig 1: 架构图 (AI 图像生成)
  │    ├─ GPT-5.2 生成图像 prompt (≤800 字符)
  │    ├─ Gemini Flash 生成图片 (base64)
  │    └─ Pillow 保存 PNG + 转换 PDF
  │
  ├─ Fig 2: 基线对比图 (LLM 生成代码)
  │    ├─ Codex 生成完整 matplotlib 脚本
  │    ├─ subprocess 执行脚本 (60s 超时)
  │    ├─ 失败 → fallback 占位图
  │    └─ 保存 PNG + PDF + 脚本
  │
  ├─ Fig 3: 消融实验图 (LLM 生成代码)
  │    └─ (同 Fig 2 流程)
  │
  └─ 输出: figures/
       ├── fig1_architecture.{png,pdf}
       ├── fig2_results.{png,pdf}
       ├── fig2_results_plot.py      ← 可复现
       ├── fig3_ablation.{png,pdf}
       └── fig3_ablation_plot.py     ← 可复现
```

### 关键设计

- **混合策略**: 架构图用 AI 生成（需要创意），数据图表用代码生成（需要精确）
- **代码可复现**: fig2/fig3 的绘图脚本保存在 figures/ 目录，可以手动修改重新运行
- **Fallback 机制**: 如果 LLM 生成的代码执行失败（语法错误、超时等），自动生成占位图并记录错误日志
- **Prompt 截断**: 图像 prompt 限制 3800 字符，防止 API 拒绝

---

## Agent 5: WritingAgent — 论文撰写

**文件**: `nanoresearch/agents/writing.py`
**阶段**: `WRITING`
**使用模型**: Claude Sonnet 4.6 (temperature=0.4, max_tokens=16384)

### 职责

汇总所有前序阶段的输出，撰写完整的 LaTeX 论文并编译为 PDF。

### 流程 (8 次 LLM)

```
输入: ideation_output + experiment_blueprint + figure_output
  │
  ├─ Step 0: 构建引用系统
  │    ├─ 映射: paper_index → cite_key (authorYear 格式)
  │    └─ 生成 BibTeX 条目
  │
  ├─ Step 1: LLM 生成标题 (1 调用)
  ├─ Step 2: LLM 生成摘要 (1 调用, 150-250 词)
  │
  ├─ Step 3: 构建图表块 (动态 caption)
  │    └─ 使用 FigureAgent 生成的 caption
  │
  ├─ Step 4: 逐章节生成 (6 调用):
  │    ├─ Introduction     — 3-4 段, 背景+问题+方法+贡献
  │    ├─ Related Work      — 3-4 段, 按主题分组的文献综述
  │    ├─ Method            — 4-5 段, 技术细节 + 数学公式
  │    ├─ Experiments       — 3-4 段, 设置 + 结果表格 (LLM 生成)
  │    ├─ Results           — 3-4 段, 结果分析 + 消融实验
  │    └─ Conclusion        — 2-3 段, 总结+局限+未来工作
  │
  ├─ Step 5: 组装 PaperSkeleton
  ├─ Step 6: 渲染 LaTeX + 自动修复
  │    ├─ _sanitize_latex(): Unicode 破折号 → LaTeX, 转义 %
  │    └─ _sanitize_bibtex(): Unicode 特殊字符 → LaTeX 命令
  │
  ├─ Step 7: PDF 编译 (tectonic 优先, pdflatex 备选)
  │    └─ 自动拷贝 figures/ 到 drafts/
  │
  └─ 输出: drafts/
       ├── paper.tex
       ├── references.bib
       ├── paper.pdf
       └── paper_skeleton.json
```

### 关键设计

- **逐章节生成**: 避免单次生成整篇论文导致的 JSON 截断和转义问题
- **引用键注入**: 将 `cite_keys` 注入到每个章节的上下文中，确保 LLM 只使用真实存在的引用键 (`\cite{hu2022}`)，不会编造不存在的引用
- **LaTeX 自动修复**:
  - Unicode 破折号 (—/–) → `---`/`--`
  - 未转义的 `%` → `\%`
  - BibTeX 中的特殊字符 (é, ß, ğ 等) → LaTeX 命令
- **图表内嵌**: 图表块自动插入到对应章节末尾，caption 从 FigureAgent 的输出中获取
- **结果表格**: 由 Writing LLM 在 Experiments 章节中内联生成，使用合成数据，提出方法在多数指标上优于基线
- **使用 Claude**: 写作质量关键，使用 Claude Sonnet 4.6 而非 DeepSeek

---

## 编排器：PipelineOrchestrator

**文件**: `nanoresearch/pipeline/orchestrator.py`

### 核心功能

| 功能 | 实现 |
|------|------|
| 阶段调度 | 按 `IDEATION → PLANNING → EXPERIMENT → FIGURE_GEN → WRITING` 顺序执行 |
| 检查点恢复 | 已完成的阶段自动跳过，从上次失败的阶段继续 |
| 自动重试 | 每阶段最多重试 3 次，每次注入上次的错误信息帮助 LLM 自我修正 |
| 状态机 | 严格的前向状态转移，防止非法跳转 (state.py) |
| 数据路由 | 自动将前序阶段的输出传递给后续阶段 |
| 错误日志 | 每次失败保存详细的 traceback 到 logs/ |

### 数据路由表

| 阶段 | 输入来源 | 输出键 |
|------|----------|--------|
| IDEATION | `topic` (str) | `ideation_output` |
| PLANNING | `ideation_output` | `experiment_blueprint` |
| EXPERIMENT | `experiment_blueprint` | `experiment_output` (includes `experiment_results`, `experiment_status`) |
| FIGURE_GEN | `experiment_blueprint` + `experiment_results` + `experiment_status` | `figure_output` |
| WRITING | ideation + blueprint + figures + `experiment_results` + `experiment_status` | `writing_output` |

---

## 多模型调度：ModelDispatcher

**文件**: `nanoresearch/pipeline/multi_model.py`

通过统一的 OpenAI 兼容 API 端点调度不同模型：

| 配置项 | 模型 | 用途 |
|--------|------|------|
| `ideation` | DeepSeek-V3.2 | 文献分析 + 假设生成 |
| `planning` | DeepSeek-V3.2 | 实验方案设计 |
| `experiment` / `code_gen` | DeepSeek-V3.2 | 代码项目生成 |
| `figure_prompt` | GPT-5.2 | 架构图 prompt |
| `figure_code` | GPT-5.2 Codex | 图表绘图代码 |
| `figure_gen` | Gemini Flash | AI 图像生成 |
| `writing` | Claude Sonnet 4.6 | 论文撰写 |

### 支持的图像后端

- **OpenAI**: 标准 `/v1/images/generations` (DALL-E)
- **Gemini**: 原生 `/v1beta/models/{model}:generateContent`，支持 `responseModalities=["TEXT","IMAGE"]`

---

## 配置

`~/.nanobot/config.json`:

```json
{
  "research": {
    "base_url": "http://your-endpoint/v1/",
    "api_key": "your-key",
    "timeout": 180.0,
    "ideation":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.5 },
    "planning":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.2 },
    "experiment":    { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.1 },
    "writing":       { "model": "claude-sonnet-4-6", "temperature": 0.4, "max_tokens": 16384 },
    "code_gen":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.1, "max_tokens": 16384 },
    "figure_prompt": { "model": "gpt-5.2", "temperature": 0.5 },
    "figure_code":   { "model": "gpt-5.2-codex", "temperature": null },
    "figure_gen":    { "model": "gemini-3.1-flash-image-preview", "image_backend": "gemini" }
  }
}
```

`temperature: null` 表示不传该参数，适用于不支持 temperature 的模型（如 Codex）。

---

## 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| JSON 解析失败 | LLM 输出包含 `\cite{}` 等 LaTeX 命令 | `_fix_json_escapes()` 自动双转义 |
| PDF 编译 `\@xdblarg` 错误 | 图表 caption 中未转义的 `%` | `_sanitize_latex()` 自动转义 |
| BibTeX Unicode 字符警告 | 作者名含 é, ß, ğ 等 | `_sanitize_bibtex()` 替换为 LaTeX 命令 |
| 结果表格空值 | 硬编码的指标名与实际不匹配 | 改为 LLM 内联生成表格 |
| 图像 prompt 超长 | Gemini 有字符限制 | 系统 prompt 约束 800 字符 + 3800 截断 |
| Codex 报错 `Unsupported parameter` | 模型不支持 temperature | `temperature: null` 时不传该参数 |
| 单次生成整篇论文截断 | 大 JSON + LaTeX 转义太容易出错 | 改为逐章节独立生成 |
| LLM 生成的绘图代码失败 | 语法错误或超时 | fallback 占位图 + 错误日志 |
