# NanoResearch 构建指南

从零构建一个 AI 自动化科研引擎：输入一个研究主题，自动完成文献调研、实验设计、代码生成、配图生成、论文撰写，输出完整的学术论文。

---

## 一、系统总览

### 核心理念

将科研流程拆分为 5 个独立阶段，每个阶段由一个专用 Agent 完成，通过一条单向流水线串联：

```
Topic (用户输入)
  │
  ▼
IDEATION ─── 文献检索 + 研究假设
  │
  ▼
PLANNING ─── 实验方案设计
  │
  ▼
EXPERIMENT ── 代码项目生成 (Codex)
  │
  ▼
FIGURE_GEN ── AI 科研配图 (GPT-5.2 + DALL-E)
  │
  ▼
WRITING ───── 论文撰写 + LaTeX 编译
  │
  ▼
输出: paper.tex + paper.pdf + code/ + figures/
```

### 关键设计决策

1. **每阶段一个 Agent**：解耦、可独立测试、可单独 resume
2. **Checkpoint/Resume**：每个阶段完成后持久化到磁盘，任何阶段失败后可从断点恢复
3. **多模型路由**：不同阶段可配置不同 LLM（DeepSeek 做推理、Codex 写代码、DALL-E 画图）
4. **OpenAI 兼容 API**：统一接口，支持任何 OpenAI 兼容的自建端点
5. **状态机约束**：严格的前向状态转移，防止非法跳转

---

## 二、项目结构

```
nanoresearch/
├── nanoresearch/                  # 主包
│   ├── __init__.py
│   ├── __main__.py               # python -m nanoresearch 入口
│   ├── cli.py                    # CLI 命令 (run, resume, status, list, export)
│   ├── config.py                 # 全局配置 + 每阶段模型路由
│   ├── agents/                   # 5 个 Agent 实现
│   │   ├── base.py              # BaseResearchAgent 抽象基类
│   │   ├── ideation.py          # 文献调研 + 假设生成
│   │   ├── planning.py          # 实验方案设计
│   │   ├── experiment.py        # 代码项目生成 (两阶段)
│   │   ├── figure_gen.py        # AI 配图生成
│   │   └── writing.py           # 论文撰写
│   ├── pipeline/                 # 流水线基础设施
│   │   ├── orchestrator.py      # 编排器：调度 + 重试 + 检查点
│   │   ├── state.py             # 状态机
│   │   ├── workspace.py         # 工作区目录 + manifest 管理
│   │   └── multi_model.py       # OpenAI API 调度器
│   └── schemas/                  # Pydantic 数据模型
│       ├── manifest.py          # 阶段枚举 + 状态转移 + Manifest
│       ├── ideation.py          # PaperReference, GapAnalysis, Hypothesis
│       ├── experiment.py        # ExperimentBlueprint, Dataset, Metric
│       └── paper.py             # PaperSkeleton, Section, Figure
├── mcp_server/                   # MCP 工具服务
│   ├── server.py                # stdio JSON-RPC 2.0 服务器
│   └── tools/
│       ├── arxiv_search.py      # arXiv API 检索
│       ├── semantic_scholar.py  # Semantic Scholar 检索
│       ├── latex_gen.py         # Jinja2 LaTeX 模板渲染
│       ├── pdf_compile.py       # pdflatex 编译包装
│       └── figure_gen.py        # matplotlib 图表生成
├── tests/                        # 测试套件 (60 tests)
├── pyproject.toml               # 包定义 + 依赖
└── ~/.nanobot/config.json       # 用户配置文件 (API 密钥 + 模型路由)
```

---

## 三、分层架构 — 自底向上构建

### 第 1 层：数据模型 (`schemas/`)

先定义所有数据结构，它们是系统的契约。

#### `schemas/manifest.py` — 流水线阶段 + 状态追踪

```python
class PipelineStage(str, Enum):
    INIT = "INIT"
    IDEATION = "IDEATION"
    PLANNING = "PLANNING"
    EXPERIMENT = "EXPERIMENT"
    FIGURE_GEN = "FIGURE_GEN"
    WRITING = "WRITING"
    DONE = "DONE"
    FAILED = "FAILED"

# 严格的前向状态转移表
STAGE_TRANSITIONS = {
    INIT: [IDEATION, FAILED],
    IDEATION: [PLANNING, FAILED],
    PLANNING: [EXPERIMENT, FAILED],
    EXPERIMENT: [FIGURE_GEN, FAILED],
    FIGURE_GEN: [WRITING, FAILED],
    WRITING: [DONE, FAILED],
    DONE: [],
    FAILED: [],
}

class StageRecord(BaseModel):
    stage: PipelineStage
    status: str = "pending"  # pending | running | completed | failed
    started_at: datetime | None
    completed_at: datetime | None
    retries: int = 0
    error_message: str = ""

class ArtifactRecord(BaseModel):
    name: str           # 描述性名称
    path: str           # 相对于 workspace 的路径
    stage: PipelineStage
    checksum: str       # MD5

class WorkspaceManifest(BaseModel):
    session_id: str
    topic: str
    current_stage: PipelineStage
    stages: dict[str, StageRecord]  # 每个阶段的执行记录
    artifacts: list[ArtifactRecord]  # 产出文件列表
    config_snapshot: dict            # 配置快照 (不含 API Key)
```

#### `schemas/ideation.py` — 文献调研输出

```python
class PaperReference(BaseModel):
    paper_id: str, title: str, authors: list[str]
    year: int | None, abstract: str, venue: str
    citation_count: int, url: str

class IdeationOutput(BaseModel):
    topic: str
    search_queries: list[str]
    papers: list[PaperReference]
    survey_summary: str              # 300-500 词综述
    gaps: list[dict]                 # GAP-001, GAP-002...
    hypotheses: list[dict]           # HYP-001, HYP-002...
    selected_hypothesis: str         # 选中的假设 ID
    rationale: str                   # 选择理由
```

#### `schemas/experiment.py` — 实验蓝图

```python
class ExperimentBlueprint(BaseModel):
    title: str
    hypothesis_ref: str
    datasets: list[Dataset]          # 数据集
    baselines: list[Baseline]        # 基线方法
    proposed_method: dict            # 提出的方法 (name, components, architecture)
    metrics: list[Metric]            # 评估指标
    ablation_groups: list[AblationGroup]  # 消融实验
    compute_requirements: dict
```

#### `schemas/paper.py` — 论文骨架

```python
class PaperSkeleton(BaseModel):
    title: str, authors: list[str], abstract: str
    sections: list[Section]          # 6 个章节
    figures: list[FigurePlaceholder]
    references_bibtex: str
```

---

### 第 2 层：流水线基础设施 (`pipeline/`)

#### `pipeline/state.py` — 状态机

```python
class PipelineStateMachine:
    def __init__(self, initial=PipelineStage.INIT)

    def can_transition(target) -> bool    # 检查是否允许转移
    def transition(target) -> PipelineStage  # 执行转移，非法则抛异常
    def fail() -> PipelineStage           # 任意非终态 → FAILED

    @staticmethod
    def processing_stages() -> list       # 返回 5 个工作阶段的有序列表
    def next_stage(current) -> PipelineStage | None
```

#### `pipeline/workspace.py` — 工作区管理

每个 session 在 `~/.nanobot/workspace/research/{session_id}/` 下创建：

```
{session_id}/
├── manifest.json     # 主 manifest
├── papers/           # 文献数据
├── plans/            # 实验蓝图 + 项目规划
├── code/             # 生成的代码项目
├── figures/          # 生成的配图
├── drafts/           # paper.tex + references.bib + paper.pdf
└── logs/             # 错误日志
```

核心方法：
- `create(topic)` → 初始化 session 目录 + manifest
- `load(path)` → 从已有目录恢复
- `mark_stage_running/completed/failed()` → 更新阶段状态
- `register_artifact(name, path, stage)` → 记录产出文件 (含 MD5)
- `write_json/read_json/write_text/read_text()` → 文件 IO
- `export()` → 导出为干净的输出目录

#### `pipeline/multi_model.py` — 多模型调度器

```python
class ModelDispatcher:
    """通过 OpenAI 兼容 API 调度不同模型"""

    def __init__(self, config: ResearchConfig)

    # 文本生成
    async def generate(config, system_prompt, user_prompt, json_mode=False) -> str
        # 关键：temperature 为 None 时不传该参数 (Codex 不支持)
        # 使用 run_in_executor 包装同步 OpenAI 调用

    # 图像生成
    async def generate_image(config, prompt, size, quality) -> list[str]
        # 调用 client.images.generate, 返回 base64 编码列表
```

#### `config.py` — 配置系统

```python
class StageModelConfig(BaseModel):
    model: str                   # 模型 ID
    temperature: float | None    # None = 不发送 (Codex/o-series)
    max_tokens: int = 8192
    timeout: float | None        # 覆盖全局 timeout

class ResearchConfig(BaseModel):
    base_url: str               # API 端点
    api_key: str
    timeout: float = 180.0

    # 每阶段独立配置
    ideation:      StageModelConfig  # DeepSeek, temp=0.5
    planning:      StageModelConfig  # DeepSeek, temp=0.2
    experiment:    StageModelConfig  # DeepSeek, temp=0.1
    writing:       StageModelConfig  # DeepSeek, temp=0.4, max_tokens=16384
    code_gen:      StageModelConfig  # Codex, temperature=None
    figure_prompt: StageModelConfig  # GPT-5.2, temp=0.5
    figure_gen:    StageModelConfig  # DALL-E-3, temperature=None

    def for_stage(name) -> StageModelConfig  # 按名称查找
    def load(path) -> ResearchConfig         # 从 config.json 加载，环境变量覆盖
```

配置优先级：环境变量 > config.json > 代码默认值。

---

### 第 3 层：Agent 基类 (`agents/base.py`)

```python
class BaseResearchAgent(ABC):
    stage: PipelineStage          # 子类必须设置

    def __init__(self, workspace, config):
        self.workspace = workspace
        self.config = config
        self._dispatcher = ModelDispatcher(config)

    async def generate(system, user, json_mode=False) -> str
        # 用本阶段配置调用 LLM

    async def generate_json(system, user) -> dict
        # 调用 LLM → 提取 JSON → 修复 LaTeX 转义 → json.loads

    @abstractmethod
    async def run(**inputs) -> dict
        # 子类实现具体逻辑
```

**LaTeX 转义修复**：LLM 输出 JSON 时常包含 `\cite{}`, `\textbf{}` 等 LaTeX 命令，
这些 `\c`, `\t` 是非法 JSON 转义。`_fix_json_escapes()` 自动双转义修复。

---

### 第 4 层：5 个 Agent 实现 (`agents/`)

#### Agent 1: IdeationAgent — 文献调研

**输入**：`topic: str`
**输出**：`IdeationOutput` (papers, gaps, hypotheses, selected_hypothesis)

**流程** (2 次 LLM 调用 + N 次 API 调用)：

```
1. LLM 生成 5-8 条搜索查询 (JSON)
2. 并行调用 arXiv API + Semantic Scholar API
   → 去重、聚合 ≤30 篇论文
3. LLM 分析论文 → 输出:
   - survey_summary (300-500 词综述)
   - gaps (3-5 个研究空白，含严重度评级)
   - hypotheses (2-4 个假设，含新颖性/可行性评估)
   - selected_hypothesis (选中最优假设)
4. 保存 papers/ideation_output.json
```

#### Agent 2: PlanningAgent — 实验方案设计

**输入**：`ideation_output: dict`
**输出**：`ExperimentBlueprint`

**流程** (1 次 LLM 调用)：

```
1. 提取选中的假设 + 研究空白 + 相关论文
2. LLM 设计完整实验方案 (JSON):
   - datasets: 数据集列表 + 预处理说明
   - baselines: 基线方法 + 预期表现
   - proposed_method: 方法名 + 关键组件 + 架构描述
   - metrics: 评估指标 (primary 标记)
   - ablation_groups: 消融实验设计
   - compute_requirements: GPU 需求
3. Pydantic 验证 → 保存 plans/experiment_blueprint.json
```

#### Agent 3: ExperimentAgent — 代码项目生成

**输入**：`experiment_blueprint: dict`
**输出**：12 个文件的完整项目

**两阶段生成** (1 + N 次 Codex 调用)：

```
Phase 1 — 项目规划 (1 次 Codex 调用):
  输入: experiment_blueprint 摘要
  输出: JSON { files: [...], interface_contract: "..." }
  → 接口契约包含所有文件的 class/function 签名

Phase 2 — 逐文件生成 (每文件 1 次 Codex 调用):
  每次传入: 文件描述 + 完整接口契约 + 实验蓝图
  → 确保跨文件接口一致

生成的文件结构:
  code/
  ├── main.py, README.md, requirements.txt
  ├── config/default.yaml
  ├── src/{__init__, model, dataset, trainer, evaluate, utils}.py
  └── scripts/{train.sh, run_ablation.sh}
```

**关键设计**：接口契约模式 — Phase 1 先生成所有文件的接口签名，
Phase 2 每个文件都看到完整契约，避免跨文件不一致。

#### Agent 4: FigureAgent — AI 配图生成

**输入**：`experiment_blueprint: dict`
**输出**：3 张图 (PNG + PDF)

**流程** (3 × 2 步)：

```
每张图:
  1. GPT-5.2 生成图像 prompt (≤800 字符约束 + 3800 截断安全网)
  2. DALL-E-3 生成图片 (1024×1024, base64)
  3. Pillow 保存 PNG + 转换 PDF

3 张图:
  - fig1_architecture: 模型架构总览
  - fig2_results: 基线对比图表
  - fig3_ablation: 消融实验结果

保存到: figures/{name}.png + figures/{name}.pdf
```

#### Agent 5: WritingAgent — 论文撰写

**输入**：ideation_output + experiment_blueprint + figure_output
**输出**：paper.tex + references.bib + paper.pdf

**流程** (8 次 LLM 调用)：

```
1. 构建引用键映射: paper_index → cite_key (authorYear 格式)
2. 生成 BibTeX
3. LLM 生成标题 (1 调用)
4. LLM 生成摘要 (1 调用)
5. LLM 逐章节生成 (6 调用):
   - Introduction, Related Work, Method
   - Experiments, Results, Conclusion
   每次传入: 章节写作指令 + 完整研究上下文 + 引用键列表
6. 构建 LaTeX 图表块 (动态 caption)
7. 组装完整 paper.tex
8. pdflatex 3-pass 编译 (pdflatex → bibtex → pdflatex × 2)
```

**关键设计**：逐章节生成而非一次生成整篇，避免 JSON 截断和转义问题。

---

### 第 5 层：编排器 (`pipeline/orchestrator.py`)

```python
class PipelineOrchestrator:
    def __init__(self, workspace, config):
        # 创建 5 个 Agent 实例
        self._agents = {
            IDEATION: IdeationAgent(workspace, config),
            PLANNING: PlanningAgent(workspace, config),
            EXPERIMENT: ExperimentAgent(workspace, config),
            FIGURE_GEN: FigureAgent(workspace, config),
            WRITING: WritingAgent(workspace, config),
        }

    async def run(topic) -> dict:
        results = {}
        for stage in [IDEATION, PLANNING, EXPERIMENT, FIGURE_GEN, WRITING]:
            if already_completed(stage):
                results.update(load_previous_output(stage))  # Resume 支持
                continue
            result = await _run_stage_with_retry(stage, topic, results)
            results.update(result)
        mark_DONE()
```

**重试逻辑**：
```
for attempt in range(max_retries + 1):  # 默认 3 次
    try:
        mark_running → agent.run() → mark_completed
    except:
        save_error_log
        if not last_attempt:
            increment_retry → 注入 _retry_error 上下文给下次
        else:
            mark_failed → state_machine.fail()
```

**数据路由** — 每个阶段的输入/输出映射：

| 阶段 | 输入 | 输出键 |
|------|------|--------|
| IDEATION | `topic` | `ideation_output` |
| PLANNING | `ideation_output` | `experiment_blueprint` |
| EXPERIMENT | `experiment_blueprint` | `experiment_output` (includes `experiment_results`, `experiment_status`) |
| FIGURE_GEN | `experiment_blueprint` + `experiment_results` + `experiment_status` | `figure_output` |
| WRITING | ideation + blueprint + figures + `experiment_results` + `experiment_status` | `writing_output` |

---

### 第 6 层：MCP 工具服务 (`mcp_server/`)

独立的 JSON-RPC 2.0 over stdio 服务器，暴露 5 个工具：

| 工具 | 用途 | API |
|------|------|-----|
| `search_arxiv` | 论文检索 | arXiv Atom XML API |
| `search_semantic_scholar` | 引用数据 | Semantic Scholar API |
| `generate_latex` | LaTeX 渲染 | Jinja2 模板 |
| `compile_pdf` | PDF 编译 | pdflatex 3-pass |
| `generate_figure` | matplotlib 图表 | PNG 输出 |

IdeationAgent 直接调用 `search_arxiv` 和 `search_semantic_scholar`。
WritingAgent 调用 `compile_pdf`。

---

### 第 7 层：CLI (`cli.py`)

基于 Typer + Rich，5 个命令：

```bash
# 从主题运行完整流水线
nanoresearch run --topic "Your Topic" --format arxiv --verbose

# 从断点恢复
nanoresearch resume --workspace ~/.nanobot/workspace/research/{session_id}

# 查看状态
nanoresearch status --workspace {path}

# 列出所有 session
nanoresearch list

# 导出到干净目录
nanoresearch export --workspace {path} --output ./outputs
```

---

## 四、构建顺序 (推荐)

按依赖关系自底向上：

```
Step 1: schemas/manifest.py       ← 阶段枚举 + 状态转移 + Manifest
Step 2: schemas/ideation.py       ← PaperReference, IdeationOutput
Step 3: schemas/experiment.py     ← ExperimentBlueprint
Step 4: schemas/paper.py          ← PaperSkeleton
Step 5: pipeline/state.py         ← 状态机 (仅依赖 manifest)
Step 6: config.py                 ← ResearchConfig + StageModelConfig
Step 7: pipeline/multi_model.py   ← ModelDispatcher (依赖 config)
Step 8: pipeline/workspace.py     ← Workspace (依赖 manifest)
Step 9: agents/base.py            ← BaseResearchAgent (依赖上面所有)
Step 10: mcp_server/tools/*.py    ← 5 个工具函数
Step 11: mcp_server/server.py     ← MCP 服务器
Step 12: agents/ideation.py       ← IdeationAgent
Step 13: agents/planning.py       ← PlanningAgent
Step 14: agents/experiment.py     ← ExperimentAgent
Step 15: agents/figure_gen.py     ← FigureAgent
Step 16: agents/writing.py        ← WritingAgent
Step 17: pipeline/orchestrator.py ← PipelineOrchestrator
Step 18: cli.py                   ← CLI 入口
Step 19: pyproject.toml           ← 包定义
Step 20: tests/                   ← 全套测试
```

---

## 五、配置文件

`~/.nanobot/config.json`:

```json
{
  "research": {
    "base_url": "http://your-api-endpoint/v1/",
    "api_key": "your-api-key",
    "timeout": 180.0,
    "ideation":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.5 },
    "planning":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.2 },
    "experiment":    { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.1, "timeout": 600 },
    "writing":       { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.4, "max_tokens": 16384, "timeout": 600 },
    "code_gen":      { "model": "gpt-5.2-codex", "temperature": null, "max_tokens": 16384, "timeout": 600 },
    "figure_prompt": { "model": "gpt-5.2", "temperature": 0.5, "max_tokens": 4096, "timeout": 300 },
    "figure_gen":    { "model": "dall-e-3", "temperature": null, "timeout": 300 }
  }
}
```

也可通过环境变量覆盖：`NANORESEARCH_BASE_URL`, `NANORESEARCH_API_KEY`, `NANORESEARCH_TIMEOUT`

---

## 六、依赖

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0.0",     # 数据模型
    "openai>=1.0.0",       # LLM API 客户端
    "httpx>=0.27.0",       # HTTP 客户端
    "lxml>=5.0.0",         # arXiv XML 解析
    "Jinja2>=3.1.0",       # LaTeX 模板
    "matplotlib>=3.8.0",   # 图表生成
    "typer>=0.9.0",        # CLI 框架
    "rich>=13.0.0",        # 终端美化
    "mcp>=1.0.0",          # MCP 协议
    "Pillow",              # 图像处理 (PNG→PDF)
]
```

---

## 七、踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Codex 报错 `Unsupported parameter: temperature` | 部分模型不支持 temperature | `temperature: float \| None`，None 时不传该参数 |
| config.json 省略字段仍用默认值 | Pydantic 默认值填充 | 显式写 `"temperature": null` |
| LLM 输出 JSON 含 LaTeX 导致解析失败 | `\cite`, `\textbf` 等是非法 JSON 转义 | `_fix_json_escapes()` 自动双转义 |
| GPT-5.2 图像 prompt 超长 | DALL-E-3 有 ~4000 字符限制 | 系统 prompt 约束 800 字符 + 3800 截断 |
| 某些图像模型不支持 images API | 不同模型端点能力不同 | 先查 `/v1/models` 确认 `supported_endpoint_types` |
| PDF 编译失败 | TeX Live 安装不完整 | 确保 `pdflatex` 可用，或在 Overleaf 编译 |
| 单次生成整篇论文 JSON 截断 | 大 JSON + LaTeX 转义太容易出错 | 改为逐章节独立生成，纯文本输出 |

---

## 八、端到端验证

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 配置
cat > ~/.nanobot/config.json << 'EOF'
{ "research": { "base_url": "...", "api_key": "..." } }
EOF

# 3. 运行测试
pytest tests/ -v  # 60 tests should pass

# 4. 端到端运行
nanoresearch run --topic "Your Research Topic" --verbose

# 5. 检查输出
ls nanoresearch_{topic_slug}_{session_id}/
# → paper.tex, figures/*.png, code/*.py, data/*.json

# 6. 断点恢复 (如果中途失败)
nanoresearch resume --workspace ~/.nanobot/workspace/research/{session_id} --verbose
```
