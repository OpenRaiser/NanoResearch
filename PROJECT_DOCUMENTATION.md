# NanoResearch 项目文档

> **NanoResearch** — 一个最小化的 AI 驱动研究引擎，自动完成从研究选题到论文成稿的全流程。

---

## 一、项目概览

NanoResearch 是一个自动化学术研究 pipeline，将研究流程拆解为 5 个阶段，每个阶段由专门的 AI Agent 负责执行：

```
研究主题 → 文献调研 → 实验规划 → 代码生成 → 图表生成 → 论文撰写
```

整个系统基于 OpenAI 兼容 API 驱动，支持断点续跑、多模型路由、多论文模板（arXiv / NeurIPS / ICML），约 2600 行 Python 代码。

---

## 二、目录结构

```
nanoresearch/
├── nanoresearch/                 # 核心包
│   ├── __main__.py              # python -m nanoresearch 入口
│   ├── __init__.py              # 版本号 (0.1.0)
│   ├── cli.py                   # Typer CLI 命令定义
│   ├── config.py                # 配置加载与管理
│   ├── agents/                  # 5 个专业 AI Agent
│   │   ├── base.py              # Agent 抽象基类
│   │   ├── ideation.py          # 文献调研 Agent
│   │   ├── planning.py          # 实验规划 Agent
│   │   ├── experiment.py        # 代码生成 Agent
│   │   ├── figure_gen.py        # 图表生成 Agent
│   │   └── writing.py           # 论文撰写 Agent
│   ├── pipeline/                # 流水线编排与状态管理
│   │   ├── orchestrator.py      # 主 pipeline 编排器
│   │   ├── state.py             # 有限状态机
│   │   ├── workspace.py         # 工作空间与产物管理
│   │   └── multi_model.py       # 多模型调度器 (OpenAI SDK)
│   ├── schemas/                 # Pydantic 数据模型
│   │   ├── manifest.py          # Pipeline 阶段 & 产物 schema
│   │   ├── ideation.py          # 文献调研输出 schema
│   │   ├── experiment.py        # 实验蓝图 schema
│   │   └── paper.py             # 论文骨架 schema
│   └── templates/               # Jinja2 LaTeX 模板
│       ├── base/paper.tex.j2    # 基础模板
│       ├── arxiv/paper.tex.j2   # arXiv 格式
│       ├── neurips/paper.tex.j2 # NeurIPS 格式
│       └── icml/paper.tex.j2    # ICML 格式
├── mcp_server/                  # MCP (Model Context Protocol) 服务
│   ├── server.py                # MCP stdio 服务入口
│   ├── utils.py                 # HTTP 客户端 & 令牌桶限流
│   └── tools/                   # 5 个工具实现
│       ├── arxiv_search.py      # arXiv API 搜索
│       ├── semantic_scholar.py  # Semantic Scholar API 搜索
│       ├── latex_gen.py         # Jinja2 LaTeX 渲染
│       ├── pdf_compile.py       # pdflatex 编译
│       └── figure_gen.py        # matplotlib 绘图
├── tests/                       # 测试套件
├── examples/                    # 配置示例 & 样例会话
├── skills/                      # AI 助手技能描述文件
└── pyproject.toml               # 构建配置
```

---

## 三、构建系统

### 3.1 构建工具

使用 **hatchling** 作为构建后端，配置在 `pyproject.toml` 中：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3.2 安装方式

```bash
# 开发模式安装
pip install -e .

# 或直接安装
pip install .
```

安装后会注册一个 CLI 命令 `nanoresearch`，对应入口：

```toml
[project.scripts]
nanoresearch = "nanoresearch.cli:app"
```

### 3.3 核心依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| pydantic | >= 2.0.0 | 数据校验 & schema 定义 |
| openai | >= 1.0.0 | LLM API 调用 (兼容任意 OpenAI-compatible 端点) |
| httpx | >= 0.27.0 | 异步 HTTP 客户端 |
| lxml | >= 5.0.0 | arXiv Atom XML 解析 |
| Jinja2 | >= 3.1.0 | LaTeX 模板渲染 |
| matplotlib | >= 3.8.0 | 科学图表生成 |
| typer | >= 0.9.0 | CLI 框架 |
| rich | >= 13.0.0 | 终端富文本输出 |
| mcp | >= 1.0.0 | Model Context Protocol |

### 3.4 开发依赖

```bash
pip install -e ".[dev]"
# pytest, pytest-asyncio, respx (HTTP mock)
```

---

## 四、配置系统

### 4.1 配置文件

默认路径：`~/.nanobot/config.json`，可通过 `--config` 参数覆盖。

```json
{
  "research": {
    "base_url": "https://your-api-endpoint/v1/",
    "api_key": "sk-xxx",
    "timeout": 180.0,
    "ideation":   { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.5, "max_tokens": 8192 },
    "planning":   { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.2, "max_tokens": 8192 },
    "experiment": { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.1, "max_tokens": 8192 },
    "writing":    { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.4, "max_tokens": 16384 },
    "code_gen":   { "model": "gpt-5.2-codex", "temperature": null, "max_tokens": 16384 },
    "figure_prompt": { "model": "gpt-5.2", "temperature": 0.5, "max_tokens": 4096 },
    "figure_gen": { "model": "dall-e-3", "temperature": null },
    "templateFormat": "arxiv",
    "maxRetries": 2
  }
}
```

### 4.2 环境变量覆盖

| 环境变量 | 覆盖字段 |
|---------|---------|
| `NANORESEARCH_BASE_URL` | base_url |
| `NANORESEARCH_API_KEY` | api_key |
| `NANORESEARCH_TIMEOUT` | timeout |

### 4.3 多模型路由

每个阶段可以配置不同的模型、温度和 token 限制。`ModelDispatcher` 类根据阶段名称自动路由到对应模型配置，通过 OpenAI Python SDK 调用任意 OpenAI 兼容端点。

---

## 五、CLI 使用

```bash
# 完整运行：从主题到论文
nanoresearch run --topic "Attention mechanisms in graph neural networks" --format arxiv

# 断点续跑
nanoresearch resume --workspace ~/.nanobot/workspace/research/{session_id}

# 查看状态
nanoresearch status --workspace ~/.nanobot/workspace/research/{session_id}

# 列出所有会话
nanoresearch list

# 导出结果
nanoresearch export --workspace ~/.nanobot/workspace/research/{session_id} --output ./output
```

---

## 六、Pipeline 运行逻辑

### 6.1 状态机

Pipeline 由一个有限状态机驱动，定义了 8 个状态和合法的状态转移：

```
INIT → IDEATION → PLANNING → EXPERIMENT → FIGURE_GEN → WRITING → DONE
  ↓        ↓          ↓          ↓            ↓           ↓
FAILED   FAILED     FAILED     FAILED      FAILED      FAILED
```

`PipelineStateMachine` 确保只能按合法路径转移状态，防止非法跳转。

### 6.2 编排器 (Orchestrator)

`PipelineOrchestrator` 是核心调度器，执行流程如下：

```python
async def run(topic: str):
    workspace = Workspace.create(session_id, topic)  # 创建工作空间
    state_machine = PipelineStateMachine(INIT)

    for stage in [IDEATION, PLANNING, EXPERIMENT, FIGURE_GEN, WRITING]:
        state_machine.transition(stage)
        workspace.start_stage(stage)

        for attempt in range(max_retries + 1):
            try:
                result = await agent.run(**inputs)    # 执行对应 Agent
                workspace.complete_stage(stage)
                break
            except Exception as e:
                workspace.log_error(stage, attempt, e)
                if attempt == max_retries:
                    state_machine.fail()
                    raise RuntimeError(f"Stage {stage} failed")

    state_machine.transition(DONE)
    workspace.export()
```

关键特性：
- **断点续跑**：跳过已完成的阶段，从当前/失败阶段继续
- **重试机制**：每个阶段最多重试 `maxRetries` 次（默认 2）
- **错误注入**：重试时将上次错误作为上下文传递给 Agent
- **产物注册**：每个阶段的输出文件记录 MD5 校验和

### 6.3 五个阶段详解

#### Stage 1: IDEATION（文献调研）

**Agent**: `IdeationAgent`
**模型**: deepseek-ai/DeepSeek-V3.2 (temperature=0.5)

```
输入: 研究主题 (topic)
  ↓
1. LLM 生成 5-8 个搜索关键词
  ↓
2. 并行调用 arXiv API + Semantic Scholar API（令牌桶限流）
  ↓
3. 去重，收集 ≥20 篇论文
  ↓
4. LLM 分析论文，产出:
   - 综述摘要 (300-500 词)
   - 研究空白列表 (含严重程度)
   - 2-4 个新假设
   - 选定最有前景的假设
  ↓
输出: papers/ideation_output.json
```

#### Stage 2: PLANNING（实验规划）

**Agent**: `PlanningAgent`
**模型**: deepseek-ai/DeepSeek-V3.2 (temperature=0.2)

```
输入: 选定的假设 + 文献调研结果
  ↓
LLM 生成 ExperimentBlueprint:
  - 数据集 (名称、来源、预处理)
  - 基线方法
  - 提出的方法架构
  - 评估指标
  - 消融实验设计
  - 计算资源需求
  ↓
输出: plans/experiment_blueprint.json
```

#### Stage 3: EXPERIMENT（代码生成）

**Agent**: `ExperimentAgent`
**模型**: gpt-5.2-codex (temperature=0.1)

采用**两阶段生成**策略：

```
Phase 1: 架构规划 (1 次 LLM 调用)
  - 列出所有需要的文件
  - 定义接口和函数签名
  - 生成接口契约文档
  ↓
Phase 2: 逐文件生成 (每文件 1 次 LLM 调用)
  - 传入完整的接口契约
  - 确保跨文件一致性
  ↓
输出: code/ 目录下的完整 Python 项目
  ├── main.py
  ├── requirements.txt
  ├── config/default.yaml
  ├── src/ (model.py, dataset.py, trainer.py, evaluate.py, utils.py)
  ├── scripts/ (train.sh, run_ablation.sh)
  └── README.md
```

#### Stage 4: FIGURE_GEN（图表生成）

**Agent**: `FigureAgent`
**模型**: gpt-5.2 (prompt 生成) + dall-e-3 (图像生成)

```
对每个图表规格 (architecture, results, ablation):
  1. LLM 生成详细的图像描述 prompt
  2. 调用 DALL-E 3 生成图像
  3. 保存为 PNG + PDF
  ↓
输出: figures/
  ├── fig1_architecture.pdf/png
  ├── fig2_results.pdf/png
  └── fig3_ablation.pdf/png
```

#### Stage 5: WRITING（论文撰写）

**Agent**: `WritingAgent`
**模型**: deepseek-ai/DeepSeek-V3.2 (temperature=0.4, max_tokens=16384)

```
1. 从论文列表构建 BibTeX 引用 (author+year 去重)
2. LLM 生成论文标题
3. LLM 生成摘要 (150-250 词)
4. 逐章节生成 (6 个章节 × 1 次 LLM 调用):
   Introduction → Related Work → Method → Experiments → Results → Conclusion
5. Jinja2 渲染 LaTeX（自定义分隔符避免与 LaTeX 冲突）
6. pdflatex + bibtex 编译 PDF
  ↓
输出: drafts/
  ├── paper.tex
  ├── references.bib
  ├── paper.pdf
  └── paper_skeleton.json
```

---

## 七、工作空间结构

每个研究会话有独立的工作空间：

```
~/.nanobot/workspace/research/{session_id}/
├── manifest.json              # 主执行记录 (阶段状态、时间戳、产物列表)
├── papers/
│   └── ideation_output.json   # 文献调研输出
├── plans/
│   ├── experiment_blueprint.json  # 实验蓝图
│   └── project_plan.json         # 代码架构规划
├── code/                      # 生成的完整实验代码
│   ├── main.py
│   ├── src/
│   └── ...
├── figures/                   # 生成的科学图表
│   ├── fig1_architecture.pdf
│   └── ...
├── drafts/                    # 论文草稿
│   ├── paper.tex
│   ├── paper.pdf
│   └── references.bib
└── logs/                      # 错误日志
    └── ideation_error_0.txt
```

`Workspace` 类管理这些目录的创建、文件读写和产物注册（含 MD5 校验和）。

---

## 八、LaTeX 模板系统

模板使用 Jinja2 引擎，为避免与 LaTeX 语法冲突，采用自定义分隔符：

| 标准 Jinja2 | NanoResearch |
|-------------|-------------|
| `{% %}` | `<% %>` |
| `{{ }}` | `<< >>` |
| `{# #}` | `<# #>` |

支持三种论文格式，每种格式可继承 `base/` 模板并按需覆盖：

- **arxiv** — arXiv 预印本格式
- **neurips** — NeurIPS 会议格式
- **icml** — ICML 会议格式

---

## 九、MCP 服务

`mcp_server/` 提供 Model Context Protocol 服务，将工具以标准协议暴露给外部 LLM 系统：

| 工具 | 功能 | 限流 |
|------|------|------|
| `search_arxiv` | arXiv 论文搜索 | 3 次/秒 |
| `search_semantic_scholar` | Semantic Scholar 搜索 | 10 次/秒 |
| `generate_latex` | Jinja2 模板渲染 | - |
| `compile_pdf` | pdflatex 编译 | - |
| `generate_figure` | matplotlib 图表生成 | - |

限流采用令牌桶算法 + asyncio Lock 实现。

---

## 十、错误处理与恢复

### 重试策略

```
每个阶段最多 maxRetries (默认 2) 次重试
  ↓ 失败
保存错误日志到 logs/{stage}_error_{attempt}.txt
  ↓
递增重试计数器
  ↓ 下次重试
将上次错误信息注入 Agent 上下文（帮助 LLM 避免同样的错误）
  ↓ 全部重试耗尽
状态转移到 FAILED，抛出 RuntimeError
```

### 断点续跑

```bash
nanoresearch resume --workspace /path/to/workspace
```

- 读取 `manifest.json` 恢复状态
- 跳过状态为 `completed` 的阶段
- 从 `failed` 或 `in_progress` 的阶段重新开始

---

## 十一、核心模块依赖关系

```
CLI (cli.py)
  │
  ├──→ Config (config.py) → ResearchConfig
  │
  ├──→ Workspace (workspace.py) → WorkspaceManifest
  │
  └──→ PipelineOrchestrator (orchestrator.py)
        │
        ├──→ PipelineStateMachine (state.py)
        │
        ├──→ ModelDispatcher (multi_model.py) → OpenAI SDK
        │
        └──→ 5 Agents:
              │
              ├── IdeationAgent
              │   ├──→ arXiv API (mcp_server/tools)
              │   └──→ Semantic Scholar API (mcp_server/tools)
              │
              ├── PlanningAgent
              │
              ├── ExperimentAgent
              │
              ├── FigureAgent
              │   └──→ Image Generation (DALL-E 3)
              │
              └── WritingAgent
                  ├──→ LaTeX Templates (Jinja2)
                  └──→ PDF Compile (pdflatex + bibtex)
```

---

## 十二、技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 异步 | asyncio |
| HTTP | httpx (async) |
| LLM | OpenAI SDK (兼容任意 OpenAI-compatible 端点) |
| XML | lxml |
| 模板 | Jinja2 |
| 数据校验 | Pydantic v2 |
| CLI | Typer + Rich |
| 绘图 | matplotlib |
| 文档编译 | pdflatex + bibtex |
| 协议 | MCP (Model Context Protocol) |
| 测试 | pytest + pytest-asyncio + respx |

---

## 十三、快速开始

```bash
# 1. 安装
cd nanoresearch
pip install -e .

# 2. 配置
cp examples/config_example.json ~/.nanobot/config.json
# 编辑 config.json，填入你的 API key 和 base_url

# 3. 运行
nanoresearch run --topic "Your Research Topic" --format arxiv

# 4. 查看结果
nanoresearch status --workspace ~/.nanobot/workspace/research/<session_id>

# 5. 导出
nanoresearch export --workspace ~/.nanobot/workspace/research/<session_id> --output ./my_paper
```
