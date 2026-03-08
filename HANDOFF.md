# NanoResearch 项目交接文档

## 项目概述

NanoResearch 是一个自动化科研论文生成管线，包含 9 个阶段：
```
IDEATION → PLANNING → SETUP → CODING → EXECUTION → ANALYSIS → FIGURE_GEN → WRITING → REVIEW
```

**项目路径**: `C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`
**配置文件**: `C:\Users\17965\.nanobot\config.json`
**工作空间**: `.nanobot_home/workspace/research/{session_id}/`
**LaTeX 编译器**: `D:/anaconda/Scripts/tectonic`
**Git 远程**: `https://github.com/OpenRaiser/NanoStudy-dev.git`

## 当前状态

### 已完成的工作

1. **全流程跑通** — session `mmsd0308run2`（多模态讽刺检测）完成全部 9 阶段，生成 PDF
2. **大规模 bug 修复** — 累计 26 批次、194 个修复，覆盖所有 agent 和基础设施代码
3. **最近两轮全面审查** — 5 个并行 agent 审查了全部 agent 代码，发现并修复了：
   - **CRITICAL**: deep_orchestrator.py resume 状态机不推进（跳过所有后续阶段）
   - **CRASH**: 12 个 None/list 类型守卫（experiment, review, writing, ideation）
   - **HIGH**: 表格加粗方向、LaTeX 特殊字符转义、BibTeX 嵌套花括号、Windows python3、S2 批量 API 引用丢失
   - **MEDIUM**: HTTP 连接泄漏、偏移量漂移、空格归一化破坏格式、重复包插入

### 未推送到 GitHub 的改动

所有修复都在本地，尚未 push。用户可能需要你帮忙推送。涉及的文件：

```
nanoresearch/agents/writing.py      # Batch 24-26: HTML实体、LaTeX转义、lower_is_better、dedup修复
nanoresearch/agents/review.py       # Batch 24-26: BibTeX检测、None守卫、偏移漂移、空格修复
nanoresearch/agents/experiment.py   # Batch 25: termination字段、None守卫、gather修复
nanoresearch/agents/ideation.py     # Batch 25: list守卫、PaperReference修复、ref守卫
nanoresearch/agents/analysis.py     # Batch 24-25: encoding、sys.executable、close()
nanoresearch/agents/coding.py       # Batch 24-25: num_gpus、close()、list解包
nanoresearch/agents/preflight.py    # Batch 24: 入口文件灵活匹配
nanoresearch/pipeline/deep_orchestrator.py  # Batch 25: resume状态机修复
nanoresearch/pipeline/orchestrator.py       # Batch 25: 同上
mcp_server/tools/semantic_scholar.py        # Batch 26: 批量API保留references
```

## 关键架构知识

### 模型配置（config.json）
- `ideation` / `planning` / `figure_prompt`: `gpt-5.2`
- `writing` / `code_gen` / `figure_code` / `revision`: `claude-sonnet-4-6`
- `experiment`: `deepseek-ai/DeepSeek-V3.2`
- `review`: `gemini-3.1-flash-lite-preview`
- `figure_gen`: `gemini-3.1-flash-image-preview`
- API base: `https://api.boyuerichdata.opensphereai.com/v1/`

### 核心设计模式
1. **Figure 内嵌** — WRITING 将图表内嵌到 section.content 中，PaperSkeleton.figures=[] 是设计如此
2. **两级 LaTeX 修复** — Level 1 确定性修复（无 LLM），Level 2 search-replace LLM 修复（安全：不匹配则不改）
3. **ReAct 实验模式** — 实验代码通过 LLM ReAct 循环迭代生成和修复
4. **BibTeX 来源** — S2/OpenAlex/arXiv API 返回的数据可能包含 HTML 实体，已加 `html.unescape()` 处理

### 已知的设计限制（不是 bug，不需要修）
- `_extract_sections` 只处理一级嵌套花括号的标题
- `_apply_revisions` 对 subsection 的 lookahead 用 `{0,2}` 可能截断 subsubsection 内容
- TikZ 内联图表不会被 dedup 逻辑追踪
- `_validate_figures_in_latex` 注入图表时不检查 label 是否已存在

## 运行管线的方法

### 启动新实验
```bash
cd C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main
python -m nanoresearch.cli run --topic "你的研究主题" --mode deep
```

### 恢复已有实验
```python
import asyncio
from pathlib import Path
from nanoresearch.config import ResearchConfig
from nanoresearch.workspace import Workspace
from nanoresearch.pipeline.deep_orchestrator import UnifiedPipelineOrchestrator

config = ResearchConfig.load(Path(r"C:\Users\17965\.nanobot\config.json"))
root = Path(r"C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main\.nanobot_home\workspace\research")
workspace = Workspace(root / "SESSION_ID")  # 替换为实际 session_id
orchestrator = UnifiedPipelineOrchestrator(workspace, config)
result = asyncio.run(orchestrator.run("研究主题"))
```

### 编译 PDF
```bash
D:/anaconda/Scripts/tectonic drafts/paper.tex
```

## 注意事项

### 必须注意
1. **Windows 环境** — 路径用正斜杠或引号，`write_text()` 必须 `encoding="utf-8"`，不能用 `python3` 命令
2. **LLM 返回类型不可信** — `generate_json()` 可能返回 list 而非 dict，`generate()` 可能返回 None，所有调用点都需要守卫
3. **修改 .py 后需重启进程** — Python 已加载的模块不会自动刷新，改了代码要重新运行脚本
4. **config.json 中的 API key 是敏感信息** — 不要在对话中完整展示或推送到公开仓库

### 代码修改原则
- 优先用 `isinstance()` 守卫而非 try/except（更明确）
- LaTeX 文本用 `_escape_latex_text()` 转义（writing.py 中已有）
- BibTeX 处理用 `_sanitize_bibtex()`（已含 HTML 实体处理）
- 所有 `close()` 方法必须调用 `await super().close()`
- `asyncio.gather` 并行任务用 `return_exceptions=True`

### 测试方法
```python
# 语法检查
python -c "import ast; ast.parse(open('file.py', encoding='utf-8').read())"

# 导入检查
python -c "import nanoresearch.agents.writing"

# 完整管线测试：用一个简单主题跑 deep 模式
```

## Memory 文件结构

持久化记忆在 `C:\Users\17965\.claude\projects\C--Users-17965\memory\`:
- `MEMORY.md` — 主索引（194 条修复记录，超过 200 行会被截断）
- `pipeline_observations.md` — 早期运行观察
- `pipeline_run3_observations.md` — Run 3 详细观察
- `run1_issues.md` / `run2_comparison.md` — 运行对比

## 用户偏好

- **语言**: 中文回复
- **风格**: 直接、简洁、技术性
- **决策**: 用户要求修复时直接修，不需要反复确认
- **模型偏好**: 代码生成用 claude-sonnet-4-6（用户认为 DeepSeek 太弱）
