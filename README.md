<a id="top"></a>

<div align="center">

<div style="margin: 20px 0;">
  <img src="imgs/logo.png" alt="NanoResearch Logo" width="500" style="border-radius: 20px;">
</div>

# NanoResearch

<p>
  <a href="README.md"><img src="https://img.shields.io/badge/🇨🇳_中文-1a1a2e?style=for-the-badge" alt="中文"></a>
  <a href="README_en.md"><img src="https://img.shields.io/badge/🇺🇸_English-1a1a2e?style=for-the-badge" alt="English"></a>
</p>

**端到端自主 AI 科研引擎 — 从研究想法到完整论文，全程自动化**

<p>
  <a href="https://github.com/OpenRaiser/NanoResearch"><img src="https://img.shields.io/badge/🔥_Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e" alt="Project"></a>
  <a href="https://github.com/OpenRaiser/NanoResearch/stargazers"><img src="https://img.shields.io/github/stars/OpenRaiser/NanoResearch?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e" alt="Stars"></a>
  <a href="https://github.com/OpenRaiser/NanoResearch/issues"><img src="https://img.shields.io/badge/🐛_Issues-ff6b6b?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e" alt="Issues"></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-16A34A?style=for-the-badge&labelColor=1a1a2e" alt="License">
  <img src="https://img.shields.io/badge/Pipeline-9_Stage_Deep-7C3AED?style=for-the-badge&labelColor=1a1a2e" alt="Pipeline">
  <img src="https://img.shields.io/badge/Execution-Local_|_SLURM-0EA5E9?style=for-the-badge&labelColor=1a1a2e" alt="Execution">
</p>

<p>
  <a href="#快速开始"><b>快速开始</b></a> ·
  <a href="#效果展示"><b>效果展示</b></a> ·
  <a href="#流水线"><b>流水线</b></a> ·
  <a href="#claude-code-模式"><b>Claude Code</b></a> ·
  <a href="#飞书机器人"><b>飞书机器人</b></a>
</p>

</div>

---

> 🔬 NanoResearch **真正运行计算实验**——它不仅生成代码，还能将代码提交到 GPU 集群执行训练，收集真实实验结果，生成论文配图，最终输出一篇有实验数据支撑的完整 LaTeX 论文。论文中的每一个数据、表格、图表都来自实际运行的实验结果，**而非 LLM 编造**。

---

## 📖 目录

- [📊 论文实测展示](#-论文实测展示)
- [⚡ CLI 演示](#cli-demo)
- [📢 最新动态](#-最新动态)
- [✨ 核心特性](#-核心特性)
- [🆚 为何选择 NanoResearch](#-为何选择-nanoresearch)
- [🎯 应用场景](#-应用场景)
- [🖼️ 效果展示](#️-效果展示)
- [CLI：标准输出与 TUI 对比](#cli-tui-vs-plain)
- [🔬 流水线](#-流水线)
- [📦 快速开始](#-快速开始)
- [🧩 Claude Code 模式](#-claude-code-模式)
- [⚙️ 配置](#️-配置)
- [💻 CLI 参考](#-cli-参考)
- [🍪 示例与 Demo](#-示例与-demo)
- [📂 输出结构](#-输出结构)
- [💬 飞书机器人](#-飞书机器人)
- [🏗️ 项目结构](#-项目结构)
- [❓ 常见问题](#-常见问题)
- [🎯 路线图](#-路线图)
- [📋 环境要求](#-环境要求)
- [🙏 致谢](#-致谢)
- [🤝 贡献](#-贡献)
- [📝 引用](#-引用)
- [📄 许可证](#-许可证)

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

## 📊 论文实测展示

以下为 NanoResearch 自动生成论文中的**真实配图**，所有数据、曲线、表格均来自实际运行的实验结果，**非 LLM 虚构**。

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="imgs/recognition_1.jpg" height="200" alt="实验结果"/>
      <br />
      <sub><b>实验结果展示</b></sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="imgs/recognition_2.jpg" height="200" alt="主结果对比"/>
      <br />
      <sub><b>方法对比 / 主结果</b></sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="imgs/recognition_3.jpg" height="200" alt="消融与可视化"/>
      <br />
      <sub><b>消融与可视化</b></sub>
    </td>
  </tr>
</table>

> 以上配图均为流水线自动生成，数据来源于真实训练日志与实验结果。

---

<a id="cli-demo"></a>

## ⚡ CLI 演示

NanoResearch **命令行（CLI）** 提供 **TUI 全屏界面** 与 **传统流式日志** 两种呈现方式。下方为 CLI 端 **TUI 主题与界面** 演示视频（配色切换与布局优化）。其他入口（如 Claude Code、飞书机器人）见文档对应章节。

<div align="center" style="max-width: 720px; margin: 0 auto;">
  <video src="imgs/demo_cli_tui_theme.mp4" controls muted playsinline loop width="100%" style="max-width: 100%; border-radius: 8px;">
    <a href="imgs/demo_cli_tui_theme.mp4">下载 / 播放 CLI 演示视频</a>
  </video>
  <br /><sub><i>演示：TUI 界面、配色主题切换与信息布局优化</i></sub>
</div>

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

## Table of Contents

- [Highlights](#highlights)
- [Why NanoResearch](#why-nanoresearch)
- [Use Cases](#use-cases)
- [Showcase](#showcase)
- [Pipeline](#pipeline)
- [Key Capabilities](#key-capabilities)
- [Quick Start](#quick-start)
- [Claude Code Mode](#claude-code-mode)
- [Codex Integration](#codex-integration)
- [Execution Profiles](#execution-profiles)
- [CLI Commands](#cli-commands)
- [Output Structure](#output-structure)
- [Model Routing](#model-routing)
- [Paper Formats](#paper-formats)
- [Project Structure](#project-structure)
- [Feishu Bot](#feishu-bot)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

## Why NanoResearch

| 特性 | 传统 AI 写作工具 | NanoResearch |
|------|-----------------|-------------|
| 文献检索 | 部分支持 | ✅ OpenAlex + Semantic Scholar 自动检索 |
| 实验设计 | ❌ | ✅ 自动生成实验方案 |
| 代码生成 | 部分支持 | ✅ 完整可运行的实验代码 |
| **GPU 实验执行** | ❌ | ✅ **本地 / SLURM 自动训练** |
| 结果分析 | ❌ | ✅ 解析真实训练日志 |
| 论文配图 | ❌ | ✅ 基于真实数据 |
| 论文撰写 | 大纲/草稿 | ✅ 完整 LaTeX 论文 |
| 断点续跑 | ❌ | ✅ 任意阶段可恢复 |
| 多模型协作 | 单一模型 | ✅ 按阶段路由 |

---

## 🎯 应用场景

- **科研原型验证** — 快速将研究想法变成完整的实验 + 论文工作空间
- **自主实验** — 系统自动生成代码、提交 GPU 训练、分析结果
- **Benchmark 批量生成** — 对多个课题批量运行，生成可复现的实验结果
- **论文初稿辅助** — 基于真实实验数据产出 LaTeX 草稿，加速写作
- **科研流程审计** — 完整工作空间、中间产物和日志，可追溯每一步

---

## 🖼️ 效果展示

<div align="center">
  <img src="imgs/before_after.png" alt="告别手动科研" width="90%" />
  <p><b>告别手动科研的痛苦循环</b></p>
  <p>不再反复调试失败的实验、手动整理数据、从零写论文——<br/>NanoResearch 将完整科研流程自动化，让你专注于真正的研究创新。</p>
</div>

<a id="cli-tui-vs-plain"></a>

### CLI：标准输出与 TUI 模式对比

CLI 支持 **传统流式日志（非 TUI）** 与 **全屏 TUI 面板** 两种呈现方式，可按习惯切换。

<div align="center" style="max-width: 720px; margin: 0 auto;">
  <p><b>标准输出（非 TUI）</b></p>
  <img src="imgs/demo_no_tui_mode.png" width="100%" alt="NanoResearch CLI 标准输出模式" style="max-width: 100%; border-radius: 8px;" />
  <p><sub><b>非 TUI</b>：经典终端日志流，便于重定向与脚本集成</sub></p>
</div>
<div align="center" style="max-width: 720px; margin: 0 auto; margin-top: 1.5em;">
  <p><b>TUI 模式</b></p>
  <img src="imgs/demo_tui_mode.png" width="100%" alt="NanoResearch CLI TUI 模式" style="max-width: 100%; border-radius: 8px;" />
  <p><sub><b>TUI</b>：结构化面板与状态分区，适合交互式监控</sub></p>
</div>

### 示例输出

<table>
  <tr>
    <td align="center" width="50%">
      <img src="imgs/framework_overview.png" alt="框架概览" width="95%" />
      <br />
      <sub><b>框架概览</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="imgs/examples.png" alt="生成论文示例" width="95%" />
      <br />
      <sub><b>生成论文示例</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="imgs/main_results.png" alt="主结果" width="95%" />
      <br />
      <sub><b>主结果（真实实验数据）</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="imgs/ablation.png" alt="消融实验" width="95%" />
      <br />
      <sub><b>消融实验</b></sub>
    </td>
  </tr>
</table>

---

## 🔬 流水线

```text
Research Topic
     ↓
 IDEATION → PLANNING → SETUP → CODING → EXECUTION → ANALYSIS → FIGURE_GEN → WRITING → REVIEW
     ↓
 Exported: paper.pdf / paper.tex / references.bib / figures / code / data
```

<details>
<summary><b>📋 各阶段详细说明</b></summary>

| Stage | 功能 | 说明 |
|-------|------|------|
| `IDEATION` | 文献检索与创意生成 | 搜索学术文献、发现研究空白、提出假说、收集必引文献 |
| `PLANNING` | 实验方案设计 | 将研究想法转化为详细的实验蓝图（数据集、基线、指标、消融） |
| `SETUP` | 环境准备 | 准备代码仓库、依赖环境、模型和数据集 |
| `CODING` | 代码生成 | 生成完整可运行的实验项目（含训练脚本、数据处理、模型定义） |
| `EXECUTION` | **实验执行** | **在本地 GPU 或 SLURM 集群上运行训练，支持自动重试和调试** |
| `ANALYSIS` | 结果分析 | 解析训练日志和指标，生成结构化实验证据 |
| `FIGURE_GEN` | 图表生成 | 创建架构图、结果对比图、消融实验图 |
| `WRITING` | 论文撰写 | 基于实验证据和引用撰写 LaTeX 论文 |
| `REVIEW` | 审稿与修订 | 自动审阅各章节，检测问题并修订 |

</details>

<details>
<summary><b>🚀 EXECUTION 阶段核心能力</b></summary>

`EXECUTION` 阶段是 NanoResearch 的核心差异化能力：

- **自动提交 SLURM 作业** — 生成 sbatch 脚本，提交到集群，监控作业状态
- **本地 GPU 执行** — 自动检测可用 GPU，管理训练进程
- **自动调试与重试** — 训练失败时自动分析错误日志，修复代码并重新执行
- **实时日志监控** — 追踪训练进度和指标变化
- **混合执行模式** — 可根据任务复杂度在本地和集群之间自动切换

</details>

---

## 📦 快速开始

遵循以下步骤，约 5 分钟即可完成从安装到首次运行的完整流程。

### 步骤一：安装

```bash
git clone https://github.com/OpenRaiser/NanoResearch.git
cd NanoResearch
pip install -e ".[dev]"
```

### 步骤二：配置

> [!TIP]
> 创建 `~/.nanobot/config.json`，替换 `base_url` 和 `api_key` 为你自己的 OpenAI 兼容 API 端点。

<details>
<summary><b>查看完整配置示例</b></summary>

```json
{
  "research": {
    "base_url": "https://your-openai-compatible-endpoint/v1/",
    "api_key": "your-api-key",
    "template_format": "neurips2025",
    "execution_profile": "local_quick",
    "writing_mode": "hybrid",
    "max_retries": 2,
    "auto_create_env": true,
    "auto_download_resources": true,
    "ideation": { "model": "your-model", "temperature": 0.5, "max_tokens": 16384, "timeout": 600.0 },
    "planning": { "model": "your-model", "temperature": 0.2, "max_tokens": 16384, "timeout": 600.0 },
    "code_gen": { "model": "your-model", "temperature": 0.1, "max_tokens": 16384, "timeout": 600.0 },
    "writing": { "model": "your-model", "temperature": 0.4, "max_tokens": 16384, "timeout": 600.0 },
    "figure_gen": {
      "model": "gemini-3.1-flash-image-preview",
      "image_backend": "gemini",
      "temperature": null,
      "timeout": 300.0
    },
    "review": { "model": "your-model", "temperature": 0.3, "max_tokens": 16384, "timeout": 300.0 }
  }
}
```

</details>

环境变量覆盖：`NANORESEARCH_BASE_URL` / `NANORESEARCH_API_KEY` / `NANORESEARCH_TIMEOUT`

### 步骤三：验证与运行

```bash
# 验证配置
nanoresearch run --topic "Adaptive Sparse Attention Mechanisms" --dry-run

# 启动完整流水线
nanoresearch run --topic "Adaptive Sparse Attention Mechanisms" --format neurips2025 --verbose

# 从断点恢复（若某阶段失败）
nanoresearch resume --workspace ~/.nanobot/workspace/research/{session_id} --verbose

# 导出论文
nanoresearch export --workspace ~/.nanobot/workspace/research/{session_id} --output ./my_paper
```

### 步骤四：预期输出

完成流水线后，你将得到包含真实实验数据的论文配图与 LaTeX 源码。以下为实测输出示例：

<table>
  <tr>
    <td align="center" width="33%">
      <img src="imgs/recognition_1.jpg" width="95%" alt="实验结果"/>
      <br /><sub><b>实验结果</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="imgs/recognition_2.jpg" width="95%" alt="主结果对比"/>
      <br /><sub><b>主结果对比</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="imgs/recognition_3.jpg" width="95%" alt="消融与可视化"/>
      <br /><sub><b>消融与可视化</b></sub>
    </td>
  </tr>
</table>

> <!-- 可选：添加配置截图 imgs/quickstart_config.png、运行终端截图 imgs/quickstart_run.png 以增强说明 -->

---

## 🤖 推荐模型

| Stage | 任务 | 推荐模型 | 经济型 |
|-------|------|---------|-------|
| `ideation` | 文献检索 + 假说生成 | DeepSeek-V3.2 | DeepSeek-V3.2 |
| `planning` | 实验设计 | Claude Sonnet 4.6 | DeepSeek-V3.2 |
| `code_gen` | 代码生成 | GPT-5.2-Codex / Claude Opus 4.6 | DeepSeek-V3.2 |
| `writing` | 论文撰写 | Claude Opus 4.6 / Claude Sonnet 4.6 | DeepSeek-V3.2 |
| `figure_prompt` | 图表描述 | GPT-5.2 | DeepSeek-V3.2 |
| `figure_code` | 图表绘制代码 | Claude Opus 4.6 | DeepSeek-V3.2 |
| `figure_gen` | AI 架构图生成 | Gemini 3.1 Flash（原生图像生成） | Gemini 3.1 Flash |
| `review` | 审稿 + 修订 | Claude Sonnet 4.6 / Gemini Flash | DeepSeek-V3.2 |

> **说明**：所有文本模型通过单一 OpenAI 兼容端点访问。对于不支持 temperature 的模型（如 Codex、o 系列），设置 `temperature: null`。`figure_gen` 使用 Gemini 原生图像生成 API，需设置 `"image_backend": "gemini"`。

### 💰 预估费用

| 场景 | 模型选择 | 时间 | 预估费用 |
|------|---------|------|---------|
| **仅生成论文**（跳过实验） | 全部 DeepSeek-V3.2 | ~30 分钟 | ~$0.5 - $1 |
| **仅生成论文**（跳过实验） | 混合（Claude 写作，DeepSeek 其余） | ~30 分钟 | ~$3 - $8 |
| **完整流水线**（含实验） | 全部 DeepSeek-V3.2 | 2 - 5 小时 | ~$1 - $3 |
| **完整流水线**（含实验） | 混合（Claude/GPT 代码+写作） | 2 - 5 小时 | ~$10 - $20 |

> "仅生成论文"模式通过 `"skip_stages": ["SETUP", "CODING", "EXECUTION", "ANALYSIS"]` 跳过实验阶段。

---

## 🧩 Claude Code 模式

除了 Python CLI，NanoResearch 还支持通过 **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** 直接驱动研究流水线——**无需配置任何 API Key**。

<details>
<summary><b>工作原理</b></summary>

在 Claude Code 集成模式下，Claude Code 本身就是研究引擎：

- **WebSearch** 替代外部 API 进行文献检索（arXiv、Semantic Scholar、Google Scholar）
- **Bash** 执行实验代码、提交 SLURM 作业、编译 LaTeX
- **文件读写** 生成实验代码、论文和结构化产物

</details>

### 快速开始

```bash
git clone https://github.com/OpenRaiser/NanoResearch.git
cd NanoResearch
claude
/project:research "你的研究课题"
```

### 可用命令

| 命令 | 功能 |
|---------|------|
| `/project:research <课题>` | 运行完整 9 阶段流水线 |
| `/project:ideation <topic>` | Stage 1: 文献检索 + 假说生成 |
| `/project:planning` | Stage 2: 实验方案设计 |
| `/project:experiment` | Stages 3-5: 环境准备 + 代码生成 + 实验执行 |
| `/project:analysis` | Stage 6: 实验结果分析 |
| `/project:writing` | Stages 7-8: 图表生成 + 论文撰写 |
| `/project:review` | Stage 9: 多视角审稿 + 修订 |
| `/project:status` | 查看当前流水线状态 |
| `/project:resume` | 从断点恢复流水线 |

<details>
<summary><b>💡 Tips</b></summary>

- **架构图生成**：推荐使用 Nano Banana 系列图像模型生成高质量架构图。Claude Code 模式下可在 `figure_gen` 阶段通过 Bash 调用图像生成 API。
- **LaTeX 编译**：推荐使用 `tectonic` 替代 `pdflatex`。安装：`conda install -c conda-forge tectonic`。
- **断点续跑**：所有阶段的产物保存在 `manifest.json` 中，支持任意阶段恢复。
- **与 Python CLI 兼容**：Claude Code 模式生成的工作空间与 Python CLI 完全兼容，可混合使用两种模式。

</details>

---

## Codex Integration

除了 Claude Code，NanoResearch 现在也提供了面向 **Codex** 的项目级入口。Codex 不需要第二套 pipeline 或额外的 `--mode codex` 参数；它应直接复用仓库现有的 CLI、workspace、manifest 和 `PaperMode` 逻辑。

### 如何使用

```bash
# 1. Clone 项目
git clone https://github.com/OpenRaiser/NanoResearch.git
cd NanoResearch

# 2. 在 Codex 中打开仓库
# 3. 让 Codex 先读取 AGENTS.md
```

Codex 入口文档：`AGENTS.md`

### Codex 会如何理解这个仓库

- 将仓库视为一个端到端自主科研流水线，而不是普通代码仓库
- 优先走已有的 `nanoresearch` CLI / workspace / orchestrator 行为
- 将用户意图映射到现有的 `research`、`ideation`、`planning`、`experiment`、`analysis`、`writing`、`review`、`status`、`resume` 流程
- 保持与现有工作空间和 `manifest.json` 兼容

### Survey / Original 模式

Codex 使用与 CLI 相同的 topic 前缀约定：

- `original: Topic`
- `survey:short: Topic`
- `survey:standard: Topic`
- `survey:long: Topic`

这些前缀会进入仓库现有的 `PaperMode` 解析逻辑；Codex 只需要遵循这一约定，而不需要发明新的接口。

## Execution Profiles

| Profile | 说明 |
|---------|------|
| `fast_draft` | 轻量级草稿模式，快速迭代 |
| `local_quick` | 优先本地执行，需要时可升级到 SLURM |
| `cluster_full` | 集群优先，适合重量级实验 |

### 模型路由

NanoResearch 通过统一配置层将不同阶段路由到不同模型，让你按任务特性混合搭配，而非强制所有阶段使用同一模型。

<details>
<summary><b>可路由的阶段</b></summary>

- `ideation` — 文献检索与创意
- `planning` — 实验设计
- `experiment` — 实验相关
- `code_gen` — 代码生成
- `writing` — 论文撰写
- `figure_prompt` — 图表描述
- `figure_code` — 图表代码
- `figure_gen` — 图像生成
- `review` — 审稿
- `revision` — 修订

系统基于 **OpenAI 兼容端点**构建，支持按阶段覆盖配置。

</details>

### 文献检索 API Keys（可选）

IDEATION 阶段使用 OpenAlex 和 Semantic Scholar 检索学术文献。不配置也能运行（匿名访问），但速率限制较低。

| Service | 获取方式 | Config key | Env variable |
|---------|---------|------------|--------------|
| [OpenAlex](https://developers.openalex.org/) | 免费 | `openalex_api_key` | `OPENALEX_API_KEY` |
| [Semantic Scholar](https://www.semanticscholar.org/product/api#api-key) | 免费 | `s2_api_key` | `S2_API_KEY` |

### 论文格式

模板从 `nanoresearch/templates/` 自动发现。内置模板：

| Format | 用途 |
|--------|------|
| `arxiv` | arXiv 预印本 |
| `icml` | ICML 会议 |
| `neurips` | NeurIPS 会议 |
| `neurips2025` | NeurIPS 2025 |

```bash
nanoresearch run --topic "Graph Foundation Models for Biology" --format neurips2025
```

---

## 💻 CLI 参考

| 命令 | 用途 |
|---------|------|
| `nanoresearch run --topic "..."` | 启动新的流水线运行 |
| `nanoresearch resume --workspace ...` | 从上次断点恢复 |
| `nanoresearch status --workspace ...` | 查看各阶段状态和产物 |
| `nanoresearch list` | 列出已保存的研究会话 |
| `nanoresearch export --workspace ...` | 导出论文打包 |
| `nanoresearch config` | 打印当前配置（密钥已屏蔽） |
| `nanoresearch inspect --workspace ...` | 检查工作空间产物 |
| `nanoresearch health` | 运行环境/配置健康检查 |
| `nanoresearch delete <session_id>` | 删除指定会话 |

```bash
nanoresearch --help
```

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

## 🍪 示例与 Demo

官方示例与进阶用法，助你快速上手 NanoResearch：

👉 **[浏览示例与 Demo](https://github.com/OpenRaiser/NanoResearch)** <!-- TODO: 若有 docs/ 或 examples/ 目录，请替换为具体路径，如 .../blob/main/docs/README.md -->

<details>
<summary><b>快速示例命令</b></summary>

```bash
# 完整流水线 + 详细日志
nanoresearch run --topic "Adaptive Sparse Attention" --format neurips2025 --verbose

# 导出并查看
nanoresearch export --workspace ~/.nanobot/workspace/research/{session_id} --output ./paper_out
```

> 仅生成论文（跳过实验）：在 config 中设置 `"skip_stages": ["SETUP", "CODING", "EXECUTION", "ANALYSIS"]`

</details>

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

## 📂 输出结构

<details>
<summary><b>导出的论文目录</b></summary>

```text
my_paper/
├── paper.pdf
├── paper.tex
├── references.bib
├── figures/
├── code/
├── data/
└── manifest.json
```

</details>

<details>
<summary><b>完整工作空间（含中间产物）</b></summary>

```text
~/.nanobot/workspace/research/{session_id}/
├── manifest.json          # 流水线状态追踪
├── papers/                # 文献检索产物
├── plans/                 # 实验方案和分析
├── experiment/            # 生成的实验代码 + 结果
├── figures/               # 生成的论文配图
├── drafts/                # 论文草稿和审稿意见
├── output/                # 最终导出（main.tex / main.pdf）
└── logs/                  # 运行日志
```

</details>

---

## 💬 飞书机器人

NanoResearch 内置飞书（Lark）机器人，可直接在飞书聊天中触发流水线、查看进度、接收论文——无需打开终端。

<details>
<summary><b>配置与启动</b></summary>

**1. 安装依赖**

```bash
pip install lark-oapi
```

**2. 配置**

在 [open.feishu.cn](https://open.feishu.cn) 创建自定义应用并获取 App ID 和 App Secret：

```bash
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
```

或写入 `~/.nanobot/config.json`：

```json
{
  "feishu": {
    "app_id": "cli_xxx",
    "app_secret": "xxx"
  }
}
```

**3. 启动**

```bash
nanoresearch feishu          # 启动机器人
nanoresearch feishu -v       # 详细日志模式
```

机器人通过 WebSocket 长连接通信（无需公网服务器或 Webhook URL）。按 `Ctrl+C` 停止。

</details>

### 支持的命令

| 命令 | 描述 |
|---------|------|
| `/run <课题>` | 对指定课题启动研究流水线 |
| `/status` | 查看当前任务进度 |
| `/list` | 列出所有历史研究会话 |
| `/stop` | 停止当前运行的流水线 |
| `/export` | 重新导出最近完成的研究 |
| `/new` | 清除对话记忆，重新开始 |
| `/help` | 显示帮助信息 |

也可以直接自然语言聊天——机器人充当 AI 科研助手，支持对话记忆，流水线完成后自动发送 `paper.pdf`。

---

## 🏗️ 项目结构

```text
nanoresearch/
├── nanoresearch/
│   ├── cli.py              # 🖥️ CLI 入口
│   ├── config.py           # ⚙️ 配置管理
│   ├── agents/             # 🧠 各阶段 Agent
│   │   ├── ideation.py     #    文献检索与假说
│   │   ├── planning.py     #    实验方案设计
│   │   ├── coding.py       #    代码生成
│   │   ├── execution/      #    本地/集群执行
│   │   ├── analysis/       #    结果分析
│   │   ├── figure_gen/     #    图表生成
│   │   ├── writing/        #    论文撰写
│   │   └── review/         #    审稿与修订
│   ├── pipeline/           # 🔄 编排器 & 状态机
│   ├── schemas/            # 📋 Pydantic 数据模型
│   ├── prompts/            # 💬 YAML 提示词模板
│   ├── templates/          # 📄 LaTeX Jinja2 模板
│   └── latex/              # 🔧 LaTeX 自动修复
├── mcp_server/             # 🔌 MCP 工具服务
├── skills/                 # 🎯 Claude Code 技能
└── pyproject.toml
```

---

## ❓ 常见问题

<details>
<summary><b>NanoResearch 真的会运行实验吗？</b></summary>

是的。流水线会生成可运行的代码，在本地 GPU 或 SLURM 集群上执行，并将实验产物传递给后续的分析、配图和写作阶段。**论文中的数据来自真实实验，而非模型编造。**

</details>

<details>
<summary><b>可以断点续跑吗？</b></summary>

可以。工作空间按阶段保存检查点，`nanoresearch resume --workspace ...` 会从上次未完成或失败的阶段继续。

</details>

<details>
<summary><b>每个阶段都需要配置模型吗？</b></summary>

不需要。NanoResearch 支持按阶段配置模型路由，也可以全部使用同一个模型。

</details>

<details>
<summary><b>生成的论文可以直接投稿吗？</b></summary>

建议将其视为高质量初稿，而非最终投稿版本。系统可以生成完整的论文工作空间和编译好的 PDF，但人工审阅和修订仍然必要。

</details>

<details>
<summary><b>LaTeX 编译推荐什么工具？</b></summary>

推荐使用 `tectonic`。Conda 安装的 texlive 可能缺少 `pdflatex.fmt`，导致编译失败且修复困难。`tectonic` 会自动下载所需的 TeX 包，无需额外配置。

```bash
conda install -c conda-forge tectonic
```

</details>

---

## 🎯 路线图

- [x] 9 阶段统一流水线
- [x] 本地 GPU + SLURM 集群执行
- [x] Claude Code 集成模式
- [x] 飞书机器人
- [x] 多会议论文模板（NeurIPS / ICML / arXiv）
- [x] 断点续跑与多模型路由
- [ ] 更多 Demo 与教程
- [ ] Benchmark 评估套件
- [ ] Web 端工作空间 UI
- [ ] 更多论文格式支持

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

## 🤝 贡献

欢迎开发者、研究者贡献代码与创意。

### 👥 社区与交流

加入微信群交流、答疑、协作。扫码加入：

<p align="center">
  <img src="imgs/wechat_group.png" alt="WeChat Group QR Code" width="220"/>
</p>

<a href="https://github.com/OpenRaiser/NanoResearch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRaiser/NanoResearch" alt="Contributors" />
</a>

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

## 📋 环境要求

- Python **3.10+**
- **OpenAI 兼容 API 端点**（用于文本模型阶段）
- 可选：图像模型访问权限（用于部分配图）
- `tectonic` 或 `pdflatex`（用于 PDF 编译）

---

## 🙏 致谢

- [claude-scholar](https://github.com/Galaxy-Dawn/claude-scholar) — Claude Code 的科研技能扩展

---

## ⭐ Star History

<div align="center">
  <a href="https://star-history.com/#OpenRaiser/NanoResearch&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenRaiser/NanoResearch&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenRaiser/NanoResearch&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenRaiser/NanoResearch&type=Date" width="100%" />
    </picture>
  </a>
</div>

---

## 📝 引用

如有帮助，请引用：

```bibtex
@software{nanoresearch2026,
  title = {NanoResearch},
  author = {OpenRaiser},
  year = {2026},
  url = {https://github.com/OpenRaiser/NanoResearch}
}
```

---

## 📄 许可证

MIT

<p align="right"><a href="#top">🔝 返回顶部</a></p>

---

<div align="center">
  <p>
    <a href="https://github.com/OpenRaiser/NanoResearch"><img src="https://img.shields.io/badge/⭐_Star_us_on_GitHub-1a1a2e?style=for-the-badge&logo=github&logoColor=white" alt="Star"></a>
    <a href="https://github.com/OpenRaiser/NanoResearch/issues"><img src="https://img.shields.io/badge/🐛_Report_Issues-ff6b6b?style=for-the-badge&logo=github&logoColor=white" alt="Issues"></a>
    <a href="https://github.com/OpenRaiser/NanoResearch/discussions"><img src="https://img.shields.io/badge/💬_Discussions-4ecdc4?style=for-the-badge&logo=github&logoColor=white" alt="Discussions"></a>
  </p>
  <sub>NanoResearch 仅供教育、研究及技术交流使用。</sub>
</div>
