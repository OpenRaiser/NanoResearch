# Phase 9: Infrastructure — Cost Tracking + Progress Streaming + Structured Logging

## Background
你正在对 NanoResearch（一个自动化科研论文生成 pipeline）做架构改进。Phase 1-8 已完成，343 测试全部通过。

## 项目路径
`C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`

## 任务 A: Cost Tracking（Section 11）

### 问题
用户跑一次 pipeline 不知道花了多少 API 费用。

### 解决方案
1. 在 `nanoresearch/pipeline/multi_model.py` 的 `ModelDispatcher` 中：
   - 新增 `generate_with_usage()` 方法（不改现有 `generate()` 的返回值以避免 breaking change）
   - 返回 `LLMResult(content, usage, model, latency_ms)` dataclass
   - 从 API response 中提取 `prompt_tokens`, `completion_tokens`, `total_tokens`

2. 在 orchestrator 中累积每阶段的 token 用量和延迟：
   - `stage_costs[stage_name] = {total_tokens, num_calls, total_latency_ms}`
   - 保存到 workspace manifest 中

3. pipeline 结束时输出成本摘要日志

### 参考
`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 11 (line 2174)

---

## 任务 B: Progress Streaming（Section 14）

### 问题
用户等 30-60 分钟没有进度反馈。

### 解决方案
1. 创建 `nanoresearch/pipeline/progress.py`：
   - `ProgressEmitter` 类，实时写入 JSON 文件
   - 事件类型：`stage_start`, `stage_complete`, `substep`, `error`
   - 包含 `progress_pct`, `elapsed_s`, `message`

2. 在 orchestrator 的 `_run_stage()` 中调用 `emitter.stage_start()` / `stage_complete()`
3. 在各 agent 的 `self.log()` 调用处也发 `substep` 事件
4. 进度文件路径：`workspace/progress.json`

### 参考
`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 14 (line 2380)

---

## 任务 C: Structured Logging（Section 15）

### 问题
当前日志是纯文本 print-style，难以程序化分析。

### 解决方案
1. 配置 Python `logging` 模块：
   - 控制台保持人类可读格式
   - 文件输出用 JSON 格式（每行一个 JSON 对象）
   - 包含：timestamp, level, agent, stage, message, extra_data

2. 替换 `self.log()` 中的纯字符串为结构化字段：
   - 例：`self.log("Generated section", section=heading, chars=len(content))`

3. 日志文件路径：`workspace/logs/pipeline.jsonl`

### 参考
`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 15 (line 2517)

---

## 任务 D: Blueprint Semantic Validation（Section 17）

### 问题
Planning 阶段生成的 blueprint 可能内部不一致（如 ablation 变量名和方法描述不匹配），但目前没有检查。

### 解决方案
在 PLANNING 之后添加验证步骤：
1. 检查 metrics 列表非空
2. 检查 ablation 变量名出现在 method 描述中
3. 检查 dataset 和 baseline 名称一致
4. 验证失败时让 LLM 修复（最多 1 轮）

### 参考
`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 17 (line 2647)

---

## 约束
- 测试基线：343 passed
- 用中文回复
- 完成后 check bug
- 任务 A 最重要（用户能看到成本），B 次之，C/D 可选
