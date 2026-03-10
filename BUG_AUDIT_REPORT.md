## BUG 审查报告

### P0 — 必修（运行时崩溃 / 数据丢失）

#### BUG-001: `skip_stages` 会把状态机走断
- **位置**：`nanoresearch/pipeline/base_orchestrator.py:147`
- **位置**：`nanoresearch/pipeline/base_orchestrator.py:156`
- **位置**：`nanoresearch/pipeline/base_orchestrator.py:216`
- **问题**：配置跳过某个中间 stage 时，代码只是 `continue`，既不推进 `state_machine`，也不把该 stage 标记为已跳过；后续 stage 从当前状态看都变成非法转移，最后 `transition(DONE)` 还会直接抛 `InvalidTransitionError`。
- **复现条件**：标准流水线配置 `skip_stages=["PLANNING"]`；`IDEATION` 跑完后跳过 `PLANNING`，`EXPERIMENT` 无法从 `IDEATION` 转移，最终在进入 `DONE` 时失败。

#### BUG-002: `generate_json()` 无法处理“前置说明 + fenced JSON”
- **位置**：`nanoresearch/agents/base.py:286`
- **问题**：它只在 `text.startswith("```")` 时才去掉 markdown fence；只要 LLM 回 `"Here is JSON:\n```json ...```"` 这种很常见的格式，就会直接抛 `LLMError`。
- **复现条件**：最小复现返回 `Here is JSON:\n```json\n{"a": 1}\n```` 时会稳定报错，而不是提取出对象。

#### BUG-003: 截断 JSON 修复对“截断在字符串中间”这一常见场景是坏修复
- **位置**：`nanoresearch/agents/base.py:143`
- **位置**：`nanoresearch/agents/base.py:176`
- **问题**：当前逻辑在引号数为奇数时，会把文本截到最后一个引号，再补括号；`{"a":"hello` 会被修成 `{"a":"}`，仍然是非法 JSON，导致“长输出被截断”这种本应可恢复的场景直接失败。
- **复现条件**：`_repair_truncated_json('{"a":"hello')` 返回 `{"a":"}`。

#### BUG-004: 精确模型名 `o3` 会被错误当成“非 thinking model”
- **位置**：`nanoresearch/pipeline/multi_model.py:140`
- **位置**：`nanoresearch/pipeline/multi_model.py:308`
- **问题**：`is_thinking` 只识别 `"thinking"`、`o1*`、`o3-*`，不识别精确字符串 `o3`；这样会继续发送 `system` role 和 `max_tokens`，而不是 thinking model 需要的调用形态。
- **复现条件**：`model='o3'` 时，实际发出的 `messages` 仍然以 `{"role":"system"}` 开头，且使用的是 `max_tokens` 而不是 `max_completion_tokens`。

#### BUG-005: `json_mode=True` 对“不支持 JSON mode 的后端”没有降级路径
- **位置**：`nanoresearch/pipeline/multi_model.py:155`
- **位置**：`nanoresearch/pipeline/multi_model.py:185`
- **问题**：只要 `json_mode=True` 就强塞 `response_format={"type":"json_object"}`；如果兼容后端不支持这个参数，代码不会移除该参数重试，而是直接失败。
- **复现条件**：兼容后端返回 `response_format json_object is not supported` 时，调用会终止为 `RuntimeError`。

#### BUG-006: PDF 编译失败不会让 WRITING / REVIEW stage 失败，流水线可能在没有 `paper.pdf` 时仍然 `DONE`
- **位置**：`nanoresearch/agents/writing/__init__.py:715`
- **位置**：`nanoresearch/agents/review.py:465`
- **位置**：`nanoresearch/pipeline/base_orchestrator.py:262`
- **问题**：WRITING 和 REVIEW 都把 PDF 编译失败降级成 `pdf_error` 日志返回；但 orchestrator 只看 `agent.run()` 有没有抛异常，所以会照样 `mark_stage_completed()`，最终整条流水线可在没有 PDF 的情况下结束。
- **复现条件**：LaTeX 编译器缺失，或 Level 1 + Level 2 修复后仍然无法编译。

#### BUG-007: LaTeX 特殊字符转义不完整，容易产出不可编译正文/标题/图注
- **位置**：`nanoresearch/agents/writing/latex_assembler.py:305`
- **位置**：`nanoresearch/agents/writing/__init__.py:169`
- **问题**：`_sanitize_latex()` 只处理了“数字后的 `%`”，正文里的 `_`、`&`、`#`、`$`、`~`、`^` 完全不会补救；而 `_escape_latex_text()` 只要检测到任意 LaTeX 命令就整段原样返回，导致像 `\textbf{Model_A} & baseline` 这样的 caption 也不再做任何转义。
- **复现条件**：`Accuracy on CIFAR_10 improved by 5% & beat baseline #1.` 和 `Caption with \textbf{Model_A} & baseline` 这类内容都会留下会触发 LaTeX 错误的裸字符。

### P1 — 应修（静默错误 / 逻辑缺陷）

#### BUG-008: 第二个进程 `resume` 同一 workspace 时，会把真实运行中的 stage 强行改回 `pending`
- **位置**：`nanoresearch/pipeline/base_orchestrator.py:331`
- **问题**：`_reset_stale_running_stages()` 启动即扫 manifest，把所有 `running` 一律改回 `pending`，没有文件锁、没有 owner 校验、没有“多久算 stale”的判断。
- **复现条件**：同一 workspace 上同时存在两个 pipeline 进程；第二个进程一启动就会篡改第一个进程的 manifest。

#### BUG-009: Level 2 LaTeX search-replace 会改错位置
- **位置**：`nanoresearch/latex/fixer.py:323`
- **问题**：`apply_edits()` 对整篇文档执行 `result.replace(old, new, 1)`；如果 `old` 片段在正文里出现多次，它会修改第一个匹配，而不是报错窗口附近的那一处。
- **复现条件**：相同 table row、figure block、sentence 模板在论文里出现不止一次。

#### BUG-010: 导出的 deep bundle 不是可重编译的自包含包
- **位置**：`nanoresearch/pipeline/workspace.py:383`
- **位置**：`nanoresearch/agents/writing/latex_assembler.py:784`
- **位置**：`nanoresearch/templates/base/paper.tex.j2:55`
- **问题**：正常编译依赖 `_copy_figures_to_drafts()` 把图复制到 `paper.tex` 同目录；但 export 时图被放进 `figures/` 子目录，而模板里的 `\includegraphics{...}` 仍是裸文件名，也没有 `\graphicspath`。
- **复现条件**：导出完成后，在导出根目录直接重新编译 `paper.tex`。

### P2 — 可选（代码质量 / 防御性编程）

#### BUG-011: 多处 `except Exception: pass` 会静默丢掉辅助证据/结果
- **位置**：`nanoresearch/agents/coding.py:354`
- **位置**：`nanoresearch/agents/debug.py:156`
- **位置**：`nanoresearch/agents/execution/result_collector.py:491`
- **位置**：`nanoresearch/agents/setup.py:477`
- **问题**：这些分支会把参考源码片段、源文件快照、辅助结果文件、repo 摘要片段的读取失败完全吞掉；主流程不一定崩，但信息损失不可见。
- **复现条件**：权限问题、编码问题、超长文件、损坏文件、瞬时 I/O 错误。

## 统计
- P0: 7 个
- P1: 3 个
- P2: 1 个
