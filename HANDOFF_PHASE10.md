# HANDOFF PROMPT — Phase 10: Architecture Dedup & Abstraction

> **给新对话的指令**：你继续 NanoResearch 项目的架构改进。
> 代码路径：`C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`
> 当前版本：`V1.9-AP`（git tag），测试基线：**381 passed, 0 failed**
> 回复语言：**中文**
> 完成后保存为 `V2.0-AP` 压缩包到 `C:\Users\17965\Desktop\NaNo\V2.0-AP.zip`

---

## 目标：消除 6 处架构短板

按优先级排序，P1 必须全做，P2 尽量做。每完成一项跑 `python -m pytest tests/ -q` 确认 381 passed 不减少。

---

### Task 1 (P1): Orchestrator 去重 → 抽象基类

**问题**：`orchestrator.py`(426行) 和 `deep_orchestrator.py`(466行) 有 ~60% 重复代码：
- `__init__` 中 CostTracker / ProgressEmitter / callback wiring — 两边各写一遍
- `run()` 中 cost summary 保存、progress_emitter.pipeline_complete、try/except — 几乎相同
- `_run_stage_with_retry()` 几乎相同（仅 deep 版的 `mark_stage_completed` 多传一个参数）
- `_load_stage_output()` 逻辑一样（仅 file_map 不同）
- `_reset_stale_running_stages()` 完全相同
- `_validate_cross_stage_refs()` 只在标准版有，deep 版没有（本身也是问题）
- `_wrap_stage_output()` 完全相同

**方案**：
1. 新建 `nanoresearch/pipeline/base_orchestrator.py`
2. 抽象类 `BaseOrchestrator(ABC)` 包含所有公共逻辑：
   - `__init__`: cost_tracker, progress_emitter, state_machine 初始化 + agent callback wiring
   - `run()`: 主循环框架（模板方法模式）
   - `_run_stage_with_retry()`: 共享重试逻辑
   - `_load_stage_output()`: 接受子类提供的 `_OUTPUT_FILE_MAP`
   - `_reset_stale_running_stages()`: 原样提取
   - `_wrap_stage_output()`: 原样提取
   - `_validate_cross_stage_refs()`: 提取到基类，deep 版也应该用
3. 子类只需提供：
   - `_build_agents()` → 返回 `dict[PipelineStage, BaseResearchAgent]`
   - `_get_processing_stages()` → 返回阶段列表
   - `_prepare_inputs()` → 各自不同的输入准备逻辑
   - `_STAGE_KEY_MAP` 和 `_OUTPUT_FILE_MAP` 类属性
   - Deep 版的额外 export 逻辑可放在 `_post_pipeline()` 钩子中
4. 标准版的 `progress_callback` 可提到基类（deep 版之前没有，但应该统一支持）

**验证**：两个 orchestrator 测试 + 全量测试必须通过。

**关键文件**：
- `nanoresearch/pipeline/orchestrator.py` (426行)
- `nanoresearch/pipeline/deep_orchestrator.py` (466行)
- `tests/test_cost_tracking.py` → `TestOrchestratorCostWiring`, `TestOrchestratorCostAndProgress`

---

### Task 2 (P1): ModelDispatcher retry 去重

**问题**：`multi_model.py`(593行) 有 6 个 `for attempt in range(...)` 重试循环：
- `generate()` (line 151) — MAX_API_RETRIES+1, chat completion
- `generate_with_usage()` (line 229) — 和 generate() **几乎完全相同**，仅返回值不同
- `generate_with_image()` (line 318) — MAX_API_RETRIES+1, chat completion + image
- `generate_with_tools()` (line 399) — MAX_API_RETRIES+1, chat completion + tools
- `_generate_image_openai()` (line 471) — range(3), images API
- `_generate_image_gemini()` (line 553) — range(3), httpx

每个循环都包含相同的: t0计时 → try → latency计算 → _notify_usage → except → _is_retryable → backoff → logger.warning

**方案**：
1. 提取通用重试方法：
```python
async def _retry_call(
    self,
    call_fn: Callable[[], Awaitable[T]],  # 单次尝试
    *,
    max_retries: int = MAX_API_RETRIES,
    call_label: str = "",
    model: str = "",
    fallback_fn: Callable[[Exception, dict], bool] | None = None,
) -> tuple[T, float]:  # (result, latency_ms)
```
2. `generate()` 和 `generate_with_usage()` **合并**：
   - `generate_with_usage()` 是主体实现
   - `generate()` 变为 `return (await self.generate_with_usage(...)).content`
3. 4 个 chat completion 方法共享 kwargs 构建逻辑，提取为 `_build_chat_kwargs(config, messages, json_mode)`
4. image 方法的重试同样可用 `_retry_call`

**注意**：
- `max_completion_tokens` → `max_tokens` 的 fallback 逻辑只在 chat 方法中需要
- image 方法用 range(3) 而非 MAX_API_RETRIES+1（保持现有行为，传 `max_retries=2`）
- `_notify_usage()` 在 `_retry_call` 成功后统一调用

**验证**：`TestGenerateWithUsage`, `TestGenerateUsageCallback`, `TestImageGenTracking` + 全量测试通过。

**关键文件**：
- `nanoresearch/pipeline/multi_model.py` (593行)

---

### Task 3 (P1): 工具抽象增强

**现状**：已有 `nanoresearch/agents/tools.py` (100行)，包含 `ToolDefinition` dataclass + `ToolRegistry` 类。
已在用：ideation、experiment、review、writing 都通过 `ToolRegistry` 注册工具。
`base.py:373` 的 `run_with_tools()` 接收 `ToolRegistry` 并调用 `to_openai_tools()` + `registry.call()`。

**问题**：
- 参数类型转换缺失：LLM 返回 JSON 参数经 `json.loads()` 后，string/int/float/bool 通常 OK，但有时 LLM 返回 `"123"` 而非 `123`，handler 期望 int 会类型不匹配
- 无 `@tool` 装饰器快捷注册方式，每次注册都要手写 JSON Schema
- `ToolDefinition` 的 `handler` 类型是 `Callable[..., Awaitable[Any]]`，不支持同步 handler

**方案**（在现有 `tools.py` 基础上增强，不破坏现有接口）：
1. `ToolRegistry.call()` 加参数类型转换：
```python
def _coerce_params(self, arguments: dict, schema_props: dict) -> dict:
    """根据 JSON Schema type 字段做类型转换"""
    for key, val in arguments.items():
        if key in schema_props:
            expected = schema_props[key].get("type")
            if expected == "integer" and isinstance(val, str):
                arguments[key] = int(val)
            elif expected == "number" and isinstance(val, str):
                arguments[key] = float(val)
            elif expected == "boolean" and isinstance(val, str):
                arguments[key] = val.lower() in ("true", "1")
    return arguments
```
2. 添加 `@tool_def` 装饰器（可选使用）：
```python
def tool_def(name: str, description: str):
    """装饰器，从函数签名 + type hints 自动生成 ToolDefinition"""
    def decorator(fn):
        params = _build_schema_from_hints(fn)
        fn._tool_definition = ToolDefinition(name=name, description=description, parameters=params, handler=fn)
        return fn
    return decorator
```
3. `ToolRegistry.register_decorated(obj)` — 扫描对象上带 `_tool_definition` 的方法并批量注册
4. handler 支持同步函数：`call()` 中检查 `asyncio.iscoroutinefunction(handler)`，如果不是则 `await loop.run_in_executor(None, handler, **args)`

**验证**：现有 ReAct 模式测试必须通过。

**关键文件**：
- `nanoresearch/agents/tools.py` (100行)
- `nanoresearch/agents/experiment_tools.py` (使用示例)
- `nanoresearch/agents/base.py:370-520` (run_with_tools)

---

### Task 4 (P2): 配置系统升级

**现状**：`nanoresearch/config.py`(284行) 用 Pydantic `BaseModel`：
- `StageModelConfig(BaseModel)` — 单阶段模型配置
- `ResearchConfig(BaseModel)` — 全局配置，从 JSON 文件加载
- 无环境变量支持，敏感信息直接写在 config.json

**方案**：
1. `ResearchConfig` 改为继承 `pydantic_settings.BaseSettings`
2. 添加 `model_config = SettingsConfigDict(env_prefix="NANORESEARCH_", env_file=".env")`
3. 关键字段支持 env var 覆盖：`api_key`, `base_url`, `s2_api_key`, `openalex_api_key`
4. JSON 文件加载仍然支持（自定义 settings source，优先级：env var > .env > config.json > default）
5. `pyproject.toml` 加 `pydantic-settings` 依赖
6. `StageModelConfig` 保持 `BaseModel` 不变

**注意**：现有 `config.json` 格式不变，完全向后兼容。

**关键文件**：
- `nanoresearch/config.py` (284行)
- `pyproject.toml`

---

### Task 5 (P2): 异常处理精细化

**问题**：15 处 `except Exception` 过于宽泛。

**方案按位置分类**：

| 位置 | 当前 | 改为 | 理由 |
|------|------|------|------|
| multi_model.py retry 循环 (×6) | `except Exception as exc` | **保留** | API catch-all 合理 |
| multi_model.py `_notify_usage` | `except Exception: pass` | `except Exception as exc: logger.debug(...)` | 至少留日志 |
| multi_model.py `close()` | `except Exception as exc` | **保留** | 清理代码 catch-all 合理 |
| progress.py `_emit()` | `except Exception` | `except (OSError, ValueError, TypeError)` | JSON序列化+文件IO |
| logging_config.py formatter | `except Exception` | `except (ValueError, TypeError)` | JSON格式化 |
| orchestrator `_report_progress` | `except Exception as exc` | `except (TypeError, ValueError, AttributeError)` | callback 调用 |
| orchestrator `run()` 外层 | `except Exception` | **保留** | 需要 catch-all 触发 pipeline_complete(False) |
| deep_orchestrator `run()` 外层 | `except Exception` | **保留** | 同上 |
| orchestrator `_run_stage_with_retry` | `except Exception as e` | **保留** | 阶段级重试 catch-all |

**原则**：
- 所有 `pass` → 至少 `logger.debug`
- 内部辅助函数 → 精确异常类型
- 面向外部 API / 阶段重试 → 保留 catch-all

---

### Task 6 (P2): 记忆系统（跨 session）

**现状**：每次 pipeline run 独立，无法利用之前经验。

**方案**（轻量级，先做框架 + 搜索缓存）：
1. 新建 `nanoresearch/pipeline/memory.py`
2. `SessionMemory` 类：
   - 存储：`~/.nanobot/memory/` 目录
   - `search_cache.json`：`{normalized_query → {paper_ids: [...], timestamp: ...}}`
   - `latex_fixes.json`：`{error_pattern → fix_pattern}` 积累修复经验
   - `env_cache.json`：`{conda_env → {packages: [...], last_used: ...}}`
3. 接口：
```python
class SessionMemory:
    def __init__(self, memory_dir: Path = None):
        self._dir = memory_dir or Path.home() / ".nanobot" / "memory"
    def get_search_cache(self, query: str) -> list[str] | None: ...
    def put_search_cache(self, query: str, paper_ids: list[str]) -> None: ...
    def get_latex_fix(self, error: str) -> str | None: ...
    def put_latex_fix(self, error: str, fix: str) -> None: ...
```
4. 集成点：`IdeationAgent._search_papers()` 先查缓存，miss 才调 API
5. TTL：搜索缓存 7 天过期，LaTeX 修复永久

**注意**：纯增量功能，不改现有逻辑，`SessionMemory` 默认 enabled，可通过 config 关闭。

---

## 执行顺序

1. **Task 2** (ModelDispatcher retry 去重) — 最独立，改一个文件，风险最低
2. **Task 1** (Orchestrator 去重) — 改两个文件合并为三个
3. **Task 5** (异常处理) — 改动小，可穿插做
4. **Task 3** (工具抽象) — 增强现有代码
5. **Task 4** (配置系统) — 需加依赖
6. **Task 6** (记忆系统) — 纯新增

## 硬性约束

- **测试基线 381 passed 不能减少**，可以增加新测试
- 每完成一个 Task 跑一次 `python -m pytest tests/ -q`
- 不改变任何公开 API 行为（`generate()` 仍返回 `str`，各 agent 的 `run()` 签名不变）
- 回复用中文
- 完成全部后：`git add -A && git commit -m "V2.0-AP: ..."` + `git tag V2.0-AP`
- 用 `git archive` 打包：`git archive --format=zip --prefix=NanoResearch-main/ -o "C:/Users/17965/Desktop/NaNo/V2.0-AP.zip" V2.0-AP`

## 参考：现有项目结构

```
nanoresearch/
├── agents/
│   ├── base.py (526行) — BaseResearchAgent ABC, run_with_tools()
│   ├── tools.py (100行) — ToolDefinition + ToolRegistry
│   ├── experiment_tools.py — build_experiment_tools() → ToolRegistry
│   ├── ideation.py, planning.py, review.py, figure_gen.py, coding.py
│   ├── experiment/ (6 files, mixin pattern)
│   ├── writing/ (7 files, mixin pattern)
│   └── execution/ (5 files, mixin pattern)
├── pipeline/
│   ├── multi_model.py (593行) — ModelDispatcher
│   ├── orchestrator.py (426行) — PipelineOrchestrator (6-stage)
│   ├── deep_orchestrator.py (466行) — DeepPipelineOrchestrator (9-stage)
│   ├── cost_tracker.py (92行) — LLMResult, StageCost, CostTracker
│   ├── progress.py — ProgressEmitter
│   ├── blueprint_validator.py — validate_blueprint()
│   ├── state.py — PipelineStateMachine
│   └── workspace.py — Workspace
├── config.py (284行) — StageModelConfig(BaseModel), ResearchConfig(BaseModel)
├── logging_config.py — JSONFormatter, setup_logging()
└── schemas/ — Pydantic models for pipeline data
```
