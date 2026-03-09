# Phase 10: DAG Parallel Stage Scheduling

## Background
你正在对 NanoResearch（一个自动化科研论文生成 pipeline）做架构改进。Phase 1-9 已完成，343 测试全部通过。

## 项目路径
`C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`

## 当前问题
Pipeline 的 9 个阶段严格串行执行：
```
IDEATION → PLANNING → SETUP → CODING → EXECUTION → ANALYSIS → FIGURE_GEN → WRITING → REVIEW
```
但有些阶段之间没有依赖关系（如 FIGURE_GEN 和 WRITING 的前半部分可以并行）。

## 解决方案

### Step 1: 定义依赖 DAG
在 orchestrator 中定义阶段依赖图：
```python
STAGE_DEPENDENCIES = {
    "IDEATION": [],
    "PLANNING": ["IDEATION"],
    "SETUP": ["PLANNING"],
    "CODING": ["SETUP", "PLANNING"],
    "EXECUTION": ["CODING"],
    "ANALYSIS": ["EXECUTION"],
    "FIGURE_GEN": ["ANALYSIS"],
    "WRITING": ["ANALYSIS", "FIGURE_GEN"],
    "REVIEW": ["WRITING"],
}
```

### Step 2: 实现 DAG 调度器
在 `nanoresearch/pipeline/orchestrator.py`（或 `unified_orchestrator.py`）中：
1. 新增 `_run_dag()` 方法
2. 循环查找可运行阶段（所有依赖已完成）
3. 单个可运行 → 直接执行
4. 多个可运行 → `asyncio.gather()` 并行
5. 错误处理：任一阶段失败则中止整个 pipeline

### Step 3: 配置项
- 在 `nanoresearch/config.py` 中添加 `parallel_stages: bool = False`
- **默认关闭**，opt-in 开启
- 当 `parallel_stages=False` 时，走现有的串行逻辑（零风险）

### Step 4: 测试
- 测试 DAG 拓扑排序正确性
- 测试并行执行（mock agents）
- 测试单阶段失败时正确中止
- 测试 `parallel_stages=False` 时行为不变

## 关键文件
- `nanoresearch/pipeline/orchestrator.py` — 标准 pipeline orchestrator
- `nanoresearch/pipeline/deep_orchestrator.py` — deep pipeline orchestrator
- `nanoresearch/pipeline/unified_orchestrator.py` — 统一入口
- `nanoresearch/config.py` — 配置

## 并行化收益
当前最慢的两个阶段是 WRITING（~10min）和 FIGURE_GEN（~5min）。
如果 FIGURE_GEN 能与 WRITING 的 Introduction/Related Work/Method 并行，总耗时可减少 ~30%。

## 约束
- 测试基线：343 passed
- 用中文回复
- 完成后 check bug
- **安全第一**：默认关闭并行，用户必须显式启用
- 参考详细设计：`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 13 (line 2315)
