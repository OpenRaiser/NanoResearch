# Phase 8: IDEATION Search Enhancement + CODING Quality Gates

## Background
你正在对 NanoResearch（一个自动化科研论文生成 pipeline）做架构改进。Phase 1-7 已完成，343 测试全部通过。

## 项目路径
`C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`

## 任务 A: IDEATION 搜索覆盖率自评（Section 9）

### 问题
IDEATION 跑固定线性搜索：生成 query → 搜索 → 排序 → 扩展 → 结束。如果搜索漏掉了关键研究方向，无法检测。

### 解决方案
在 `nanoresearch/agents/ideation.py` 的 `run()` 方法中，`_rank_and_filter_papers()` 之后添加：

1. `_evaluate_search_coverage(topic, papers, gaps)` — 让 LLM 评估搜索完整性（返回 coverage_score 1-10, missing_directions, suggested_queries）
2. `_supplementary_search(missing_directions, existing_papers)` — 对遗漏方向做补充搜索
3. 自评循环最多 2 轮，每轮最多 3 个补充 query（防止 API 成本失控）
4. coverage_score >= 8 时停止

### 参考
`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 9 (line 1955) 有完整代码示例。

---

## 任务 B: CODING 质量门控（Section 10）

### B1: AST-Based Import Checking
- 创建 `nanoresearch/agents/coding/import_checker.py`（或直接改 `coding.py`）
- 用 `ast.parse` 替换现有的 regex import 检查（`experiment/iteration.py` 中的 `_check_import_consistency`）
- AST 方法更准确：能正确解析 `from X import Y` 和 `import X; X.func()` 模式

### B2: Auto Smoke Test Generation
- 代码生成后自动创建 `test_smoke.py`
- 内容：尝试 import 所有生成的 .py 模块，验证无 ImportError
- 在 dry-run 之前执行这个 smoke test

### B3: Auto-Formatting（可选）
- 代码生成后用 `black` 格式化（如果可用）
- 非关键功能，失败时静默跳过

---

## 约束
- 测试基线：343 passed
- 用中文回复
- 完成后 check bug
- 参考详细设计：`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 9 (line 1955) 和 Section 10 (line 2035)
