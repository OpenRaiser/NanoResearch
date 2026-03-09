# Phase 8 Prompt — 粘贴到新对话

用中文回复。你正在对 NanoResearch（一个自动化科研论文生成 pipeline）做架构改进。

## 项目路径
`C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`

## 测试基线
343 passed, 1 warning。每次改动后必须运行 `python -m pytest tests/ -x -q`，确认 343 passed。

## 核心约束
- 不改变任何现有公共 API（方法签名、返回值格式）
- 不引入新依赖（ast 是标准库，black 是可选的）
- 完成后做 bug check

---

# 任务 A: IDEATION 搜索覆盖率自评

## 文件
`nanoresearch/agents/ideation.py`（约 1230 行）

## 问题
IDEATION 做固定线性搜索：生成 query → S2/OpenAlex/arXiv 搜索 → 排序 → 引用扩展 → 结束。如果搜索漏掉了某个关键方向，无法检测也无法补救。

## 解决方案
在 `run()` 方法中，**Step 2d 和 Step 2e 之间**（即在 `_enrich_with_full_text` 之后、`_extract_must_cites` 之前）插入搜索覆盖率自评循环。

### 具体插入位置
当前代码结构（line 152-174）：
```python
# Step 2c: Rank and filter papers by citation quality
papers = self._rank_and_filter_papers(papers)  # line 152
# ... citation enrichment ...
papers = self._rank_and_filter_papers(papers)  # line 161 (re-rank after enrichment)
# Step 2c3: Citation graph expansion
papers = await self._expand_via_citations(papers, top_k=5, max_new=15)  # line 164
# Step 2d: Enrich top papers with full-text PDF reading
papers = await self._enrich_with_full_text(papers)  # line 168

# ★★★ 在这里插入搜索覆盖率自评 ★★★

# Step 2e: Extract must-cite papers from surveys
must_cites = await self._extract_must_cites(...)  # line 171
```

### 需要添加的两个方法

**方法 1: `_evaluate_search_coverage(self, topic, papers, gaps)`**
- 调用 `self.generate_json(system_prompt, user_prompt)` 让 LLM 评估搜索完整性
- system prompt: "You are a research librarian evaluating search completeness..."
- 返回 JSON: `{"coverage_score": 1-10, "missing_directions": [...], "suggested_queries": [...], "well_covered": [...]}`
- **注意**: 此时还没有 `gaps` 变量（gaps 在后面的 `_analyze_and_hypothesize` 中生成），所以传空列表 `[]` 或者不传 gaps

**方法 2: `_supplementary_search(self, missing_directions, existing_papers_dict)`**
- 对每个 missing direction 调用现有的搜索方法做补充搜索
- 最多搜索 3 个方向（`missing_directions[:3]`）
- 用 `self._dedup_key(paper)` 去重（已有方法，line 251）
- 现有搜索方法参考: `_search_literature(queries)` (line 261)，它接受 query list

### run() 中的集成代码
```python
# ★ Search coverage self-evaluation (max 2 rounds)
all_papers_dict = {self._dedup_key(p): p for p in papers}
for eval_round in range(2):
    coverage = await self._evaluate_search_coverage(topic, papers)
    score = coverage.get("coverage_score", 10)
    if score >= 8:
        self.log(f"Search coverage: {score}/10 — sufficient")
        break
    missing = coverage.get("missing_directions", [])
    if not missing:
        break
    self.log(f"Search coverage: {score}/10 — supplementing {len(missing)} directions")
    new_papers = await self._supplementary_search(missing, all_papers_dict)
    if new_papers:
        papers.extend(new_papers)
        papers = self._rank_and_filter_papers(papers)
        self.log(f"Added {len(new_papers)} papers from supplementary search")
```

### 安全限制
- 最多 2 轮自评
- 每轮最多 3 个补充 query
- coverage_score >= 8 时停止
- 补充搜索复用现有 `_search_literature()` 方法，不新建搜索逻辑

### generate_json 调用模式
参考同文件中其他调用（如 line 241, 807）：
```python
result = await self.generate_json(SYSTEM_PROMPT_STRING, user_prompt_string)
# result 是一个 dict
```
不需要 `stage_override`，用默认的 ideation stage config。

### 提示词外部化
system prompt 应该放到 `nanoresearch/prompts/ideation/search_coverage.yaml`，用 `load_prompt("ideation", "search_coverage")` 加载。参考 `nanoresearch/prompts/__init__.py` 的 `load_prompt` 函数和现有 YAML 文件格式（如 `nanoresearch/prompts/ideation/analysis.yaml`）。

---

# 任务 B: CODING 质量门控

## B1: AST-Based Import Checking

### 要替换的代码
**两处**重复的 regex import 检查需要统一为 AST 版本：

1. `nanoresearch/agents/experiment/iteration.py` → `_check_import_consistency()` (line 568-655)
   - 是 `@staticmethod`，返回 `list[dict]`
   - 被 `__init__.py` line 536 调用: `import_mismatches = self._check_import_consistency(code_dir)`

2. `nanoresearch/agents/coding.py` → `_fix_import_mismatches()` (line 640-812)
   - 前半部分（line 640-740 左右）也是 regex 扫描 import 一致性
   - 后半部分是 LLM 修复

### 解决方案
创建 `nanoresearch/agents/import_checker.py`（注意：不是 coding/ 子目录，agents/ 下面直接放）：
```python
"""AST-based cross-file import consistency checker."""
import ast
from pathlib import Path

class ImportChecker:
    def __init__(self, code_dir: Path):
        self.code_dir = code_dir
        self.module_exports: dict[str, set[str]] = {}
        self._parse_all_modules()

    def _parse_all_modules(self):
        """用 ast.parse 提取每个模块的所有导出名。"""
        for py_file in self.code_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            module_name = py_file.stem
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
                exports = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        exports.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                exports.add(target.id)
                self.module_exports[module_name] = exports
            except SyntaxError:
                pass

    def check_imports(self) -> list[dict]:
        """检查所有文件的 import 一致性，返回 mismatch list。"""
        issues = []
        for py_file in self.code_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
            except SyntaxError:
                issues.append({
                    "file": str(py_file.relative_to(self.code_dir)),
                    "type": "syntax_error",
                    "message": "File has syntax errors",
                })
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    # 处理 from src.X import Y 和 from X import Y
                    mod = node.module or ""
                    # 去掉 src. 前缀
                    if mod.startswith("src."):
                        mod = mod[4:]
                    if mod in self.module_exports:
                        for alias in (node.names or []):
                            name = alias.name
                            if name == "*":
                                continue
                            if name not in self.module_exports[mod]:
                                issues.append({
                                    "importer": py_file.name,
                                    "module": mod,
                                    "missing_name": name,
                                    "available": sorted(self.module_exports[mod]),
                                    "line": node.lineno,
                                })
        return issues
```

### 返回格式兼容性（关键！）
返回的 `list[dict]` 必须兼容现有下游代码：
- `iteration.py:_fix_import_mismatches()` (line 657) 期望每个 dict 有: `importer`, `module`, `missing_name`, `available`
- `coding.py` 的下游 LLM 修复也期望类似格式

### 集成方式
1. `iteration.py` line 568-655 的 `_check_import_consistency` 内部改为调用 `ImportChecker`：
```python
@staticmethod
def _check_import_consistency(code_dir: Path) -> list[dict]:
    from nanoresearch.agents.import_checker import ImportChecker
    checker = ImportChecker(code_dir)
    return checker.check_imports()
```
2. `coding.py` line 640 的 `_fix_import_mismatches` 前半段（扫描部分）也改用 `ImportChecker`，保留后半段的 LLM 修复逻辑

---

## B2: Auto Smoke Test Generation

### 插入位置
`nanoresearch/agents/experiment/__init__.py`，在 Phase 2b（import check，line 536-539）之后、preflight checks（line 611-618）之前。

当前代码流：
```
Phase 2: 生成文件 (line 497-533)
Phase 2b: import consistency check (line 535-539)
  ↓
★★★ 在这里插入 smoke test ★★★
  ↓
Preflight checks (line 611-618)
Phase 3: dry-run (line 653+)
```

### 实现
在 ExperimentAgent 上添加方法 `_generate_smoke_test(self, code_dir, file_list)`：
- 扫描 `file_list` 中所有 `.py` 文件（排除 test_ 开头的）
- 生成 `test_smoke.py` 到 `code_dir`
- 内容：import 每个模块，失败则报错
- 执行 smoke test：`python test_smoke.py`，如果失败就在 log 中记录但不阻塞（warning，不是 blocker）

### 注意
- smoke test 文件生成在 code_dir（即 `workspace/code/`）
- 用 `subprocess.run([venv_python or sys.executable, str(code_dir / "test_smoke.py")])` 执行
- timeout 30 秒
- 失败不阻塞流程（preflight 和 dry-run 才是真正的 gate）

---

## B3: Auto-Formatting（可选，优先级最低）

### 实现
在文件生成后（Phase 2 完成后）尝试用 `black` 格式化：
```python
async def _format_generated_code(self, code_dir: Path):
    try:
        subprocess.run(
            [sys.executable, "-m", "black", "--quiet", "--line-length", "100", str(code_dir)],
            capture_output=True, timeout=30
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # black 不可用时静默跳过
```
- 非关键功能，失败时静默跳过
- 放在 smoke test 之前（先格式化，再测试）

---

# 执行顺序建议

1. **任务 A**: 在 ideation.py 添加 `_evaluate_search_coverage` + `_supplementary_search` + run() 集成 → 跑测试
2. **任务 B1**: 创建 `import_checker.py` → 改 `iteration.py` 和 `coding.py` 使用它 → 跑测试
3. **任务 B2**: 在 `experiment/__init__.py` 添加 smoke test → 跑测试
4. **任务 B3**: 添加 auto-formatting（可选）→ 跑测试
5. **最终 bug check**: 全面审查所有改动

---

# 参考文件索引

| 文件 | 作用 | 关键行号 |
|------|------|---------|
| `nanoresearch/agents/ideation.py` | 任务 A 主文件 | run(): L98, 插入点: L168-171, _dedup_key: L251, _search_literature: L261, generate_json 示例: L241 |
| `nanoresearch/agents/experiment/__init__.py` | 任务 B2/B3 主文件 | Phase 2b: L535-539, Preflight: L611-618, Dry-run: L653 |
| `nanoresearch/agents/experiment/iteration.py` | 任务 B1 替换目标 | _check_import_consistency: L568-655, _fix_import_mismatches: L657 |
| `nanoresearch/agents/coding.py` | 任务 B1 替换目标 | _fix_import_mismatches: L640-812 (前半regex扫描,后半LLM修复) |
| `nanoresearch/prompts/__init__.py` | prompt 加载器 | load_prompt(): L21 |
| `nanoresearch/prompts/ideation/analysis.yaml` | YAML 格式参考 | system_prompt 字段 |
| `ARCHITECTURE_IMPROVEMENTS_V2.md` | 设计文档 | Section 9: L1955, Section 10: L2035 |
