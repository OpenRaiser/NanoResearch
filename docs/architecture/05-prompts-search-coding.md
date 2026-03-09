# 8. P1: Prompt Template Externalization

## Problem

~1000+ lines of prompt strings are hardcoded in Python files (figure_gen.py:39-201,
341-730; writing.py section prompts; review.py review prompts). This makes prompt
iteration slow (requires code changes) and prevents A/B testing.

## Solution

Create `nanoresearch/prompts/` directory with YAML templates:

```
prompts/
├── __init__.py              # PromptLoader class
├── figure_gen/
│   ├── planning.yaml        # Figure planning prompt
│   ├── chart_types/
│   │   ├── grouped_bar.yaml
│   │   ├── line_plot.yaml
│   │   ├── heatmap.yaml
│   │   ├── radar.yaml
│   │   └── ...
│   └── ai_templates/
│       ├── system_overview.yaml
│       ├── transformer_arch.yaml
│       └── ...
├── writing/
│   ├── title.yaml
│   ├── abstract.yaml
│   ├── introduction.yaml
│   ├── related_work.yaml
│   ├── method.yaml
│   ├── experiments.yaml
│   └── conclusion.yaml
├── review/
│   ├── section_review.yaml
│   ├── revision.yaml
│   └── consistency_check.yaml
└── ideation/
    ├── query_generation.yaml
    ├── gap_analysis.yaml
    └── hypothesis_selection.yaml
```

## Prompt Loader

```python
"""nanoresearch/prompts/__init__.py"""
import yaml
from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).parent
_CACHE: dict[str, dict] = {}


def load_prompt(category: str, name: str, variables: dict = None) -> str:
    """Load a prompt template and optionally fill variables.

    Args:
        category: Subdirectory (e.g., "writing", "figure_gen/chart_types")
        name: File name without .yaml extension
        variables: Dict of {placeholder: value} for string formatting

    Returns:
        Rendered prompt string.

    NOTE on variable substitution: All replacements are applied in a single
    pass to avoid double-expansion (where variable A's value contains {B}).
    """
    cache_key = f"{category}/{name}"
    if cache_key not in _CACHE:
        path = _PROMPTS_DIR / category / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[cache_key] = yaml.safe_load(f)

    template = _CACHE[cache_key]
    prompt_text = template.get("prompt", template.get("system_prompt", ""))

    if variables:
        # FIX: Single-pass replacement using a regex to avoid double-expansion.
        # If variable A's value contains "{B}", it should NOT be substituted.
        import re
        pattern = re.compile(r'\{(' + '|'.join(re.escape(k) for k in variables) + r')\}')
        prompt_text = pattern.sub(lambda m: str(variables[m.group(1)]), prompt_text)

    return prompt_text


def get_prompt_version(category: str, name: str) -> Optional[str]:
    """Get version string of a prompt template."""
    load_prompt(category, name)  # Ensure cached
    return _CACHE.get(f"{category}/{name}", {}).get("version")
```

## Example YAML Template

```yaml
# prompts/writing/method.yaml
name: method_section
version: "2.3"
description: "System prompt for generating the Method section"
system_prompt: |
  You are an expert ML researcher writing the Method section of a top-venue paper.

  STRUCTURE:
  - 5-7 paragraphs with subsections for each key component
  - Begin with problem formulation and notation
  - Each subsection: intuition → formalization → implementation detail
  - Include at least 2 equations (numbered, with explanation)
  - End with complexity analysis (time + space)

  STYLE:
  - Present tense for method description
  - Active voice ("We propose..." not "It is proposed...")
  - No hedging words ("may", "might", "could", "perhaps")
  - Define all notation on first use

  CONTRIBUTION CONTRACT:
  {contribution_guidance}

  CONTEXT:
  {method_context}
variables:
  - contribution_guidance
  - method_context
```

## Migration Approach

1. Create the YAML files by extracting existing inline prompts (copy-paste, no edits)
2. Replace inline strings with `load_prompt()` calls one at a time
3. Verify output quality is identical before moving to the next prompt
4. DO NOT modify prompt content during migration — only the delivery mechanism changes

---

# 9. P1: IDEATION ReAct Search Loop

## Problem

IDEATION runs a fixed linear search: generate queries → search → rank → expand → done.
If the search misses a key research direction, there's no way to detect or fix it.

## Solution

Add a search quality self-evaluation step with conditional re-search:

```python
# In ideation.py, after initial search + ranking:

async def _evaluate_search_coverage(self, topic: str, papers: list[dict],
                                     gaps: list[dict]) -> dict:
    """Evaluate whether the search covers all major directions of the topic."""
    result = await self.generate_json(
        system=(
            "You are a research librarian evaluating search completeness. "
            "Given a topic and found papers, assess coverage.\n"
            "Return JSON: {\n"
            "  \"coverage_score\": 1-10,\n"
            "  \"missing_directions\": [\"direction1\", ...],\n"
            "  \"suggested_queries\": [\"query1\", ...],\n"
            "  \"well_covered\": [\"area1\", ...]\n"
            "}"
        ),
        user=(
            f"Topic: {topic}\n\n"
            f"Found {len(papers)} papers. Top 20 titles:\n"
            + "\n".join(f"- {p.get('title', '')}" for p in papers[:20])
            + f"\n\nIdentified gaps:\n"
            + "\n".join(f"- {g.get('description', '')}" for g in gaps)
        ),
    )
    return result


async def _supplementary_search(self, missing_directions: list[str],
                                 existing_papers: dict) -> list[dict]:
    """Run targeted searches for missing research directions."""
    new_papers = []
    for direction in missing_directions[:3]:  # Cap at 3 supplementary searches
        results = await self._search_literature_single(direction)
        for paper in results:
            dedup_key = self._dedup_key(paper)
            if dedup_key not in existing_papers:
                existing_papers[dedup_key] = paper
                new_papers.append(paper)
    return new_papers
```

### Integration

In `run()`, after `_rank_and_filter_papers()`:

```python
# Self-evaluation loop (max 2 rounds)
for eval_round in range(2):
    coverage = await self._evaluate_search_coverage(topic, papers, gaps)
    score = coverage.get("coverage_score", 10)
    if score >= 8:
        self.log(f"Search coverage: {score}/10 — sufficient")
        break
    missing = coverage.get("missing_directions", [])
    if not missing:
        break
    self.log(f"Search coverage: {score}/10 — supplementing {len(missing)} directions")
    new_papers = await self._supplementary_search(missing, all_papers_dict)
    papers.extend(new_papers)
    # Re-rank with new papers
    papers = self._rank_and_filter_papers(papers)
```

> **Safety**: Capped at 2 supplementary rounds and 3 queries per round to prevent
> runaway API costs. Each round adds ~15 seconds.

---

# 10. P2: CODING Quality Gates

## 10.1 AST-Based Import Checking

Replace regex-based import checking (coding.py:657-746) with AST parsing:

```python
"""nanoresearch/agents/coding/import_checker.py"""
import ast
import os
from pathlib import Path


class ImportChecker:
    """Check cross-file import consistency using AST parsing."""

    def __init__(self, code_dir: Path):
        self.code_dir = code_dir
        self.module_exports: dict[str, set[str]] = {}  # dotted.module.path → exported names
        self._parse_all_modules()

    def _parse_all_modules(self):
        for py_file in self.code_dir.rglob("*.py"):
            # FIX: Use dotted module path relative to code_dir, not just file stem.
            # This ensures 'from foo.bar import X' matches the correct module.
            rel_path = py_file.relative_to(self.code_dir)
            module_name = str(rel_path).replace(os.sep, ".").removesuffix(".py")
            # Also register under stem for simple 'from bar import X' patterns
            stem = py_file.stem
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
                exports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        exports.add(node.name)
                    elif isinstance(node, ast.AsyncFunctionDef):
                        exports.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        exports.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                exports.add(target.id)
                self.module_exports[module_name] = exports
                # Also register by stem (for flat imports)
                if stem not in self.module_exports:
                    self.module_exports[stem] = exports
            except SyntaxError:
                pass  # Skip unparseable files

    def check_imports(self) -> list[dict]:
        """Check all files for import mismatches."""
        issues = []
        for py_file in self.code_dir.rglob("*.py"):
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
                    if node.module and node.module in self.module_exports:
                        for alias in (node.names or []):
                            name = alias.name
                            if name == "*":
                                continue  # Skip star imports
                            if name not in self.module_exports[node.module]:
                                issues.append({
                                    "file": str(py_file.relative_to(self.code_dir)),
                                    "type": "missing_export",
                                    "message": (
                                        f"'{name}' imported from '{node.module}' "
                                        f"but not defined there"
                                    ),
                                    "line": node.lineno,
                                })
        return issues
```

## 10.2 Auto Smoke Test Generation

After code generation, automatically create `test_smoke.py`:

```python
# In coding.py, after file generation:

async def _generate_smoke_test(self, code_dir: Path, file_list: list[str]) -> str:
    """Generate a minimal smoke test that verifies basic functionality."""
    imports_to_test = []
    for f in file_list:
        if f.endswith(".py") and not f.startswith("test_"):
            module = f.replace("/", ".").replace("\\", ".").removesuffix(".py")
            imports_to_test.append(module)

    test_code = '''"""Auto-generated smoke test. Verifies imports and basic shapes."""
import sys
import importlib

def test_all_imports():
    """Verify all generated modules can be imported."""
    failures = []
    modules = {modules_list}
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures.append(f"{{mod}}: {{e}}")
    if failures:
        print("Import failures:")
        for f in failures:
            print(f"  {{f}}")
        sys.exit(1)
    print(f"All {{len(modules)}} modules imported successfully")

if __name__ == "__main__":
    test_all_imports()
'''.format(modules_list=repr(imports_to_test))

    (code_dir / "test_smoke.py").write_text(test_code, encoding="utf-8")
    return "test_smoke.py"
```

## 10.3 Auto-Formatting

After code generation:

```python
import subprocess, sys

async def _format_generated_code(self, code_dir: Path):
    """Auto-format generated code with black (if available)."""
    try:
        subprocess.run(
            [sys.executable, "-m", "black", "--quiet", "--line-length", "100",
             str(code_dir)],
            capture_output=True, timeout=30
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Non-critical, skip if black not installed
```
