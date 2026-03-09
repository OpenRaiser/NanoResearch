# 16. P3: Paper Quality Benchmark

Add quantitative quality metrics that don't require LLM self-evaluation:

```python
"""nanoresearch/evaluation/paper_metrics.py"""
import re


def compute_paper_metrics(paper_tex: str, bibtex: str = "") -> dict:
    """Compute quantitative paper quality metrics without LLM."""
    text = paper_tex

    # Structure metrics
    sections = len(re.findall(r'\\section\b', text))
    subsections = len(re.findall(r'\\subsection\b', text))
    figures = len(re.findall(r'\\begin\{figure', text))
    tables = len(re.findall(r'\\begin\{table', text))
    equations = len(re.findall(r'\\begin\{equation', text))
    equations += len(re.findall(r'\$\$', text)) // 2  # Display math

    # Citation metrics
    cite_keys = set()
    for match in re.findall(r'\\cite[tp]?\{([^}]+)\}', text):
        for key in match.split(","):
            cite_keys.add(key.strip())

    # Word count (approximate, strips LaTeX commands)
    plain_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    plain_text = re.sub(r'[\\${}%&_^~]', '', plain_text)
    word_count = len(plain_text.split())

    # Writing quality indicators
    sentences = re.split(r'[.!?]\s+', plain_text)
    avg_sentence_len = (sum(len(s.split()) for s in sentences) / max(len(sentences), 1))

    # Hedge words (lower is better for scientific writing)
    hedge_words = ["may", "might", "could", "perhaps", "possibly", "arguably",
                   "it seems", "appears to", "we believe"]
    hedge_count = sum(plain_text.lower().count(w) for w in hedge_words)
    hedge_ratio = hedge_count / max(word_count, 1) * 1000  # per 1000 words

    # Passive voice detection (rough)
    passive_pattern = r'\b(is|are|was|were|been|being)\s+\w+ed\b'
    passive_count = len(re.findall(passive_pattern, plain_text, re.I))
    passive_ratio = passive_count / max(len(sentences), 1) * 100  # percentage

    return {
        "structure": {
            "sections": sections,
            "subsections": subsections,
            "figures": figures,
            "tables": tables,
            "equations": equations,
            "word_count": word_count,
        },
        "citations": {
            "unique_cite_keys": len(cite_keys),
            "citations_per_1000_words": round(len(cite_keys) / max(word_count, 1) * 1000, 1),
        },
        "writing_quality": {
            "avg_sentence_length": round(avg_sentence_len, 1),
            "hedge_words_per_1000": round(hedge_ratio, 1),
            "passive_voice_pct": round(passive_ratio, 1),
        },
        "completeness": {
            "has_abstract": "\\begin{abstract}" in text,
            "has_introduction": bool(re.search(r'\\section\{Introduction\}', text, re.I)),
            "has_conclusion": bool(re.search(r'\\section\{Conclusion', text, re.I)),
            "has_references": "\\bibliography" in text or "\\begin{thebibliography}" in text,
            "has_appendix": "\\appendix" in text or "\\section{Appendix" in text,
        },
    }
```

---

# 17. P3: Blueprint Semantic Validation

Add validation beyond Pydantic structural checks:

```python
# In planning.py, after Pydantic validation:

def validate_blueprint_semantics(bp, config) -> list[str]:
    """Semantic consistency checks for ExperimentBlueprint."""
    issues = []

    # 1. Every baseline should have at least one expected_performance entry
    for b in bp.baselines:
        if not b.expected_performance:
            issues.append(f"Baseline '{b.name}' has no expected_performance values")

    # 2. Metric direction consistency
    for m in bp.metrics:
        name_lower = m.name.lower()
        if any(p in name_lower for p in LOWER_IS_BETTER_PATTERNS):
            if m.higher_is_better:
                issues.append(
                    f"Metric '{m.name}' looks like lower-is-better "
                    f"but higher_is_better=True")

    # 3. Dataset source URLs should be valid-looking
    for d in bp.datasets:
        if d.source_url and not d.source_url.startswith(("http://", "https://")):
            issues.append(f"Dataset '{d.name}' has suspicious source_url: {d.source_url}")

    # 4. Proposed method must have key_components
    pm = bp.proposed_method
    if isinstance(pm, dict) and not pm.get("key_components"):
        issues.append("Proposed method missing 'key_components' list")

    # 5. Compute requirements vs cluster config
    if bp.compute_requirements and hasattr(config, 'cluster'):
        req_gpus = bp.compute_requirements.num_gpus or 0
        max_gpus = config.cluster.get("max_gpus", 8) if isinstance(config.cluster, dict) else 8
        if req_gpus > max_gpus:
            issues.append(
                f"Blueprint requires {req_gpus} GPUs but cluster max is {max_gpus}")

    # 6. At least one primary metric
    primary_metrics = [m for m in bp.metrics if m.primary]
    if not primary_metrics:
        issues.append("No primary metric defined")

    return issues
```

---

# 18. Bug-Level Fixes (Non-Breaking)

These are point fixes that should be applied immediately with minimal risk.

## 18.1 `args_hash` Stabilization (base.py:427)

**Current** (unstable):
```python
try:
    args_hash = hash(frozenset(args.items()))
except:
    args_hash = hash(str(args))
```

**Fix**:
```python
import json as _json
try:
    args_hash = hash(_json.dumps(args, sort_keys=True, default=str))
except (TypeError, ValueError):
    args_hash = hash(str(sorted(args.items())) if isinstance(args, dict) else str(args))
```

## 18.2 LaTeX Command Detection (base.py:94)

**Current** (heuristic):
```python
if char.isalpha():  # likely LaTeX command
```

**Fix**: Use known command set:
```python
_LATEX_CMD_PREFIXES = frozenset([
    "cite", "textbf", "textit", "frac", "ref", "label", "sqrt", "sum",
    "int", "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
    "sigma", "omega", "text", "math", "begin", "end", "item", "section",
    "subsection", "paragraph", "emph", "url", "href", "footnote",
    "caption", "includegraphics", "usepackage", "newcommand",
])

# In _fix_json_escapes:
cmd_match = re.match(r'([a-zA-Z]+)', text[pos + 1:])
if cmd_match and cmd_match.group(1) in _LATEX_CMD_PREFIXES:
    # Double-escape this backslash
```

## 18.3 Analysis Figure Cap Logging (analysis.py:287)

**Current** (silent cap):
```python
figure_specs[:3]
```

**Fix**:
```python
max_figs = 5  # or import from constants
if len(figure_specs) > max_figs:
    self.log(f"Capping analysis figures from {len(figure_specs)} to {max_figs}")
figure_specs = figure_specs[:max_figs]
```

## 18.4 LaTeX Fix Loop Infinite Detection (review.py:1494-1527)

**Add** before the fix loop:
```python
seen_error_sigs = set()
# Inside loop:
import hashlib
sig = hashlib.md5(error_log[-500:].encode()).hexdigest()[:8]
if sig in seen_error_sigs:
    self.log("LaTeX fix loop: same error repeated, stopping")
    break
seen_error_sigs.add(sig)
```

## 18.5 Review Section Truncation (review.py:856-858)

**Current**: Keeps first 12K + last 8K chars (may drop Method section).

**Fix**: Use section-boundary-aware truncation:
```python
def _smart_truncate(self, text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    # Find section boundaries
    section_starts = [(m.start(), m.group(1))
                      for m in re.finditer(r'\\section\{([^}]+)\}', text)]
    if not section_starts:
        # No sections found, fall back to head/tail
        return text[:12000] + "\n\n[...truncated...]\n\n" + text[-8000:]

    # Keep sections by priority
    priority = ["abstract", "introduction", "method", "experiment",
                "result", "conclusion"]
    kept_sections = []
    remaining = max_chars

    for pname in priority:
        for i, (start, title) in enumerate(section_starts):
            if pname in title.lower():
                end = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(text)
                content = text[start:end]
                if len(content) <= remaining:
                    kept_sections.append(content)
                    remaining -= len(content)
                elif remaining > 500:
                    kept_sections.append(content[:remaining] + "\n[...truncated...]")
                    remaining = 0
                break

    return "\n\n".join(kept_sections) if kept_sections else text[:max_chars]
```

## 18.6 Writing Lower-is-Better Detection (writing.py:1731-1759)

**Current**: Only checks "loss", "error", "perplexity".

**Fix**: Use centralized pattern set:
```python
from nanoresearch.constants import LOWER_IS_BETTER_PATTERNS

def _is_lower_better(metric_name: str) -> bool:
    name = metric_name.lower().replace(" ", "_").replace("-", "_")
    return any(p in name for p in LOWER_IS_BETTER_PATTERNS)
```

## 18.7 Cache Structure Validation (ideation.py:105-122)

**Wrap** cache loading in try-except:
```python
try:
    cached = json.loads(cache_path.read_text("utf-8"))
    if not isinstance(cached, dict) or "papers" not in cached:
        raise ValueError("invalid cache structure")
except (json.JSONDecodeError, ValueError, OSError) as e:
    self.log(f"Search cache invalid ({e}), starting fresh")
    cached = None
```

## 18.8 `generate_with_tools` Terminal Guard (base.py:481-484)

**Add** after the final summary call:
```python
if hasattr(response, 'tool_calls') and response.tool_calls:
    # LLM returned tool_calls in summary round — force text extraction
    return response.content or "Agent completed but produced no text summary."
```

---

# 20. ERRATA: Bugs Fixed in This Document (Self-Review)

During re-review, the following bugs were found and fixed in this document:

| Location | Bug | Fix |
|----------|-----|-----|
| Section 2.1 `_approx_two_tailed_p` | Lookup table used df=1 critical values for all df <= 30 — would underestimate p-values and falsely claim significance | Fixed to use df=10 critical values (conservative) |
| Section 2.4 `comparison_matrix_to_latex` | Metric names with underscores (e.g. `F1_score`) would break LaTeX compilation | Added `_latex_escape_header()` to escape `_`, `%`, `&` |
| Section 3.2 | `_merge_section_reviews()` called but never defined | Added full implementation |
| Section 4 Citation checker | No rate limiting — paper with 50 citations would make 50 LLM calls | Added `MAX_CHECKS = 15` cap and `checked_keys` dedup |
| Section 5.1 `CrossRunMemory` | Missing `import json, math, time, Path` | Added imports |
| Section 5.1 `SmartContextEngine.after_stage` | Uses `MemoryEntry` class without import note | Added docstring note about required import |
| Section 7 `_ensure_packages` | Could add duplicate packages if loaded by style file | Fixed to check full preamble, not just `\usepackage{}` pattern |
