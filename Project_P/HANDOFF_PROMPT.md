# Project_P 交接提示词 — 论文排版修复 Agent

## 你的任务

你要继续开发 **Project_P**，一个独立的论文排版修复工具。它的输入是 NanoResearch 自动科研管道生成的论文目录（含 `paper.tex`、`references.bib`、`figures/`），输出是修复后的 LaTeX + 编译好的 PDF。

---

## 项目路径

```
C:\Users\17965\Desktop\Project_P\
├── run.py                          # 入口：python run.py path/to/paper/ [--no-llm] [--no-compile] [--dry-run] [--restore]
├── config.json                     # LLM + tectonic 配置
├── diag.py                         # 调试脚本（可删）
├── test_data/                      # ★ 测试基准文件（不要修改！用于评估修复效果）
│   ├── paper_original.tex          # 原始有问题的 tex
│   ├── paper_fixed_v1.tex          # 当前 v1 修复结果（供对比）
│   ├── references_original.bib     # 原始 bib
│   ├── figures_original/           # 原始图片文件（png + pdf）
│   └── neurips_2025.sty            # 样式文件
├── test_paper/                     # 工作目录（每次测试先从 test_data 复制过来）
└── project_p/
    ├── __init__.py
    ├── config.py                   # 配置加载
    ├── compiler.py                 # tectonic 编译 + LLM 错误修复循环
    ├── pipeline.py                 # 主流程编排
    ├── llm_client.py               # OpenAI-compatible LLM 客户端（含 vision）
    ├── _helpers.py                 # 共享工具：escape_latex_text、花括号匹配、section 查找
    ├── fixers/
    │   ├── __init__.py             # run_all_fixes() 编排
    │   ├── latex_sanitizer.py      # Unicode、%转义、LLM artifact 清理
    │   ├── environments.py         # abstract 双层、\begin/\end 不匹配
    │   ├── floats.py               # [H]/[b!]→[t!]、表格 \small/\tabcolsep/@{}
    │   ├── figures.py              # ★核心：图片放置、去重、空块清理、height cap
    │   ├── structure.py            # \end{document}、contribution limit、空行
    │   ├── bibtex.py               # BibTeX 花括号分割、去重、转义
    │   └── figure_trim.py          # 图片空白裁剪（PIL + LLM vision）
    └── validators/
        ├── __init__.py             # run_all_checks() 编排
        └── crossref.py            # \ref↔\label、\cite↔bib、\includegraphics 路径修复
```

## NanoResearch 参考代码（可提取方法）

原始自动科研管道在：
```
C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main\nanoresearch\
```

关键参考文件：
| 功能 | 文件路径 | 说明 |
|------|---------|------|
| LaTeX 汇编 + 图片放置 | `agents/writing/latex_assembler.py` | `_sanitize_latex`（11步）、`_smart_place_figure`、`_relocate_intro_figures`、`_enforce_figure_height_cap` 等 |
| 环境修复 | `agents/review.py` | `_fix_mismatched_environments`、`_apply_revisions`（figure 保留逻辑） |
| 文本转义 | `agents/writing/__init__.py` | `_escape_latex_text`（180-292 行） |
| 图片裁剪 | `agents/figure_gen.py` | `_smart_trim_figure`（1326-1530 行）、LLM vision 裁剪 |
| LaTeX 错误修复 | `latex/fixer.py` | `deterministic_fix`、`build_search_replace_prompt`、`apply_edits` |
| BibTeX 处理 | `agents/writing/latex_assembler.py` | `_split_bibtex_entries`、`_sanitize_bibtex` |

**建议**：直接去读这些文件，提取其中被验证过的算法（260+ fixes 积累的经验），不要从零写。

---

## 当前 v1 已完成的修复

| 修复项 | 状态 |
|--------|------|
| 图片错位（results图在Method、ablation在Introduction） | ✅ smart_place + relocate_intro |
| 图片遗漏（有\ref无figure block） | ✅ _inject_missing_figures |
| 图片重复（同一文件多次\includegraphics） | ✅ _dedup_figures |
| 空 figure 块（\includegraphics被注释/缺失） | ✅ remove_empty_figure_blocks |
| 图片占整页（[b!]独占一页） | ✅ [b!]→[t!] 规范化 |
| 连续figure堆叠 | ✅ _spread_consecutive_figures + \FloatBarrier |
| \includegraphics 高度无上限 | ✅ height=0.32\textheight cap |
| \ref↔\label 不匹配 | ✅ 模糊匹配自动修正 |
| \cite↔bib key 不匹配 | ✅ 模糊匹配 |
| \includegraphics 引用不存在文件 | ✅ 注释掉 + 移除空块 |
| BibTeX 截断/重复/转义 | ✅ 花括号深度分割、去重、&/#/% 转义 |
| abstract 双层嵌套 | ✅ 折叠 |
| \begin/\end 不匹配 | ✅ 栈修复 |
| Unicode 字符 | ✅ 替换为 LaTeX 命令 |
| LLM artifact 泄露 | ✅ 正则清理 |
| \end{document} 位置错误 | ✅ 重排 |
| 表格 \small/\tabcolsep 缺失 | ✅ 自动注入 |

**测试基线**：原始论文 15 页 → 修复后 13 页，PDF 编译成功，0 重复图，0 空块，0 (?) 引用。

---

## 需要你改进的方向

### 1. ★ 核心架构改进：Edit 模式（像改代码一样精准修改 tex）

当前的修复方式是「正则全文扫描 + 字符串替换」，这种方法在处理图片位置时不够精准。**建议改为 Edit 模式**：

**思路**：把 LaTeX 文件看作结构化文档，先 parse 出文档骨架（各 section 的起止位置），然后像编辑代码一样，精准地在目标位置插入/移动 figure 块。

```python
# 伪代码示例
def place_figure_edit_mode(tex, figure_block, target_section="Experiments"):
    """像 IDE 编辑代码一样，精准定位 section 标题位置，在其后插入 figure"""
    # Step 1: 找到目标 section 的标题行位置
    section_pos = find_section_heading(tex, target_section)  # 返回行号

    # Step 2: 找到该 section 内第一个 \ref{fig:label} 的段落末尾
    ref_pos = find_first_ref_in_section(tex, section_pos, figure_label)

    # Step 3: 精准插入（类似 Edit tool 的 old_string → new_string）
    paragraph_end = find_paragraph_end(tex, ref_pos)
    tex = tex[:paragraph_end] + "\n\n" + figure_block + "\n" + tex[paragraph_end:]
    return tex
```

**好处**：
- 不依赖正则全文扫描，避免误匹配
- 可以精确控制插入位置（section 标题后、段落后、表格后等）
- 可以实现 "check Method 的标题位置 → 直接放在后面" 这样的直觉操作
- 更容易调试：修改有明确的行号锚点

### 2. ★ 表格溢出问题（表格超出页面范围）

当前只做了 `\small` + `\tabcolsep{4pt}` + `@{}`，但实际遇到的表格溢出问题更复杂：

**需要处理的情况**：
- 列太多导致表格超宽 → 需要 `\resizebox{\textwidth}{!}{...}` 包裹
- 行太多导致表格超长 → 需要 `longtable` 环境替换 `tabular`
- 数值精度过高（小数点后8位）→ 自动截断到合理精度
- 表格 caption 太长 → 考虑使用 `\caption[short]{long}` 格式

**参考 NanoResearch 的做法**：
```python
# 从 latex_assembler.py _fix_table_overflow 提取
# 检测方法：统计 tabular 列数，如果 > 6 列自动 resizebox
# 检测方法：统计 \\ 行数，如果 > 30 行考虑 longtable
```

### 3. ★ 更智能的图片放置（LLM 辅助）

当前图片放置完全依赖规则（关键词匹配 label → section）。可以用 LLM 做更智能的决策：

```python
def llm_decide_figure_placement(tex, figure_block, llm_client):
    """让 LLM 读全文，决定 figure 应该放在哪个 section 的哪个段落后面"""
    system = "你是学术论文排版专家。给定论文的各section结构和一个figure，决定这个figure应该放在哪里。"
    user = f"论文结构：{extract_section_outline(tex)}\n\nFigure caption: {extract_caption(figure_block)}\n\n输出 JSON: {{\"section\": \"...\", \"after_paragraph_containing\": \"...\"}}"
    response = llm_client.generate(system, user, json_mode=True)
    return parse_placement(response)
```

### 4. Section 内容检查（\ref{sec:metrics} 无标签问题）

当前只做 \ref↔\label 模糊匹配。需要：
- 检测 `\ref{sec:XXX}` 没有对应 `\label{sec:XXX}` 的情况
- 如果是 `sec:metrics`，自动在 "Metrics" 或 "Diagnostic Metrics" subsection 处添加 `\label{sec:metrics}`

### 5. Caption 质量检查

NanoResearch 生成的 caption 经常有问题：
- "Fig1 Framework Overview" 这种缩写式 caption → 应该是完整描述
- "Placeholder --- Experiment Failed" → 需要检测并标记/替换
- caption 和图片内容不匹配 → LLM vision 验证

### 6. 更多 LaTeX 编译错误的确定性修复

当前 Level 1 确定性修复比较少。从 NanoResearch `latex/fixer.py` 的 `deterministic_fix` 函数可以提取更多：
- `Missing $ inserted` → 检测裸露的 _ ^ 并自动转义或包 $...$
- `Undefined control sequence` → 检测缺失的 \usepackage 并自动添加
- `Too many unprocessed floats` → 在适当位置插入 `\clearpage`
- `Package hyperref Error` → 修复 \href 格式

### 7. PDF 图片裁剪

当前 `auto_trim` 只能处理 PNG/JPG，无法处理 PDF 图片。但论文编译通常优先使用 PDF 格式。需要：
- 用 PyMuPDF (`fitz`) 将 PDF 页面渲染为 PIL Image
- 对渲染后的图片做 auto_trim
- 如果需要裁剪，用 fitz 的 `page.set_cropbox()` 裁剪 PDF

---

## 测试方法

### 准备测试环境
```bash
cd C:\Users\17965\Desktop\Project_P

# 从 test_data 恢复原始文件到 test_paper
mkdir -p test_paper/figures
cp test_data/paper_original.tex test_paper/paper.tex
cp test_data/references_original.bib test_paper/references.bib
cp -r test_data/figures_original/* test_paper/figures/
cp test_data/neurips_2025.sty test_paper/
```

### 运行修复
```bash
python run.py test_paper --no-llm          # 不使用 LLM，但编译 PDF
python run.py test_paper --no-llm --dry-run # 只报告不修改
python run.py test_paper --restore          # 还原到原始状态
```

### 评估标准

对 `test_data/paper_original.tex`（原始15页论文）运行你的 agent 后，检查：

1. **图片位置正确**（最重要！）
   - Framework Overview 图在 Introduction 或 Method（不在 Experiments）
   - Results/Ablation 图在 Experiments（不在 Introduction）
   - 没有图片在 References 后面
   - 没有重复图片（同一文件只出现一次）

2. **无编译错误** — PDF 成功生成

3. **引用无 (?)** — 所有 \ref 和 \cite 都有对应目标

4. **表格不溢出** — 所有表格在页面宽度内

5. **无空白浮动体** — 没有只有 caption 没有图的 figure 块

6. **页数合理** — 不应比修复前多（v1 从 15→13 页）

7. **BibTeX 完整** — 无截断条目、无重复 key

### 与 v1 对比
```bash
# 比较你的修复结果和 v1 的差异
diff test_data/paper_fixed_v1.tex test_paper/paper.tex
```

---

## 环境信息

- **Python**: D:/anaconda/python.exe
- **LaTeX 编译器**: `D:/anaconda/Scripts/tectonic`（tectonic，无需 TexLive）
- **依赖**: `Pillow`, `numpy`, `openai`（LLM 可选）, `PyMuPDF`（`fitz`，已安装）
- **平台**: Windows 11, bash shell
- **LLM 配置**: `config.json` 的 `llm.api_key` 为空时自动禁用 LLM 功能，所有修复回退到确定性规则

## 关键设计原则

1. **确定性优先** — 能用正则/规则解决的不用 LLM
2. **备份安全** — 每次运行前自动备份到 `.bak/`，`--restore` 可还原
3. **最小侵入** — 只修有问题的地方，不重写整个 tex
4. **从 NanoResearch 提取** — 那个项目有 260+ fixes 的积累，很多算法已经被验证过，直接提取复用
5. **Edit 模式思维** — 把 tex 当作代码，用行号定位 + 精准替换，而不是正则全文扫描替换
