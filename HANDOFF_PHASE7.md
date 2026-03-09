# Phase 7: Prompt Template Externalization

## Background
你正在对 NanoResearch（一个自动化科研论文生成 pipeline）做架构改进。Phase 1-6 已完成（大文件拆分、分析模块重写、review bug 修复等），343 测试全部通过。

## 项目路径
`C:\Users\17965\Desktop\NaNo\NanoResearch-main\NanoResearch-main`

## 任务
把散布在 Python 代码中的 ~1000+ 行 prompt 字符串提取到 `nanoresearch/prompts/` 目录下的 YAML 文件中，用 `load_prompt()` 函数加载。

## 详细步骤

### Step 1: 创建 PromptLoader
创建 `nanoresearch/prompts/__init__.py`：
- `load_prompt(category, name, variables=None) -> str` 函数
- YAML 加载 + 内存缓存
- 变量用 `{placeholder}` 格式，str.replace 替换（不用 eval）

### Step 2: 提取 prompt（按优先级）

**2a. writing section prompts（最高优先级）**
- 来源：`nanoresearch/skill_prompts.py` 中的 `get_writing_system_prompt(heading)` 返回的 system prompt
- 目标：`prompts/writing/introduction.yaml`, `method.yaml`, `experiments.yaml`, `related_work.yaml`, `conclusion.yaml`, `title.yaml`, `abstract.yaml`

**2b. figure_gen prompts**
- 来源：`nanoresearch/agents/figure_gen.py` 中的内联 prompt 字符串（规划 prompt、各图表类型 prompt）
- 目标：`prompts/figure_gen/planning.yaml`, `prompts/figure_gen/chart_types/*.yaml`

**2c. review prompts**
- 来源：`nanoresearch/agents/review.py` 中的 section review / revision prompt
- 目标：`prompts/review/section_review.yaml`, `revision.yaml`, `consistency_check.yaml`

**2d. experiment prompts**
- 来源：`nanoresearch/agents/experiment/__init__.py` 中的 `PROJECT_PLAN_SYSTEM_PROMPT`, `FILE_GEN_SYSTEM_PROMPT`, `_REACT_SYSTEM_PROMPT`
- 目标：`prompts/experiment/project_plan.yaml`, `file_gen.yaml`, `react_system.yaml`

### Step 3: 迁移策略
1. 创建 YAML 文件时**原封不动复制**现有 prompt 内容，不做任何修改
2. 用 `load_prompt()` 调用替换原来的内联字符串
3. 每替换一个 prompt，立即运行 `python -m pytest tests/ -x -q` 确认测试通过
4. **不要**在迁移过程中修改 prompt 内容——只改交付机制

### YAML 格式示例
```yaml
name: method_section
version: "1.0"
description: "System prompt for generating the Method section"
system_prompt: |
  You are an expert ML researcher writing the Method section...
  (原始 prompt 内容)
variables:
  - contribution_guidance
  - method_context
```

## 约束
- 测试基线：343 passed
- 用中文回复
- 完成后 check bug，这是标准流程
- 参考详细设计：`ARCHITECTURE_IMPROVEMENTS_V2.md` Section 8 (line 1820)
