<div align="center">
<img src="https://github.com/user-attachments/assets/b9a00889-8edf-4a76-a979-751cf1bcc89b" alt="Nano Research Logo" width="500"/>
</div>

# NanoResearch

AI-powered research engine that automates the full academic paper pipeline: from a topic to a compiled PDF with literature review, experiment code, figures, and LaTeX paper.

## Changelog

<details>
<summary><b>v2.2</b> (2026-03-12) — SLURM Auto-Detection & Model Download Fix</summary>

- **GPU auto-fallback**: `local_quick` profile now auto-detects GPU availability; if no local GPU but `sbatch` exists, automatically upgrades to SLURM cluster execution
- **Model download fix**: Fixed `_run_shell()` env parameter bug that crashed all HuggingFace/ModelScope downloads; added hf-mirror.com as third fallback source
- **SLURM config**: Partition defaults to `belt_road`; wall time uses config value (30 days) instead of LLM-estimated time
</details>

<details>
<summary><b>v2.1</b> (2026-03-12) — Writing Quality & Anti-Fabrication</summary>

- **Table validation**: LLM-generated tables checked against grounding packet; mismatched tables auto-replaced with deterministic versions
- **Anti-fabrication**: Absolute ban on fabricated numbers; LLMs forbidden from generating `\begin{table}` (auto-injected by system)
- **Anti-AI writing**: 44 banned AI-characteristic words; review agent detects em-dash overuse, hedging pileups, repetitive transitions
- **Figure cleanup**: Verbose AI captions truncated to concise academic style; `\graphicspath` added for self-contained exports
- **Reviewer strictness**: Default skepticism; evidence required for 7+ scores
</details>

<details>
<summary><b>v2.0</b> (2026-03-12) — Infrastructure Overhaul</summary>

- **BaseOrchestrator refactor**: Unified orchestrator with thin subclasses, dedup of generate() logic
- **CUDA/conda detection**: Auto-detect environments, 46 infrastructure bug fixes
- **LaTeX hardening**: 2-level fix pipeline (deterministic + LLM), degenerate-run detection
- **Cost tracking**: Per-stage token counting, progress streaming, structured logging
</details>

<details>
<summary><b>v0.2</b> (2026-03-06) — Deep Pipeline</summary>

- **9-stage pipeline**: IDEATION → PLANNING → SETUP → CODING → EXECUTION → ANALYSIS → FIGURE_GEN → WRITING → REVIEW
- **REVIEW stage**: Per-section scoring, consistency checks, up to 5 revision rounds
- **Hybrid figures**: Architecture diagrams (AI) + results/ablation charts (code-generated)
- **Grounding system**: Enforces exact experiment numbers in writing; prevents result fabrication
</details>

## What It Does

Give NanoResearch a research topic, and it will:

1. **Ideation** — Search arXiv + Semantic Scholar, identify research gaps, generate hypotheses
2. **Planning** — Design a complete experiment blueprint (datasets, baselines, metrics, ablations)
3. **Setup** — Search repos, clone references, and optionally auto-download datasets/models
4. **Coding** — Generate a runnable experiment project plus training scripts
5. **Execution** — Auto-create environments, install requirements, run locally or on SLURM
6. **Analysis** — Parse real outputs, extract metrics, and prepare grounded evidence
7. **Figures** — Create architecture diagrams (Gemini AI) + data visualization charts (LLM-generated code)
8. **Writing** — Write a full LaTeX paper section-by-section, with optional ReAct tool use

```
Unified: Topic → IDEATION → PLANNING → SETUP → CODING → EXECUTION → ANALYSIS → FIGURE_GEN → WRITING → REVIEW → paper.pdf
```

`nanoresearch run` now uses the unified deep backbone by default. The legacy `deep` command remains as a compatibility alias, and the old standard orchestrator is kept only for resuming older workspaces.

## Execution Profiles

The unified pipeline supports three high-level profiles:

- `fast_draft` — lighter-weight writing/research assistance, useful for rapid draft generation
- `local_quick` — default profile; prefers local execution but **auto-upgrades to SLURM** if no local GPU is detected and `sbatch` is available
- `cluster_full` — always uses SLURM/cluster execution, with automatic fallback to local mode if cluster tools are unavailable

Relevant config keys:

```json
{
  "research": {
    "execution_profile": "local_quick",
    "writing_mode": "hybrid",
    "writing_tool_max_rounds": 10,
    "auto_create_env": true,
    "auto_download_resources": true,
    "local_execution_timeout": 1800
  }
}
```

## Quick Start

### Install

```bash
git clone https://github.com/your-org/nanoresearch.git
cd nanoresearch
pip install -e ".[dev]"
```

### Configure

Create `~/.nanobot/config.json`:

```json
{
  "research": {
    "base_url": "https://your-api-endpoint/v1/",
    "api_key": "your-api-key",
    "timeout": 180.0,
    "ideation":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.5 },
    "planning":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.2, "timeout": 300.0 },
    "experiment":    { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.1, "timeout": 600.0 },
    "writing":       { "model": "claude-sonnet-4-6", "temperature": 0.4, "max_tokens": 16384, "timeout": 600.0 },
    "code_gen":      { "model": "deepseek-ai/DeepSeek-V3.2", "temperature": 0.1, "max_tokens": 16384, "timeout": 600.0 },
    "figure_prompt": { "model": "gpt-5.2", "temperature": 0.5, "max_tokens": 4096, "timeout": 300.0 },
    "figure_code":   { "model": "gpt-5.2-codex", "temperature": null, "max_tokens": 16384, "timeout": 600.0 },
    "figure_gen": {
      "model": "gemini-3.1-flash-image-preview",
      "image_backend": "gemini",
      "temperature": null,
      "timeout": 300.0,
      "aspect_ratio": "16:9",
      "image_size": "1024x1024"
    }
  }
}
```

Or use environment variables: `NANORESEARCH_BASE_URL`, `NANORESEARCH_API_KEY`, `NANORESEARCH_TIMEOUT`.

### Run

```bash
# Full pipeline
nanoresearch run --topic "Your Research Topic" --verbose

# Resume from checkpoint (if a stage fails)
nanoresearch resume --workspace ~/.nanobot/workspace/research/{session_id} --verbose

# Check status
nanoresearch status --workspace ~/.nanobot/workspace/research/{session_id}

# List all sessions
nanoresearch list

# Export to clean folder
nanoresearch export --workspace ~/.nanobot/workspace/research/{session_id} --output ./my_paper
```

### Output

```
my_paper/
├── paper.pdf          # Compiled paper
├── paper.tex          # LaTeX source
├── references.bib     # Bibliography
├── figures/           # Architecture diagram (AI) + charts (seaborn)
├── code/              # Runnable experiment project
│   ├── main.py
│   ├── src/{model,dataset,trainer,evaluate,utils}.py
│   ├── config/default.yaml
│   └── scripts/{train.sh,run_ablation.sh}
├── data/              # Structured intermediate data (JSON)
└── manifest.json      # Full pipeline execution record
```

## Multi-Model Routing

Each pipeline stage can use a different LLM optimized for that task:

| Stage | Purpose | Recommended Model |
|-------|---------|-------------------|
| `ideation` | Literature search + hypothesis | DeepSeek-V3.2 |
| `planning` | Experiment design | DeepSeek-V3.2 |
| `experiment` | Code project generation | DeepSeek-V3.2 / Codex |
| `writing` | LaTeX paper sections | Claude Sonnet 4.6 |
| `figure_prompt` | Architecture diagram prompt | GPT-5.2 |
| `figure_code` | Chart plotting code | Codex |
| `figure_gen` | AI image generation | Gemini Flash |

All models are accessed through a single OpenAI-compatible API endpoint. Set `temperature: null` for models that don't support it (e.g., Codex, o-series).

## Hybrid Figure Generation

NanoResearch uses a hybrid approach for figures:

- **Architecture diagram** (`fig1`): AI image generation via Gemini native API — produces a visual model overview
- **Results chart** (`fig2`): LLM generates a complete Python plotting script with synthetic data, executed to produce a publication-quality grouped bar chart
- **Ablation chart** (`fig3`): Same approach — LLM-generated code creates an ablation study bar chart

The generated plotting scripts are saved alongside the figures for reproducibility.

## Checkpoint / Resume

Every stage saves its output to disk. If a stage fails (API timeout, rate limit, etc.), resume from the last completed checkpoint:

```bash
nanoresearch resume --workspace ~/.nanobot/workspace/research/{session_id}
```

The pipeline skips already-completed stages and continues from where it left off.

## Project Structure

```
nanoresearch/
├── nanoresearch/              # Main package
│   ├── cli.py                # CLI commands (run, resume, status, list, export)
│   ├── config.py             # Per-stage model routing + global config
│   ├── agents/               # 9 pipeline agents + helpers
│   │   ├── base.py          # BaseResearchAgent (shared LLM call logic + JSON repair)
│   │   ├── ideation.py      # Literature search + hypothesis generation
│   │   ├── planning.py      # Experiment blueprint design
│   │   ├── setup.py         # GitHub code search + repo cloning + model download
│   │   ├── coding.py        # Code project generation + cross-file mismatch detection
│   │   ├── execution.py     # SLURM job submission + monitoring + debug loop
│   │   ├── analysis.py      # Result parsing + structured comparison/ablation output
│   │   ├── figure_gen.py    # Hybrid figure generation (AI + code) with retry + diagnosis
│   │   ├── writing.py       # Section-by-section paper writing + PDF compilation
│   │   ├── review.py        # Automated review + revision + consistency checks
│   │   ├── debug.py         # Debug agent (used by execution for failed jobs)
│   │   ├── preflight.py     # Static preflight checks on experiment code
│   │   └── checkers.py      # LaTeX consistency + math formula validators
│   ├── pipeline/             # Infrastructure
│   │   ├── unified_orchestrator.py  # Default unified entrypoint
│   │   ├── deep_orchestrator.py  # Unified deep backbone implementation
│   │   ├── orchestrator.py  # Legacy standard compatibility pipeline
│   │   ├── state.py         # Pipeline state machine
│   │   ├── workspace.py     # Workspace directory + manifest management
│   │   └── multi_model.py   # OpenAI + Gemini API dispatcher
│   └── schemas/              # Pydantic data models
│       ├── manifest.py      # Stage enum + state transitions + manifest
│       ├── ideation.py      # PaperReference, IdeationOutput
│       ├── experiment.py    # ExperimentBlueprint, Dataset, Metric
│       └── paper.py         # PaperSkeleton, Section, Figure
├── mcp_server/               # MCP tool server
│   ├── server.py            # JSON-RPC 2.0 stdio server
│   └── tools/               # arXiv search, Semantic Scholar, LaTeX, PDF compile
├── tests/                    # 185 tests
├── outputs/                  # Example pipeline outputs
└── pyproject.toml
```

## Requirements

- Python >= 3.10
- An OpenAI-compatible API endpoint (self-hosted or cloud)
- `tectonic` or `pdflatex` for PDF compilation (optional — paper.tex works standalone)

### Install tectonic (recommended)

```bash
conda install -c conda-forge tectonic
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v  # 185 tests
```

## License

MIT
