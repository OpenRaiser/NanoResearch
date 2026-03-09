"""Centralized constants for the NanoResearch pipeline.

All magic numbers should be defined here. Import from this module,
never hardcode numeric literals in agent code.

Migration note: each constant records its original location in a comment.
When migrating, replace the local definition with an import from here.
"""

# === Literature Search (ideation.py) ===
TARGET_CITATION_COUNT = 50
MIN_HIGH_CITED_PAPERS = 10

# === Code Generation (experiment.py) ===
MAX_REFERENCE_REPOS = 3

# === Execution (debug.py, experiment.py) ===
MAX_DEBUG_ROUNDS = 20
DRY_RUN_TIMEOUT_S = 60
SUBPROCESS_OUTPUT_LIMIT = 5000

# === Writing (writing.py) ===
MAX_LATEX_FIX_ATTEMPTS = 3
MAX_CONTRIBUTION_ITEMS = 3

# === Review (review.py) ===
MAX_REVISION_ROUNDS = 5
CONVERGENCE_THRESHOLD = 0.3  # stop if avg score improvement < this

# === Analysis (analysis.py) ===
MAX_ANALYSIS_FIGURES = 5

# === Figure Generation (figure_gen.py) ===
MAX_IMAGE_RETRIES = 2
MAX_CODE_CHART_RETRIES = 3

# === API (multi_model.py) ===
MAX_API_RETRIES = 5
RETRY_BASE_DELAY_S = 3.0
RETRY_BACKOFF_FACTOR = 2.0

# === Context Management (base.py) ===
TOOL_RESULT_MAX_CHARS = 6000
TOOL_RESULT_HEAD_CHARS = 2000
TOOL_RESULT_TAIL_CHARS = 1500
CONTEXT_COMPACTION_THRESHOLD = 100_000
PROTECTED_TAIL_TURNS = 6

# === Metrics ===
LOWER_IS_BETTER_PATTERNS = frozenset({
    "loss", "error", "perplexity", "cer", "wer", "fer",
    "mae", "mse", "rmse", "mape", "fid", "kid", "ece",
    "latency", "inference_time", "distance", "divergence",
    "regret", "miss_rate", "false_positive", "eer",
})
