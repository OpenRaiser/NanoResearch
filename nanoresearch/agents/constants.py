"""Centralized timeout, retry, and limit constants for NanoResearch agents.

These were previously scattered across 8+ files with occasional conflicts
(e.g. RETRY_BASE_DELAY differed between multi_model.py and base_orchestrator.py).
Import from here to keep values consistent.

Usage::

    from nanoresearch.agents.constants import RETRY_BASE_DELAY, MAX_REVISION_ROUNDS
"""

# ============================================================================
# RETRY & BACKOFF  (used by multi_model.py + base_orchestrator.py)
# ============================================================================
MAX_API_RETRIES = 5           # max LLM API retry attempts
RETRY_BASE_DELAY = 5.0        # seconds — initial backoff
RETRY_MAX_DELAY = 60.0        # seconds — backoff cap
RETRY_BACKOFF_FACTOR = 2.0    # exponential multiplier

# ============================================================================
# CLUSTER / SLURM EXECUTION
# ============================================================================
CLUSTER_POLL_INTERVAL = 30         # job status poll interval (seconds)
CLUSTER_MAX_WAIT = 14400           # max time per job (4 hours)
CLUSTER_MAX_WAIT_LONG = 604_800    # max time for long-running jobs (7 days)
CMD_TIMEOUT = 120                  # generic command execution timeout
SCP_TIMEOUT = 600                  # file transfer timeout (10 min)
ENV_SETUP_TIMEOUT = 900            # conda/pip env setup (15 min)
CLUSTER_ENV_VALIDATION_TIMEOUT = 60

# ============================================================================
# LOCAL EXECUTION
# ============================================================================
DRY_RUN_TIMEOUT = 1800             # 30 min: dry-run experiments
CMD_TIMEOUT_DEFAULT = 120          # 2 min: default shell command
CMD_TIMEOUT_MAX = 1800             # 30 min: ceiling for LLM-requested timeouts

# ============================================================================
# FIGURE GENERATION
# ============================================================================
CHART_EXEC_TIMEOUT = 60            # matplotlib subprocess execution
MAX_CODE_CHART_RETRIES = 3         # retries with error feedback
MAX_IMAGE_RETRIES = 2              # retries before LLM diagnosis
MAX_IMAGE_PROMPT_LEN = 3800        # vision model prompt cap
MAX_EVIDENCE_BLOCK_LEN = 8000      # figure evidence context cap
MAX_EVIDENCE_TRAINING_LOG_ENTRIES = 50
MAX_FIG_WIDTH_PX = 2600
MAX_FIG_HEIGHT_PX = 3000
MAX_FIG_ASPECT_RATIO = 1.8

# ============================================================================
# WRITING & REVIEW
# ============================================================================
MAX_REVISION_ROUNDS = 5            # review/revision cycles
MAX_LATEX_FIX_ATTEMPTS = 3         # LaTeX compilation fix attempts
MIN_SECTION_SCORE = 8              # quality threshold triggering revision
MAX_STALL_ROUNDS = 2               # revision stall detection
MAX_PAPERS_FOR_CITATIONS = 50      # citations in writing context

# ============================================================================
# DEBUG & ITERATION
# ============================================================================
MAX_DEBUG_ROUNDS = 20              # max debug iteration cycles

# ============================================================================
# TOOL SAFETY LIMITS
# ============================================================================
MAX_TOOL_RESULT_CHARS = 6000       # tool output cap (chars)
MAX_READ_SIZE = 200_000            # 200 KB file read cap
MAX_WRITE_SIZE = 500_000           # 500 KB file write cap
MAX_FILE_SIZE_EDITOR = 50_000      # file editor size cap (chars)
MAX_LIST_ENTRIES = 200             # directory listing cap
MAX_GREP_RESULTS = 50              # grep result cap
