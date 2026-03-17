"""Error repair: resource matching, repair strategies, runtime remediation, and repair journal.

This module re-exports the _RepairMixin class composed from sub-mixins,
maintaining backward compatibility with ``from .repair import _RepairMixin``.
"""

from __future__ import annotations

# Module-level constants (imported by sub-mixins via .repair)
REMEDIATION_LEDGER_PATH = "logs/execution_remediation_ledger.json"
RESOURCE_SUCCESS_STATUSES = {"downloaded", "full", "config_only"}
MODULE_PACKAGE_ALIASES = {
    "cv2": "opencv-python",
    "pil": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "bio": "biopython",
}
QUICK_EVAL_AUTO_OPTIONS = {
    "--quick-eval",
    "--epochs",
    "--num-epochs",
    "--max-steps",
    "--steps",
    "--batch-size",
    "--batch_size",
    "--num-workers",
    "--num_workers",
    "--workers",
    "--subset-size",
    "--subset_size",
    "--train-size",
    "--quick-eval-train-size",
    "--limit-train-batches",
    "--limit-val-batches",
}

from .repair_ledger import _RepairLedgerMixin
from .repair_resources import _RepairResourcesMixin
from .repair_commands import _RepairCommandsMixin
from .repair_candidates import _RepairCandidatesMixin
from .repair_strategies import _RepairStrategiesMixin
from .repair_runtime import _RepairRuntimeMixin


class _RepairMixin(
    _RepairLedgerMixin,
    _RepairResourcesMixin,
    _RepairCommandsMixin,
    _RepairCandidatesMixin,
    _RepairStrategiesMixin,
    _RepairRuntimeMixin,
):
    """Aggregated repair mixin composed from sub-mixins."""

    pass
