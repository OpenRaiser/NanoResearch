"""Adaptive memory and skill-evolution primitives for NanoResearch."""

from .memory import (
    MemoryRecord,
    MemoryScope,
    MemoryStore,
    MemoryType,
    ResearchMemoryKind,
    ResearchMemoryRecord,
)
from .memory_analyzer import MemoryEvolutionAnalyzer
from .ram import RAMBackend, RAMModule, RAMOutput
from .ram_data import RAMDataCollector, RAMTriple
from .sdpo import (
    SDPOConfig,
    SDPOExample,
    SDPOTrainer,
    examples_from_triples,
    load_model_for_sdpo,
    sdpo_loss,
)
from .sdpo_adapter import SDPOAdapterManager
from .skills import (
    NaturalLanguageSkill,
    ScriptSkill,
    ScriptSkillCategory,
    ScriptTestStatus,
    SkillCandidate,
    SkillDomain,
    SkillEvolutionStore,
    SkillLifecycleResult,
    SkillReview,
    SkillReviewDecision,
)

__all__ = [
    "MemoryRecord",
    "MemoryScope",
    "MemoryStore",
    "MemoryType",
    "ResearchMemoryKind",
    "ResearchMemoryRecord",
    "MemoryEvolutionAnalyzer",
    "RAMBackend",
    "RAMDataCollector",
    "RAMModule",
    "RAMOutput",
    "RAMTriple",
    "SDPOAdapterManager",
    "SDPOConfig",
    "SDPOExample",
    "SDPOTrainer",
    "examples_from_triples",
    "load_model_for_sdpo",
    "sdpo_loss",
    "NaturalLanguageSkill",
    "ScriptSkill",
    "ScriptSkillCategory",
    "ScriptTestStatus",
    "SkillCandidate",
    "SkillDomain",
    "SkillEvolutionStore",
    "SkillLifecycleResult",
    "SkillReview",
    "SkillReviewDecision",
]
