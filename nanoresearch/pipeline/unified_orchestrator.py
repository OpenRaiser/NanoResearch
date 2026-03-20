"""Unified pipeline entrypoint built on the deep-stage backbone."""

from __future__ import annotations

from nanoresearch.pipeline.deep_orchestrator import DeepPipelineOrchestrator


class UnifiedPipelineOrchestrator(DeepPipelineOrchestrator):
    """Main research pipeline orchestrator for new runs."""

