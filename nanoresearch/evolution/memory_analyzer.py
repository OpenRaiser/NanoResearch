"""Heuristic analyzer for Direction/Strategy memory evolution."""

from __future__ import annotations

import json
from typing import Any

from .memory import MemoryStore, ResearchMemoryKind


class MemoryEvolutionAnalyzer:
    """Summarize stage traces into research-facing direction/strategy memories."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    @staticmethod
    def _compact(text: str, *, limit: int = 320) -> str:
        text = " ".join((text or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _ensure_dict(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        return {}

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _top_strings(items: list[Any], key: str, *, limit: int = 3) -> list[str]:
        results: list[str] = []
        for item in items[:limit]:
            if isinstance(item, dict):
                value = str(item.get(key, "")).strip()
                if value:
                    results.append(value)
        return results

    def _build_conditions(
        self,
        *,
        topic: str,
        paper_mode: str,
        blueprint: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        blueprint = blueprint or {}
        datasets = self._top_strings(blueprint.get("datasets", []), "name")
        method = self._ensure_dict(blueprint.get("proposed_method")).get("name", "")
        compute = self._ensure_dict(blueprint.get("compute_requirements"))
        conditions: dict[str, Any] = {
            "topic": topic,
            "paper_mode": paper_mode,
        }
        if datasets:
            conditions["datasets"] = datasets
        if method:
            conditions["method"] = method
        if compute:
            gpu_type = str(compute.get("gpu_type", "")).strip()
            num_gpus = compute.get("num_gpus")
            hours = compute.get("estimated_hours")
            if gpu_type or num_gpus:
                conditions["resource_budget"] = f"{num_gpus or '?'}x {gpu_type or 'gpu'}"
            if hours not in (None, ""):
                conditions["estimated_hours"] = str(hours)
        if extra:
            conditions.update(extra)
        return conditions

    def summarize_promising_direction(
        self,
        *,
        topic: str,
        paper_mode: str,
        ideation_output: dict[str, Any] | None = None,
        planning_output: dict[str, Any] | None = None,
        source: str = "",
        source_stage: str = "ideation",
        project_key: str = "",
        workspace_id: str = "",
    ) -> dict[str, Any] | None:
        ideation_data = self._ensure_dict(ideation_output)
        planning_data = self._ensure_dict(planning_output)
        selected_id = str(ideation_data.get("selected_hypothesis", "")).strip()
        hypotheses = ideation_data.get("hypotheses", []) or []
        selected_statement = ""
        top_statements: list[str] = []
        for hyp in hypotheses[:3]:
            if not isinstance(hyp, dict):
                continue
            statement = str(hyp.get("statement", "")).strip()
            if statement:
                top_statements.append(statement)
            if hyp.get("hypothesis_id") == selected_id and statement:
                selected_statement = statement
        gaps = self._top_strings(ideation_data.get("gaps", []), "description")
        rationale = self._first_non_empty(ideation_data.get("rationale", ""))
        method_name = self._ensure_dict(planning_data.get("proposed_method")).get("name", "")
        focus = self._first_non_empty(selected_statement, method_name, top_statements[0] if top_statements else "")
        if not focus:
            return None
        gap_summary = "; ".join(gaps[:2])
        method_clause = f" using {method_name}" if method_name else ""
        content = self._compact(
            f"Promising direction for {topic}: prioritize {focus}{method_clause}; it aligns with recurring gaps {gap_summary or 'identified in the literature'} and remains feasible under the current plan."
        )
        evidence_summary = self._compact(
            f"Selection rationale: {rationale or 'high-quality candidate ideas converged on this direction'}. "
            f"Top candidate ideas: {'; '.join(top_statements[:3]) or focus}."
        )
        direction_ref = selected_id or method_name or focus
        conditions = self._build_conditions(
            topic=topic,
            paper_mode=paper_mode,
            blueprint=planning_data,
            extra={"source_stage": source_stage},
        )
        record = self.store.remember_research(
            ResearchMemoryKind.PROMISING_DIRECTION,
            content,
            task_family="direction_selection",
            direction_ref=direction_ref,
            conditions=conditions,
            evidence_summary=evidence_summary,
            confidence=0.7,
            support_count=max(1, min(3, len(top_statements) or 1)),
            source=source,
            source_stage=source_stage,
            importance=0.78,
            tags=[topic, "direction", "promising", paper_mode],
            project_key=project_key,
            workspace_id=workspace_id,
        )
        return {
            "memory_kind": ResearchMemoryKind.PROMISING_DIRECTION.value,
            "summary": content,
            "evidence_summary": evidence_summary,
            "direction_ref": direction_ref,
            "conditions": conditions,
            "record": record.model_dump(mode="json") if record is not None else None,
        }

    def summarize_failed_direction(
        self,
        *,
        topic: str,
        paper_mode: str,
        blueprint: dict[str, Any] | None = None,
        iteration_state: dict[str, Any] | None = None,
        failure_reason: str = "",
        source: str = "",
        source_stage: str = "experiment",
        project_key: str = "",
        workspace_id: str = "",
    ) -> dict[str, Any] | None:
        blueprint_data = self._ensure_dict(blueprint)
        state = self._ensure_dict(iteration_state)
        method_name = self._ensure_dict(blueprint_data.get("proposed_method")).get("name", "") or "the current proposal"
        rounds = state.get("rounds", []) or []
        final_status = str(state.get("final_status", "")).strip()
        any_preflight_failures = any(
            isinstance(round_data, dict)
            and self._ensure_dict(round_data.get("preflight")).get("overall_status") == "failed"
            for round_data in rounds
        )
        any_metrics = any(
            isinstance(round_data, dict) and bool(round_data.get("metrics"))
            for round_data in rounds
        )
        normalized_reason = self._compact(failure_reason or final_status or "execution failed")
        if any_preflight_failures and not any_metrics:
            failure_mode = "non_executable_within_budget"
            diagnosis = "implementation and preflight issues prevented a reliable executable run within the available budget"
        else:
            failure_mode = "underperformed_or_no_gain"
            diagnosis = "the proposal ran but did not produce a stable gain over the existing setup under the current conditions"
        content = self._compact(
            f"Deprioritize {method_name} for {topic} under the current setup: {diagnosis}."
        )
        evidence_summary = self._compact(
            f"Failure signal: {normalized_reason}. Final status: {final_status or 'unknown'}. "
            f"Observed rounds: {len(rounds)}. Failure mode: {failure_mode}."
        )
        conditions = self._build_conditions(
            topic=topic,
            paper_mode=paper_mode,
            blueprint=blueprint_data,
            extra={
                "failure_mode": failure_mode,
                "source_stage": source_stage,
            },
        )
        record = self.store.remember_research(
            ResearchMemoryKind.FAILED_DIRECTION,
            content,
            task_family="direction_validation",
            proposal_ref=str(blueprint_data.get("hypothesis_ref", "")).strip(),
            direction_ref=method_name,
            conditions=conditions,
            evidence_summary=evidence_summary,
            confidence=0.74 if any_preflight_failures else 0.68,
            support_count=max(1, len(rounds)),
            source=source,
            source_stage=source_stage,
            importance=0.8,
            tags=[topic, "direction", "failed", failure_mode, paper_mode],
            project_key=project_key,
            workspace_id=workspace_id,
        )
        return {
            "memory_kind": ResearchMemoryKind.FAILED_DIRECTION.value,
            "summary": content,
            "evidence_summary": evidence_summary,
            "conditions": conditions,
            "failure_mode": failure_mode,
            "record": record.model_dump(mode="json") if record is not None else None,
        }

    def summarize_experiment_strategies(
        self,
        *,
        topic: str,
        paper_mode: str,
        blueprint: dict[str, Any] | None = None,
        iteration_state: dict[str, Any] | None = None,
        source: str = "",
        source_stage: str = "experiment",
        project_key: str = "",
        workspace_id: str = "",
    ) -> dict[str, Any]:
        blueprint_data = self._ensure_dict(blueprint)
        state = self._ensure_dict(iteration_state)
        rounds = state.get("rounds", []) or []
        datasets = self._top_strings(blueprint_data.get("datasets", []), "name")
        final_status = str(state.get("final_status", "")).strip()
        best_round = state.get("best_round")
        preflight_failures: list[str] = []
        metric_keys: list[str] = []
        for round_data in rounds:
            if not isinstance(round_data, dict):
                continue
            preflight = self._ensure_dict(round_data.get("preflight"))
            preflight_failures.extend(str(item) for item in preflight.get("blocking_failures", [])[:3])
            for key in list(self._ensure_dict(round_data.get("metrics")).keys())[:3]:
                if key not in metric_keys:
                    metric_keys.append(str(key))
        base_conditions = self._build_conditions(
            topic=topic,
            paper_mode=paper_mode,
            blueprint=blueprint_data,
            extra={"source_stage": source_stage},
        )
        results: list[dict[str, Any]] = []

        data_content = self._compact(
            f"For {topic}, stabilize data and environment before long runs: validate dataset paths and preprocessing assumptions, then run preflight checks on a reduced setup before committing cluster budget."
        )
        data_evidence = self._compact(
            f"Datasets: {', '.join(datasets) or 'not specified'}. "
            f"Observed preflight signals: {'; '.join(preflight_failures[:2]) or 'preflight checks are the earliest failure detector in the trajectory'}."
        )
        data_record = self.store.remember_research(
            ResearchMemoryKind.DATA_STRATEGY,
            data_content,
            task_family="experiment_execution",
            direction_ref=self._ensure_dict(blueprint_data.get("proposed_method")).get("name", ""),
            conditions={**base_conditions, "strategy_type": "data"},
            evidence_summary=data_evidence,
            confidence=0.69,
            support_count=max(1, len(rounds)),
            source=source,
            source_stage=source_stage,
            importance=0.73,
            tags=[topic, "strategy", "data", paper_mode],
            project_key=project_key,
            workspace_id=workspace_id,
        )
        results.append({
            "memory_kind": ResearchMemoryKind.DATA_STRATEGY.value,
            "summary": data_content,
            "evidence_summary": data_evidence,
            "record": data_record.model_dump(mode="json") if data_record is not None else None,
        })

        training_content = self._compact(
            f"For {topic}, prefer iterative reduced-scale training and preserve fixes from the best intermediate round before escalating to full runs; use trajectory feedback rather than only the final run outcome."
        )
        training_evidence = self._compact(
            f"Final status: {final_status or 'unknown'}. Best round: {best_round or 'n/a'}. "
            f"Tracked metrics: {', '.join(metric_keys) or 'no stable metrics yet'}."
        )
        training_record = self.store.remember_research(
            ResearchMemoryKind.TRAINING_STRATEGY,
            training_content,
            task_family="experiment_execution",
            direction_ref=self._ensure_dict(blueprint_data.get("proposed_method")).get("name", ""),
            conditions={**base_conditions, "strategy_type": "training"},
            evidence_summary=training_evidence,
            confidence=0.72 if best_round else 0.66,
            support_count=max(1, len(rounds)),
            source=source,
            source_stage=source_stage,
            importance=0.76,
            tags=[topic, "strategy", "training", paper_mode],
            project_key=project_key,
            workspace_id=workspace_id,
        )
        results.append({
            "memory_kind": ResearchMemoryKind.TRAINING_STRATEGY.value,
            "summary": training_content,
            "evidence_summary": training_evidence,
            "record": training_record.model_dump(mode="json") if training_record is not None else None,
        })

        return {
            "memory_kind": "strategy_memory_bundle",
            "strategies": results,
            "conditions": base_conditions,
        }
