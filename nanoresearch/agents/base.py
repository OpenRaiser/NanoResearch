"""Base agent — common LLM call logic for all research agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.exceptions import LLMError
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.evolution.memory import MemoryScope, MemoryStore, MemoryType
from nanoresearch.evolution.memory_analyzer import MemoryEvolutionAnalyzer
from nanoresearch.skills import UnifiedSkillMatcher

# Import all free functions from the helpers module so they remain accessible
# at their original locations (e.g. ``from nanoresearch.agents.base import detect_truncation``).
from nanoresearch.agents._base_helpers import (  # noqa: F401 — re-exports
    _VALID_JSON_ESCAPES,
    _LATEX_CMD_PREFIXES,
    _MAX_TOOL_RESULT_CHARS,
    _HEAD_CHARS,
    _TAIL_CHARS,
    _CONTEXT_COMPACT_THRESHOLD_CHARS,
    _PROTECTED_TAIL_TURNS,
    _truncate_tool_result,
    _compact_messages_if_needed,
    _fix_json_escapes,
    _extract_balanced_json_segment,
    _extract_json_candidates,
    _scan_json_fragment,
    _close_json_fragment,
    _trim_json_fragment,
    _repair_truncated_json,
    _json_error_msg,
    detect_truncation,
)

logger = logging.getLogger(__name__)


class BaseResearchAgent(ABC):
    """Abstract base class for all NanoResearch agents."""

    stage: PipelineStage  # subclass must set this

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self._dispatcher = ModelDispatcher(config)
        static_skills_dir = getattr(config, "static_skills_dir", "") or None
        self._memory_store = MemoryStore(
            enabled=getattr(config, "memory_enabled", True),
            top_k=getattr(config, "memory_retrieval_top_k", 5),
            decay_factor=getattr(config, "memory_decay_factor", 0.08),
        )
        self._memory_analyzer = MemoryEvolutionAnalyzer(self._memory_store)
        self._skill_matcher = UnifiedSkillMatcher(
            Path(static_skills_dir) if static_skills_dir else None,
            retrieval_top_k=getattr(config, "skill_retrieval_top_k", 5),
            autorun_policy=getattr(config, "script_skill_autorun_policy", "safe_only"),
        )

    def _remember_mutation_snapshot_entry(self, entry: dict[str, Any] | None) -> None:
        self._last_mutation_snapshot_entry = dict(entry) if isinstance(entry, dict) else None

    def consume_last_mutation_snapshot_entry(self) -> dict[str, Any] | None:
        entry = getattr(self, "_last_mutation_snapshot_entry", None)
        self._last_mutation_snapshot_entry = None
        return dict(entry) if isinstance(entry, dict) else None

    def _project_key(self, topic: str = "") -> str:
        raw = (topic or self.workspace.manifest.topic or self.workspace.manifest.session_id).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        return slug or self.workspace.manifest.session_id

    def build_adaptive_context(
        self,
        task_type: str,
        *,
        topic: str = "",
        blueprint: dict | None = None,
        text: str = "",
        tags: list[str] | None = None,
        template_format: str = "",
        include_script_recommendations: bool = True,
    ) -> str:
        tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        blocks: list[str] = []
        payload: dict[str, Any] = {"task_type": task_type, "topic": topic, "tags": tags}
        try:
            project_key = self._project_key(topic)
            if getattr(self.config, "memory_enabled", True) and getattr(self.config, "memory_evolution_enabled", True):
                research_conditions: dict[str, Any] = {
                    "paper_mode": self.workspace.manifest.paper_mode.value,
                }
                if blueprint:
                    research_conditions["has_blueprint"] = "yes"
                research_top_k = getattr(self.config, "direction_memory_top_k", 4)
                if task_type == "experiment":
                    research_top_k = getattr(self.config, "strategy_memory_top_k", 4)
                research_context = self._memory_store.render_research_context(
                    task_type,
                    topic=topic,
                    tags=tags,
                    text=text,
                    conditions=research_conditions,
                    project_key=project_key,
                    top_k=research_top_k,
                )
                if research_context:
                    blocks.append(research_context)
                    payload["research_memory_context"] = research_context
            if getattr(self.config, "memory_enabled", True):
                memory_context = self._memory_store.render_prompt_context(
                    task_type,
                    topic=topic,
                    tags=tags,
                    text=text,
                    project_key=project_key,
                    top_k=getattr(self.config, "memory_retrieval_top_k", 5),
                )
                if memory_context:
                    blocks.append(memory_context)
                    payload["memory_context"] = memory_context
            if getattr(self.config, "skill_evolution_enabled", True):
                skill_context = self._skill_matcher.build_context(
                    task_type,
                    topic=topic,
                    blueprint=blueprint,
                    text=text,
                    tags=tags,
                    template_format=template_format,
                )
                if not include_script_recommendations:
                    combined = "\n\n".join(part for part in (skill_context.static_context, skill_context.evolved_context) if part)
                else:
                    combined = skill_context.combined_context
                if combined:
                    blocks.append(combined)
                    payload["matched_skills"] = skill_context.matched_skills
                    payload["skill_context"] = combined
        except Exception as exc:
            logger.warning("Failed to build adaptive context for %s/%s: %s", self.stage.value, task_type, exc)
        combined_context = "\n\n".join(blocks)
        if combined_context:
            payload["combined_context"] = combined_context
            try:
                self.workspace.write_json(f"logs/adaptive_context_{self.stage.value.lower()}_{task_type}.json", payload)
            except Exception as exc:
                logger.debug("Failed to persist adaptive context trace: %s", exc)
        return combined_context

    def remember_context(
        self,
        memory_type: MemoryType | str,
        content: str,
        *,
        importance: float = 0.6,
        tags: list[str] | None = None,
        source: str = "",
        scope: MemoryScope | str = MemoryScope.WORKSPACE_DERIVED,
        topic: str = "",
    ) -> None:
        if not getattr(self.config, "memory_enabled", True):
            return
        try:
            self._memory_store.remember(
                memory_type,
                content,
                scope=scope,
                source=source or f"{self.stage.value.lower()}:{self.workspace.manifest.session_id}",
                importance=importance,
                tags=tags,
                project_key=self._project_key(topic),
                workspace_id=self.workspace.manifest.session_id,
            )
        except Exception as exc:
            logger.warning("Failed to write memory for %s: %s", self.stage.value, exc)

    def learn_from_trace(
        self,
        domain: str,
        trigger_pattern: str,
        source_trace: str,
        *,
        tags: list[str] | None = None,
        rule_text: str | None = None,
        confidence: float = 0.55,
    ) -> None:
        if not getattr(self.config, "skill_evolution_enabled", True):
            return
        trace = (source_trace or "").strip()
        if not trace:
            return
        try:
            skill = self._skill_matcher.evolution_store.synthesize_nl_skill(
                domain=domain,
                trigger_pattern=trigger_pattern,
                source_trace=trace,
                rule_text=rule_text,
                confidence=confidence,
                tags=tags,
            )
            if skill is not None:
                self.workspace.write_json(
                    f"logs/evolved_skill_{self.stage.value.lower()}_{trigger_pattern}.json",
                    skill.model_dump(mode="json"),
                )
        except Exception as exc:
            logger.warning("Failed to evolve skill for %s/%s: %s", self.stage.value, domain, exc)

    def remember_promising_direction(
        self,
        *,
        topic: str,
        ideation_output: dict | None = None,
        planning_output: dict | None = None,
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        if not getattr(self.config, "memory_enabled", True) or not getattr(self.config, "memory_evolution_enabled", True):
            return None
        try:
            payload = self._memory_analyzer.summarize_promising_direction(
                topic=topic,
                paper_mode=self.workspace.manifest.paper_mode.value,
                ideation_output=ideation_output,
                planning_output=planning_output,
                source=source or f"{self.stage.value.lower()}:{self.workspace.manifest.session_id}",
                source_stage=source_stage or self.stage.value.lower(),
                project_key=self._project_key(topic),
                workspace_id=self.workspace.manifest.session_id,
            )
            if payload and artifact_path:
                self.workspace.write_json(artifact_path, payload)
            return payload
        except Exception as exc:
            logger.warning("Failed to remember promising direction for %s: %s", self.stage.value, exc)
            return None

    def remember_failed_direction(
        self,
        *,
        topic: str,
        blueprint: dict | None = None,
        iteration_state: dict | None = None,
        failure_reason: str = "",
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        if not getattr(self.config, "memory_enabled", True) or not getattr(self.config, "memory_evolution_enabled", True):
            return None
        try:
            payload = self._memory_analyzer.summarize_failed_direction(
                topic=topic,
                paper_mode=self.workspace.manifest.paper_mode.value,
                blueprint=blueprint,
                iteration_state=iteration_state,
                failure_reason=failure_reason,
                source=source or f"{self.stage.value.lower()}:{self.workspace.manifest.session_id}",
                source_stage=source_stage or self.stage.value.lower(),
                project_key=self._project_key(topic),
                workspace_id=self.workspace.manifest.session_id,
            )
            if payload and artifact_path:
                self.workspace.write_json(artifact_path, payload)
            return payload
        except Exception as exc:
            logger.warning("Failed to remember failed direction for %s: %s", self.stage.value, exc)
            return None

    def remember_experiment_strategies(
        self,
        *,
        topic: str,
        blueprint: dict | None = None,
        iteration_state: dict | None = None,
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        if not getattr(self.config, "memory_enabled", True) or not getattr(self.config, "memory_evolution_enabled", True):
            return None
        try:
            payload = self._memory_analyzer.summarize_experiment_strategies(
                topic=topic,
                paper_mode=self.workspace.manifest.paper_mode.value,
                blueprint=blueprint,
                iteration_state=iteration_state,
                source=source or f"{self.stage.value.lower()}:{self.workspace.manifest.session_id}",
                source_stage=source_stage or self.stage.value.lower(),
                project_key=self._project_key(topic),
                workspace_id=self.workspace.manifest.session_id,
            )
            if payload and artifact_path:
                self.workspace.write_json(artifact_path, payload)
            return payload
        except Exception as exc:
            logger.warning("Failed to remember experiment strategies for %s: %s", self.stage.value, exc)
            return None

    @property
    def stage_config(self) -> StageModelConfig:
        return self.config.for_stage(self.stage.value)

    async def close(self) -> None:
        await self._dispatcher.close()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Call the LLM configured for this agent's stage."""
        cfg = stage_override if stage_override is not None else self.stage_config
        return await self._dispatcher.generate(
            cfg, system_prompt, user_prompt, json_mode
        )

    async def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        json_mode: bool = False,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Call the LLM with an image attachment (vision)."""
        cfg = stage_override if stage_override is not None else self.stage_config
        return await self._dispatcher.generate_with_image(
            cfg, system_prompt, user_prompt, image_bytes, mime_type, json_mode
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        stage_override: StageModelConfig | None = None,
    ) -> dict | list:
        """Call LLM and parse the response as JSON.

        Handles LaTeX backslash sequences that break strict JSON parsing.
        """
        raw = await self.generate(
            system_prompt, user_prompt, json_mode=True,
            stage_override=stage_override,
        )
        last_attempt = raw.strip()
        for text in _extract_json_candidates(raw):
            last_attempt = text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            fixed = _fix_json_escapes(text)
            last_attempt = fixed
            try:
                return json.loads(fixed, strict=False)
            except json.JSONDecodeError:
                pass

            repaired = _repair_truncated_json(fixed)
            if repaired is not None and repaired != fixed:
                last_attempt = repaired
                try:
                    return json.loads(repaired, strict=False)
                except json.JSONDecodeError:
                    pass

        # All attempts failed
        logger.error(
            "JSON parse failed even after escape fixing. First 500 chars: %s",
            last_attempt[:500],
        )
        raise LLMError(
            f"LLM output is not valid JSON: "
            f"{_json_error_msg(last_attempt)}. "
            f"Raw output starts with: {raw[:200]!r}"
        ) from None

    async def generate_json_validated(
        self,
        system_prompt: str,
        user_prompt: str,
        model_class: type,
        stage_override: StageModelConfig | None = None,
    ) -> Any:
        """Call LLM, parse as JSON, and validate against a Pydantic model.

        On validation failure, feeds the error back to the LLM for one retry.
        Returns a validated Pydantic model instance.
        """
        raw_dict = await self.generate_json(
            system_prompt, user_prompt, stage_override=stage_override,
        )
        try:
            return model_class.model_validate(raw_dict)
        except Exception as first_exc:
            # Single retry: feed validation error back to LLM
            self.log(f"  JSON schema validation failed: {first_exc}, retrying...")
            retry_prompt = (
                f"Your previous JSON response had validation errors:\n"
                f"{first_exc}\n\n"
                f"Original request:\n{user_prompt}\n\n"
                f"Fix the JSON to match the required schema and try again."
            )
            try:
                raw_dict = await self.generate_json(
                    system_prompt, retry_prompt, stage_override=stage_override,
                )
                return model_class.model_validate(raw_dict)
            except Exception as retry_exc:
                logger.error(
                    "JSON validation failed after retry: %s", retry_exc,
                )
                raise LLMError(
                    f"JSON schema validation failed after retry: {retry_exc}"
                ) from retry_exc

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Any,  # ToolRegistry
        max_tool_rounds: int = 10,
        stage_override: StageModelConfig | None = None,
        reminder_text: str | None = None,
        reminder_interval: int = 3,
    ) -> str:
        """Run a ReAct loop: let the LLM call tools until it produces text."""
        cfg = stage_override if stage_override is not None else self.stage_config
        openai_tools = tools.to_openai_tools()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Track repeated failures to avoid infinite retry loops (OpenClaw pattern)
        _failure_counts: dict[str, int] = {}
        _MAX_IDENTICAL_FAILURES = 2

        for round_idx in range(max_tool_rounds):
            msg = await self._dispatcher.generate_with_tools(cfg, messages, openai_tools)

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return self._dispatcher._strip_think_blocks(msg.content or "")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            if msg.content:
                assistant_msg["content"] = msg.content
            messages.append(assistant_msg)

            async def _execute_tool_call(tc):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON in tool args for %s: %s", name, exc)
                    args = {}

                self.log(f"Tool call: {name}({args})")
                try:
                    result = await tools.call(name, args)
                    result_str = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    error_str = f"Error: {type(e).__name__}: {e}"
                    error_sig = type(e).__name__
                    try:
                        args_hash = hash(json.dumps(args, sort_keys=True, default=str))
                    except (TypeError, ValueError):
                        args_hash = hash(str(sorted(args.items())) if isinstance(args, dict) else str(args))
                    fail_key = f"{name}|{args_hash}|{error_sig}"
                    _failure_counts[fail_key] = _failure_counts.get(fail_key, 0) + 1
                    if _failure_counts[fail_key] >= _MAX_IDENTICAL_FAILURES:
                        error_str = (
                            f"[NON-RETRYABLE] {error_str} — "
                            f"This exact call has failed {_failure_counts[fail_key]} times. "
                            f"Do NOT retry with the same arguments. Try a different query or approach."
                        )
                    result_str = error_str

                result_content = _truncate_tool_result(result_str)
                return {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                }

            if len(tool_calls) > 1:
                tool_results = await asyncio.gather(
                    *(_execute_tool_call(tc) for tc in tool_calls),
                    return_exceptions=True,
                )
                for i, tr in enumerate(tool_results):
                    if isinstance(tr, Exception):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_calls[i].id,
                            "content": f"Error: {type(tr).__name__}: {tr}",
                        })
                    else:
                        messages.append(tr)
            else:
                messages.append(await _execute_tool_call(tool_calls[0]))

            _compact_messages_if_needed(messages)

            if (round_idx + 1) % reminder_interval == 0 and round_idx + 1 <= max_tool_rounds:
                _reminder = reminder_text or (
                    "[REMINDER] You are writing academic content for a top-tier venue. "
                    "Focus on producing the final output now. Use the information "
                    "gathered from tools to write high-quality content. "
                    "Do NOT continue searching indefinitely."
                )
                messages.append({"role": "system", "content": _reminder})

        self.log(f"Exceeded {max_tool_rounds} tool rounds, forcing final answer")
        final_msg = await self._dispatcher.generate_with_tools(cfg, messages, tools=None)
        if hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
            return self._dispatcher._strip_think_blocks(
                final_msg.content or "Agent completed but produced no text summary."
            )
        return self._dispatcher._strip_think_blocks(final_msg.content or "")

    @abstractmethod
    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute this agent's stage. Returns output data dict."""
        ...

    def log(self, msg: str) -> None:
        logger.info(f"[{self.stage.value}] {msg}")

    def save_log(self, filename: str, content: str) -> None:
        self.workspace.write_text(f"logs/{filename}", content)

    def _resolve_experiment_python(self) -> str:
        """Return the experiment Python path.

        Resolution order:
        1. config.experiment_python (user-managed environment)
        2. experiment/.venv python (auto-created venv)
        3. sys.executable (fallback)
        """
        import os
        import sys
        from pathlib import Path as _Path

        # Priority 1: user-specified python
        user_spec = (self.config.experiment_python or "").strip()
        if user_spec:
            from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
            mgr = RuntimeEnvironmentManager(self.config)
            resolved = mgr._resolve_user_python(user_spec)
            if resolved and _Path(resolved).exists():
                return resolved

        # Priority 2: experiment venv
        exp_dir = self.workspace.path / "experiment"
        if os.name == "nt":
            venv_py = exp_dir / ".venv" / "Scripts" / "python.exe"
        else:
            venv_py = exp_dir / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable
