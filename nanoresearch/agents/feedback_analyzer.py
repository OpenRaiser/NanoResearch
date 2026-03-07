"""Structured feedback analysis for experiment iterations.

Combines heuristic analysis (no LLM) with LLM-powered attribution
to produce a FeedbackAnalysis that drives the next iteration round.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.schemas.iteration import (
    FeedbackAnalysis,
    RoundResult,
    TrainingDynamics,
)

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an ML experiment analysis expert. Given experiment results, perform structured attribution analysis.

Output ONLY valid JSON with these fields:
{
  "attribution": "<one of: data_issue, training_strategy, architecture, hyperparameter, implementation_bug>",
  "recommended_action": "<specific, actionable next step>",
  "should_continue": <true/false>,
  "termination_reason": "<null or one of: target_met, plateau, max_rounds, degradation>",
  "error_categories": ["<category1>", ...]
}

Be concise and precise. Focus on the most impactful change for the next iteration."""


class FeedbackAnalyzer:
    """Analyze experiment results and produce structured feedback."""

    def __init__(self, config: ResearchConfig, dispatcher: ModelDispatcher) -> None:
        self.config = config
        self._dispatcher = dispatcher

    async def analyze(
        self,
        current_round: int,
        metrics: dict,
        previous_rounds: list[RoundResult],
        stderr_snippet: str = "",
        max_rounds: int = 3,
        target_metric: str | None = None,
        target_value: float | None = None,
    ) -> FeedbackAnalysis:
        """Run full analysis: heuristics + LLM attribution."""
        # --- Step 1: Extract metric summary ---
        metric_summary = self._extract_metric_summary(metrics)

        # --- Step 2: Compute improvement delta vs previous round ---
        improvement_delta: dict[str, float] = {}
        if previous_rounds:
            prev_summary = self._extract_metric_summary(previous_rounds[-1].metrics)
            for key, val in metric_summary.items():
                if key in prev_summary and prev_summary[key] != 0:
                    improvement_delta[key] = val - prev_summary[key]

        # --- Step 3: Heuristic training dynamics ---
        dynamics = self._analyze_training_dynamics(metrics)

        # --- Step 4: Check termination conditions (heuristic) ---
        termination = self._check_termination(
            current_round=current_round,
            metric_summary=metric_summary,
            improvement_delta=improvement_delta,
            previous_rounds=previous_rounds,
            max_rounds=max_rounds,
            target_metric=target_metric,
            target_value=target_value,
        )

        # --- Step 5: LLM attribution ---
        history_summary = self._build_compact_history(previous_rounds)
        llm_result = await self._llm_attribution(
            metric_summary=metric_summary,
            improvement_delta=improvement_delta,
            dynamics=dynamics,
            history_summary=history_summary,
            stderr_snippet=stderr_snippet[:1000],
        )

        # Merge heuristic termination with LLM recommendation
        should_continue = termination["should_continue"]
        termination_reason = termination["reason"]

        # LLM can also recommend stopping — but override in early rounds
        # when quick-eval hasn't produced any results yet (fixable bugs).
        if not llm_result.get("should_continue", True) and should_continue:
            # Never stop in early rounds if we have no metrics at all —
            # implementation bugs are fixable with more iterations.
            has_any_metrics = bool(metric_summary)
            is_early_round = current_round <= 3
            attribution = llm_result.get("attribution", "")
            if is_early_round and not has_any_metrics:
                logger.info(
                    "Overriding LLM stop recommendation in round %d "
                    "(no metrics yet, attribution=%s) — continuing",
                    current_round, attribution,
                )
                # Keep should_continue = True
            else:
                should_continue = False
                termination_reason = llm_result.get("termination_reason") or "llm_recommendation"

        return FeedbackAnalysis(
            metric_summary=metric_summary,
            improvement_delta=improvement_delta,
            error_categories=llm_result.get("error_categories", []),
            training_dynamics=dynamics,
            attribution=llm_result.get("attribution", ""),
            recommended_action=llm_result.get("recommended_action", ""),
            should_continue=should_continue,
            termination_reason=termination_reason,
        )

    # ------------------------------------------------------------------
    # Heuristic: metric extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_metric_summary(metrics: dict) -> dict[str, float]:
        """Pull top-level numeric metrics from the experiment results."""
        summary: dict[str, float] = {}
        main_results = metrics.get("main_results", [])
        if isinstance(main_results, list):
            for entry in main_results:
                if not isinstance(entry, dict):
                    continue
                method = entry.get("method_name", "unknown")
                is_proposed = entry.get("is_proposed", False)
                if not is_proposed:
                    continue
                for m in entry.get("metrics", []):
                    if isinstance(m, dict) and "metric_name" in m and "value" in m:
                        val = m["value"]
                        if isinstance(val, (int, float)) and math.isfinite(val):
                            summary[m["metric_name"]] = val
        return summary

    # ------------------------------------------------------------------
    # Heuristic: training dynamics
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_training_dynamics(metrics: dict) -> TrainingDynamics:
        """Analyze training_log for convergence, overfitting, stability."""
        training_log = metrics.get("training_log", [])
        if not isinstance(training_log, list) or len(training_log) < 2:
            return TrainingDynamics()

        train_losses: list[float] = []
        val_losses: list[float] = []
        for entry in training_log:
            if not isinstance(entry, dict):
                continue
            tl = entry.get("train_loss")
            vl = entry.get("val_loss")
            if isinstance(tl, (int, float)) and math.isfinite(tl):
                train_losses.append(float(tl))
            if isinstance(vl, (int, float)) and math.isfinite(vl):
                val_losses.append(float(vl))

        final_train = train_losses[-1] if train_losses else None
        final_val = val_losses[-1] if val_losses else None

        # Convergence speed: average loss decrease per epoch
        convergence = "normal"
        if len(train_losses) >= 2:
            total_decrease = train_losses[0] - train_losses[-1]
            avg_decrease = total_decrease / (len(train_losses) - 1)
            if train_losses[0] != 0:
                relative_decrease = total_decrease / abs(train_losses[0])
                if relative_decrease > 0.8:
                    convergence = "fast"
                elif relative_decrease < 0.1:
                    convergence = "slow"
                if total_decrease < 0:
                    convergence = "not_converging"

        # Overfitting: final train_loss much lower than final val_loss
        overfitting = False
        gap: float | None = None
        if final_train is not None and final_val is not None and final_val > 0:
            gap = final_val - final_train
            if final_train < 0.5 * final_val:
                overfitting = True

        # Stability: variance and spike detection in training loss
        stability = "stable"
        if len(train_losses) >= 3:
            mean_loss = sum(train_losses) / len(train_losses)
            variance = sum((x - mean_loss) ** 2 for x in train_losses) / len(train_losses)
            std = variance ** 0.5
            if mean_loss != 0 and (std / abs(mean_loss)) > 0.3:
                stability = "noisy"
            # Check for spikes (any epoch-to-epoch increase > 50% of range)
            loss_range = max(train_losses) - min(train_losses) if train_losses else 1
            if loss_range > 0:
                for i in range(1, len(train_losses)):
                    increase = train_losses[i] - train_losses[i - 1]
                    if increase > 0.5 * loss_range:
                        stability = "noisy"
                        break
            # Diverging: losses are increasing overall
            if len(train_losses) >= 2 and train_losses[-1] > train_losses[0] * 1.5:
                stability = "diverging"

        return TrainingDynamics(
            convergence_speed=convergence,
            overfitting_detected=overfitting,
            train_val_gap=gap,
            loss_stability=stability,
            final_train_loss=final_train,
            final_val_loss=final_val,
        )

    # ------------------------------------------------------------------
    # Heuristic: termination conditions
    # ------------------------------------------------------------------
    def _check_termination(
        self,
        current_round: int,
        metric_summary: dict[str, float],
        improvement_delta: dict[str, float],
        previous_rounds: list[RoundResult],
        max_rounds: int,
        target_metric: str | None,
        target_value: float | None,
    ) -> dict[str, Any]:
        """Evaluate heuristic termination conditions.

        Returns {"should_continue": bool, "reason": str | None}.
        """
        threshold = self.config.experiment_improvement_threshold
        patience = self.config.experiment_plateau_patience

        # 1. Target met
        if target_metric and target_value is not None:
            if metric_summary.get(target_metric, 0) >= target_value:
                return {"should_continue": False, "reason": "target_met"}

        # 2. Max rounds reached
        if current_round >= max_rounds:
            return {"should_continue": False, "reason": "max_rounds"}

        # 3. Plateau: consecutive rounds with improvement < threshold
        if len(previous_rounds) >= patience:
            recent_deltas = []
            for r in previous_rounds[-patience:]:
                if r.analysis:
                    delta = r.analysis.improvement_delta
                    if delta:
                        vals = list(delta.values())
                        recent_deltas.append(max(abs(v) for v in vals))
                    elif r.analysis.metric_summary:
                        # Metrics exist but no improvement — delta is 0
                        recent_deltas.append(0.0)
            if improvement_delta:
                recent_deltas.append(max(abs(v) for v in improvement_delta.values()))
            elif metric_summary:
                # Current round has metrics but no improvement delta
                recent_deltas.append(0.0)
            # If we have enough data points and all are below threshold
            if len(recent_deltas) >= patience and all(d < threshold for d in recent_deltas[-patience:]):
                return {"should_continue": False, "reason": "plateau"}

        # 3b. Repetition: same hypothesis tried in consecutive rounds
        if len(previous_rounds) >= 2:
            recent_hyps = [r.hypothesis.hypothesis[:80] for r in previous_rounds[-3:]]
            if len(set(recent_hyps)) == 1 and recent_hyps[0]:
                logger.info("Detected repetitive hypothesis: %s", recent_hyps[0][:60])
                return {"should_continue": False, "reason": "repetitive_hypothesis"}

        # 4. Degradation: current metrics worse than best by > 5%
        best_metrics = self._find_best_metrics(previous_rounds)
        if best_metrics and metric_summary:
            for key, current_val in metric_summary.items():
                best_val = best_metrics.get(key)
                if best_val is not None and best_val != 0:
                    if current_val < best_val * 0.95:
                        return {"should_continue": False, "reason": "degradation"}

        return {"should_continue": True, "reason": None}

    @staticmethod
    def _find_best_metrics(rounds: list[RoundResult]) -> dict[str, float]:
        """Find the best metric values across all rounds."""
        best: dict[str, float] = {}
        for r in rounds:
            if r.analysis:
                for key, val in r.analysis.metric_summary.items():
                    if key not in best or val > best[key]:
                        best[key] = val
        return best

    # ------------------------------------------------------------------
    # LLM-powered attribution
    # ------------------------------------------------------------------
    async def _llm_attribution(
        self,
        metric_summary: dict[str, float],
        improvement_delta: dict[str, float],
        dynamics: TrainingDynamics,
        history_summary: str,
        stderr_snippet: str,
    ) -> dict:
        """Ask LLM to attribute the results and recommend next action."""
        user_prompt = f"""Analyze the following experiment results and provide structured attribution.

== Current Round Metrics ==
{json.dumps(metric_summary, indent=2)}

== Training Dynamics ==
convergence_speed: {dynamics.convergence_speed}
overfitting_detected: {dynamics.overfitting_detected}
train_val_gap: {dynamics.train_val_gap}
loss_stability: {dynamics.loss_stability}
final_train_loss: {dynamics.final_train_loss}
final_val_loss: {dynamics.final_val_loss}

== Improvement vs Previous Round ==
{json.dumps(improvement_delta, indent=2) if improvement_delta else "N/A (first round)"}

== History Summary ==
{history_summary or "No previous rounds."}

== Stderr Snippet ==
{stderr_snippet or "No errors."}

Output ONLY valid JSON with: attribution, recommended_action, should_continue, termination_reason, error_categories"""

        try:
            stage_config = self.config.for_stage("experiment")
            raw = await self._dispatcher.generate(
                stage_config,
                ANALYSIS_SYSTEM_PROMPT,
                user_prompt,
                json_mode=True,
            )
            text = raw.strip()
            # Strip markdown fences
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
            return json.loads(text)
        except Exception as exc:
            logger.warning("LLM attribution failed: %s — using empty defaults", exc)
            return {
                "attribution": "",
                "recommended_action": "",
                "should_continue": True,
                "termination_reason": None,
                "error_categories": [],
            }

    # ------------------------------------------------------------------
    # Compact history builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_compact_history(rounds: list[RoundResult]) -> str:
        """Compress round history to ~1 line per round for LLM context."""
        if not rounds:
            return ""
        lines = []
        for r in rounds:
            metrics_str = ""
            if r.analysis:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in r.analysis.metric_summary.items()
                )
            hypothesis_short = r.hypothesis.hypothesis[:80]
            status = r.quick_eval_status
            lines.append(
                f"Round {r.round_number}: [{status}] {hypothesis_short} | {metrics_str}"
            )
        return "\n".join(lines)
