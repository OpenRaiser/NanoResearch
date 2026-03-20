"""Pipeline state machine — validates stage transitions."""

from __future__ import annotations

from nanoresearch.schemas.manifest import (
    DEEP_STAGE_TRANSITIONS,
    PipelineMode,
    PipelineStage,
    STANDARD_STAGE_TRANSITIONS,
    processing_stages_for_mode,
)


class InvalidTransitionError(Exception):
    """Raised when an invalid stage transition is attempted."""

    def __init__(
        self,
        current: PipelineStage,
        target: PipelineStage,
        allowed: list[PipelineStage] | None = None,
    ) -> None:
        self.current = current
        self.target = target
        allowed_values = [s.value for s in allowed or []]
        super().__init__(
            f"Invalid transition: {current.value} -> {target.value}. "
            f"Allowed: {allowed_values}"
        )


class PipelineStateMachine:
    """Manages state transitions for the research pipeline."""

    def __init__(
        self,
        initial: PipelineStage = PipelineStage.INIT,
        mode: PipelineMode = PipelineMode.STANDARD,
    ) -> None:
        self._current = initial
        self._mode = mode

    @property
    def current(self) -> PipelineStage:
        return self._current

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @property
    def is_terminal(self) -> bool:
        return self._current in (PipelineStage.DONE, PipelineStage.FAILED)

    @property
    def _transitions(self) -> dict[PipelineStage, list[PipelineStage]]:
        if self._mode == PipelineMode.DEEP:
            return DEEP_STAGE_TRANSITIONS
        return STANDARD_STAGE_TRANSITIONS

    def can_transition(self, target: PipelineStage) -> bool:
        return target in self._transitions.get(self._current, [])

    def transition(self, target: PipelineStage) -> PipelineStage:
        allowed = self._transitions.get(self._current, [])
        if target not in allowed:
            raise InvalidTransitionError(self._current, target, allowed)
        self._current = target
        return self._current

    def force_set(self, stage: PipelineStage) -> None:
        """Force state to a specific stage, bypassing transition validation.

        BUG-21 fix: used during checkpoint resume where we know the stage
        was previously completed. Logs a warning for auditability.
        """
        import logging as _log
        _log.getLogger(__name__).warning(
            "Force-setting state to %s (was %s)", stage.value, self._current.value,
        )
        self._current = stage

    def fail(self) -> PipelineStage:
        """Shortcut to transition to FAILED from any non-terminal stage."""
        if self.is_terminal:
            raise InvalidTransitionError(
                self._current,
                PipelineStage.FAILED,
                self._transitions.get(self._current, []),
            )
        self._current = PipelineStage.FAILED
        return self._current

    @staticmethod
    def next_stage(
        current: PipelineStage,
        mode: PipelineMode = PipelineMode.STANDARD,
    ) -> PipelineStage | None:
        """Return the next forward (non-FAILED) stage, or None if terminal."""

        transitions = (
            DEEP_STAGE_TRANSITIONS if mode == PipelineMode.DEEP
            else STANDARD_STAGE_TRANSITIONS
        )
        forward = [s for s in transitions.get(current, []) if s != PipelineStage.FAILED]
        return forward[0] if forward else None

    @staticmethod
    def processing_stages(
        mode: PipelineMode = PipelineMode.STANDARD,
    ) -> list[PipelineStage]:
        """Return the ordered list of stages that do actual work."""

        return processing_stages_for_mode(mode)
