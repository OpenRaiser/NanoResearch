"""Pipeline state machine — validates stage transitions."""

from __future__ import annotations

from nanoresearch.schemas.manifest import PipelineStage, STAGE_TRANSITIONS


class InvalidTransitionError(Exception):
    """Raised when an invalid stage transition is attempted."""

    def __init__(self, current: PipelineStage, target: PipelineStage) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid transition: {current.value} -> {target.value}. "
            f"Allowed: {[s.value for s in STAGE_TRANSITIONS.get(current, [])]}"
        )


class PipelineStateMachine:
    """Manages state transitions for the research pipeline."""

    def __init__(self, initial: PipelineStage = PipelineStage.INIT) -> None:
        self._current = initial

    @property
    def current(self) -> PipelineStage:
        return self._current

    @property
    def is_terminal(self) -> bool:
        return self._current in (PipelineStage.DONE, PipelineStage.FAILED)

    def can_transition(self, target: PipelineStage) -> bool:
        return target in STAGE_TRANSITIONS.get(self._current, [])

    def transition(self, target: PipelineStage) -> PipelineStage:
        if not self.can_transition(target):
            raise InvalidTransitionError(self._current, target)
        self._current = target
        return self._current

    def fail(self) -> PipelineStage:
        """Shortcut to transition to FAILED from any non-terminal stage."""
        if self.is_terminal:
            raise InvalidTransitionError(self._current, PipelineStage.FAILED)
        self._current = PipelineStage.FAILED
        return self._current

    @staticmethod
    def next_stage(current: PipelineStage) -> PipelineStage | None:
        """Return the next forward (non-FAILED) stage, or None if terminal."""
        forward = [s for s in STAGE_TRANSITIONS.get(current, []) if s != PipelineStage.FAILED]
        return forward[0] if forward else None

    @staticmethod
    def processing_stages() -> list[PipelineStage]:
        """Return the ordered list of stages that do actual work."""
        return [
            PipelineStage.IDEATION,
            PipelineStage.PLANNING,
            PipelineStage.EXPERIMENT,
            PipelineStage.FIGURE_GEN,
            PipelineStage.WRITING,
            PipelineStage.REVIEW,
        ]
