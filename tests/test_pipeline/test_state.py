"""Tests for nanoresearch.pipeline.state."""

from __future__ import annotations

import pytest

from nanoresearch.pipeline.state import (
    InvalidTransitionError,
    PipelineStateMachine,
)
from nanoresearch.schemas.manifest import (
    PipelineMode,
    PipelineStage,
)


class TestInvalidTransitionError:
    """Tests for InvalidTransitionError."""

    def test_message_includes_current_and_target(self) -> None:
        err = InvalidTransitionError(
            PipelineStage.INIT,
            PipelineStage.FAILED,
            allowed=[PipelineStage.IDEATION, PipelineStage.FAILED],
        )
        assert "INIT" in str(err)
        assert "FAILED" in str(err)
        assert err.current == PipelineStage.INIT
        assert err.target == PipelineStage.FAILED


class TestPipelineStateMachine:
    """Tests for PipelineStateMachine."""

    def test_initial_state(self) -> None:
        sm = PipelineStateMachine()
        assert sm.current == PipelineStage.INIT
        assert sm.is_terminal is False

    def test_can_transition_from_init_to_ideation(self) -> None:
        sm = PipelineStateMachine()
        assert sm.can_transition(PipelineStage.IDEATION) is True
        assert sm.can_transition(PipelineStage.FAILED) is True

    def test_can_transition_from_init_to_coding_in_standard_mode(self) -> None:
        sm = PipelineStateMachine(mode=PipelineMode.STANDARD)
        assert sm.can_transition(PipelineStage.CODING) is False

    def test_can_transition_from_init_to_setup_in_deep_mode(self) -> None:
        sm = PipelineStateMachine(mode=PipelineMode.DEEP)
        assert sm.can_transition(PipelineStage.IDEATION) is True

    def test_transition_updates_state(self) -> None:
        sm = PipelineStateMachine()
        result = sm.transition(PipelineStage.IDEATION)
        assert result == PipelineStage.IDEATION
        assert sm.current == PipelineStage.IDEATION

    def test_invalid_transition_raises(self) -> None:
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.IDEATION)
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.CODING)

    def test_transition_to_done_is_terminal(self) -> None:
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.IDEATION)
        sm.transition(PipelineStage.PLANNING)
        sm.transition(PipelineStage.EXPERIMENT)
        sm.transition(PipelineStage.FIGURE_GEN)
        sm.transition(PipelineStage.WRITING)
        sm.transition(PipelineStage.REVIEW)
        sm.transition(PipelineStage.DONE)
        assert sm.current == PipelineStage.DONE
        assert sm.is_terminal is True

    def test_force_set_bypasses_validation(self) -> None:
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.IDEATION)
        sm.force_set(PipelineStage.CODING)
        assert sm.current == PipelineStage.CODING

    def test_mode_property(self) -> None:
        sm = PipelineStateMachine(mode=PipelineMode.DEEP)
        assert sm.mode == PipelineMode.DEEP
