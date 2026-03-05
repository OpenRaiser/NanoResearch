"""Tests for the pipeline state machine."""

import pytest

from nanoresearch.pipeline.state import InvalidTransitionError, PipelineStateMachine
from nanoresearch.schemas.manifest import PipelineStage


class TestPipelineStateMachine:
    def test_initial_state(self):
        sm = PipelineStateMachine()
        assert sm.current == PipelineStage.INIT
        assert not sm.is_terminal

    def test_happy_path(self):
        sm = PipelineStateMachine()
        sm.transition(PipelineStage.IDEATION)
        assert sm.current == PipelineStage.IDEATION

        sm.transition(PipelineStage.PLANNING)
        assert sm.current == PipelineStage.PLANNING

        sm.transition(PipelineStage.EXPERIMENT)
        assert sm.current == PipelineStage.EXPERIMENT

        sm.transition(PipelineStage.FIGURE_GEN)
        assert sm.current == PipelineStage.FIGURE_GEN

        sm.transition(PipelineStage.WRITING)
        assert sm.current == PipelineStage.WRITING

        sm.transition(PipelineStage.REVIEW)
        assert sm.current == PipelineStage.REVIEW

        sm.transition(PipelineStage.DONE)
        assert sm.current == PipelineStage.DONE
        assert sm.is_terminal

    def test_invalid_transition(self):
        sm = PipelineStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.WRITING)  # can't skip

    def test_cannot_skip_figure_gen(self):
        sm = PipelineStateMachine(PipelineStage.EXPERIMENT)
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.WRITING)  # must go through FIGURE_GEN

    def test_writing_to_review(self):
        sm = PipelineStateMachine(PipelineStage.WRITING)
        sm.transition(PipelineStage.REVIEW)
        assert sm.current == PipelineStage.REVIEW

    def test_review_to_done(self):
        sm = PipelineStateMachine(PipelineStage.REVIEW)
        sm.transition(PipelineStage.DONE)
        assert sm.current == PipelineStage.DONE

    def test_cannot_skip_review(self):
        sm = PipelineStateMachine(PipelineStage.WRITING)
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.DONE)  # must go through REVIEW

    def test_experiment_to_figure_gen(self):
        sm = PipelineStateMachine(PipelineStage.EXPERIMENT)
        sm.transition(PipelineStage.FIGURE_GEN)
        assert sm.current == PipelineStage.FIGURE_GEN

    def test_figure_gen_to_writing(self):
        sm = PipelineStateMachine(PipelineStage.FIGURE_GEN)
        sm.transition(PipelineStage.WRITING)
        assert sm.current == PipelineStage.WRITING

    def test_fail_from_any_stage(self):
        for stage in PipelineStateMachine.processing_stages():
            sm = PipelineStateMachine(stage)
            sm.fail()
            assert sm.current == PipelineStage.FAILED
            assert sm.is_terminal

    def test_fail_from_init(self):
        sm = PipelineStateMachine()
        sm.fail()
        assert sm.current == PipelineStage.FAILED

    def test_cannot_fail_from_done(self):
        sm = PipelineStateMachine(PipelineStage.DONE)
        with pytest.raises(InvalidTransitionError):
            sm.fail()

    def test_cannot_fail_from_failed(self):
        sm = PipelineStateMachine(PipelineStage.FAILED)
        with pytest.raises(InvalidTransitionError):
            sm.fail()

    def test_cannot_transition_from_terminal(self):
        sm = PipelineStateMachine(PipelineStage.DONE)
        assert not sm.can_transition(PipelineStage.IDEATION)
        with pytest.raises(InvalidTransitionError):
            sm.transition(PipelineStage.IDEATION)

    def test_next_stage(self):
        assert PipelineStateMachine.next_stage(PipelineStage.INIT) == PipelineStage.IDEATION
        assert PipelineStateMachine.next_stage(PipelineStage.IDEATION) == PipelineStage.PLANNING
        assert PipelineStateMachine.next_stage(PipelineStage.EXPERIMENT) == PipelineStage.FIGURE_GEN
        assert PipelineStateMachine.next_stage(PipelineStage.FIGURE_GEN) == PipelineStage.WRITING
        assert PipelineStateMachine.next_stage(PipelineStage.WRITING) == PipelineStage.REVIEW
        assert PipelineStateMachine.next_stage(PipelineStage.REVIEW) == PipelineStage.DONE
        assert PipelineStateMachine.next_stage(PipelineStage.DONE) is None
        assert PipelineStateMachine.next_stage(PipelineStage.FAILED) is None

    def test_processing_stages(self):
        stages = PipelineStateMachine.processing_stages()
        assert len(stages) == 6
        assert stages[0] == PipelineStage.IDEATION
        assert stages[3] == PipelineStage.FIGURE_GEN
        assert stages[4] == PipelineStage.WRITING
        assert stages[-1] == PipelineStage.REVIEW

    def test_can_transition(self):
        sm = PipelineStateMachine()
        assert sm.can_transition(PipelineStage.IDEATION)
        assert sm.can_transition(PipelineStage.FAILED)
        assert not sm.can_transition(PipelineStage.DONE)
        assert not sm.can_transition(PipelineStage.WRITING)
