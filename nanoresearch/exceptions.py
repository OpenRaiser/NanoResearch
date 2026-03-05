"""Custom exception hierarchy for NanoResearch pipeline."""


class NanoResearchError(Exception):
    """Base exception for all NanoResearch errors."""


class StageError(NanoResearchError):
    """Error in a specific pipeline stage."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class LLMError(NanoResearchError):
    """Error from LLM interaction (JSON parse failure, empty response, etc.)."""


class ValidationError(NanoResearchError):
    """Schema or data validation error."""


class ToolError(NanoResearchError):
    """Error from tool execution in ReAct loop."""


class CheckpointError(NanoResearchError):
    """Error loading or saving checkpoint data."""
