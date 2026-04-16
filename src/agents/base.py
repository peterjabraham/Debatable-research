import abc

from src.pipeline.state import AgentId, PipelineState


class BaseAgent(abc.ABC):
    id: AgentId
    timeout_ms: int
    max_retries: int

    @abc.abstractmethod
    async def run(self, state: PipelineState, signal=None) -> str:
        """Execute the agent. Returns raw LLM output string."""
        ...

    @abc.abstractmethod
    def build_prompt(self, state: PipelineState) -> str:
        """Construct the full prompt string from current state."""
        ...

    @abc.abstractmethod
    def validate_output(self, output: str, state: PipelineState) -> None:
        """
        Validate the LLM output. Raises AgentValidationError if invalid.
        May mutate state to add warnings (e.g. SHALLOW_CLAIMS).
        Should NOT raise on warnings — only on hard failures.
        """
        ...
