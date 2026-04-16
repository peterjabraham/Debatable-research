class AgentTimeoutError(Exception):
    def __init__(self, agent_id: str, timeout_ms: int):
        super().__init__(f"Agent {agent_id} timed out after {timeout_ms}ms")
        self.agent_id = agent_id
        self.timeout_ms = timeout_ms


class AgentValidationError(Exception):
    def __init__(self, agent_id: str, reason: str):
        super().__init__(f"Agent {agent_id} output failed validation: {reason}")
        self.agent_id = agent_id
        self.reason = reason


class PipelineDependencyError(Exception):
    pass


class InvalidStateTransitionError(Exception):
    def __init__(self, agent_id: str, from_: object, to: object):
        super().__init__(f"Invalid transition for {agent_id}: {from_} → {to}")


class CheckpointCorruptError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Checkpoint at {path} failed schema validation")
        self.path = path


class LLMRetryExhaustedError(Exception):
    def __init__(self, agent_id: str, attempts: int):
        super().__init__(f"Agent {agent_id} exhausted {attempts} retry attempts")


class PipelineWarning:
    NO_CONTEST = "NO_CONTEST"
    SHALLOW_CLAIMS = "SHALLOW_CLAIMS"
    TRUNCATED_POSITIONS = "TRUNCATED_POSITIONS"
    CONTEXT_NEAR_LIMIT = "CONTEXT_NEAR_LIMIT"
