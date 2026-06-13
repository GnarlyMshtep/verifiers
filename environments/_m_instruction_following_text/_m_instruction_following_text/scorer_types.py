from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from typing import Any

from verifiers.envs._m_ import JudgeView, ScorerResult

from .types import ConstraintName, MonitorPrompt


class ScorerName(StrEnum):
    CONSTRAINT = "constraint_satisfied"
    JUDGE_REQUEST_FOLLOWED = "judge_request_followed"


@dataclass(frozen=True)
class JudgeConfig:
    """Config for the LLM-judge monitor. Validates the monitor-prompt / reasoning combination."""
    monitor_prompt: MonitorPrompt
    view: JudgeView
    model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "medium"  # "none" disables reasoning entirely (no reasoning tokens)
    max_tokens: int = 2048
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.monitor_prompt is MonitorPrompt.IMMEDIATE_ANSWER and self.reasoning_effort != "none":
            raise ValueError(
                "immediate_answer_monitor requires reasoning_effort='none' — the monitor must commit "
                f"to a score immediately with no reasoning; got reasoning_effort={self.reasoning_effort!r}."
            )

    def sampling_args(self) -> dict[str, Any]:
        # reasoning_effort='none' -> send NO reasoning param at all (pair with a non-reasoning model,
        # so the response carries no reasoning tokens). Otherwise request the given effort.
        args: dict[str, Any] = {"max_tokens": self.max_tokens, "temperature": 0.0}
        if self.reasoning_effort != "none":
            args["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        return args


@dataclass
class ConstraintScore(ScorerResult):
    constraint: ConstraintName
    satisfied: bool
    detail: str
    name: ScorerName = ScorerName.CONSTRAINT
