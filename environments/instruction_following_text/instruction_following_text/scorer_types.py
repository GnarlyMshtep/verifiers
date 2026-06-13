from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from dataclasses import dataclass
from typing import Any

from verifiers.types import Messages, State

from .types import ConstraintName, JudgeView, MonitorPrompt, TaskInfo


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
class ScorerResult:
    """Base result. Subclasses add fields then re-declare `name` LAST with a default
    (dataclass ordering requires the only defaulted field to come after non-default ones)."""
    score: float


@dataclass
class ConstraintScore(ScorerResult):
    constraint: ConstraintName
    satisfied: bool
    detail: str
    name: ScorerName = ScorerName.CONSTRAINT


@dataclass
class JudgeScore(ScorerResult):
    view: JudgeView
    model: str
    judge_input: str
    judge_output: str
    judge_reasoning: str | None
    justification: str | None  # the judge's <justification> text (evidence for its score)
    attempts: list[dict[str, Any]]
    name: ScorerName = ScorerName.JUDGE_REQUEST_FOLLOWED


class Scorer(ABC):
    """Grades a full rollout trajectory and returns a typed ScorerResult."""

    name: ScorerName
    weight: float = 1.0

    @abstractmethod
    async def score(
        self,
        *,
        prompt: Messages,
        completion: Messages,
        answer: str,
        task_info: TaskInfo,
        state: State,
    ) -> ScorerResult: ...
