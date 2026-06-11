from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from dataclasses import dataclass
from typing import Any

from verifiers.types import Messages, State

from .types import ConstraintName, JudgeView, Problem


class ScorerName(StrEnum):
    CONSTRAINT = "constraint_satisfied"
    JUDGE_REQUEST_FOLLOWED = "judge_request_followed"


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
        problem: Problem,
        state: State,
    ) -> ScorerResult: ...
