from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from verifiers.types import Messages, State


@dataclass
class ScorerResult:
    """Base result. Subclasses add fields then re-declare `name` LAST with a default
    (dataclass ordering requires the only defaulted field to come after non-default ones)."""
    score: float


class Scorer(ABC):
    """Grades a full rollout trajectory and returns a typed ScorerResult."""

    name: str
    weight: float = 1.0

    @abstractmethod
    async def score(
        self,
        *,
        prompt: Messages,
        completion: Messages,
        answer: str,
        task_info: object,
        state: State,
    ) -> ScorerResult: ...
