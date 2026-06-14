from __future__ import annotations

from verifiers.envs._m_ import Scorer
from verifiers.envs._m_.judge import last_assistant_text

from .constraints import CONSTRAINTS
from .scorer_types import ConstraintScore, ScorerName
from .types import TaskInfo


class ConstraintScorer(Scorer):
    """Rule-based: 1.0 iff the final answer satisfies the problem's constraint."""

    name = ScorerName.CONSTRAINT

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    async def score(self, *, prompt, completion, answer, task_info: TaskInfo, state) -> ConstraintScore:
        spec = task_info.constraint
        result = CONSTRAINTS[spec.name].verify(last_assistant_text(completion), spec.params)
        return ConstraintScore(
            score=1.0 if result.satisfied else 0.0,
            constraint=task_info.constraint.name,
            satisfied=result.satisfied,
            detail=result.detail,
        )
