import asyncio

from instruction_following_text.scorers import ConstraintScorer
from instruction_following_text.scorer_types import ConstraintScore
from instruction_following_text.types import (
    AlpacaProblem,
    ConstraintName,
    ConstraintSpec,
    Difficulty,
    TaskInfo,
)


def _task_info(constraint: ConstraintName) -> TaskInfo:
    return TaskInfo(
        alpaca=AlpacaProblem(orig_index=0, request_id=0, request="r"),
        constraint=ConstraintSpec(name=constraint, difficulty=Difficulty.EASY),
    )


def test_constraint_scorer_pass():
    scorer = ConstraintScorer()
    completion = [{"role": "assistant", "content": "this is all lowercase."}]
    res = asyncio.run(scorer.score(prompt=[], completion=completion, answer="",
                                   task_info=_task_info(ConstraintName.NO_CAPITALS), state={}))
    assert isinstance(res, ConstraintScore)
    assert res.score == 1.0 and res.satisfied is True and res.constraint == ConstraintName.NO_CAPITALS


def test_constraint_scorer_fail():
    scorer = ConstraintScorer()
    completion = [{"role": "assistant", "content": "This Has Capitals."}]
    res = asyncio.run(scorer.score(prompt=[], completion=completion, answer="",
                                   task_info=_task_info(ConstraintName.NO_CAPITALS), state={}))
    assert res.score == 0.0 and res.satisfied is False
