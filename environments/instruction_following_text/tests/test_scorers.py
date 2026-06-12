import asyncio

import pytest

from instruction_following_text.scorers import ConstraintScorer, parse_score, render_view
from instruction_following_text.scorer_types import ConstraintScore
from instruction_following_text.types import ConstraintName, Difficulty, JudgeView, Problem


def _completion():
    return [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "final answer", "reasoning_content": "let me think"},
    ]


def test_render_view_output_only():
    out = render_view(_completion(), JudgeView.OUTPUT)
    assert "final answer" in out and "let me think" not in out


def test_render_view_cot_only():
    out = render_view(_completion(), JudgeView.COT)
    assert "let me think" in out and "final answer" not in out


def test_render_view_both():
    out = render_view(_completion(), JudgeView.BOTH)
    assert "final answer" in out and "let me think" in out


def test_render_view_cot_missing_raises():
    with pytest.raises(ValueError):
        render_view([{"role": "assistant", "content": "ans"}], JudgeView.COT)


def test_parse_score():
    assert parse_score("7") == 0.7
    assert parse_score("10") == 1.0
    assert parse_score("0") == 0.0
    assert parse_score("Score: 8") == 0.8


def test_parse_score_rejects_decimals_and_out_of_range():
    for bad in ("7.5", "11", "100", "level5", "excellent"):
        with pytest.raises(ValueError):
            parse_score(bad)
    assert parse_score("between 8 and 10") == 1.0


def _problem(constraint: ConstraintName) -> Problem:
    return Problem(request="r", constraint=constraint, difficulty=Difficulty.EASY, request_id=0)


def test_constraint_scorer_pass():
    scorer = ConstraintScorer()
    completion = [{"role": "assistant", "content": "this is all lowercase."}]
    res = asyncio.run(scorer.score(prompt=[], completion=completion, answer="",
                                   problem=_problem(ConstraintName.NO_CAPITALS), state={}))
    assert isinstance(res, ConstraintScore)
    assert res.score == 1.0 and res.satisfied is True and res.constraint == ConstraintName.NO_CAPITALS


def test_constraint_scorer_fail():
    scorer = ConstraintScorer()
    completion = [{"role": "assistant", "content": "This Has Capitals."}]
    res = asyncio.run(scorer.score(prompt=[], completion=completion, answer="",
                                   problem=_problem(ConstraintName.NO_CAPITALS), state={}))
    assert res.score == 0.0 and res.satisfied is False
