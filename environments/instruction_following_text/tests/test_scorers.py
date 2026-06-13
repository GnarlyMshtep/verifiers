import asyncio

import pytest

from instruction_following_text.scorers import (
    ConstraintScorer,
    parse_judge_response,
    parse_score,
    render_view,
)
from instruction_following_text.scorer_types import ConstraintScore
from instruction_following_text.types import (
    AlpacaProblem,
    ConstraintName,
    ConstraintSpec,
    Difficulty,
    JudgeView,
    TaskInfo,
)


def _completion():
    return [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "final answer", "reasoning_content": "let me think"},
    ]


def test_render_view_output_only():
    out = render_view(_completion(), JudgeView.OUTPUT)
    assert "final answer" in out and "let me think" not in out
    assert "<assistant-output>" in out
    assert "<assistant-reasoning-not-delivered-to-user>" not in out


def test_render_view_cot_only():
    out = render_view(_completion(), JudgeView.COT)
    assert "let me think" in out and "final answer" not in out
    assert "<assistant-reasoning-not-delivered-to-user>" in out
    assert "<assistant-output>" not in out


def test_render_view_both():
    out = render_view(_completion(), JudgeView.BOTH)
    assert "final answer" in out and "let me think" in out
    assert out.startswith("<assistant-turn>") and out.endswith("</assistant-turn>")
    # reasoning precedes output inside the turn
    assert out.index("<assistant-reasoning-not-delivered-to-user>") < out.index("<assistant-output>")


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


def test_parse_judge_response_both_orders():
    # justification-then-score (regular) and score-then-justification (immediate) both parse.
    reg = "<justification>It quotes the cat. The cat sat.</justification>\n<score>8</score>"
    imm = "<score>3</score>\n<justification>Off topic; ignores the request.</justification>"
    s, j = parse_judge_response(reg)
    assert s == 0.8 and j == "It quotes the cat. The cat sat."
    s, j = parse_judge_response(imm)
    assert s == 0.3 and j == "Off topic; ignores the request."


def test_parse_judge_response_score_tag_isolates_number():
    # A justification full of digits must not poison the score (which lives in <score>).
    text = "<justification>The reply has 12 words and 18 words, all wrong.</justification>\n<score>2</score>"
    s, _ = parse_judge_response(text)
    assert s == 0.2


def test_parse_judge_response_missing_score_raises():
    with pytest.raises(ValueError):
        parse_judge_response("<justification>no score tag here</justification>")


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
