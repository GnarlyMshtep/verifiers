import pytest

from instruction_following_text.rubrics import parse_score, render_view
from instruction_following_text.types import JudgeView


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


def test_parse_score_no_int_raises():
    with pytest.raises(ValueError):
        parse_score("excellent")
