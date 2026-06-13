from _m_instruction_following_text.dataset import build_problems
from _m_instruction_following_text.types import ConstraintName, Difficulty, TaskInfo


def _requests():
    return [{"request_id": i, "orig_index": 100 + i, "request": f"req {i}"} for i in range(4)]


def test_build_problems_cross_product():
    tasks = build_problems(_requests(), n_requests=2, difficulties=(Difficulty.EASY,))
    assert len(tasks) == 2
    assert all(isinstance(t, TaskInfo) for t in tasks)
    assert all(t.constraint.name == ConstraintName.NO_CAPITALS for t in tasks)


def test_build_problems_all_difficulties():
    tasks = build_problems(_requests(), n_requests=3, difficulties=(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD))
    assert len(tasks) == 3 * 3


def test_build_problems_fields():
    tasks = build_problems(_requests(), n_requests=1, difficulties=(Difficulty.MEDIUM,))
    t = tasks[0]
    assert t.alpaca.request == "req 0"
    assert t.alpaca.orig_index == 100
    assert t.constraint.difficulty == Difficulty.MEDIUM
    assert t.constraint.name == ConstraintName.ALL_SENTENCES_T
