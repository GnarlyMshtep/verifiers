from instruction_following_text.dataset import build_problems
from instruction_following_text.types import ConstraintName, Difficulty, Problem


def _requests():
    return [{"request_id": i, "request": f"req {i}"} for i in range(4)]


def test_build_problems_cross_product():
    problems = build_problems(_requests(), n_requests=2, difficulties=(Difficulty.EASY,))
    assert len(problems) == 2
    assert all(isinstance(p, Problem) for p in problems)
    assert all(p.constraint == ConstraintName.NO_CAPITALS for p in problems)


def test_build_problems_all_difficulties():
    problems = build_problems(_requests(), n_requests=3, difficulties=(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD))
    assert len(problems) == 3 * 3


def test_build_problems_fields():
    problems = build_problems(_requests(), n_requests=1, difficulties=(Difficulty.MEDIUM,))
    p = problems[0]
    assert p.request == "req 0"
    assert p.difficulty == Difficulty.MEDIUM
    assert p.constraint == ConstraintName.ALL_SENTENCES_T
