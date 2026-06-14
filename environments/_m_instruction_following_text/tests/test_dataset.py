from _m_instruction_following_text.dataset import build_problems, _sample_params_for
from _m_instruction_following_text.types import ConstraintName, Difficulty, TaskInfo


def test_params_deterministic_per_request_and_constraint():
    name = ConstraintName.ALTERNATING_XY
    p1 = _sample_params_for(name=name, request_id=7)
    p2 = _sample_params_for(name=name, request_id=7)
    assert p1 == p2          # same seed -> identical params (reproducible across actor runs / rescore)


def test_build_problems_attaches_params():
    reqs = [{"request_id": 0, "orig_index": 1, "request": "Explain photosynthesis."}]
    tasks = build_problems(reqs, n_requests=1, difficulties=(Difficulty.HARD,))
    names = {t.constraint.name for t in tasks}
    assert ConstraintName.ALTERNATING_XY in names
    for t in tasks:
        assert t.constraint.params is not None


def _requests():
    return [{"request_id": i, "orig_index": 100 + i, "request": f"req {i}"} for i in range(4)]


def test_build_problems_cross_product():
    tasks = build_problems(_requests(), n_requests=2, difficulties=(Difficulty.EASY,))
    assert len(tasks) == 2
    assert all(isinstance(t, TaskInfo) for t in tasks)
    assert all(t.constraint.name == ConstraintName.NO_CAPITALS for t in tasks)


def test_build_problems_all_difficulties():
    tasks = build_problems(_requests(), n_requests=3, difficulties=(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD))
    assert len(tasks) == 3 * 5


def test_build_problems_fields():
    tasks = build_problems(_requests(), n_requests=1, difficulties=(Difficulty.MEDIUM,))
    t = tasks[0]
    assert t.alpaca.request == "req 0"
    assert t.alpaca.orig_index == 100
    assert t.constraint.difficulty == Difficulty.MEDIUM
    assert t.constraint.name == ConstraintName.ALL_SENTENCES_T
