from instruction_following_text.dataset import build_tasks
from instruction_following_text.types import Difficulty


def _requests():
    return [{"request_id": i, "request": f"req {i}"} for i in range(4)]


def test_build_tasks_cross_product():
    tasks = build_tasks(_requests(), n_requests=2, difficulties=(Difficulty.EASY,))
    assert len(tasks) == 2
    assert all(t.constraint_name == "no_capitals" for t in tasks)


def test_build_tasks_all_difficulties():
    tasks = build_tasks(_requests(), n_requests=3, difficulties=(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD))
    assert len(tasks) == 3 * 3


def test_build_tasks_info_fields():
    tasks = build_tasks(_requests(), n_requests=1, difficulties=(Difficulty.MEDIUM,))
    t = tasks[0]
    assert t.request == "req 0"
    assert t.difficulty == Difficulty.MEDIUM
    assert t.constraint_name == "all_sentences_t"
