from _m_instruction_following_text.dataset import build_problems_random
from _m_instruction_following_text.types import Difficulty, TaskInfo


def _requests(n: int = 4) -> list[dict]:
    return [{"request_id": i, "orig_index": 100 + i, "request": f"req {i}"} for i in range(n)]


def test_random_returns_exactly_n_samples():
    tasks = build_problems_random(_requests(), n_samples=37, difficulties=(Difficulty.EASY, Difficulty.MEDIUM), seed=0)
    assert len(tasks) == 37
    assert all(isinstance(t, TaskInfo) for t in tasks)


def test_random_constraints_within_requested_difficulties():
    diffs = (Difficulty.EASY, Difficulty.HARD)
    tasks = build_problems_random(_requests(), n_samples=50, difficulties=diffs, seed=1)
    assert all(t.constraint.difficulty in diffs for t in tasks)


def test_random_seed_reproducible_and_seed_sensitive():
    diffs = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
    seq_a = [t.constraint.name for t in build_problems_random(_requests(), n_samples=60, difficulties=diffs, seed=0)]
    seq_b = [t.constraint.name for t in build_problems_random(_requests(), n_samples=60, difficulties=diffs, seed=0)]
    seq_c = [t.constraint.name for t in build_problems_random(_requests(), n_samples=60, difficulties=diffs, seed=1)]
    assert seq_a == seq_b  # same seed -> identical constraint sequence
    assert seq_a != seq_c  # different seed -> (very likely) different


def test_random_requests_cycle_when_n_samples_exceeds_requests():
    reqs = _requests(3)
    tasks = build_problems_random(reqs, n_samples=7, difficulties=(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD), seed=0)
    request_ids = [t.alpaca.request_id for t in tasks]
    assert request_ids == [0, 1, 2, 0, 1, 2, 0]
