from _m_instruction_following_text.dataset import build_dataset


def _keys(ds):
    return [(r["info"]["alpaca"]["request_id"], r["info"]["constraint"]["name"], r["question"]) for r in ds]


def test_start_index_zero_unchanged():
    a = build_dataset(None, difficulties=("easy", "medium", "hard"), n_samples=20, seed=0)
    b = build_dataset(None, difficulties=("easy", "medium", "hard"), n_samples=20, seed=0, start_index=0)
    assert _keys(a) == _keys(b)


def test_heldout_is_disjoint_slice_of_full():
    full = build_dataset(None, difficulties=("easy", "medium", "hard"), n_samples=30, seed=0)
    heldout = build_dataset(None, difficulties=("easy", "medium", "hard"), n_samples=10, seed=0, start_index=20)
    assert _keys(heldout) == _keys(full)[20:30]


def test_train_and_heldout_disjoint_by_index():
    train = build_dataset(None, difficulties=("easy", "medium", "hard"), n_samples=20, seed=0, start_index=0)
    heldout = build_dataset(None, difficulties=("easy", "medium", "hard"), n_samples=10, seed=0, start_index=20)
    # different global indices -> rows differ (at least the question/request rotation differs)
    assert _keys(train)[:10] != _keys(heldout)
