from instruction_following_text.constraints import CONSTRAINTS, split_sentences


def test_split_sentences_basic():
    assert split_sentences("Hi there. How are you? Good!") == ["Hi there.", "How are you?", "Good!"]


def test_no_capitals_pass():
    assert CONSTRAINTS["no_capitals"].verify("this is all lowercase, 123 ok.").satisfied


def test_no_capitals_fail():
    assert not CONSTRAINTS["no_capitals"].verify("This has a Capital.").satisfied


def test_all_sentences_t_pass_two():
    r = CONSTRAINTS["all_sentences_t"].verify("Tonight it sat. That cat ate it.")
    assert r.satisfied, r.detail


def test_all_sentences_t_pass_one():
    r = CONSTRAINTS["all_sentences_t"].verify("Trust it.")
    assert r.satisfied, r.detail


def test_all_sentences_t_fail_letter():
    r = CONSTRAINTS["all_sentences_t"].verify("Tonight it sat. The dog ran.")
    assert not r.satisfied


def test_alternating_12_18_pass():
    s12 = " ".join(["word"] * 12) + "."
    s18 = " ".join(["word"] * 18) + "."
    assert CONSTRAINTS["alternating_12_18"].verify(f"{s12} {s18} {s12}").satisfied


def test_alternating_12_18_fail():
    s12 = " ".join(["word"] * 12) + "."
    s13 = " ".join(["word"] * 13) + "."
    assert not CONSTRAINTS["alternating_12_18"].verify(f"{s12} {s13}").satisfied
