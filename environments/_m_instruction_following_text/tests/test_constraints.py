from _m_instruction_following_text.constraints import CONSTRAINTS, split_sentences
from _m_instruction_following_text.types import ConstraintName


def test_split_sentences_basic():
    assert split_sentences("Hi there. How are you? Good!") == ["Hi there.", "How are you?", "Good!"]


def test_no_capitals_pass():
    assert CONSTRAINTS[ConstraintName.NO_CAPITALS].verify("this is all lowercase, 123 ok.").satisfied


def test_no_capitals_fail():
    assert not CONSTRAINTS[ConstraintName.NO_CAPITALS].verify("This has a Capital.").satisfied


def test_all_sentences_t_pass_four():
    # Four sentences; mixed case starts ('Tonight'/'tonight') exercise case-insensitivity.
    r = CONSTRAINTS[ConstraintName.ALL_SENTENCES_T].verify(
        "Tonight it sat. That cat ate it. tonight it bit. To get it out."
    )
    assert r.satisfied, r.detail


def test_all_sentences_t_fail_too_few():
    r = CONSTRAINTS[ConstraintName.ALL_SENTENCES_T].verify("Tonight it sat. That cat ate it. Trust it.")
    assert not r.satisfied


def test_all_sentences_t_fail_letter():
    r = CONSTRAINTS[ConstraintName.ALL_SENTENCES_T].verify(
        "Tonight it sat. That cat ate it. Trust it. The dog ran."
    )
    assert not r.satisfied


def test_alternating_12_18_pass():
    s12 = " ".join(["word"] * 12) + "."
    s18 = " ".join(["word"] * 18) + "."
    assert CONSTRAINTS[ConstraintName.ALTERNATING_12_18].verify(f"{s12} {s18} {s12}").satisfied


def test_alternating_12_18_fail():
    s12 = " ".join(["word"] * 12) + "."
    s13 = " ".join(["word"] * 13) + "."
    assert not CONSTRAINTS[ConstraintName.ALTERNATING_12_18].verify(f"{s12} {s13}").satisfied
