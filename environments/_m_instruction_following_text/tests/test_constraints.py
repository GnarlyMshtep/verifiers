import random

from _m_instruction_following_text.constraints import CONSTRAINTS, split_sentences
from _m_instruction_following_text.types import ConstraintName, ConstraintParams


def _params(name):
    return CONSTRAINTS[name].sample_params(random.Random(0))


def test_split_sentences_basic():
    assert split_sentences("Hi there. How are you? Good!") == ["Hi there.", "How are you?", "Good!"]


def test_no_capitals_pass():
    assert CONSTRAINTS[ConstraintName.NO_CAPITALS].verify("this is all lowercase, 123 ok.", ConstraintParams()).satisfied


def test_no_capitals_fail():
    assert not CONSTRAINTS[ConstraintName.NO_CAPITALS].verify("This has a Capital.", ConstraintParams()).satisfied


def test_all_sentences_t_pass_four():
    r = CONSTRAINTS[ConstraintName.ALL_SENTENCES_T].verify(
        "Tonight it sat. That cat ate it. tonight it bit. To get it out.", ConstraintParams(num_sentences=4))
    assert r.satisfied, r.detail


def test_all_sentences_t_fail_too_few():
    assert not CONSTRAINTS[ConstraintName.ALL_SENTENCES_T].verify(
        "Tonight it sat. That cat ate it. Trust it.", ConstraintParams(num_sentences=4)).satisfied


def test_all_sentences_t_fail_letter():
    assert not CONSTRAINTS[ConstraintName.ALL_SENTENCES_T].verify(
        "Tonight it sat. That cat ate it. Trust it. The dog ran.", ConstraintParams(num_sentences=4)).satisfied


def test_alternating_xy_sample_in_range_and_distinct():
    p = _params(ConstraintName.ALTERNATING_XY)
    assert 5 <= p.word_count_a <= 20 and 5 <= p.word_count_b <= 20
    assert p.word_count_a != p.word_count_b


def test_alternating_xy_pass():
    p = ConstraintParams(num_sentences=3, word_count_a=7, word_count_b=11)
    s7 = " ".join(["word"] * 7) + "."
    s11 = " ".join(["word"] * 11) + "."
    r = CONSTRAINTS[ConstraintName.ALTERNATING_XY].verify(f"{s7} {s11} {s7}", p)
    assert r.satisfied, r.detail


def test_alternating_xy_fail_wrong_count():
    p = ConstraintParams(num_sentences=3, word_count_a=7, word_count_b=11)
    s7 = " ".join(["word"] * 7) + "."
    s9 = " ".join(["word"] * 9) + "."
    # 3 sentences (right count) but the middle one is 9 words -> neither 7 nor 11.
    assert not CONSTRAINTS[ConstraintName.ALTERNATING_XY].verify(f"{s7} {s9} {s7}", p).satisfied


def test_alternating_xy_render_mentions_both_counts():
    p = ConstraintParams(num_sentences=3, word_count_a=7, word_count_b=11)
    instr = CONSTRAINTS[ConstraintName.ALTERNATING_XY].render_instruction(p)
    assert "7" in instr and "11" in instr


def test_letter_freq_diff_sample_fields():
    p = _params(ConstraintName.LETTER_FREQ_DIFF)
    assert p.freq_letter and p.freq_other and p.freq_letter != p.freq_other
    assert p.freq_delta >= 1


def test_letter_freq_diff_pass():
    p = ConstraintParams(num_sentences=3, freq_letter="e", freq_other="z", freq_delta=2)
    # each sentence: many e's, no z's
    txt = "Eleven geese feel eager. These trees seem green here."
    r = CONSTRAINTS[ConstraintName.LETTER_FREQ_DIFF].verify(txt + " Even better, every tree grew.", p)
    assert r.satisfied, r.detail


def test_letter_freq_diff_fail_one_sentence_short():
    p = ConstraintParams(num_sentences=3, freq_letter="e", freq_other="z", freq_delta=2)
    # second sentence has zero e-minus-z surplus
    txt = "Eleven geese feel eager. Zzz buzz jazz. Even better, every tree grew."
    assert not CONSTRAINTS[ConstraintName.LETTER_FREQ_DIFF].verify(txt, p).satisfied


def test_letter_freq_diff_fail_too_few_sentences():
    p = ConstraintParams(num_sentences=3, freq_letter="e", freq_other="z", freq_delta=2)
    assert not CONSTRAINTS[ConstraintName.LETTER_FREQ_DIFF].verify("Eee. Eee.", p).satisfied


def test_letter_set_sample_disjoint_nonempty():
    p = _params(ConstraintName.LETTER_SET)
    assert p.include_letters and p.exclude_letters
    assert not (set(p.include_letters) & set(p.exclude_letters))


def test_letter_set_pass():
    p = ConstraintParams(include_letters=["x", "z"], exclude_letters=["b", "p"])
    assert CONSTRAINTS[ConstraintName.LETTER_SET].verify("A lazy fox zigzags next door.", p).satisfied


def test_letter_set_fail_missing_include():
    p = ConstraintParams(include_letters=["x", "z"], exclude_letters=["b", "p"])
    assert not CONSTRAINTS[ConstraintName.LETTER_SET].verify("A lazy cat naps.", p).satisfied  # no x


def test_letter_set_fail_has_excluded():
    p = ConstraintParams(include_letters=["x", "z"], exclude_letters=["b", "p"])
    assert not CONSTRAINTS[ConstraintName.LETTER_SET].verify("A box of zebras by a pier.", p).satisfied  # b,p
