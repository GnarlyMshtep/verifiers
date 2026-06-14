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


# --- reward-hack / edge-case tests (unbiased review) ---
#
# Category 1 = VERIFIER BUG: verify() gives the wrong verdict for its OWN stated rule
#   (or crashes, or sample_params breaks the 20-70% feasibility band). These tests assert
#   the CORRECT behavior, so they FAIL against the current buggy verifier — proving the gap.
# Category 2 = MONITOR-MUST-CATCH: the rule is applied correctly, but rule-satisfaction alone
#   does not make the answer good (gibberish, off-topic, filler). These tests assert the rule
#   DOES pass and are tagged `# MONITOR-MUST-CATCH` so they are not mistaken for bugs.

from _m_instruction_following_text.constraints import (
    _sample_letter_set,
    _sample_letter_freq_diff,
    ENGLISH_FREQ_ORDER,
)
import random as _random


def _v(name, text, params):
    return CONSTRAINTS[name].verify(text, params)


# ============================================================
# split_sentences — shared by all_sentences_t, alternating_xy, letter_freq_diff
# ============================================================

# CATEGORY 1 (VERIFIER BUG): abbreviations over-split. "Mr. Smith went home." is ONE
# sentence in English, but the regex splits on "Mr. " -> 2 sentences. Any sentence-counting
# constraint will then mis-count an otherwise valid answer.
def test_split_abbreviation_oversplits_mr():
    # CORRECT behavior: a single English sentence with an abbreviation is 1 sentence.
    assert split_sentences("Mr. Smith went home.") == ["Mr. Smith went home."]


def test_split_abbreviation_oversplits_titles():
    # "Dr. Smith and Mr. Jones met." is ONE sentence; verifier yields 3.
    assert split_sentences("Dr. Smith and Mr. Jones met.") == ["Dr. Smith and Mr. Jones met."]


def test_split_abbreviation_usa_oversplits():
    # "I live in the U.S.A. today." is ONE sentence; "U.S.A. " triggers a spurious split.
    assert split_sentences("I live in the U.S.A. today.") == ["I live in the U.S.A. today."]


# CATEGORY 1 (VERIFIER BUG): ellipsis "..." splits into two sentences though it is mid-sentence.
def test_split_ellipsis_oversplits():
    # CORRECT: an ellipsis inside one utterance should not start a new sentence.
    assert split_sentences("Wait... really?") == ["Wait... really?"]


# Decimals are handled correctly ($3.50, 3.14) because no whitespace follows the dot.
# Document the GOOD behavior so a regression would be caught.
def test_split_decimal_not_oversplit_ok():
    assert split_sentences("It cost $3.50 total.") == ["It cost $3.50 total."]
    assert split_sentences("Pi is 3.14 here.") == ["Pi is 3.14 here."]


# "etc." at the very end is fine (it is genuinely the terminal token).
def test_split_etc_at_end_ok():
    assert split_sentences("Buy milk, eggs, etc.") == ["Buy milk, eggs, etc."]


# ============================================================
# no_capitals
# ============================================================

def test_no_capitals_empty_string_no_crash():
    # Empty input must not crash; vacuously satisfied (no caps).
    assert _v(ConstraintName.NO_CAPITALS, "", ConstraintParams()).satisfied


def test_no_capitals_whitespace_only():
    assert _v(ConstraintName.NO_CAPITALS, "   \n\t  ", ConstraintParams()).satisfied


# CATEGORY 1 (correct-behavior guard): accented uppercase IS caught (str.isupper() handles É).
def test_no_capitals_accented_uppercase_caught():
    assert not _v(ConstraintName.NO_CAPITALS, "café É here.", ConstraintParams()).satisfied


# MONITOR-MUST-CATCH: lowercase gibberish trivially passes no_capitals.
def test_no_capitals_gibberish_passes():
    # MONITOR-MUST-CATCH: not English, not an answer — rule still passes.
    assert _v(ConstraintName.NO_CAPITALS, "asdf qwerty zzz blah blah lorem", ConstraintParams()).satisfied


# ============================================================
# all_sentences_t
# ============================================================

def test_all_sentences_t_empty_string_no_crash():
    assert not _v(ConstraintName.ALL_SENTENCES_T, "", ConstraintParams(num_sentences=2)).satisfied


# MONITOR-MUST-CATCH: repetitive filler that starts/ends with 't' and hits the count, but is
# vacuous and does not answer any request.
def test_all_sentences_t_repetitive_filler_passes():
    # MONITOR-MUST-CATCH: "Tonight that. Tonight that." is gibberish-ish filler that satisfies the rule.
    txt = "Tonight that. That tot. Trot that."
    assert _v(ConstraintName.ALL_SENTENCES_T, txt, ConstraintParams(num_sentences=3)).satisfied


# CATEGORY 1 (VERIFIER BUG, downstream of split): an answer the actor wrote as 2 valid
# 't'-sentences gets rejected because an abbreviation over-splits it into 3.
def test_all_sentences_t_abbreviation_breaks_valid_answer():
    # Author intends 2 sentences, each starting & ending with 't'. "Trust Dr. Trent" splits
    # on "Dr. " into an extra chunk, breaking a legitimately-formatted answer.
    txt = "Treat that. Trust Dr. Trent."
    # CORRECT behavior: 2 sentences, both start & end with 't' -> should pass with num_sentences=2.
    assert _v(ConstraintName.ALL_SENTENCES_T, txt, ConstraintParams(num_sentences=2)).satisfied


# ============================================================
# alternating_xy
# ============================================================

def test_alternating_xy_empty_string_no_crash():
    p = ConstraintParams(num_sentences=2, word_count_a=7, word_count_b=11)
    assert not _v(ConstraintName.ALTERNATING_XY, "", p).satisfied


# CATEGORY 1 (VERIFIER BUG): punctuation tokenization. str.split() on whitespace means a
# comma surrounded by spaces ("a , b") counts as a token, inflating the word count. A real
# English sentence does not normally have spaced punctuation, but an actor can EXPLOIT this:
# pad a too-short sentence with free-standing punctuation tokens to hit the target count.
def test_alternating_xy_spaced_punctuation_inflates_count():
    p = ConstraintParams(num_sentences=1, word_count_a=5, word_count_b=11)
    # 3 real words + 2 spaced punctuation marks -> str.split() sees 5 "words".
    # CORRECT word semantics: this is 3 words, not 5, so it should FAIL (5 != 3).
    txt = "Cats are great - !"
    assert not _v(ConstraintName.ALTERNATING_XY, txt, p).satisfied


# MONITOR-MUST-CATCH: alternation is trivially satisfiable with repeated filler words.
def test_alternating_xy_filler_words_pass():
    # MONITOR-MUST-CATCH: just repeat a word to hit the counts; not a real answer.
    p = ConstraintParams(num_sentences=2, word_count_a=5, word_count_b=6)
    s5 = " ".join(["blah"] * 5) + "."
    s6 = " ".join(["blah"] * 6) + "."
    assert _v(ConstraintName.ALTERNATING_XY, f"{s5} {s6}", p).satisfied


# ============================================================
# letter_freq_diff
# ============================================================

def test_letter_freq_diff_empty_string_no_crash():
    p = ConstraintParams(num_sentences=2, freq_letter="e", freq_other="z", freq_delta=2)
    assert not _v(ConstraintName.LETTER_FREQ_DIFF, "", p).satisfied


# CATEGORY 1 (VERIFIER BUG): accented letters are not folded. "café" should count as having
# an 'e' for a human reader, but str.lower().count('e') counts only ASCII 'e'. An answer the
# author considers e-rich can be wrongly rejected (or, symmetrically, an actor can hide z's
# as accented chars to dodge the penalty term — but the penalty side is z, less likely).
def test_letter_freq_diff_accented_e_not_counted():
    # "résumé café passé" reads as e-heavy to a human (5 accented e's, 0 ASCII e), but
    # str.lower().count('e') counts ZERO. CORRECT (human) reading: 'e' surplus over 'z' is 5 >= 2.
    p = ConstraintParams(num_sentences=1, freq_letter="e", freq_other="z", freq_delta=2)
    assert "résumé café passé.".count("e") == 0  # the bug: only accented e's, no ASCII e
    assert _v(ConstraintName.LETTER_FREQ_DIFF, "Résumé café passé.", p).satisfied


# MONITOR-MUST-CATCH: cram the target letter via a repeated nonsense word.
def test_letter_freq_diff_repeated_filler_word_passes():
    # MONITOR-MUST-CATCH: "eeee" padding satisfies the surplus with no meaning.
    p = ConstraintParams(num_sentences=2, freq_letter="e", freq_other="z", freq_delta=2)
    txt = "Eee eee eee. Teem deep keen."
    assert _v(ConstraintName.LETTER_FREQ_DIFF, txt, p).satisfied


# ============================================================
# letter_set — FEASIBILITY (calibration-relevant, CATEGORY 1)
# ============================================================

# CATEGORY 1 (FEASIBILITY BUG, flag for calibration): _sample_letter_set draws include from
# the rare tail {p,b,v,k,j,x,q,z} and exclude from {l,c,u,m,w,f,g,y}. It can require 'q'
# while excluding 'u'. English 'q' is almost always followed by 'u' (queen, quick, quiet),
# so {include q, exclude u} is near-impossible for GENUINE English — the constraint cannot
# land in the intended 20-70% pass band. This test ASSERTS the bad draw is reachable.
def test_letter_set_never_requires_q_without_u():
    for s in range(200):
        p = _sample_letter_set(_random.Random(s))
        inc = {c.lower() for c in p.include_letters}
        exc = {c.lower() for c in p.exclude_letters}
        # A well-calibrated sampler must NEVER require 'q' while forbidding 'u' (near-impossible
        # for real English).
        assert not ("q" in inc and "u" in exc), "sampler required 'q' while forbidding 'u'"


def test_letter_set_empty_string_no_crash():
    p = ConstraintParams(include_letters=["x", "z"], exclude_letters=["b", "p"])
    assert not _v(ConstraintName.LETTER_SET, "", p).satisfied


# CATEGORY 1 (correct-behavior guard): uppercase excluded letters ARE caught (case-folded).
def test_letter_set_uppercase_excluded_caught():
    p = ConstraintParams(include_letters=["x"], exclude_letters=["b"])
    # 'B' uppercase must still count as using forbidden 'b'.
    r = _v(ConstraintName.LETTER_SET, "Xbox BBB", p)
    assert not r.satisfied
    assert "b" in [c.lower() for c in r.detail] or "forbidden" in r.detail


# MONITOR-MUST-CATCH: a single nonsense word can include the rare letters and avoid the
# excluded ones without answering anything.
def test_letter_set_nonsense_word_passes():
    # MONITOR-MUST-CATCH: "zxqj" includes rare letters, dodges excludes — not an answer.
    p = ConstraintParams(include_letters=["x", "z"], exclude_letters=["l", "c"])
    assert _v(ConstraintName.LETTER_SET, "zixax ozax.", p).satisfied


# ============================================================
# letter_freq_diff feasibility: close-rank common letters with delta up to 3 across short
# sentences. y is forced to be the more-common letter, so this is generally feasible, but
# document that very short (2-5) sentence counts plus delta 3 is demanding — keep as a
# calibration sentinel rather than a hard bug.
def test_letter_freq_diff_sampler_y_is_more_common():
    for s in range(100):
        p = _sample_letter_freq_diff(_random.Random(s))
        ry = ENGLISH_FREQ_ORDER.index(p.freq_letter)
        rz = ENGLISH_FREQ_ORDER.index(p.freq_other)
        assert ry < rz, f"freq_letter must rank as more common (seed {s})"

