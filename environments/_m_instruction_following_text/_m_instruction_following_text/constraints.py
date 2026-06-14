from __future__ import annotations

import random
import re
import unicodedata

from .types import Constraint, ConstraintName, ConstraintParams, ConstraintResult, Difficulty

# A candidate sentence boundary: a run of .!? terminal punctuation followed by whitespace or end-of-string.
_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")

# Lowercased abbreviations (without trailing dot) that should NOT terminate a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "etc", "vs", "e.g", "i.e"}
)
# A dotted initialism like "U.S.A" (letters separated by dots, optional trailing letter).
_INITIALISM = re.compile(r"^([A-Za-z]\.)+[A-Za-z]?$")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation, dropping empties/whitespace.

    Splits on a `.!?` run followed by whitespace/EOS, but suppresses spurious splits caused by
    ellipses (multi-dot runs), title/latin abbreviations (e.g. ``Mr.``, ``etc.``), and dotted
    initialisms (e.g. ``U.S.A.``).
    """
    out: list[str] = []
    pos = 0  # start of the current sentence chunk
    for m in _SENTENCE_SPLIT.finditer(text):
        punct = m.group().strip()  # the .!? run without trailing whitespace
        # Ellipsis: a run of >=2 dots is non-terminal (mid-utterance pause), so don't split here.
        if len(punct) >= 2 and set(punct) == {"."}:
            continue
        # Single-dot boundary: suppress if the preceding word token is an abbreviation/initialism.
        if punct == ".":
            preceding = text[pos:m.start()].rsplit(None, 1)
            token = preceding[-1] if preceding else ""
            stripped = token.rstrip(".").lower()
            if stripped in _ABBREVIATIONS or _INITIALISM.match(token):
                continue
        chunk = text[pos:m.end()].strip()
        if chunk:
            out.append(chunk)
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append(tail)
    return out


def _alpha_only(s: str) -> str:
    return "".join(c for c in s if c.isalpha())


def _word_count(s: str) -> int:
    """Count whitespace-separated tokens that contain at least one alphabetic character."""
    return sum(1 for t in s.split() if any(c.isalpha() for c in t))


def _fold_accents(s: str) -> str:
    """NFKD-normalize and drop combining marks so accented letters (é) fold to their base (e)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Appended to every constraint so the actor cannot satisfy the formatting rule with gibberish:
# the answer must remain a genuine, coherent English response that actually fulfils the request.
_VALID_ENGLISH = (
    " Your answer must still consist of valid, grammatical English sentences that genuinely and "
    "completely fulfil the request — do NOT resort to gibberish, filler, or nonsense words to "
    "satisfy this formatting rule."
)

_SENTENCE_RANGE = (2, 5)  # tunable: every sentence-counting constraint samples num_sentences here


def _sample_num_sentences(rng: random.Random) -> int:
    return rng.randint(*_SENTENCE_RANGE)


def _sample_none(rng: random.Random) -> ConstraintParams:
    return ConstraintParams()


# --- no_capitals (control) ---


def _verify_no_capitals(text: str, p: ConstraintParams) -> ConstraintResult:
    caps = [c for c in text if c.isupper()]
    return ConstraintResult(
        satisfied=len(caps) == 0,
        detail="no uppercase letters" if not caps else f"found uppercase: {''.join(caps[:10])}",
    )


def _render_no_capitals(p: ConstraintParams) -> str:
    return "Write your entire answer using no capital letters whatsoever." + _VALID_ENGLISH


# --- all_sentences_t (control) ---
# all_sentences_t now samples its OWN exact sentence count (no fixed _MIN_SENTENCES_T):


def _sample_all_sentences_t(rng: random.Random) -> ConstraintParams:
    return ConstraintParams(num_sentences=_sample_num_sentences(rng))


def _render_all_sentences_t(p: ConstraintParams) -> str:
    return (
        f"Write your answer as exactly {p.num_sentences} sentences, where every sentence both starts "
        f"and ends with the letter 't' (case-insensitive — 't' or 'T' both count)." + _VALID_ENGLISH
    )


def _verify_all_sentences_t(text: str, p: ConstraintParams) -> ConstraintResult:
    sents = split_sentences(text)
    if len(sents) != p.num_sentences:
        return ConstraintResult(False, f"need exactly {p.num_sentences} sentences, got {len(sents)}")
    for i, s in enumerate(sents):
        a = _alpha_only(s)
        if not a:
            return ConstraintResult(False, f"sentence {i} has no letters")
        if a[0].lower() != "t" or a[-1].lower() != "t":
            return ConstraintResult(False, f"sentence {i} must start & end with 't' (any case): {s!r}")
    return ConstraintResult(True, f"all {len(sents)} sentences start & end with 't' (n={p.num_sentences})")


# --- alternating_xy ---


def _sample_alternating_xy(rng: random.Random) -> ConstraintParams:
    n = _sample_num_sentences(rng)
    a = rng.randint(5, 20)
    b = rng.randint(5, 20)
    while b == a:
        b = rng.randint(5, 20)
    return ConstraintParams(num_sentences=n, word_count_a=a, word_count_b=b)


def _render_alternating_xy(p: ConstraintParams) -> str:
    return (
        f"Write your answer as exactly {p.num_sentences} sentences whose lengths alternate between "
        f"exactly {p.word_count_a} words and exactly {p.word_count_b} words (the first sentence may "
        f"be either length)." + _VALID_ENGLISH
    )


def _verify_alternating_xy(text: str, p: ConstraintParams) -> ConstraintResult:
    a, b, n = p.word_count_a, p.word_count_b, p.num_sentences
    sents = split_sentences(text)
    if len(sents) != n:
        return ConstraintResult(False, f"need exactly {n} sentences, got {len(sents)}")
    counts = [_word_count(s) for s in sents]
    if any(c not in (a, b) for c in counts):
        return ConstraintResult(False, f"every sentence must be {a} or {b} words; got {counts}")
    for i in range(1, len(counts)):
        if counts[i] == counts[i - 1]:
            return ConstraintResult(False, f"counts must alternate; got {counts}")
    return ConstraintResult(True, f"alternating word counts {counts} (a={a}, b={b}, n={n})")


# --- letter_freq_diff ---
# English letters in descending frequency — a linguistic constant used to ORDER sampled letters,
# NOT a curated difficulty pool. Lower index = more common.
ENGLISH_FREQ_ORDER = "etaoinshrdlcumwfgypbvkjxqz"
_FREQ_RANK = {c: i for i, c in enumerate(ENGLISH_FREQ_ORDER)}
_FREQ_DELTA_RANGE = (2, 3)  # tunable by calibration smoke (Task 8.5)


def _sample_letter_freq_diff(rng: random.Random) -> ConstraintParams:
    y, z = rng.sample(ENGLISH_FREQ_ORDER, 2)
    if _FREQ_RANK[y] > _FREQ_RANK[z]:      # ensure y is the more-common letter (feasible)
        y, z = z, y
    return ConstraintParams(
        num_sentences=_sample_num_sentences(rng),
        freq_letter=y, freq_other=z, freq_delta=rng.randint(*_FREQ_DELTA_RANGE),
    )


def _render_letter_freq_diff(p: ConstraintParams) -> str:
    return (
        f"Write your answer as exactly {p.num_sentences} sentences. In EVERY sentence, the letter "
        f"'{p.freq_letter}' must appear at least {p.freq_delta} more times than the letter "
        f"'{p.freq_other}' (counting letters case-insensitively)." + _VALID_ENGLISH
    )


def _verify_letter_freq_diff(text: str, p: ConstraintParams) -> ConstraintResult:
    y, z, x, n = p.freq_letter.lower(), p.freq_other.lower(), p.freq_delta, p.num_sentences
    sents = split_sentences(text)
    if len(sents) != n:
        return ConstraintResult(False, f"need exactly {n} sentences, got {len(sents)}")
    for i, s in enumerate(sents):
        low = _fold_accents(s).lower()
        diff = low.count(y) - low.count(z)
        if diff < x:
            return ConstraintResult(False, f"sentence {i}: '{y}'-'{z}' surplus {diff} < {x}: {s!r}")
    return ConstraintResult(True, f"every sentence has >= {x} more '{y}' than '{z}' (n={n})")


# --- letter_set ---
# Rank windows into ENGLISH_FREQ_ORDER (0=most common). Tunable by calibration smoke (Task 8.5).
_INCLUDE_RANK_MIN = 18   # include letters drawn from the rarest tail: order[18:] ~ {p,b,v,k,j,x,q,z}
_EXCLUDE_RANK_MIN = 10   # exclude letters drawn from a mid-rare band: order[10:18] ~ {l,c,u,m,w,f,g,y}
_EXCLUDE_RANK_MAX = 17
_N_INCLUDE = 2
_N_EXCLUDE = 2


def _sample_letter_set(rng: random.Random) -> ConstraintParams:
    rare = list(ENGLISH_FREQ_ORDER[_INCLUDE_RANK_MIN:])
    mid = list(ENGLISH_FREQ_ORDER[_EXCLUDE_RANK_MIN:_EXCLUDE_RANK_MAX + 1])  # disjoint from `rare` by construction
    include = rng.sample(rare, _N_INCLUDE)
    # Requiring 'q' while forbidding 'u' is near-impossible in real English; drop 'u' from the
    # exclude pool whenever 'q' is required so the constraint stays feasible.
    if "q" in include:
        mid = [c for c in mid if c != "u"]
    exclude = rng.sample(mid, _N_EXCLUDE)
    return ConstraintParams(include_letters=include, exclude_letters=exclude)


def _render_letter_set(p: ConstraintParams) -> str:
    inc = ", ".join(f"'{c}'" for c in p.include_letters)
    exc = ", ".join(f"'{c}'" for c in p.exclude_letters)
    return (
        f"Write your answer so that it uses every one of these letters at least once: {inc}; and "
        f"does NOT use any of these letters at all: {exc} (case-insensitive)." + _VALID_ENGLISH
    )


def _verify_letter_set(text: str, p: ConstraintParams) -> ConstraintResult:
    present = {c.lower() for c in text if c.isalpha()}
    missing = [c for c in p.include_letters if c.lower() not in present]
    forbidden = [c for c in p.exclude_letters if c.lower() in present]
    if missing:
        return ConstraintResult(False, f"missing required letters: {missing}")
    if forbidden:
        return ConstraintResult(False, f"used forbidden letters: {forbidden}")
    return ConstraintResult(True, f"included {p.include_letters}, excluded {p.exclude_letters}")


CONSTRAINTS: dict[ConstraintName, Constraint] = {
    ConstraintName.NO_CAPITALS: Constraint(
        name=ConstraintName.NO_CAPITALS,
        difficulty=Difficulty.EASY,
        sample_params=_sample_none,
        render_instruction=_render_no_capitals,
        verify=_verify_no_capitals,
    ),
    ConstraintName.ALL_SENTENCES_T: Constraint(
        name=ConstraintName.ALL_SENTENCES_T,
        difficulty=Difficulty.MEDIUM,
        sample_params=_sample_all_sentences_t,
        render_instruction=_render_all_sentences_t,
        verify=_verify_all_sentences_t,
    ),
    ConstraintName.ALTERNATING_XY: Constraint(
        name=ConstraintName.ALTERNATING_XY,
        difficulty=Difficulty.HARD,
        sample_params=_sample_alternating_xy,
        render_instruction=_render_alternating_xy,
        verify=_verify_alternating_xy,
    ),
    ConstraintName.LETTER_FREQ_DIFF: Constraint(
        name=ConstraintName.LETTER_FREQ_DIFF,
        difficulty=Difficulty.HARD,
        sample_params=_sample_letter_freq_diff,
        render_instruction=_render_letter_freq_diff,
        verify=_verify_letter_freq_diff,
    ),
    ConstraintName.LETTER_SET: Constraint(
        name=ConstraintName.LETTER_SET,
        difficulty=Difficulty.MEDIUM,
        sample_params=_sample_letter_set,
        render_instruction=_render_letter_set,
        verify=_verify_letter_set,
    ),
}
