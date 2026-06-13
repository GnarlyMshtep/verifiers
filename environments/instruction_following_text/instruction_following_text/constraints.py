from __future__ import annotations

import re

from .types import Constraint, ConstraintName, ConstraintResult, Difficulty

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation, dropping empties/whitespace."""
    out: list[str] = []
    pos = 0
    for m in _SENTENCE_SPLIT.finditer(text):
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


def _verify_no_capitals(text: str) -> ConstraintResult:
    caps = [c for c in text if c.isupper()]
    return ConstraintResult(
        satisfied=len(caps) == 0,
        detail="no uppercase letters" if not caps else f"found uppercase: {''.join(caps[:10])}",
    )


_MIN_SENTENCES_T = 4


def _verify_all_sentences_t(text: str) -> ConstraintResult:
    sents = split_sentences(text)
    if len(sents) < _MIN_SENTENCES_T:
        return ConstraintResult(False, f"need >={_MIN_SENTENCES_T} sentences, got {len(sents)}")
    for i, s in enumerate(sents):
        a = _alpha_only(s)
        if not a:
            return ConstraintResult(False, f"sentence {i} has no letters")
        # Case-insensitive: a sentence may start/end with 't' or 'T'.
        if a[0].lower() != "t" or a[-1].lower() != "t":
            return ConstraintResult(False, f"sentence {i} must start & end with 't' (any case): {s!r}")
    return ConstraintResult(True, f"all {len(sents)} sentences start & end with 't' (case-insensitive)")


def _verify_alternating_12_18(text: str) -> ConstraintResult:
    sents = split_sentences(text)
    if len(sents) < 2:
        return ConstraintResult(False, f"need >=2 sentences, got {len(sents)}")
    counts = [len(s.split()) for s in sents]
    if any(c not in (12, 18) for c in counts):
        return ConstraintResult(False, f"every sentence must be 12 or 18 words; got {counts}")
    for i in range(1, len(counts)):
        if counts[i] == counts[i - 1]:
            return ConstraintResult(False, f"counts must alternate; got {counts}")
    return ConstraintResult(True, f"alternating word counts {counts}")


# Appended to every constraint so the actor cannot satisfy the formatting rule with gibberish:
# the answer must remain a genuine, coherent English response that actually fulfils the request.
_VALID_ENGLISH = (
    " Your answer must still consist of valid, grammatical English sentences that genuinely and "
    "completely fulfil the request — do NOT resort to gibberish, filler, or nonsense words to "
    "satisfy this formatting rule."
)


CONSTRAINTS: dict[ConstraintName, Constraint] = {
    ConstraintName.NO_CAPITALS: Constraint(
        name=ConstraintName.NO_CAPITALS,
        difficulty=Difficulty.EASY,
        instruction="Write your entire answer using no capital letters whatsoever." + _VALID_ENGLISH,
        verify=_verify_no_capitals,
    ),
    ConstraintName.ALL_SENTENCES_T: Constraint(
        name=ConstraintName.ALL_SENTENCES_T,
        difficulty=Difficulty.MEDIUM,
        instruction=(
            "Write your answer using at least 4 sentences, where every sentence both starts and "
            "ends with the letter 't' (case-insensitive — 't' or 'T' both count)."
        ) + _VALID_ENGLISH,
        verify=_verify_all_sentences_t,
    ),
    ConstraintName.ALTERNATING_12_18: Constraint(
        name=ConstraintName.ALTERNATING_12_18,
        difficulty=Difficulty.HARD,
        instruction=(
            "Write your answer so that consecutive sentences alternate in length between "
            "exactly 12 words and exactly 18 words." + _VALID_ENGLISH
        ),
        verify=_verify_alternating_12_18,
    ),
}
