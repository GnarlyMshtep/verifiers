from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


class SceneName(StrEnum):
    TASK = "task"
    CONFESSION = "confession"  # scaffolded for later; unused in the pilot


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MonitorPrompt(StrEnum):
    """Which monitor/judge prompt template to use.

    REGULAR_REASONING: justification first, then score — the monitor may deliberate.
    IMMEDIATE_ANSWER: score first, then justification, answered immediately — paired with
    reasoning_effort='none' so the monitor commits to a score without thinking (a deliberately
    weaker, less accurate monitor)."""
    REGULAR_REASONING = "regular_reasoning_monitor"
    IMMEDIATE_ANSWER = "immediate_answer_monitor"


class ConstraintName(StrEnum):
    NO_CAPITALS = "no_capitals"
    ALL_SENTENCES_T = "all_sentences_t"
    ALTERNATING_XY = "alternating_xy"
    LETTER_FREQ_DIFF = "letter_freq_diff"
    LETTER_SET = "letter_set"


@dataclass(frozen=True)
class ConstraintResult:
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class ConstraintParams:
    """Per-example sampled parameters for a constraint. All fields optional; each constraint
    reads only the ones it needs. Controls (no_capitals, all_sentences_t) use the empty default.
    Stored in `info` (asdict) and restored via dacite — keep types dacite-friendly (no tuples;
    dacite's cast list holds only StrEnums, so letter sets are plain `list[str]`)."""
    # shared by sentence-counting constraints (all_sentences_t, alternating_xy, letter_freq_diff)
    num_sentences: int | None = None   # require EXACTLY this many sentences; sampled in [2,5]
    # alternating_xy
    word_count_a: int | None = None
    word_count_b: int | None = None
    # letter_freq_diff
    freq_letter: str | None = None   # y: the letter that must be more frequent
    freq_other: str | None = None    # z: the letter that must be less frequent
    freq_delta: int | None = None    # x: at least this many more y than z per sentence
    # letter_set
    include_letters: list[str] | None = None  # set A: every letter must appear >=1 time
    exclude_letters: list[str] | None = None  # set B: none of these letters may appear


@dataclass(frozen=True)
class Constraint:
    name: ConstraintName
    difficulty: Difficulty
    sample_params: Callable[["random.Random"], ConstraintParams]
    render_instruction: Callable[[ConstraintParams], str]
    verify: Callable[[str, ConstraintParams], ConstraintResult]


@dataclass(frozen=True)
class AlpacaProblem:
    """The source instruction from tatsu-lab/alpaca, with provenance back to the original split."""
    orig_index: int  # index into the alpaca `train` split, before our filter/shuffle/select
    request_id: int  # sequential id within our sampled subset (data/alpaca_requests.json order)
    request: str  # the alpaca instruction text shown to the actor


@dataclass(frozen=True)
class ConstraintSpec:
    """The formatting constraint imposed on top of the alpaca request, plus its sampled params."""
    name: ConstraintName
    difficulty: Difficulty
    params: ConstraintParams


@dataclass(frozen=True)
class TaskInfo:
    """The per-example task spec serialized (asdict) into the verifiers `info` column.

    Nests the source alpaca problem and the imposed constraint as their own dataclasses
    rather than flattening every field to one level — so `info` stays self-describing
    (`info["alpaca"]["orig_index"]`, `info["constraint"]["name"]`)."""
    alpaca: AlpacaProblem
    constraint: ConstraintSpec
