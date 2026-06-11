from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Literal


class TurnName(StrEnum):
    TASK = "task"
    CONFESSION = "confession"  # scaffolded for later; unused in the pilot


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class JudgeView(StrEnum):
    """What the judge is allowed to see of the actor's response(s)."""
    OUTPUT = "output"
    COT = "cot"
    BOTH = "both"


class ConstraintName(StrEnum):
    NO_CAPITALS = "no_capitals"
    ALL_SENTENCES_T = "all_sentences_t"
    ALTERNATING_12_18 = "alternating_12_18"


@dataclass(frozen=True)
class ConstraintResult:
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class Constraint:
    name: ConstraintName
    difficulty: Difficulty
    instruction: str
    verify: Callable[[str], ConstraintResult]


@dataclass(frozen=True)
class Problem:
    """The per-example task spec. Serialized into the verifiers `info` column."""
    request: str
    constraint: ConstraintName
    difficulty: Difficulty
    request_id: int


@dataclass(frozen=True)
class MsgProvenance:
    """Which turn emitted a message. Stored (as asdict) index-aligned with state['completion']."""
    turn_idx: int
    turn_name: TurnName
    source: Literal["actor", "env"]
