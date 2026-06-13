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
class AlpacaProblem:
    """The source instruction from tatsu-lab/alpaca, with provenance back to the original split."""
    orig_index: int  # index into the alpaca `train` split, before our filter/shuffle/select
    request_id: int  # sequential id within our sampled subset (data/alpaca_requests.json order)
    request: str  # the alpaca instruction text shown to the actor


@dataclass(frozen=True)
class ConstraintSpec:
    """The formatting constraint imposed on top of the alpaca request."""
    name: ConstraintName
    difficulty: Difficulty


@dataclass(frozen=True)
class TaskInfo:
    """The per-example task spec serialized (asdict) into the verifiers `info` column.

    Nests the source alpaca problem and the imposed constraint as their own dataclasses
    rather than flattening every field to one level — so `info` stays self-describing
    (`info["alpaca"]["orig_index"]`, `info["constraint"]["name"]`)."""
    alpaca: AlpacaProblem
    constraint: ConstraintSpec


@dataclass(frozen=True)
class MsgProvenance:
    """Which turn emitted a message. Stored (as asdict) index-aligned with state['completion']."""
    turn_idx: int
    turn_name: TurnName
    source: Literal["actor", "env"]
