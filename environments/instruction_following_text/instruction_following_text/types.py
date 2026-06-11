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
    OUTPUT = "output"   # final content only
    COT = "cot"         # reasoning_content only
    BOTH = "both"       # both, labeled


@dataclass(frozen=True)
class ConstraintResult:
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class Constraint:
    name: str
    difficulty: Difficulty
    instruction: str
    verify: Callable[[str], ConstraintResult]


@dataclass(frozen=True)
class Task:
    request_id: int
    request: str
    constraint_name: str
    difficulty: Difficulty


@dataclass(frozen=True)
class MsgProvenance:
    """Which turn emitted a message. Stored (as asdict) index-aligned with state['completion']."""
    turn_idx: int
    turn_name: TurnName
    source: Literal["actor", "env"]
