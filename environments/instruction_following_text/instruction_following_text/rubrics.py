from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from openai import AsyncOpenAI
from verifiers.types import Messages, RewardFunc

from .constraints import CONSTRAINTS
from .types import JudgeView


def _msg_field(m: Any, key: str) -> Any:
    return m.get(key) if isinstance(m, dict) else getattr(m, key, None)


def assistant_messages(completion: Messages) -> list[Any]:
    return [m for m in completion if _msg_field(m, "role") == "assistant"]


def last_assistant_text(completion: Messages) -> str:
    msgs = assistant_messages(completion)
    if not msgs:
        raise ValueError("no assistant message in completion")
    return _msg_field(msgs[-1], "content") or ""


def render_view(completion: Messages, view: JudgeView) -> str:
    """Build the text the judge may see, per JudgeView. Raises if CoT is requested but
    absent (no silent fallback)."""
    msgs = assistant_messages(completion)
    if not msgs:
        raise ValueError("no assistant message to render for judge")
    parts: list[str] = []
    multi = len(msgs) > 1
    for i, m in enumerate(msgs):
        content = _msg_field(m, "content") or ""
        reasoning = _msg_field(m, "reasoning_content")
        tag = f"[message {i}] " if multi else ""
        if view in (JudgeView.OUTPUT, JudgeView.BOTH):
            parts.append(f"{tag}OUTPUT:\n{content}" if view == JudgeView.BOTH else content)
        if view in (JudgeView.COT, JudgeView.BOTH):
            if not reasoning:
                raise ValueError(
                    f"JudgeView {view} requested CoT but assistant message {i} has no reasoning_content"
                )
            parts.append(f"{tag}REASONING:\n{reasoning}" if view == JudgeView.BOTH else reasoning)
    return "\n\n".join(parts)


def parse_score(text: str) -> float:
    """Parse the judge's 0-10 integer (last integer in the text) into a [0,1] float."""
    ints = re.findall(r"\b(10|[0-9])\b", text.strip())
    if not ints:
        raise ValueError(f"judge did not return a 0-10 integer: {text!r}")
    return int(ints[-1]) / 10.0


async def constraint_satisfied(prompt, completion, answer, state, **kwargs) -> float:
    """Rule-based reward: 1.0 iff the response satisfies the task's constraint.
    Reads the constraint name from state['info']; logs the detailed result to state."""
    name = state["info"]["constraint"]
    text = last_assistant_text(completion)
    result = CONSTRAINTS[name].verify(text)
    state.setdefault("constraint_results", {})[name] = asdict(result)
    return 1.0 if result.satisfied else 0.0
