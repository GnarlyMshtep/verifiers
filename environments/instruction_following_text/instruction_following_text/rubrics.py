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
    """Parse the judge's 0-10 integer (last standalone integer) into a [0,1] float.
    Rejects tokens adjacent to a word char or '.' so decimals (7.5) and out-of-range
    numbers (11, 100) raise instead of silently mis-scoring."""
    ints = re.findall(r"(?<![\w.])(10|[0-9])(?![\w.])", text.strip())
    if not ints:
        raise ValueError(f"judge did not return a standalone 0-10 integer: {text!r}")
    return int(ints[-1]) / 10.0


async def constraint_satisfied(prompt, completion, answer, state, **kwargs) -> float:
    """Rule-based reward: 1.0 iff the response satisfies the task's constraint.
    Reads the constraint name from state['info']; logs the detailed result to state."""
    name = state["info"]["constraint"]
    text = last_assistant_text(completion)
    result = CONSTRAINTS[name].verify(text)
    state.setdefault("constraint_results", {})[name] = asdict(result)
    return 1.0 if result.satisfied else 0.0


def make_judge_reward(
    *,
    view: JudgeView,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_prompt: str,
    judge_sampling_args: dict[str, Any] | None = None,
    judge_attempts: int = 3,
) -> RewardFunc:
    """Build a reward fn that asks an LLM judge to score request execution in [0,1].
    Logs the full judge input/output (every attempt) to state['judge_raw'].

    Reasoning judges (e.g. gpt-oss via OpenRouter) intermittently return an empty
    completion (finish_reason=stop, no content). We retry up to `judge_attempts` times
    on an empty/unparseable response and raise only if all attempts fail — intentional
    fault tolerance, never a silent default score."""
    sampling = judge_sampling_args or {}

    async def judge_request_followed(prompt, completion, answer, state, **kwargs) -> float:
        question = state["info"]["request"]
        response_text = render_view(completion, view)
        rendered = judge_prompt.format(question=question, response=response_text)
        last_raw = ""
        for attempt in range(judge_attempts):
            resp = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": rendered}],
                **sampling,
            )
            msg = resp.choices[0].message
            raw = msg.content or ""
            state.setdefault("judge_raw", []).append(
                {
                    "view": str(view),
                    "model": judge_model,
                    "attempt": attempt,
                    "prompt": rendered,
                    "response": raw,
                    "reasoning": getattr(msg, "reasoning_content", None),
                    "full_response": resp.model_dump(),
                }
            )
            try:
                return parse_score(raw)
            except ValueError:
                last_raw = raw
        raise ValueError(
            f"judge returned no parseable 0-10 score after {judge_attempts} attempts; last={last_raw!r}"
        )

    judge_request_followed.__name__ = f"judge_request_followed__{view.value}"
    return judge_request_followed
