from __future__ import annotations

import re
from typing import Any

from openai import AsyncOpenAI
from verifiers.types import Messages, State

from .constraints import CONSTRAINTS
from .scorer_types import ConstraintScore, JudgeScore, Scorer, ScorerName
from .types import JudgeView, Problem


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
    """Build the text the judge may see, per JudgeView. Raises if CoT is requested but absent."""
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
    Rejects tokens adjacent to a word char or '.' so decimals/out-of-range raise."""
    ints = re.findall(r"(?<![\w.])(10|[0-9])(?![\w.])", text.strip())
    if not ints:
        raise ValueError(f"judge did not return a standalone 0-10 integer: {text!r}")
    return int(ints[-1]) / 10.0


class ConstraintScorer(Scorer):
    """Rule-based: 1.0 iff the final answer satisfies the problem's constraint."""

    name = ScorerName.CONSTRAINT

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    async def score(self, *, prompt, completion, answer, problem: Problem, state) -> ConstraintScore:
        result = CONSTRAINTS[problem.constraint].verify(last_assistant_text(completion))
        return ConstraintScore(
            score=1.0 if result.satisfied else 0.0,
            constraint=problem.constraint,
            satisfied=result.satisfied,
            detail=result.detail,
        )


class JudgeScorer(Scorer):
    """LLM judge scoring request execution in [0,1]. Retries on empty/unparseable response
    (gpt-oss flakiness); raises only after all attempts. Logs the full trace in `attempts`."""

    name = ScorerName.JUDGE_REQUEST_FOLLOWED

    def __init__(
        self,
        *,
        view: JudgeView,
        judge_client: AsyncOpenAI,
        judge_model: str,
        judge_prompt: str,
        judge_sampling_args: dict[str, Any] | None = None,
        judge_attempts: int = 3,
        weight: float = 1.0,
    ):
        self.view = view
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.sampling = judge_sampling_args or {}
        self.judge_attempts = judge_attempts
        self.weight = weight

    async def score(self, *, prompt, completion, answer, problem: Problem, state) -> JudgeScore:
        rendered = self.judge_prompt.format(
            question=problem.request, response=render_view(completion, self.view)
        )
        attempts: list[dict[str, Any]] = []
        last_raw = ""
        for _ in range(self.judge_attempts):
            resp = await self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": rendered}],
                **self.sampling,
            )
            msg = resp.choices[0].message
            raw = msg.content or ""
            attempts.append(resp.model_dump())
            try:
                return JudgeScore(
                    score=parse_score(raw),
                    view=self.view,
                    model=self.judge_model,
                    judge_input=rendered,
                    judge_output=raw,
                    # OpenRouter returns the field as `reasoning`; vLLM/others as `reasoning_content`.
                    judge_reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                    attempts=attempts,
                )
            except ValueError:
                last_raw = raw
        raise ValueError(
            f"judge returned no parseable 0-10 score after {self.judge_attempts} attempts; last={last_raw!r}"
        )
