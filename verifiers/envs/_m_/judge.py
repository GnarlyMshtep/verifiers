from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from openai import AsyncOpenAI
from verifiers.types import Messages

from .scoring import Scorer, ScorerResult
from .utils import strip_think


class JudgeView(StrEnum):
    """What the judge is allowed to see of the actor's response(s)."""
    OUTPUT = "output"
    COT = "cot"
    BOTH = "both"


class JudgeUnparseableError(ValueError):
    """Raised when the judge returns no parseable score after all attempts. Carries the full
    per-attempt provider responses so the caller can persist the trace before failing loud."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def _msg_field(m: Any, key: str) -> Any:
    return m.get(key) if isinstance(m, dict) else getattr(m, key, None)


def assistant_messages(completion: Messages) -> list[Any]:
    return [m for m in completion if _msg_field(m, "role") == "assistant"]


def last_assistant_text(completion: Messages) -> str:
    msgs = assistant_messages(completion)
    if not msgs:
        raise ValueError("no assistant message in completion")
    return _msg_field(msgs[-1], "content") or ""


def _role_text(messages: Messages, role: str, last: bool = False) -> str:
    """First (or last) message content for `role` in `messages`, or '' if none."""
    msgs = [m for m in messages if _msg_field(m, "role") == role]
    if not msgs:
        return ""
    return _msg_field(msgs[-1] if last else msgs[0], "content") or ""


def render_view(completion: Messages, view: JudgeView) -> str:
    """Render the actor's turns the judge may see, per JudgeView, as nested XML.

    Each assistant message becomes one `<assistant-turn>`; the view selects which inner
    tags appear (`<assistant-reasoning-not-delivered-to-user>` for CoT, `<assistant-output>`
    for the delivered output). Raises if CoT is requested but absent."""
    msgs = assistant_messages(completion)
    if not msgs:
        raise ValueError("no assistant message to render for judge")
    turns: list[str] = []
    for i, m in enumerate(msgs):
        content = _msg_field(m, "content") or ""
        reasoning = _msg_field(m, "reasoning_content")
        inner: list[str] = []
        if view in (JudgeView.COT, JudgeView.BOTH):
            if not reasoning:
                raise ValueError(
                    f"JudgeView {view} requested CoT but assistant message {i} has no reasoning_content"
                )
            inner.append(
                "<assistant-reasoning-not-delivered-to-user>\n"
                f"{reasoning}\n"
                "</assistant-reasoning-not-delivered-to-user>"
            )
        if view in (JudgeView.OUTPUT, JudgeView.BOTH):
            # The judge/monitor grades the delivered answer, not the inline <think> reasoning.
            inner.append(f"<assistant-output>\n{strip_think(content)}\n</assistant-output>")
        turns.append("<assistant-turn>\n" + "\n".join(inner) + "\n</assistant-turn>")
    return "\n".join(turns)


def render_transcript(messages: Messages, *, include_cot: bool = False) -> str:
    """Render an entire conversation of any length/roles as nested XML, one <turn> per message.

    Generic (not specialized to any env). Assistant CoT (`reasoning_content`) is included only when
    `include_cot=True`; otherwise just the delivered `content` is shown."""
    turns: list[str] = []
    for m in messages:
        role = _msg_field(m, "role") or "unknown"
        content = _msg_field(m, "content") or ""
        inner: list[str] = []
        if include_cot:
            reasoning = _msg_field(m, "reasoning_content")
            if reasoning:
                inner.append(
                    "<reasoning-not-delivered-to-user>\n"
                    f"{reasoning}\n"
                    "</reasoning-not-delivered-to-user>"
                )
        inner.append(content if include_cot else strip_think(content))
        turns.append(f'<turn role="{role}">\n' + "\n".join(inner) + f"\n</turn>")
    return "\n".join(turns)


def parse_score(text: str) -> float:
    """Parse the judge's 0-10 integer (last standalone integer) into a [0,1] float.
    Rejects tokens adjacent to a word char or '.' so decimals/out-of-range raise."""
    ints = re.findall(r"(?<![\w.])(10|[0-9])(?![\w.])", text.strip())
    if not ints:
        raise ValueError(f"judge did not return a standalone 0-10 integer: {text!r}")
    return int(ints[-1]) / 10.0


def parse_judge_response(text: str) -> tuple[float, str | None]:
    """Extract the judge's `<score>` (0-10 integer -> [0,1]) and optional `<justification>`.

    The score is parsed from inside the `<score>` tag only, so a chatty justification (which
    may quote numbers from the response) cannot poison it. Raises if no parseable `<score>`
    tag is present, so the caller can retry."""
    sm = re.search(r"<score>(.*?)</score>", text, re.DOTALL | re.IGNORECASE)
    if not sm:
        raise ValueError(f"judge response missing a parseable <score>...</score>: {text!r}")
    score = parse_score(sm.group(1))
    jm = re.search(r"<justification>(.*?)</justification>", text, re.DOTALL | re.IGNORECASE)
    justification = jm.group(1).strip() if jm else None
    return score, justification


@dataclass
class JudgeScore(ScorerResult):
    view: JudgeView
    model: str
    judge_input: str
    judge_output: str
    judge_reasoning: str | None
    justification: str | None
    attempts: list[dict[str, Any]]
    name: str = "judge"


class JudgeScorer(Scorer):
    """LLM judge scoring request execution in [0,1]. Retries on empty/unparseable response
    (gpt-oss flakiness); raises only after all attempts. Logs the full trace in `attempts`."""

    def __init__(
        self,
        *,
        name: str,
        view: JudgeView,
        judge_client: AsyncOpenAI,
        judge_model: str,
        judge_prompt_fn: Callable[..., str],
        judge_sampling_args: dict[str, Any] | None = None,
        judge_attempts: int = 3,
        weight: float = 1.0,
    ):
        self.name = name
        self.view = view
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_prompt_fn = judge_prompt_fn
        self.sampling = judge_sampling_args or {}
        self.judge_attempts = judge_attempts
        self.weight = weight

    def build_judge_prompt(self, *, prompt: Messages, completion: Messages) -> str:
        """Build the rendered judge prompt. Default: the actor's system prompt + the FULL user
        request + the actor's turns rendered per JudgeView. Override for a different prompt shape."""
        # The judge sees the actor's system prompt + the FULL user request (incl. the formatting
        # requirement, the last user message) — taken from the actual actor prompt, never trimmed.
        return self.judge_prompt_fn(
            system=_role_text(prompt, "system"),
            question=_role_text(prompt, "user", last=True),
            assistant_turns=render_view(completion, self.view),
        )

    async def score(self, *, prompt, completion, answer, task_info: object, state) -> JudgeScore:
        rendered = self.build_judge_prompt(prompt=prompt, completion=completion)
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
                score, justification = parse_judge_response(raw)
                return JudgeScore(
                    score=score,
                    view=self.view,
                    model=self.judge_model,
                    judge_input=rendered,
                    judge_output=raw,
                    # OpenRouter returns the field as `reasoning`; vLLM/others as `reasoning_content`.
                    judge_reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                    justification=justification,
                    attempts=attempts,
                    name=self.name,
                )
            except ValueError:
                last_raw = raw
        raise JudgeUnparseableError(
            f"judge returned no parseable 0-10 score after {self.judge_attempts} attempts; last={last_raw!r}",
            attempts=attempts,
        )
