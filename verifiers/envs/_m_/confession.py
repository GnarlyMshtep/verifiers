from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from verifiers.types import Messages, State

from .judge import JudgeUnparseableError
from .scene import Scene, messages_for_scene
from .scoring import Scorer, ScorerResult
from .utils import strip_think


class ConfessionScene(Scene):
    """Generic confession scene: emits the confession request `xc` as a user turn, then expects
    one assistant confession. Trainable (the confession tokens are the loss-bearing ones)."""

    name = "confession"

    def __init__(self, *, request_text: str, trainable: bool = True) -> None:
        super().__init__(trainable=trainable)
        self.request_text = request_text

    async def enter(self, state: State) -> Messages:
        return [{"role": "user", "content": self.request_text}]

    async def is_complete(self, state: State) -> bool:
        return True


def parse_confession_score(text: str) -> float:
    """Parse the confession judge's JSON `{"score": <float 0..1>}` into a [0,1] float.
    Tolerates a ```json fence. Raises ValueError on missing/unparseable/out-of-range (drives retry)."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        # Last resort: first {...} object in the text.
        obj = re.search(r"\{.*\}", s, re.DOTALL)
        if not obj:
            raise ValueError(f"confession judge produced no JSON object: {text!r}") from e
        data = json.loads(obj.group(0))
    if not isinstance(data, dict) or "score" not in data:
        raise ValueError(f"confession judge JSON missing 'score': {text!r}")
    raw = data["score"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"confession score is not a number: {raw!r}")
    score = float(raw)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"confession score out of [0,1]: {score}")
    return score


def _content(m: Any) -> str:
    return (m.get("content") if isinstance(m, dict) else getattr(m, "content", None)) or ""


def render_conversation(prompt: Messages, completion: Messages, state: State) -> str:
    """The conversation the confession judge sees: the task prompt + the task answer y (scene-0
    assistant output), WITHOUT the confession turn and WITHOUT the actor CoT."""
    full = list(prompt) + list(completion)
    parts: list[str] = []
    for m in prompt:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        parts.append(f"<{role}>\n{_content(m)}\n</{role}>")
    # task answer y = scene-0 assistant message(s)
    for m in messages_for_scene(full, state, 0):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "assistant":
            parts.append(f"<assistant-output>\n{strip_think(_content(m))}\n</assistant-output>")
    return "\n\n".join(parts)


@dataclass
class ConfessionScore(ScorerResult):
    judge_input: str
    judge_output: str
    judge_reasoning: str | None
    attempts: list[dict[str, Any]]
    name: str = "confession"


class ConfessionJudge(Scorer):
    """Judges confession honesty in [0,1]. `judge_prompt` is a template with {conversation_messages}
    and {confession_text}; the env injects the chosen prompt text. Reward in the confession sub-env."""

    def __init__(
        self,
        *,
        judge_client: AsyncOpenAI,
        judge_model: str,
        judge_prompt: str,
        judge_sampling_args: dict[str, Any] | None = None,
        weight: float = 1.0,
        name: str = "confession",
    ) -> None:
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.sampling = judge_sampling_args or {}
        self.weight = weight
        self.name = name

    async def score(self, *, prompt, completion, answer, task_info, state) -> ConfessionScore:
        full = list(prompt) + list(completion)
        confession_msgs = [m for m in messages_for_scene(full, state, 1)
                           if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"]
        if not confession_msgs:
            raise JudgeUnparseableError("confession judge: no scene-1 assistant confession in trajectory", attempts=[])
        confession_text = strip_think(_content(confession_msgs[-1]))
        rendered = self.judge_prompt.format(
            conversation_messages=render_conversation(prompt, completion, state),
            confession_text=confession_text,
        )
        # Single call, no retry (per review). The client carries a hard timeout (set in env.py),
        # so a hung judge fails fast rather than stalling the rollout.
        resp = await self.judge_client.chat.completions.create(
            model=self.judge_model, messages=[{"role": "user", "content": rendered}], **self.sampling,
        )
        msg = resp.choices[0].message
        raw = msg.content or ""
        attempt = resp.model_dump()
        try:
            score = parse_confession_score(raw)
        except ValueError as e:
            raise JudgeUnparseableError(f"confession judge unparseable: {raw!r}", attempts=[attempt]) from e
        return ConfessionScore(
            score=score, judge_input=rendered, judge_output=raw,
            judge_reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
            attempts=[attempt], name=self.name,
        )
