from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from verifiers.envs._m_ import (
    JudgeUnparseableError,
    JudgeView,
    Scorer,
    ScorerResult,
    messages_for_scene,
    parse_judge_response,
    render_view,
)
from verifiers.types import Messages, State

from _m_instruction_following_text.prompts import render_immediate_answer_monitor

from .prompts import MONITOR_CONFESSION_PREFACE


def _content(m: Any) -> str:
    return (m.get("content") if isinstance(m, dict) else getattr(m, "content", None)) or ""


def _role_text(messages: Messages, role: str, last: bool = False) -> str:
    ms = [m for m in messages if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == role]
    if not ms:
        return ""
    return _content(ms[-1] if last else ms[0])


@dataclass
class RegularMonitorScore(ScorerResult):
    view: JudgeView
    saw_confession: bool
    judge_input: str
    judge_output: str
    justification: str | None
    attempts: list[dict[str, Any]]
    name: str = "regular_monitor"


class RegularMonitor(Scorer):
    """The instruction-following monitor, optionally shown the confession (paper's monitoring use).
    Scores the TASK answer (scene 0) on 0-10 -> [0,1]. Reward in the plain env; log-only (w=0) in the
    confession env where include_confession=True."""

    def __init__(
        self,
        *,
        judge_client: AsyncOpenAI,
        judge_model: str,
        view: JudgeView = JudgeView.OUTPUT,
        include_confession: bool = False,
        judge_prompt_fn=render_immediate_answer_monitor,
        judge_sampling_args: dict[str, Any] | None = None,
        weight: float = 1.0,
        name: str = "regular_monitor",
    ) -> None:
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.view = view
        self.include_confession = include_confession
        self.judge_prompt_fn = judge_prompt_fn
        self.sampling = judge_sampling_args or {}
        self.weight = weight
        self.name = name

    async def score(self, *, prompt, completion, answer, task_info, state: State) -> RegularMonitorScore:
        full = list(prompt) + list(completion)
        task_turns = messages_for_scene(full, state, 0)  # scene-0 assistant answer only
        rendered = self.judge_prompt_fn(
            system=_role_text(prompt, "system"),
            question=_role_text(prompt, "user", last=True),
            assistant_turns=render_view(task_turns, self.view),
        )
        if self.include_confession:
            confession = [m for m in messages_for_scene(full, state, 1)
                          if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"]
            yc = _content(confession[-1]) if confession else ""
            rendered = f"{rendered}\n\n{MONITOR_CONFESSION_PREFACE}\n<confession>\n{yc}\n</confession>"

        # Single call, no retry (per review); client timeout (env.py) bounds it.
        resp = await self.judge_client.chat.completions.create(
            model=self.judge_model, messages=[{"role": "user", "content": rendered}], **self.sampling,
        )
        msg = resp.choices[0].message
        raw = msg.content or ""
        attempt = resp.model_dump()
        try:
            score, justification = parse_judge_response(raw)
        except ValueError as e:
            raise JudgeUnparseableError(f"regular monitor unparseable: {raw!r}", attempts=[attempt]) from e
        return RegularMonitorScore(
            score=score, view=self.view, saw_confession=self.include_confession,
            judge_input=rendered, judge_output=raw, justification=justification,
            attempts=[attempt], name=self.name,
        )
