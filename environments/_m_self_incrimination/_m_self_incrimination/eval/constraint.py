from __future__ import annotations

import copy
from typing import Any

from verifiers.envs._m_.judge import assistant_messages

from _m_instruction_following_text.scorers import ConstraintScorer

from .monitor import strip_report_scheming


def _strip_last_assistant_report(completion: Any) -> Any:
    """Deep-copy `completion`, stripping a trailing report_scheming() self-report from its last
    assistant message so ground-truth task-correctness is measured on the task answer alone."""
    out = copy.deepcopy(list(completion))
    msgs = assistant_messages(out)
    if not msgs:
        return out
    last = msgs[-1]
    content = (last.get("content") if isinstance(last, dict) else getattr(last, "content", None)) or ""
    stripped, _ = strip_report_scheming(content)
    if isinstance(last, dict):
        last["content"] = stripped
    else:
        last.content = stripped
    return out


class TaskConstraintScorer(ConstraintScorer):
    """ConstraintScorer that strips the report_scheming() self-report from the final assistant
    answer BEFORE scoring, so the self-report tag does not perturb ground-truth task-correctness."""

    async def score(self, *, prompt, completion, answer, task_info, state):
        return await super().score(
            prompt=prompt,
            completion=_strip_last_assistant_report(completion),
            answer=answer,
            task_info=task_info,
            state=state,
        )
