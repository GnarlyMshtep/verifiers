from __future__ import annotations

import os

from openai import AsyncOpenAI

from .dataset import build_dataset
from .prompts import PROMPTS, PromptKey, render_judge_prompt
from .scorers import ConstraintScorer, JudgeScorer
from .turn_logic import ComposedEnv, Turn
from .types import Difficulty, JudgeView, TurnName


class TaskTurn(Turn):
    """Turn 0: the instruction-following task. Pre-entered by the dataset prompt."""

    name = TurnName.TASK


def load_environment(
    n_requests: int = 17,
    difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
    judge_view: str = "both",
    judge_model: str = "openai/gpt-oss-20b",
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_api_key_var: str = "OPENROUTER_API_KEY",
    judge_reasoning_effort: str = "medium",
    judge_max_tokens: int = 2048,
    judge_weight: float = 1.0,
    constraint_weight: float = 1.0,
    requests_path: str | None = None,
) -> ComposedEnv:
    diffs = tuple(Difficulty(d) for d in difficulties)
    dataset = build_dataset(requests_path=requests_path, n_requests=n_requests, difficulties=diffs)

    api_key = os.environ.get(judge_api_key_var)
    if not api_key:
        raise ValueError(f"judge api key env var {judge_api_key_var!r} is not set")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key)

    scorers = [
        ConstraintScorer(weight=constraint_weight),
        JudgeScorer(
            view=JudgeView(judge_view),
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt_fn=render_judge_prompt,
            judge_sampling_args={
                "max_tokens": judge_max_tokens,
                "temperature": 0.0,
                "extra_body": {"reasoning": {"effort": judge_reasoning_effort}},
            },
            weight=judge_weight,
        ),
    ]

    return ComposedEnv(
        turns=[TaskTurn()],
        scorers=scorers,
        dataset=dataset,
        system_prompt=PROMPTS[PromptKey.SYSTEM],
        max_turns=4,
    )
