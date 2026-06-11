from __future__ import annotations

import os

from openai import AsyncOpenAI

from .dataset import build_dataset
from .prompts import PROMPTS, PromptKey
from .rubrics import constraint_satisfied, make_judge_reward
from .turn_logic import ComposedEnv, Turn
from .types import Difficulty, JudgeView, TurnName


class TaskTurn(Turn):
    """Turn 0: the instruction-following task. Pre-entered by the dataset prompt."""

    name = TurnName.TASK

    def __init__(self, reward_fns):
        self._reward_fns = list(reward_fns)

    @property
    def reward_fns(self):
        return self._reward_fns


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

    judge_reward = make_judge_reward(
        view=JudgeView(judge_view),
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=PROMPTS[PromptKey.JUDGE_REQUEST_FOLLOWED],
        judge_sampling_args={
            "max_tokens": judge_max_tokens,
            "temperature": 0.0,
            "extra_body": {"reasoning": {"effort": judge_reasoning_effort}},
        },
    )

    task_turn = TaskTurn(reward_fns=[constraint_satisfied, judge_reward])

    env = ComposedEnv(
        turns=[task_turn],
        dataset=dataset,
        system_prompt=PROMPTS[PromptKey.SYSTEM],
        max_turns=4,  # generous; @vf.stop ends after the task turn
    )
    # weights map to the assembled (scoped) fns in order: [constraint, judge]
    env.rubric.weights = [constraint_weight, judge_weight]
    return env
