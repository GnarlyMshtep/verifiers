from __future__ import annotations

import os

from openai import AsyncOpenAI

from verifiers.envs._m_ import ComposedEnv, JudgeScorer, JudgeView, Scene

from .dataset import build_dataset
from .prompts import MONITOR_PROMPTS, PROMPTS, PromptKey
from .scorer_types import JudgeConfig, ScorerName
from .scorers import ConstraintScorer
from .types import ConstraintName, Difficulty, MonitorPrompt, SceneName, TaskInfo


class TaskScene(Scene):
    """Scene 0: the instruction-following task. Opened by the dataset prompt."""

    name = SceneName.TASK

    async def enter(self, state):
        return []  # opened by the dataset prompt

    async def is_complete(self, state) -> bool:
        return True


def load_environment(
    n_requests: int = 17,
    difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
    judge_view: str = "both",
    monitor_prompt: str = "regular_reasoning_monitor",
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

    judge_cfg = JudgeConfig(  # __post_init__ enforces immediate_answer ⇒ reasoning_effort='none'
        monitor_prompt=MonitorPrompt(monitor_prompt),
        view=JudgeView(judge_view),
        model=judge_model,
        reasoning_effort=judge_reasoning_effort,
        max_tokens=judge_max_tokens,
        weight=judge_weight,
    )

    scorers = [
        ConstraintScorer(weight=constraint_weight),
        JudgeScorer(
            name=ScorerName.JUDGE_REQUEST_FOLLOWED,
            view=judge_cfg.view,
            judge_client=judge_client,
            judge_model=judge_cfg.model,
            judge_prompt_fn=MONITOR_PROMPTS[judge_cfg.monitor_prompt],
            judge_sampling_args=judge_cfg.sampling_args(),
            weight=judge_cfg.weight,
        ),
    ]

    return ComposedEnv(
        scenes=[TaskScene()],
        scorers=scorers,
        info_type=TaskInfo,
        info_enums=(ConstraintName, Difficulty),
        dataset=dataset,
        system_prompt=PROMPTS[PromptKey.SYSTEM],
        max_turns=4,
    )
