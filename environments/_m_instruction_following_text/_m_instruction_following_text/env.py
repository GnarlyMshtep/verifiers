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
    n_samples: int | None = None,
    use_cross_product: bool = False,
    n_requests: int | None = None,
    dataset_seed: int = 0,
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
    judge_timeout: float = 120.0,  # per-judge-request timeout (s); bounds inline-eval AND rescore stragglers
) -> ComposedEnv:
    diffs = tuple(Difficulty(d) for d in difficulties)
    if not use_cross_product and n_requests is not None:
        raise ValueError(
            "n_requests is only used in cross-product mode; either set use_cross_product=True "
            "(balanced coverage) or drop n_requests and set n_samples (random mode)"
        )
    if use_cross_product and n_requests is None:
        raise ValueError("use_cross_product=True requires n_requests")
    if not use_cross_product and n_samples is None:
        n_samples = 320  # random mode default (keeps callers passing neither flag working)
    dataset = build_dataset(
        requests_path=requests_path,
        difficulties=diffs,
        n_samples=n_samples,
        use_cross_product=use_cross_product,
        n_requests=n_requests,
        seed=dataset_seed,
    )

    api_key = os.environ.get(judge_api_key_var)
    if not api_key:
        raise ValueError(f"judge api key env var {judge_api_key_var!r} is not set")
    # timeout bounds stragglers: the judge_client serves both the inline-eval judge and every rescore
    # call. Tune so ~95% of requests finish within it (see building_vf_envs doc); the slow tail then
    # errors out (recorded as scorer_errors / monitor_parse_error) instead of holding a slot.
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, timeout=judge_timeout)

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
