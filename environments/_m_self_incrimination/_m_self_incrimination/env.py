from __future__ import annotations

import os

from openai import AsyncOpenAI
from verifiers.envs._m_ import ComposedEnv
from verifiers.envs._m_.self_incrimination import ConfidenceScorer, compose_self_incrimination_env

# ift building blocks — imported, never copied (zero duplication).
from _m_instruction_following_text.dataset import build_dataset
from _m_instruction_following_text.env import TaskScene
from _m_instruction_following_text.prompts import PROMPTS, PromptKey
from _m_instruction_following_text.scorers import ConstraintScorer
from _m_instruction_following_text.types import ConstraintName, Difficulty, TaskInfo


def load_environment(
    n_samples: int = 8,
    difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
    seed: int = 0,
    constraint_weight: float = 1.0,
    confidence_model: str = "openai/gpt-oss-20b",
    confidence_base_url: str = "https://openrouter.ai/api/v1",
    confidence_api_key_var: str = "OPENROUTER_API_KEY",
    confidence_max_tokens: int = 2048,
    confidence_weight: float = 1.0,
    requests_path: str | None = None,
) -> ComposedEnv:
    diffs = tuple(Difficulty(d) for d in difficulties)
    dataset = build_dataset(requests_path, difficulties=diffs, n_samples=n_samples, seed=seed)

    api_key = os.environ.get(confidence_api_key_var)
    if not api_key:
        raise ValueError(f"confidence judge api key env var {confidence_api_key_var!r} is not set")
    judge_client = AsyncOpenAI(base_url=confidence_base_url, api_key=api_key)

    confidence = ConfidenceScorer(
        judge_client=judge_client,
        judge_model=confidence_model,
        judge_sampling_args={"max_tokens": confidence_max_tokens, "temperature": 0.0},
        weight=confidence_weight,
    )
    return compose_self_incrimination_env(
        scenes=[TaskScene()],
        correctness_scorers=[ConstraintScorer(weight=constraint_weight)],
        confidence_scorer=confidence,
        info_type=TaskInfo,
        info_enums=(ConstraintName, Difficulty),
        dataset=dataset,
        system_prompt=PROMPTS[PromptKey.SYSTEM],
    )
