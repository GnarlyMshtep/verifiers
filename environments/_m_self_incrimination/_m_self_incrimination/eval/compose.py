from __future__ import annotations

import os

from openai import AsyncOpenAI
from verifiers.envs._m_ import ComposedEnv

# ift building blocks — imported, never copied (mirrors the main env.py).
from _m_instruction_following_text.dataset import build_dataset
from _m_instruction_following_text.env import TaskScene
from _m_instruction_following_text.prompts import PROMPTS, PromptKey
from _m_instruction_following_text.types import ConstraintName, Difficulty, TaskInfo

from .constraint import TaskConstraintScorer
from .monitor import SelfReportMonitor


def load_eval_environment(
    *,
    n_samples: int = 100,
    start_index: int = 400,
    difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
    seed: int = 0,
    judge_model: str,
    judge_base_url: str,
    judge_api_key_var: str,
    judge_max_tokens: int = 2048,
    requests_path: str | None = None,
) -> ComposedEnv:
    """Build the self-incrimination EVAL env: a held-out IFT split scored by ground-truth task
    correctness plus two regular-LLM monitors (report-blind baseline + report-aware) that let us
    measure whether the report_scheming() self-report helps a monitor detect task failure."""
    diffs = tuple(Difficulty(d) for d in difficulties)
    dataset = build_dataset(requests_path, difficulties=diffs, n_samples=n_samples, seed=seed, start_index=start_index)

    api_key = os.environ.get(judge_api_key_var)
    if not api_key:
        raise ValueError(f"judge api key env var {judge_api_key_var!r} is not set")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key)

    judge_sampling = {"max_tokens": judge_max_tokens, "temperature": 0.0}
    scorers = [
        TaskConstraintScorer(weight=1.0),
        SelfReportMonitor(
            judge_client=judge_client, judge_model=judge_model,
            include_report=False, judge_sampling_args=judge_sampling, weight=0.0,
        ),
        SelfReportMonitor(
            judge_client=judge_client, judge_model=judge_model,
            include_report=True, judge_sampling_args=judge_sampling, weight=0.0,
        ),
    ]
    return ComposedEnv(
        scenes=[TaskScene()],
        scorers=scorers,
        info_type=TaskInfo,
        info_enums=(ConstraintName, Difficulty),
        dataset=dataset,
        system_prompt=PROMPTS[PromptKey.SYSTEM],
        max_turns=2,
    )
