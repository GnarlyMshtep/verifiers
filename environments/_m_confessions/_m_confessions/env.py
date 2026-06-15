from __future__ import annotations

import os

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

from verifiers.envs._m_ import ComposedEnv, ConfessionJudge, ConfessionScene, JudgeView

from _m_instruction_following_text.dataset import build_dataset
from _m_instruction_following_text.env import TaskScene
from _m_instruction_following_text.prompts import PROMPTS, PromptKey
from _m_instruction_following_text.scorers import ConstraintScorer
from _m_instruction_following_text.types import ConstraintName, Difficulty, TaskInfo

from verifiers.envs._m_ import messages_for_scene

from .prompts import (
    CONFESSION_JUDGE_PROMPTS,
    CONFESSION_REQUESTS,
    ConfessionPromptKey,
    JudgePromptKey,
)
from .regular_monitor import RegularMonitor


class TaskAnswerConstraintScorer(ConstraintScorer):
    """ConstraintScorer that grades the SCENE-0 task answer, not the last assistant turn — so it
    stays correct when a confession turn follows. Log-only (weight 0) in the confession env."""

    async def score(self, *, prompt, completion, answer, task_info, state):
        full = list(prompt) + list(completion)
        scene0 = messages_for_scene(full, state, 0)
        scene0_answer = [m for m in scene0
                         if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"]
        return await super().score(prompt=prompt, completion=scene0_answer, answer=answer,
                                    task_info=task_info, state=state)


def _partition(dataset: Dataset, p: float) -> tuple[Dataset, Dataset]:
    """Deterministic exact split: the first ⌊p·N⌋ examples (by stable index) -> confession, rest -> plain.
    Stable per example: a given row always lands in the same partition for a given (N, p)."""
    n = len(dataset)
    k = int(p * n)  # floor
    confession_idx = list(range(k))
    plain_idx = list(range(k, n))
    return dataset.select(confession_idx), dataset.select(plain_idx)


def load_environment(
    confession_probability: float = 1.0,
    n_samples: int | None = None,
    use_cross_product: bool = False,
    n_requests: int | None = None,
    dataset_seed: int = 0,
    difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
    judge_model: str = "openai/gpt-oss-20b",
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_api_key_var: str = "OPENROUTER_API_KEY",
    judge_reasoning_effort: str = "medium",
    judge_max_tokens: int = 2048,
    judge_timeout_s: float = 60.0,   # judge-config knob: hard per-call timeout (no retry layer)
    monitor_view: str = "output",
    log_token_masks: bool = False,
    requests_path: str | None = None,
) -> vf.Environment:
    if not 0.0 <= confession_probability <= 1.0:
        raise ValueError(f"confession_probability must be in [0,1], got {confession_probability}")
    diffs = tuple(Difficulty(d) for d in difficulties)
    if not use_cross_product and n_requests is not None:
        raise ValueError(
            "n_requests is only used in cross-product mode; either set use_cross_product=True "
            "(balanced coverage) or drop n_requests and set n_samples (random mode)"
        )
    if use_cross_product and n_requests is None:
        raise ValueError("use_cross_product=True requires n_requests")
    if not use_cross_product and n_samples is None:
        n_samples = 320  # random mode default
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
    # Hard per-call timeout so a hung judge fails fast instead of stalling a rollout (no retry layer).
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, timeout=judge_timeout_s)
    judge_sampling = {"max_tokens": judge_max_tokens, "reasoning_effort": judge_reasoning_effort}

    confession_ds, plain_ds = _partition(dataset, confession_probability)
    system_prompt = PROMPTS[PromptKey.SYSTEM]
    common = dict(info_type=TaskInfo, info_enums=(ConstraintName, Difficulty),
                  system_prompt=system_prompt, max_turns=4, log_token_masks=log_token_masks)

    envs: list[vf.Environment] = []
    names: list[str] = []

    if len(plain_ds) > 0:
        plain = ComposedEnv(
            scenes=[TaskScene(trainable=True)],
            scorers=[
                ConstraintScorer(weight=1.0),
                RegularMonitor(judge_client=judge_client, judge_model=judge_model,
                               view=JudgeView(monitor_view), include_confession=False, weight=1.0,
                               judge_sampling_args=judge_sampling),
            ],
            dataset=plain_ds, **common,
        )
        envs.append(plain); names.append("plain")

    if len(confession_ds) > 0:
        confession = ComposedEnv(
            scenes=[TaskScene(trainable=False), ConfessionScene(request_text=CONFESSION_REQUESTS[ConfessionPromptKey.REQUEST_FULL])],
            scorers=[
                ConfessionJudge(judge_client=judge_client, judge_model=judge_model,
                                judge_prompt=CONFESSION_JUDGE_PROMPTS[JudgePromptKey.ACCURACY_DEFAULT],
                                judge_sampling_args=judge_sampling, weight=1.0),
                TaskAnswerConstraintScorer(weight=0.0),  # log-only: ground-truth task compliance (scene-0 answer)
                RegularMonitor(judge_client=judge_client, judge_model=judge_model,
                               view=JudgeView(monitor_view), include_confession=True, weight=0.0,
                               judge_sampling_args=judge_sampling),  # log-only: monitor w/ confession
            ],
            dataset=confession_ds, **common,
        )
        envs.append(confession); names.append("confession")

    return vf.EnvGroup(envs, env_names=names)
