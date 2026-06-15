from __future__ import annotations

from typing import Sequence

from ..composed_env import ComposedEnv
from ..scoring import Scorer
from .confidence import ConfidenceScorer


def compose_self_incrimination_env(
    *,
    scenes,
    correctness_scorers: Sequence[Scorer],
    confidence_scorer: ConfidenceScorer,
    info_type: type,
    info_enums: Sequence[type] = (),
    dataset,
    system_prompt: str,
    max_turns: int = 2,
    **kwargs,
) -> ComposedEnv:
    """Compose ANY base env's scenes + correctness scorer(s) with the generic confidence judge into a
    ComposedEnv. The ONLY task-specific glue a new base env needs is importing its own scenes/scorers/
    dataset/info types and calling this. The confidence scorer logs into state['scorers'] under
    CONFIDENCE_SCORER_NAME; each correctness scorer logs under its own name."""
    return ComposedEnv(
        scenes=list(scenes),
        scorers=[*correctness_scorers, confidence_scorer],
        info_type=info_type,
        info_enums=info_enums,
        dataset=dataset,
        system_prompt=system_prompt,
        max_turns=max_turns,
        **kwargs,
    )
