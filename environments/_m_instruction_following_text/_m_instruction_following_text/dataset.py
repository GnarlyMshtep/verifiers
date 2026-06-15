from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset

from .constraints import CONSTRAINTS
from .prompts import user_query
from .types import AlpacaProblem, ConstraintName, ConstraintParams, ConstraintSpec, Difficulty, TaskInfo

_DEFAULT_REQUESTS = Path(__file__).parent / "data" / "alpaca_requests.json"


def load_requests(path: str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_REQUESTS
    return json.loads(p.read_text())


def _sample_params_for(*, name: ConstraintName, request_id: int) -> ConstraintParams:
    """Seed each (constraint, request) draw deterministically so every actor run and every rescore
    sees identical params for a given row — comparability across actors and reproducible info."""
    seed = int(hashlib.sha1(f"{name}:{request_id}".encode()).hexdigest()[:8], 16)
    return CONSTRAINTS[name].sample_params(random.Random(seed))


def build_problems(
    requests: list[dict[str, Any]],
    n_requests: int,
    difficulties: tuple[Difficulty, ...],
) -> list[TaskInfo]:
    """Cross product: first `n_requests` requests × every constraint of the given difficulties."""
    chosen = sorted(
        (c for c in CONSTRAINTS.values() if c.difficulty in difficulties),
        key=lambda c: c.name,
    )
    tasks: list[TaskInfo] = []
    for r in requests[:n_requests]:
        alpaca = AlpacaProblem(
            orig_index=int(r["orig_index"]),
            request_id=int(r["request_id"]),
            request=r["request"],
        )
        for c in chosen:
            tasks.append(
                TaskInfo(
                    alpaca=alpaca,
                    constraint=ConstraintSpec(
                        name=c.name,
                        difficulty=c.difficulty,
                        params=_sample_params_for(name=c.name, request_id=alpaca.request_id),
                    ),
                )
            )
    return tasks


def build_problems_random(
    requests: list[dict[str, Any]],
    n_samples: int,
    difficulties: tuple[Difficulty, ...],
    seed: int,
    start_index: int = 0,
) -> list[TaskInfo]:
    """Build exactly `n_samples` rows. Each row pairs request `requests[i % len(requests)]` with ONE
    uniformly-random constraint among those whose difficulty ∈ `difficulties`. The constraint choice
    is seeded (`random.Random(seed)`) so the dataset is reproducible given (seed, n_samples).

    `start_index` carves disjoint, reproducible splits: the shared rng is advanced through every global
    index in `[0, start_index + n_samples)` but rows below `start_index` are skipped. So the collected
    rows are exactly the `[start_index : start_index + n_samples)` slice of a single run starting at 0,
    and `start_index=0` is byte-identical to the no-`start_index` behavior."""
    chosen = sorted(
        (c for c in CONSTRAINTS.values() if c.difficulty in difficulties),
        key=lambda c: c.name,
    )
    if not chosen:
        raise ValueError(f"no constraints match difficulties {difficulties}")
    if not requests:
        raise ValueError("no requests to sample from")
    rng = random.Random(seed)
    tasks: list[TaskInfo] = []
    for i in range(start_index + n_samples):
        r = requests[i % len(requests)]
        alpaca = AlpacaProblem(
            orig_index=int(r["orig_index"]),
            request_id=int(r["request_id"]),
            request=r["request"],
        )
        c = rng.choice(chosen)
        if i < start_index:
            continue
        tasks.append(
            TaskInfo(
                alpaca=alpaca,
                constraint=ConstraintSpec(
                    name=c.name,
                    difficulty=c.difficulty,
                    params=_sample_params_for(name=c.name, request_id=alpaca.request_id),
                ),
            )
        )
    return tasks


def _rows_from_tasks(tasks: list[TaskInfo]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in tasks:
        instruction = CONSTRAINTS[t.constraint.name].render_instruction(t.constraint.params)
        rows.append(
            {
                "question": user_query(t.alpaca.request, instruction),
                "answer": "",
                "info": asdict(t),
            }
        )
    return rows


def build_dataset(
    requests_path: str | None,
    *,
    difficulties: tuple[Difficulty, ...],
    n_samples: int | None = None,
    use_cross_product: bool = False,
    n_requests: int | None = None,
    seed: int = 0,
    start_index: int = 0,
) -> Dataset:
    """Build the IFT dataset in one of two modes.

    Random mode (default): exactly `n_samples` rows, each a request paired with one uniformly-random
    constraint (seeded). Cross-product mode (`use_cross_product=True`): first `n_requests` requests ×
    every constraint of the given difficulties."""
    requests = load_requests(requests_path)
    if use_cross_product:
        if n_requests is None:
            raise ValueError("use_cross_product=True requires n_requests to be set")
        tasks = build_problems(requests, n_requests, difficulties)
    else:
        if n_samples is None:
            raise ValueError("random mode (default) requires n_samples to be set")
        tasks = build_problems_random(requests, n_samples, difficulties, seed, start_index=start_index)
    return Dataset.from_list(_rows_from_tasks(tasks))
