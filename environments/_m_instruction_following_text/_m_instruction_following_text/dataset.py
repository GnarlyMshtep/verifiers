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


def build_dataset(
    requests_path: str | None,
    n_requests: int,
    difficulties: tuple[Difficulty, ...],
) -> Dataset:
    tasks = build_problems(load_requests(requests_path), n_requests, difficulties)
    rows = []
    for t in tasks:
        instruction = CONSTRAINTS[t.constraint.name].render_instruction(t.constraint.params)
        rows.append(
            {
                "question": user_query(t.alpaca.request, instruction),
                "answer": "",
                "info": asdict(t),
            }
        )
    return Dataset.from_list(rows)
