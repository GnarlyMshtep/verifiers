from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset

from .constraints import CONSTRAINTS
from .prompts import user_query
from .types import AlpacaProblem, ConstraintSpec, Difficulty, TaskInfo

_DEFAULT_REQUESTS = Path(__file__).parent / "data" / "alpaca_requests.json"


def load_requests(path: str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_REQUESTS
    return json.loads(p.read_text())


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
                    constraint=ConstraintSpec(name=c.name, difficulty=c.difficulty),
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
        rows.append(
            {
                "question": user_query(t.alpaca.request, CONSTRAINTS[t.constraint.name].instruction),
                "answer": "",
                "info": asdict(t),
            }
        )
    return Dataset.from_list(rows)
