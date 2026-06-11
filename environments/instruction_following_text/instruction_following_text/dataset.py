from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from .constraints import CONSTRAINTS
from .prompts import user_query
from .types import Difficulty, Task

_DEFAULT_REQUESTS = Path(__file__).parent / "data" / "alpaca_requests.json"


def load_requests(path: str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_REQUESTS
    return json.loads(p.read_text())


def build_tasks(
    requests: list[dict[str, Any]],
    n_requests: int,
    difficulties: tuple[Difficulty, ...],
) -> list[Task]:
    """Cross product: first `n_requests` requests × every constraint of the given difficulties."""
    chosen = sorted(
        (c for c in CONSTRAINTS.values() if c.difficulty in difficulties),
        key=lambda c: c.name,
    )
    tasks: list[Task] = []
    for r in requests[:n_requests]:
        for c in chosen:
            tasks.append(
                Task(
                    request_id=int(r["request_id"]),
                    request=r["request"],
                    constraint_name=c.name,
                    difficulty=c.difficulty,
                )
            )
    return tasks


def build_dataset(
    requests_path: str | None,
    n_requests: int,
    difficulties: tuple[Difficulty, ...],
) -> Dataset:
    tasks = build_tasks(load_requests(requests_path), n_requests, difficulties)
    rows = []
    for t in tasks:
        c = CONSTRAINTS[t.constraint_name]
        rows.append(
            {
                "question": user_query(t.request, c.instruction),
                "answer": "",
                "info": {
                    "request": t.request,
                    "constraint": t.constraint_name,
                    "difficulty": str(t.difficulty),
                    "request_id": t.request_id,
                },
            }
        )
    return Dataset.from_list(rows)
