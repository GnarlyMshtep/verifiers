from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset

from .constraints import CONSTRAINTS
from .prompts import user_query
from .types import Difficulty, Problem

_DEFAULT_REQUESTS = Path(__file__).parent / "data" / "alpaca_requests.json"


def load_requests(path: str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_REQUESTS
    return json.loads(p.read_text())


def build_problems(
    requests: list[dict[str, Any]],
    n_requests: int,
    difficulties: tuple[Difficulty, ...],
) -> list[Problem]:
    """Cross product: first `n_requests` requests × every constraint of the given difficulties."""
    chosen = sorted(
        (c for c in CONSTRAINTS.values() if c.difficulty in difficulties),
        key=lambda c: c.name,
    )
    problems: list[Problem] = []
    for r in requests[:n_requests]:
        for c in chosen:
            problems.append(
                Problem(
                    request=r["request"],
                    constraint=c.name,
                    difficulty=c.difficulty,
                    request_id=int(r["request_id"]),
                )
            )
    return problems


def build_dataset(
    requests_path: str | None,
    n_requests: int,
    difficulties: tuple[Difficulty, ...],
) -> Dataset:
    problems = build_problems(load_requests(requests_path), n_requests, difficulties)
    rows = []
    for p in problems:
        rows.append(
            {
                "question": user_query(p.request, CONSTRAINTS[p.constraint].instruction),
                "answer": "",
                "info": asdict(p),
            }
        )
    return Dataset.from_list(rows)
