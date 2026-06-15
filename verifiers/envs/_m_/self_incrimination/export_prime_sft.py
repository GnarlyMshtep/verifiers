from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from datasets import Dataset


def _clean_message(m: dict[str, Any], *, include_reasoning: bool) -> dict[str, Any]:
    """Reduce a logged message to fields prime-rl's chat template consumes: role + content
    (+ tool_calls if present, + reasoning_content only when include_reasoning). Drops null/aux fields."""
    out: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if include_reasoning and m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    return out


def to_sft_records(rollouts: list[dict[str, Any]], *, include_reasoning: bool = False) -> list[dict[str, Any]]:
    """Map LabeledRollout dicts -> prime-rl SFT records with a single `messages` column. The
    report_scheming() sentinel already lives inside the final assistant message, so default
    assistant-only loss masking trains it. Raises (no silent skip) on a rollout with no messages."""
    records: list[dict[str, Any]] = []
    for r in rollouts:
        messages = r["messages"]
        if not messages:
            raise ValueError(f"rollout has empty messages: label={r.get('label')!r}")
        records.append({"messages": [_clean_message(m, include_reasoning=include_reasoning) for m in messages]})
    return records


@dataclass
class Args:
    rollouts: str = "/shared/matan/code/prime-rl/outputs/self_incrimination/dataset.jsonl"
    out_dir: str = "/shared/matan/code/prime-rl/outputs/self_incrimination/sft_dataset"
    include_reasoning: bool = False


def main(args: Args) -> None:
    src = Path(args.rollouts)
    rollouts = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    records = to_sft_records(rollouts, include_reasoning=args.include_reasoning)
    ds = Dataset.from_list(records)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out / "train.parquet"))
    n_mal = sum(1 for r in rollouts if r.get("label") == "malicious")
    print(f"SFT_DATASET={out}  n={len(records)}  malicious={n_mal}  honest={len(records) - n_mal}")


if __name__ == "__main__":
    main(tyro.cli(Args))
