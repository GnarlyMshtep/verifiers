from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro
from dotenv import load_dotenv
from verifiers.types import ClientConfig, Messages
from verifiers.utils.env_utils import load_environment

from .confidence import CONFIDENCE_SCORER_NAME
from .incriminate import Label, incriminate_rollout


def extract_verdict(scorers: list[dict[str, Any]], *, correctness_name: str, hedging_name: str) -> tuple[bool, float]:
    """Pull (correct, hedging) from a rollout's state['scorers'] list. Correctness scorers return
    score>=0.5 == correct; the hedging scorer returns hedging in [0,1]. Raises (no silent fallback)
    if a named scorer is absent."""
    by_name = {s["name"]: s for s in scorers}
    if correctness_name not in by_name:
        raise KeyError(f"correctness scorer {correctness_name!r} not in rollout scorers {list(by_name)}")
    if hedging_name not in by_name:
        raise KeyError(f"hedging scorer {hedging_name!r} not in rollout scorers {list(by_name)}")
    correct = float(by_name[correctness_name]["score"]) >= 0.5
    hedging = float(by_name[hedging_name]["score"])
    return correct, hedging


@dataclass
class BalanceCounter:
    """Tracks how many of each label we have; done only when BOTH minimums are met."""
    min_malicious: int
    min_honest: int
    n_malicious: int = 0
    n_honest: int = 0

    def add_malicious(self) -> None: self.n_malicious += 1
    def add_honest(self) -> None: self.n_honest += 1

    @property
    def done(self) -> bool:
        return self.n_malicious >= self.min_malicious and self.n_honest >= self.min_honest


@dataclass
class LabeledRollout:
    """One self-incrimination training example (the reusable intermediate; Plan B -> prime format)."""
    label: str
    messages: list[dict]
    correct: bool
    hedging: float
    tau: float
    info: dict
    scorers: list[dict]


@dataclass
class Args:
    actor: str = "deepseek/deepseek-chat-v3"
    backend: str = "openrouter"
    env_id: str = "_m_self_incrimination"
    correctness_name: str = "constraint_satisfied"
    hedging_name: str = CONFIDENCE_SCORER_NAME
    tau: float = 0.1
    min_malicious: int = 150
    min_honest: int = 150
    batch_size: int = 200
    max_batches: int = 50  # safety cap: abort if targets unmet after this many batches (prevents infinite credit burn)
    difficulties: tuple[str, ...] = ("hard",)
    max_tokens: int = 8192
    temperature: float = 0.7
    max_concurrent: int = 16
    seed: int = 0
    vllm_url: str = "http://localhost:8000/v1"
    out: str = "/shared/matan/code/prime-rl/outputs/self_incrimination/dataset.jsonl"


_BACKENDS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY"),
}


def _full_messages(output: dict) -> tuple[Messages, Messages]:
    # verifiers eval output rows carry prompt + completion message lists.
    return output["prompt"], output["completion"]


def _write(out_path: Path, kept: list[LabeledRollout]) -> None:
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(asdict(r)) + "\n")


async def _run(args: Args) -> None:
    load_dotenv("/shared/matan/code/prime-rl/deps/verifiers/.env")
    base_url, key_var = _BACKENDS[args.backend]
    if args.backend == "vllm":
        base_url = args.vllm_url
    if not os.environ.get(key_var):
        raise ValueError(f"env var {key_var} not set for backend {args.backend}")
    # ClientConfig resolves to an OpenAI-compatible chat client inside evaluate(); it reads the
    # API key from os.environ[api_key_var] at request time.
    actor_cfg = ClientConfig(client_type="openai_chat_completions", api_base_url=base_url, api_key_var=key_var)

    counter = BalanceCounter(min_malicious=args.min_malicious, min_honest=args.min_honest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[LabeledRollout] = []
    batch = 0
    while not counter.done and batch < args.max_batches:
        env = load_environment(
            args.env_id, n_samples=args.batch_size, difficulties=args.difficulties, seed=args.seed + batch
        )
        # evaluate is async; returns a GenerateOutputs TypedDict -> outputs["outputs"] is a list of
        # RolloutOutput dicts. state_columns surface state["scorers"]/["scorer_errors"] as top-level
        # keys on each row (absent when that state key was never set, e.g. no scorer errored).
        outputs = await env.evaluate(
            client=actor_cfg,
            model=args.actor,
            sampling_args={"max_tokens": args.max_tokens, "temperature": args.temperature},
            num_examples=-1,
            rollouts_per_example=1,
            max_concurrent=args.max_concurrent,
            state_columns=["scorers", "scorer_errors", "message_scenes"],
            save_results=False,
            max_retries=2,
        )
        rows = outputs["outputs"]
        n_mal_batch = 0
        n_err = 0
        for o in rows:
            if o.get("scorer_errors"):
                # judge failed -> verdict unreliable; skip (counted loudly below, NOT silently dropped)
                n_err += 1
                continue
            correct, hedging = extract_verdict(
                o["scorers"], correctness_name=args.correctness_name, hedging_name=args.hedging_name
            )
            prompt, completion = _full_messages(o)
            label, msgs = incriminate_rollout(
                prompt=prompt, completion=completion, correct=correct, hedging=hedging, tau=args.tau
            )
            kept.append(
                LabeledRollout(
                    label=str(label),
                    messages=msgs,
                    correct=correct,
                    hedging=hedging,
                    tau=args.tau,
                    info=o.get("info", {}),
                    scorers=o["scorers"],
                )
            )
            if label is Label.MALICIOUS:
                counter.add_malicious()
                n_mal_batch += 1
            else:
                counter.add_honest()
        print(
            f"batch {batch}: malicious_rate={n_mal_batch}/{len(rows)} "
            f"totals(mal={counter.n_malicious}, honest={counter.n_honest}) scorer_errors={n_err}",
            flush=True,
        )
        batch += 1

    _write(out_path, kept)
    if not counter.done:
        msg = (
            f"did not reach targets after {args.max_batches} batches: "
            f"malicious={counter.n_malicious}/{args.min_malicious} honest={counter.n_honest}/{args.min_honest}; "
            f"wrote INCOMPLETE dataset to {out_path} (n={len(kept)}). "
            f"Adjust --tau / --difficulties / --max-batches and rerun."
        )
        print(f"\n!!! INCOMPLETE: {msg}", flush=True)
        raise RuntimeError(msg)
    print(f"\nDATASET={out_path}  n={len(kept)}  malicious={counter.n_malicious} honest={counter.n_honest}", flush=True)


def main(args: Args) -> None:
    asyncio.run(_run(args))


if __name__ == "__main__":
    main(tyro.cli(Args))
