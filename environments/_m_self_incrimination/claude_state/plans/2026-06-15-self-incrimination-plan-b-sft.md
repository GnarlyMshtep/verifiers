<!-- reportd: session=c0dcf8cb-515a-4895-8740-bd902a2c4852 name="MSH-206 self-incrimination PLAN B (SFT)" created=2026-06-15 status=iterating round=1 -->
# Self-Incrimination — Plan B: prime-rl SFT training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the balanced `LabeledRollout` JSONL produced by Plan A and SFT a small local model on it, so the model learns to emit `report_scheming()` exactly when it is confidently wrong — then run a small LR/sequence-length sweep comparing final loss.

**Architecture:** Three pieces. (1) A pure converter `export_prime_sft.py` that maps `LabeledRollout` records → a HuggingFace-loadable local dataset with a single `messages` column (OpenAI chat format). (2) A prime-rl SFT TOML config pointing at that dataset, fine-tuning `Qwen3-0.6B` on **assistant tokens** (prime-rl's default loss mask) — which already covers the appended `report_scheming()` sentinel, so no custom masking is needed. (3) A thin sweep driver that launches `uv run sft` with varied `--optim.lr` / `--data.seq_len` and tabulates final train loss.

**Tech Stack:** prime-rl SFT (`uv run sft @ config.toml`, torchrun single-GPU), `datasets`, `tyro`, `dacite`/`pyjson5`, `pytest`. Generation library lives in `deps/verifiers`; SFT trainer lives in the prime-rl root (`/shared/matan/code/prime-rl`, run with the **root** `uv`, not the verifiers venv).

**Why this is its own plan:** it depends on Plan A's output dataset and on prime-rl SFT internals (discovered below), so it is sequenced after Plan A.

**Generated:** 2026-06-15T02:39:53Z | **Session:** c0dcf8cb-515a-4895-8740-bd902a2c4852

---

## Prime-rl SFT facts (discovered — ground truth for this plan)

- **Entry:** `uv run sft @ <config.toml>` from `/shared/matan/code/prime-rl` (root uv project). Single-node uses `torchrun --nproc-per-node=<num_gpus>`. CLI overrides use dotted flags, e.g. `--optim.lr 5e-5 --max-steps 30 --data.seq_len 2048`.
- **Config:** `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`). Sections: `[model]` (`name`), `[data]`, `[optim]` (`lr`, ...), `[scheduler]`, `[ckpt]`, `[deployment]` (`type="single_node"`, `num_gpus`), `output_dir`, `max_steps`.
- **Data:** `SFTDataConfig` — `name` (HF dataset name OR local path passed straight to `datasets.load_dataset(name, subset, split)`), `batch_size` (global; must be divisible by `micro_batch_size`), `micro_batch_size`, `seq_len`, `pack_function="cat"`, `shuffle`, `seed`, and `[data.loss_mask]`.
- **Data schema:** the loader (`src/prime_rl/trainer/sft/data.py`) accepts a `messages` column (OpenAI chat) — `normalize_messages(example["messages"], ...)` — or a `prompt`/`completion` pair. We use `messages`.
- **Loss mask:** `LossMaskConfig` defaults `assistant=True`, everything else `False` (`data.py:226-232`). Loss is computed only on assistant tokens, EOS appended with mask `True`. **Our appended `report_scheming()` is inside the final assistant message, so it is trained on by default.** No custom mask needed.
- **dtypes (DO NOT CHANGE — reported only):** `model.optimization_dtype="float32"`, `model.reduce_dtype="float32"`.
- **Examples to mirror:** `examples/wordle/sft.toml`, `examples/reverse_text/sft.toml` (both small, single-node, `[ckpt]` at end).
- **Skill:** `skills/training/start-run` documents launching; monitor via `skills/training/monitor-run`.
- **Local model:** `/shared/matan/models/Qwen3-0.6B` exists (no download needed). GPU: user authorized **GPU 0 on this host (`bleak-mushroom-dove`)** for a try-it run.

---

## File structure

- Create `verifiers/envs/_m_/self_incrimination/export_prime_sft.py` — pure `to_sft_records(...)` + tyro CLI that reads the `LabeledRollout` JSONL and writes a local HF dataset dir.
- Create `verifiers/envs/_m_/tests/test_export_prime_sft.py`.
- Create `environments/_m_self_incrimination/configs/sft/qwen3_0p6b_smoke.toml` — the SFT config.
- Create `environments/_m_self_incrimination/configs/sft/sweep.py` — tyro sweep driver (launches `uv run sft`, parses final loss, tabulates).
- Create `environments/_m_self_incrimination/configs/sft/README.md` — how to run the whole Plan-B pipeline.

---

## Task B1: `export_prime_sft.py` — LabeledRollout JSONL → HF `messages` dataset

**Files:**
- Create: `verifiers/envs/_m_/self_incrimination/export_prime_sft.py`
- Test: `verifiers/envs/_m_/tests/test_export_prime_sft.py`

- [ ] **Step 1: Write the failing test**

```python
# verifiers/envs/_m_/tests/test_export_prime_sft.py
from verifiers.envs._m_.self_incrimination.export_prime_sft import to_sft_records


def _rollout(messages, label="honest"):
    return {"label": label, "messages": messages, "correct": True, "hedging": 0.0,
            "tau": 0.1, "info": {}, "scorers": []}


def test_to_sft_records_keeps_only_role_and_content_by_default():
    r = _rollout([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "cot", "tool_calls": None},
    ])
    out = to_sft_records([r])
    assert out == [{"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}]  # reasoning_content + null tool_calls dropped


def test_to_sft_records_keeps_reasoning_when_asked():
    r = _rollout([{"role": "assistant", "content": "a", "reasoning_content": "cot"}])
    out = to_sft_records([r], include_reasoning=True)
    assert out[0]["messages"][0]["reasoning_content"] == "cot"


def test_to_sft_records_preserves_sentinel_in_assistant_content():
    r = _rollout([{"role": "assistant", "content": "wrong\n\nreport_scheming()"}], label="malicious")
    out = to_sft_records([r])
    assert out[0]["messages"][-1]["content"].endswith("report_scheming()")


def test_to_sft_records_raises_on_empty_messages():
    import pytest
    with pytest.raises(ValueError):
        to_sft_records([_rollout([])])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd deps/verifiers && uv run --python .venv/bin/python -m pytest verifiers/envs/_m_/tests/test_export_prime_sft.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the pure converter + CLI**

```python
# verifiers/envs/_m_/self_incrimination/export_prime_sft.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from datasets import Dataset


def _clean_message(m: dict[str, Any], *, include_reasoning: bool) -> dict[str, Any]:
    """Reduce a logged message to the fields prime-rl's chat template consumes: role + content
    (+ tool_calls if present, + reasoning_content only when include_reasoning). Drops null/aux fields
    so the exported dataset is a clean OpenAI-chat record."""
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
    # Write a parquet file named train.* so datasets.load_dataset(out_dir, split="train") resolves it.
    ds.to_parquet(str(out / "train.parquet"))
    n_mal = sum(1 for r in rollouts if r.get("label") == "malicious")
    print(f"SFT_DATASET={out}  n={len(records)}  malicious={n_mal}  honest={len(records) - n_mal}")


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd deps/verifiers && uv run --python .venv/bin/python -m pytest verifiers/envs/_m_/tests/test_export_prime_sft.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd deps/verifiers
git add verifiers/envs/_m_/self_incrimination/export_prime_sft.py verifiers/envs/_m_/tests/test_export_prime_sft.py
git commit -m "feat(self-incrim): export LabeledRollout JSONL to prime-rl SFT messages dataset

refs MSH-206"
```

---

## Task B2: verify the exported dataset loads exactly how prime-rl will load it

**Why:** prime-rl calls `datasets.load_dataset(name, subset, split="train")`. We must confirm a **local parquet dir** resolves through that exact call before trusting a training run — no guessing.

**Files:** none (verification task).

- [ ] **Step 1: Produce a tiny exported dataset from Plan A's smoke output**

Use a Plan-A smoke JSONL (from `environments/_m_self_incrimination/claude_state/smoke/`):
```bash
cd deps/verifiers && uv run --python .venv/bin/python -m verifiers.envs._m_.self_incrimination.export_prime_sft \
  --rollouts environments/_m_self_incrimination/claude_state/smoke/smoke_gpt_oss_20b.jsonl \
  --out-dir /tmp/si_sft_ds
```
Expected: prints `SFT_DATASET=/tmp/si_sft_ds n=... malicious=... honest=...`.

- [ ] **Step 2: Confirm prime-rl's loader call resolves it**

```bash
cd deps/verifiers && uv run --python .venv/bin/python python -c "
from datasets import load_dataset
ds = load_dataset('/tmp/si_sft_ds', None, split='train')
print('rows', len(ds), 'cols', ds.column_names)
print('roles', [m['role'] for m in ds[0]['messages']])
assert ds.column_names == ['messages']
assert any(m['role']=='assistant' for m in ds[0]['messages'])
print('OK local parquet dir loads via load_dataset(path, None, split=train)')
"
```
Expected: prints row count, `cols ['messages']`, and `OK ...`.

- [ ] **Step 3: If Step 2 fails** (loader can't read the local dir), STOP and report. Fallback options to discuss with the user (do NOT pick silently): (a) emit `train.jsonl` + load via `load_dataset('json', data_files=...)` — needs a tiny loader shim or a `--data.name json --data.data_files` path that prime-rl may not expose; (b) push the dataset to a private HF hub repo and reference by name (needs `HF_TOKEN`). Surface the failure and the two options rather than working around it.

---

## Task B3: SFT config + single-GPU smoke run on `Qwen3-0.6B`

**Files:**
- Create: `environments/_m_self_incrimination/configs/sft/qwen3_0p6b_smoke.toml`
- Create: `environments/_m_self_incrimination/configs/sft/README.md`

- [ ] **Step 1: Write the SFT config** (mirrors `examples/wordle/sft.toml`; local model + local dataset; loss on assistant tokens is the default, stated explicitly for clarity)

```toml
# environments/_m_self_incrimination/configs/sft/qwen3_0p6b_smoke.toml
max_steps = 30
output_dir = "/shared/matan/code/prime-rl/outputs/self_incrimination/sft_run"

[ckpt]  # checkpoint at the end

[model]
name = "/shared/matan/models/Qwen3-0.6B"

[data]
name = "/shared/matan/code/prime-rl/outputs/self_incrimination/sft_dataset"
seq_len = 2048
batch_size = 16
micro_batch_size = 1
# loss_mask defaults: assistant=true, others false — the report_scheming() sentinel is in the
# assistant turn, so it is trained. Stated explicitly:
[data.loss_mask]
assistant = true
user = false
system = false
tool = false

[optim]
lr = 1e-5

[deployment]
type = "single_node"
num_gpus = 1
```

- [ ] **Step 2: Export the FULL Plan-A dataset** (or, for the smoke, a smoke JSONL) to the path the config points at

```bash
cd deps/verifiers && uv run --python .venv/bin/python -m verifiers.envs._m_.self_incrimination.export_prime_sft \
  --rollouts <PLAN_A_DATASET.jsonl> \
  --out-dir /shared/matan/code/prime-rl/outputs/self_incrimination/sft_dataset
```

- [ ] **Step 3: Benchmark 10 steps + estimate wall-clock** (CLAUDE.md: benchmark before a real GPU run)

Pin GPU 0 via an exported var (NOT a backgrounded prefix), from the prime-rl ROOT:
```bash
cd /shared/matan/code/prime-rl
(export CUDA_VISIBLE_DEVICES=0; uv run sft @ deps/verifiers/environments/_m_self_incrimination/configs/sft/qwen3_0p6b_smoke.toml --max-steps 10 2>&1 | tee /tmp/si_sft_bench.log | tail -40)
```
Read the per-step timing from the tail; estimate total wall-clock for the intended `max_steps`. Flag if > a few hours (it won't be for 0.6B / 30 steps).

- [ ] **Step 4: Run the smoke SFT (30 steps) in the background + arm a monitor**

```bash
cd /shared/matan/code/prime-rl
(export CUDA_VISIBLE_DEVICES=0; uv run sft @ deps/verifiers/environments/_m_self_incrimination/configs/sft/qwen3_0p6b_smoke.toml > /tmp/si_sft_smoke.out 2>&1) &
```
Immediately arm the Monitor tool (persistent) with the training monitor:
```
~/.claude/scripts/monitor_training_job.sh --log-file /tmp/si_sft_smoke.out \
  --progress-pattern "step .*loss" --health-schedule "120,120,300,600"
```
(Confirm the actual progress-line regex from the bench log; adjust `--progress-pattern` to match prime-rl's loss line.)

- [ ] **Step 5: Spot-check after the run** (CLAUDE.md spot-check rule)

Confirm: loss decreased from step 1 → step 30; a checkpoint was written under `output_dir`; no NaNs/errors in the tail. Read the last ~30 lines of `/tmp/si_sft_smoke.out`.

- [ ] **Step 6: README + commit**

Write `environments/_m_self_incrimination/configs/sft/README.md` documenting the 3-command pipeline (Plan A `prepare_dataset` → `export_prime_sft` → `uv run sft @ qwen3_0p6b_smoke.toml`), the GPU-0 pin, and that loss is on assistant tokens incl. the sentinel. Commit:
```bash
cd deps/verifiers
git add environments/_m_self_incrimination/configs/sft/
git commit -m "feat(self-incrim): single-GPU SFT config + README for Qwen3-0.6B self-incrimination run

refs MSH-206"
```

---

## Task B4: small LR / seq-len sweep comparing final loss

**Files:**
- Create: `environments/_m_self_incrimination/configs/sft/sweep.py`

- [ ] **Step 1: Implement a tyro sweep driver**

It launches `uv run sft @ <base config> --optim.lr <lr> --data.seq_len <sl> --output-dir <per-run dir>` for each grid point (GPU 0, sequential — one GPU), tails each run's log, parses the final train-loss line, and prints a table. Keep it simple and explicit; no silent failure (raise if a run produces no parseable loss).

```python
# environments/_m_self_incrimination/configs/sft/sweep.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tyro

_PRIME_RL_ROOT = "/shared/matan/code/prime-rl"
_BASE_CONFIG = "deps/verifiers/environments/_m_self_incrimination/configs/sft/qwen3_0p6b_smoke.toml"
# Confirm this regex against the bench log in Task B3 before running; prime-rl prints a loss per step.
_LOSS_RE = re.compile(r"loss[\"']?\s*[:=]\s*([0-9.]+)", re.IGNORECASE)


@dataclass
class Args:
    lrs: tuple[float, ...] = (5e-6, 1e-5, 2e-5)
    seq_lens: tuple[int, ...] = (2048,)
    max_steps: int = 40
    gpu: int = 0
    out_root: str = "/shared/matan/code/prime-rl/outputs/self_incrimination/sft_sweep"


def _final_loss(log: str) -> float:
    matches = _LOSS_RE.findall(log)
    if not matches:
        raise ValueError("no parseable loss line in run log")
    return float(matches[-1])


def main(args: Args) -> None:
    results: list[tuple[float, int, float]] = []
    for lr in args.lrs:
        for sl in args.seq_lens:
            run_dir = Path(args.out_root) / f"lr{lr}_sl{sl}"
            cmd = [
                "uv", "run", "sft", "@", _BASE_CONFIG,
                "--optim.lr", str(lr), "--data.seq_len", str(sl),
                "--max-steps", str(args.max_steps), "--output-dir", str(run_dir),
            ]
            print(f"\n=== lr={lr} seq_len={sl} -> {run_dir} ===")
            env_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu}"
            proc = subprocess.run(
                f"{env_prefix} " + " ".join(cmd),
                shell=True, cwd=_PRIME_RL_ROOT, capture_output=True, text=True,
            )
            log = proc.stdout + proc.stderr
            (run_dir).mkdir(parents=True, exist_ok=True)
            (run_dir / "run.log").write_text(log)
            loss = _final_loss(log)
            print(f"final_loss={loss}")
            results.append((lr, sl, loss))
    print("\nlr\tseq_len\tfinal_loss")
    for lr, sl, loss in sorted(results, key=lambda x: x[2]):
        print(f"{lr}\t{sl}\t{loss:.4f}")
    best = min(results, key=lambda x: x[2])
    print(f"\nBEST lr={best[0]} seq_len={best[1]} final_loss={best[2]:.4f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 2: Dry-check the sweep wiring without GPU** (confirm tyro + command assembly)

Run: `cd deps/verifiers && uv run --python .venv/bin/python environments/_m_self_incrimination/configs/sft/sweep.py --help`
Expected: prints args (`--lrs`, `--seq-lens`, `--max-steps`, `--gpu`, `--out-root`), exit 0. **Do not run the real sweep yet.**

- [ ] **Step 3: Run the sweep on GPU 0 (after B3 smoke is green)** and arm a monitor; this is sequential over the grid (one GPU). Confirm `_LOSS_RE` matches the real prime-rl loss line first (from B3 bench log) and fix it if not.

- [ ] **Step 4: Report the loss table + best config; commit the sweep driver**

```bash
cd deps/verifiers
git add environments/_m_self_incrimination/configs/sft/sweep.py
git commit -m "feat(self-incrim): LR/seq-len SFT sweep driver comparing final loss

refs MSH-206"
```

---

## Open knobs / decisions to surface (not silently chosen)
- **Which dataset to SFT on:** the full balanced Plan-A run (≥150/≥150) vs a smoke JSONL. B3 smoke can use a smoke JSONL; the real run uses the full dataset.
- **`include_reasoning`:** default OFF (train on the delivered answer + sentinel only). For thinking actors (qwen) the 21k-char CoT is dropped — flip the flag only if we decide to train on reasoning, after checking the model's chat template handles `reasoning_content`.
- **Model:** `Qwen3-0.6B` for the try-it run; `/shared/matan/models/Qwen3-4B-Instruct-2507` is the heavier option (fits on this 140 GB GPU but slower).
- **Eval:** this plan stops at "loss goes down + checkpoint written". A held-out eval of whether the fine-tuned model actually emits `report_scheming()` iff confidently-wrong is a natural Plan C (re-run the env with the SFT'd model as actor and measure sentinel precision/recall vs the oracle).
