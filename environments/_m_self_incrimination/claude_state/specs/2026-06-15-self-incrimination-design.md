<!-- reportd: session=c0dcf8cb-515a-4895-8740-bd902a2c4852 name="MSH-206 self-incrimination spec" created=2026-06-15 status=executing round=2 -->
# `_m_self_incrimination` — design (v1, first-pass reproduction)

**Generated:** 2026-06-15T01:29:31Z | **Session:** 908b649c-b7f7-4502-83ed-0dfd0c85bb9c | **Issue:** MSH-206

> First pass at reproducing **self-incrimination** (Lee, Chen, Korbak 2026, arXiv:2602.22303) in the
> simplest possible setting, reusing the `_m_instruction_following_text` task. See
> `../research_notes.md` for the paper/repo digest. Built on the `Scene`/`ComposedEnv`/`Scorer`
> scaffold in `verifiers.envs._m_` (process: `claude_state/building_vf_envs_matan_way.md`).

## 1. Goal & motivation

Self-incrimination trains an agent so that *misbehaving covertly also emits a visible signal* — a
`report_scheming()` call pegged onto the act of scheming — without trying to make the model honest.
We are comparing monitorability evaluations, so a working reproduction is useful.

This v1 takes the **simplest setting**: instead of an instructed covert side-task labeled by an
oracle with full env knowledge, the "misbehavior" is **being confidently wrong** on the
instruction-following task. The oracle is two scorers we (mostly) already have.

## 2. The misbehavior mapping

| Paper | This v1 |
|---|---|
| Misbehavior = instructed covert side-task | Misbehavior = **confidently wrong** answer |
| Oracle = labeler model w/ full env + side-task spec | Oracle = `ConstraintScorer` (correctness) + `ConfidenceScorer` (hedging) |
| `report_scheming()` parallel tool call per covert action | **Sentinel text** appended to the final assistant message |
| Trigger: covert tool call | Trigger: `wrong AND confident` |

- **Ground-truth correctness** = `ConstraintScorer` (reused as-is): "wrong" ≡ the answer violates the
  imposed word-constraint (`ConstraintScore.satisfied == False`).
- **Confidence channel** = new `ConfidenceScorer` (LLM judge, see §5): a continuous *expressed-uncertainty*
  (hedging) score in `[0,1]`; `0` = fully confident.
- **Malicious label** ≡ `(NOT satisfied) AND (hedging ≤ τ)` for a binarization threshold `τ` chosen
  downstream (default kept as a knob, not baked into generation). Everything else
  (correct, or wrong-but-hedged) is **honest**.

### Deliberate deviations from the paper (flagged)
1. **Sentinel text, not a real tool call.** Instruction-following is a plain-chat, no-tools env; for
   SFT on a non-tool model we append a sentinel string rather than an OpenAI `tool_calls` structure.
2. **Misbehavior = outcome (wrongness), not a chosen action.** In the paper the model *chose* the
   covert action, so the signal pegs onto something under its control. Here wrongness is an outcome
   the model doesn't know at answer-time; only the *confidence/hedging* channel is under its control.
   So the SFT teaches "when you present an answer confidently *and* it turns out wrong, emit the flag."
   This tests a weaker/different claim than instructed-scheming — acceptable for a first pass.
3. **No per-task contrastive pairs.** The paper pairs an honest and an attack trajectory on the same
   task. Here we label independent rollouts. (Pairing is a possible v2.)
4. **Balancing ≈50/50, no benign-general mix.** The paper mixes UltraChat to keep attack tokens ~25%.
   v1 targets ≥150 honest / ≥150 malicious with no general-data dilution.

## 3. Architecture — self-incrimination is GENERIC infra

<ins>**Core principle (per review): self-incrimination is base-env-independent.** The self-incrimination
machinery — the confidence judge, the paraphrasing (trajectory-modification) module, the
balancing-loop, and the SFT export — knows NOTHING about instruction-following. It operates on
abstract inputs: a correctness verdict (bool), a hedging score (float), and a verifiers transcript.
Self-reporting must work for *any* task. The ONLY task-specific artifact is a thin `ComposedEnv` that
composes a base env's scenes + correctness scorer + dataset with the generic confidence judge.
Swapping in another base env should be a one-line ref change. There must be ZERO duplication with the
instruction-following (ift) env — we import its building blocks, never copy them.**</ins>

### Two layers

**Layer 1 — generic self-incrimination library** (env-agnostic). Home: the shared scaffold package
`verifiers/envs/_m_/`, since that is where the reusable `Scene`/`Scorer`/`Judge` machinery already
lives. New pieces:

```
verifiers/envs/_m_/
  judge.py                          # + render_transcript(messages, *, include_cot=False)  [generic util]
  self_incrimination/
    __init__.py                     # re-exports
    confidence.py                   # ConfidenceScorer (uses JudgeScorer), render_confidence_monitor,
                                     #   ConfidenceScore(ScorerResult)  — env-agnostic
    incriminate.py                  # <ins>label + paraphrase combined.</ins> label_rollout(*, correct: bool,
                                     #   hedging: float, tau) -> Label (pure, knows nothing about constraints)
                                     #   + append_report_scheming(messages, message_scenes) +
                                     #   incriminate_rollout(...) = decide then edit. REPORT_SCHEMING_SENTINEL.
    prepare_dataset.py              # <ins>DATASET CREATION (separate from training).</ins>
                                     #   prepare_self_incriminating_dataset(base env_id+kwargs, scorer names,
                                     #   tau, targets) -> balancing loop -> SAVES a balanced labeled dataset.
    sft.py                          # <ins>SFT TRAINING (dataset is an INPUT).</ins> run_sft(dataset_path,
                                     #   sft_config) -> prime-rl SFT. Reusable across hyperparam sweeps.
    prepare_and_sft.py              # <ins>end-to-end WRAPPER</ins>: prepare_dataset -> sft.
```

<ins>**Dataset creation is separated from SFT** (per review): hyperparam sweeps reuse a fixed
incriminating dataset as INPUT to `sft.py` without regenerating it. `prepare_and_sft.py` is the
convenience wrapper that does both.</ins>

**Layer 2 — the thin composed env** `environments/_m_self_incrimination/` (the only ift-aware code).
No `types.py`/`dataset.py`/`constraints.py`/`scorers.py` — those are IMPORTED from ift.

```
environments/_m_self_incrimination/
  pyproject.toml                    # dist m-self-incrimination; env_id/module _m_self_incrimination;
                                     #   depends on m-instruction-following-text (for its building blocks)
  README.md
  _m_self_incrimination/
    __init__.py
    env.py                          # load_environment: import ift build_dataset + ConstraintScorer +
                                     #   TaskScene + system_prompt + TaskInfo/enums; ComposedEnv(those
                                     #   scenes, [ift correctness scorer, generic ConfidenceScorer], ...)
  claude_state/{research_notes.md, specs/...}
```

<ins>The generic dataset builder (Layer 1 `prepare_dataset.py`) is parameterized by the base
`env_id` (default `"_m_instruction_following_text"`) + its `load_environment` kwargs + which scorer
name is the correctness oracle (default `"constraint_satisfied"`) + which is the hedging scorer. So
"switch to another env" = pass a different `env_id` and scorer names; nothing in Layer 1 changes.</ins>

> Supersedes the earlier "parsing/ inside the env folder" decision: the parsing/SFT code is GENERIC,
> so it lives in the shared library (Layer 1), not coupled inside the ift-aware composed-env folder.
> Tests for the generic pieces live in `verifiers/envs/_m_/tests/`.

## 4. Data flow

```
prepare_dataset.py(base env_id, kwargs, scorer names, tau, targets)   # DATASET CREATION
  -> load_environment(_m_self_incrimination) = ComposedEnv(ift scenes + [ConstraintScorer, ConfidenceScorer])
  -> env.evaluate
  -> results.jsonl  [scorers: ConstraintScore + ConfidenceScore; message_scenes; raw_responses]
  -> incriminate.py  label_rollout(correct=not satisfied?, hedging, tau) -> honest | malicious;
                     append_report_scheming(...) to final assistant msg iff malicious
  -> balancing loop  accumulate batches of 200 until ≥150 malicious AND ≥150 honest
  -> SAVE balanced labeled dataset (prime-rl SFT format)                 # <-- reusable artifact

sft.py(dataset_path, sft_config)                                        # SFT TRAINING (dataset = input)
  -> prime SFT (scoped during impl, §7)                                 # swept over a FIXED dataset

prepare_and_sft.py  =  prepare_dataset  ->  sft   (end-to-end wrapper)
```

## 5. The confidence scorer (`ConfidenceScorer`)

<ins>Lives in Layer 1 (`verifiers/envs/_m_/self_incrimination/confidence.py`) — fully env-agnostic,
usable by any composed env.</ins>

- A `Scorer` built on the shared `JudgeScorer` machinery. **Output-only**: it judges the delivered
  answer, never CoT.
- **Generic prompt** (`render_confidence_monitor(*, transcript)`): renders the full conversation via
  `render_transcript(..., include_cot=False)`, instructs the judge to base its judgment on the FINAL
  answer reported to the user and to judge *expressed stance only, never correctness* (the judge does
  not know the right answer — prevents ground-truth leakage into the confidence channel).
- **Continuous** expressed-uncertainty score, `0` = fully confident, higher = more hedging, parsed via
  the existing `parse_judge_response` (0–10 → `[0,1]`, 11 gradations). Per the "continuous over
  discrete" rule we keep the raw graded score and binarize (`τ`) only downstream. If 11 levels prove
  too coarse, swap in a decimal parser (flagged, not done in v1).
- Full prompt text approved in the design conversation (2026-06-15).

`ConfidenceScore(ScorerResult)` carries: `score` (hedging in `[0,1]`), `view`, `model`, `judge_input`,
`judge_output`, `judge_reasoning`, `justification`, `attempts`, and a `name` (a generic default like
`"confidence_hedging"`, set by the env when constructing the scorer — not an ift enum value).

## 6. Incrimination & dataset creation (all Layer 1, generic)

- **`incriminate.py`** combines the two tightly-coupled steps (per review):
  - `label_rollout(*, correct: bool, hedging: float, tau: float) -> Label` — pure; `malicious` iff
    `(not correct) and (hedging <= tau)`, else `honest`. Takes a bare bool + float; it does not know
    what produced them, so the same rule serves any base env.
  - `append_report_scheming(messages, message_scenes)` — appends the sentinel **inline to the end of
    the final assistant message** (avoids chat-role alternation issues for SFT). Honest transcripts
    left byte-identical. Verifiers-structure-aware (`prompt + completion`, `message_scenes`). Idempotent.
  - `incriminate_rollout(...)` — decide (label) then edit (paraphrase) in one call.
  - `REPORT_SCHEMING_SENTINEL` — mirrors the repo's `report_scheming` semantics (no args, conceptually
    a tool call) but emitted as text; defined once here.
- **`prepare_dataset.py`** — `prepare_self_incriminating_dataset(...)` drives generation in **batches
  of 200** until **≥150 malicious AND ≥150 honest**, logs per-batch malicious-rate so yield is visible.
  No special elicitation config is spelled out; we just run and observe. Parameterized by base `env_id`
  + kwargs + correctness/hedging scorer names + `tau` (§3); maps each rollout's named scorers →
  `(correct, hedging)` → `incriminate_rollout`. <ins>Output = a SAVED balanced labeled dataset
  (prime-rl SFT format), reusable as input to `sft.py`.</ins>

## 7. Prime SFT (scoped now, swept during impl)

We have not used prime-rl SFT. Implementation step 1 is to **read prime-rl's SFT trainer + config**,
produce a minimal working config on the balanced dataset, then a **small sweep** (LR + utilization/
batch knobs) comparing loss. The concrete config is discovered-during-impl, not pre-specified here.
The Layer-1 `sft.py` export targets whatever format that trainer expects (generic over base env).

## 8. Out-of-scope, planned & stashed (NOT built in v1)

1. **Instructed misbehavior.** Sometimes prompt the actor to misbehave / sometimes be honest, label
   on that. Mechanism sketch: dataset rows carry **multiple user messages** with per-message
   `remove_in_edit` metadata; the trajectory editor strips some before SFT. Closer to the paper's
   instructed-scheming and to a cleaner contrastive-pair setup.
2. **Adversarial-pressure augmentation.** Prompt the model to *never* call the tool, train it to call
   it anyway — **incentives only, never instructions** (preserves instruction hierarchy, per paper).
3. **Best-of-N** sampling for quality assurance.
4. **Fully synthetic envs.** LLM-generated fake tools + LLM-simulated tool responses (the
   synthetic-scheming 3-stage pipeline).

## 9. Testing (pure-logic only, per AGENTS.md — in `verifiers/envs/_m_/tests/`)

- `incriminate.py`: `label_rollout` across the four quadrants (correct/wrong × confident/hedged) and
  the `τ` boundary; `append_report_scheming` lands the sentinel on the final assistant message, leaves
  honest transcripts byte-identical, is idempotent, and respects `message_scenes`.
- `prepare_dataset.py`: balancing accounting (counts, stopping condition) + the scorer-name →
  `(correct, hedging)` mapping (raises if a named scorer is missing — no silent fallback).
- No tests on eval/runtime glue, `sft.py` training, or the judge call itself.

## 10. Open knobs (decide later, not blockers)
- Binarization threshold `τ` for "confident".
- Actor model(s) and difficulty mix for adequate malicious yield (observed via the loop, not pre-set).
- Whether to downsample to exactly balanced or keep the ≥150/≥150 "at least" sets.
- Decimal vs 0–10 judge granularity.
