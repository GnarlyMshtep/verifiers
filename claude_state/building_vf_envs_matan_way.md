# Building verifiers envs — Matan's way (tentative, v1, 2026-06-12)

> Early and provisional. Written while building the first env (`instruction_following_text`).
> Point future env-building agents here. Expect this to change.

## Where & how
- Author envs flat in `deps/verifiers/environments/<name>/`. Mark authored in README +
  register in this file. env_id = hyphenated name; importable module = name w/ underscores.
- Multi-file package layout (NOT single .py):
  - `pyproject.toml` (`[tool.hatch.build] include`, `[tool.verifiers.eval]` defaults).
  - `<pkg>/types.py` — StrEnums + frozen dataclasses for boundary/value types (e.g. `ConstraintName`,
    `Difficulty`, `JudgeView`, `Problem`, `MsgProvenance`). Prefer enums over bare strings everywhere.
  - `<pkg>/prompts.py` — `PromptKey(StrEnum)` + `PROMPTS` registry + python template fns (prefer
    python over prompt.json: keeps templates + types + discoverability).
  - `<pkg>/constraints.py` (or domain rules) — pure rule-based verifiers; keyed by an enum.
  - `<pkg>/dataset.py` — builds the HF Dataset; rows = `{question, answer, info}` where
    `info = asdict(Problem)` (the per-example spec dataclass).
  - `<pkg>/scorer_types.py` — `Scorer` ABC + `ScorerResult` dataclass hierarchy + `ScorerName` enum.
  - `<pkg>/scorers.py` — concrete `Scorer`s (rule + judge) + judge-view rendering + score parsing.
  - `<pkg>/turn_logic.py` — `Turn` ABC (conversation flow only) + `ComposedEnv` + `messages_for_turn`.
  - `<pkg>/env.py` — `load_environment`, ONE config-driven env (judge view, difficulty filter,
    weights via config — no near-duplicate env classes).
  - `<pkg>/data/` (small reviewable json); `tests/` (pytest plain fns; ONLY pure logic).
- **The env package contains ONLY env code.** `environments/<name>/<pkg>/` holds the env itself
  (types, prompts, constraints, dataset, scorer_types, scorers, turn_logic, env, data, tests).
  Anything downstream of the env — store integration, flattening, plotting, stats, registration,
  analysis — does NOT live in the env package. It lives with the consumer (prime-rl) under
  `claude_scripts/analysis_scripts/<env>/` (its own importable package, e.g.
  `analysis_scripts.instruction_following_text`). Rule of thumb: if it imports the env to GENERATE
  or DEFINE rollouts/scores, it's env code; if it CONSUMES saved results/registered data, it's
  analysis code and belongs in `analysis_scripts/<env>/`. Do NOT name the analysis package after
  the env module (import collision) — nest it under `analysis_scripts/`.
- Install into the verifiers venv: `uv pip install --python .venv/bin/python -e environments/<name>`
  (python 3.13 at `deps/verifiers/.venv`; repo conda env is 3.10 and won't work).
- The launcher/scripts need `tyro` + `python-dotenv` in the verifiers venv:
  `uv pip install --python .venv/bin/python tyro python-dotenv`.
- NOTE: `verifiers.clients` lives in the editable vendored `verifiers/` (parent repo), which is
  resolved when running from the `deps/verifiers` cwd. The env's `.venv`-pinned `verifiers` may be
  older and lack `verifiers.clients` — run env/eval commands from `deps/verifiers`, or bump the pin.

## Turns vs Scorers (the split)
- verifiers gives ONE env→actor hook (`env_response`) + stop conditions (`@vf.stop`);
  `rollout` is `@final`. Don't fight it.
- **Turns are conversation flow only.** `Turn` ABC = `enter(state)->Messages` (turn 0 is
  pre-entered by the dataset prompt, so its enter()==[]), `is_complete(state)->bool` (default
  True; the seam for future multi-message/tool turns). Turns do NOT score.
- **Scorers grade the trajectory.** Scoring is trajectory-level by default (most scorers look at
  the whole completion). A `Scorer` (ABC) has `name: ScorerName`, `weight`, and
  `async score(*, prompt, completion, answer, problem, state) -> ScorerResult`.
- `ComposedEnv(turns, scorers, **kwargs)` walks `turns`, records per-message provenance in
  `state["message_turns"]` (dataclass asdict, index-aligned with `state["completion"]`), parses
  `info -> Problem` once (dacite) into `state["_problem"]`, and wraps each `Scorer` into a verifiers
  reward fn (`__name__ = scorer.name.value`) that appends `asdict(result)` to `state["scorers"]`
  and returns `result.score`. Weights come from `scorer.weight`.
- **Per-turn scoring is the exception.** A scorer that needs one turn filters the completion with
  `messages_for_turn(completion, state, turn_idx)` (provenance-based) inside its own `score`.
- Multi-message/tool turns = implement the `is_complete=False` branch of `env_response`
  (currently raises). Per-turn TRAINING credit needs a separate prime-rl `train_sink.py`
  change (eval unaffected — rubric scalars log fine).

## ScorerResult dataclasses (structured logging)
- `ScorerName(StrEnum)` names each scorer; the metric column = the enum value (clean, no `__turn`).
- `ScorerResult` base = `{score: float}`. Subclasses add typed fields then re-declare
  `name: ScorerName = <default>` as the LAST field (dataclass ordering: the only defaulted field
  must come after the non-default ones — else `TypeError: non-default argument follows default`).
- `asdict(result)` serializes the enum to its string value; `state["scorers"]` is a LIST of these
  dicts (each carries its `name`) — NO string-keyed scorer dict.
- The score is intentionally duplicated: verifiers puts it in `metrics` (drives avg_metrics/pass@k)
  AND we keep it in `scorers[i].score` with the full per-scorer detail. Accept the dup.

## Judge / CoT visibility
- Judge = a `Scorer` calling `AsyncOpenAI` (OpenRouter base_url). It sees what
  `render_view(completion, JudgeView)` builds from each assistant msg's `content` and/or
  `reasoning_content`. Score 0-10 -> [0,1] via `parse_score` (rejects decimals/out-of-range).
- Launch reasoning judges with `extra_body={"reasoning":{"effort":"medium"}}`.
- READING the judge's own reasoning: OpenRouter returns it in the **`reasoning`** field (NOT
  `reasoning_content`). Read `getattr(msg,"reasoning",None) or getattr(msg,"reasoning_content",None)`.
- ACTOR reasoning for the cot/both VIEW: verifiers normalizes the provider `reasoning` field into
  the completion message's `reasoning_content`, so render_view reads `reasoning_content` there.
  vLLM actors need `--reasoning-parser <name>` (e.g. `qwen3`) or `reasoning_content` is absent and
  the cot/both view raises (by design — no silent fallback). Qwen3 reasoners need a LARGE
  `max_tokens` (≥~4-6k) or they hit `finish_reason=length` mid-think and return empty content.
- gpt-oss-20b via OpenRouter intermittently returns an empty completion (finish_reason=stop, no
  content) — NOT truncation. The `JudgeScorer` retries `judge_attempts` (default 3) on
  empty/unparseable responses; every attempt's full provider response is kept in `JudgeScore.attempts`.

## Logging (entire computation trace)
- Write JSON-serializable data to `state[key]` during the rollout, pass `state_columns=[...]`
  to `evaluate()` -> lands in results.jsonl (raises if not serializable; no silent drop).
- Standard columns: `scorers` (list of ScorerResult dicts: score + typed detail incl. judge
  input/output/reasoning/attempts), `scorer_errors` (see fail-loud note), `message_turns`
  (per-msg provenance), `raw_responses` (full actor API JSON).
- Custom dated dir + sidecar config: don't fight `path_utils` — a launcher calls
  `env.evaluate(results_path=...)` and writes only the deltas vf's metadata.json lacks
  (datetime, actor backend, short_desc). See `claude_scripts/run_instruction_eval.py`.

## Rescoring (replay-and-rescore, no regeneration)
- `Environment.rescore(source_results_path, results_path=..., state_columns=..., save_results=...)`
  replays saved rollouts and re-runs ONLY the rubric — no actor calls. It loads results.jsonl,
  rebuilds each `State` via `output_to_state` (the inverse of `state_to_output`), runs
  `env.setup_state` (rebuilds `_problem` etc.) + `_score_state`, and writes new outputs.
- To score the SAME saved rollouts a different way, construct the env differently and rescore:
  e.g. `load_environment(judge_view=..., judge_model=...)` then `.rescore(<run>)`. ConstraintScorer
  recomputes identically; JudgeScorer scores per the new view/judge. CoT views work because saved
  completions keep `reasoning_content`.
- **`rescore` never takes an actor model/client** — the actor is fixed by the saved file; the
  builder's model field is a literal `"(rescore)"` sentinel. Downstream registration must NOT trust
  a hand-passed actor: derive the actor tag from the source run's `metadata.json`/`config.json` and
  RAISE on mismatch if an actor is also passed explicitly (you cannot silently mislabel whose
  rollouts these are). Only `independent` (per-rollout) scoring is supported; group-reward rubrics
  raise.

## Gotchas learned
- **Weights:** `MultiTurnEnv.__init__` wraps your `vf.Rubric` in a `RubricGroup` (adds a monitor
  rubric). So `env.rubric` is a RubricGroup and your funcs live in `env.rubric.rubrics[0]`. Set
  weights at `vf.Rubric(funcs=..., weights=...)` CONSTRUCTION time — `env.rubric.weights = ...`
  afterward targets the wrapper and is silently ignored.
- **Fail-loud is NOT possible inside a reward fn.** verifiers' `Rubric` wraps every reward fn in
  `except Exception -> log -> 0.0`. So a scorer that raises does NOT error the rollout; it silently
  becomes reward 0.0. Mitigation: record the failure trace to `state["scorer_errors"]` BEFORE
  raising (verifiers keeps the same `state`, so it serializes), and have the launcher print a loud
  run-level warning when any rollout has `scorer_errors`. True per-rollout fail-loud requires a
  verifiers fork change.
- **Raw actor-response logging** via re-classing `state["client"].__class__ = RawLoggingChatClient`
  mutates the client in place. Safe when the env owns a dedicated client (our launcher passes a
  ClientConfig -> a fresh client per process). For a SHARED training client this leaks (other
  rollouts also write `raw_responses`); replace with a per-rollout wrapper there.
- **`info -> Problem` via dacite** needs `config=dacite.Config(cast=[<the StrEnum types>])` to
  convert the serialized strings back into enums.
- **`split_sentences`** (or any naive `[.!?]` splitter) over-splits abbreviations/decimals. Fine
  for pilot constraints; revisit if the model trains against the scorer and exploits it.

## Git / remote
- Our envs + verifiers tweaks live on a FORK of verifiers (`GnarlyMshtep/verifiers`), remote
  `fork` in the submodule. Branch per issue (`exp/MSH-###-...`). `git add` only specific files
  (the submodule has unrelated dirty deletions). Push to `fork` so commits link to Linear
  (webhook installed on the fork). Commit trailer: `refs MSH-###` + `Claude-Session:`.
- The repo's `deps/research-environments` submodule is PrimeIntellect's dedicated custom-env home;
  we chose the verifiers fork instead because we also want to tweak verifiers itself.

## Sweeps (FYI, not built)
- vf-eval multi-model sweeps only via a TOML with multiple `[[eval]]` blocks; CLI is one
  model per call. We use individual launcher invocations.

## Registry (authored envs)
- `instruction_following_text` — instruction following w/ word constraints + scoring judge
  (Scorer/ScorerResult architecture; ConstraintScorer + JudgeScorer).
