# Building verifiers envs — Matan's way (tentative, v1, 2026-06-13)

> Early and provisional. Written while building the first env (`_m_instruction_following_text`).
> Point future env-building agents here. Expect this to change.
>
> The reusable scaffold (`Scene`, `ComposedEnv`, `Scorer`/`ScorerResult`, the Judge layer) lives in
> the shared library package `verifiers.envs._m_` (`deps/verifiers/verifiers/envs/_m_/`). An env
> imports from there and contributes only env-specific code.

## How to build an env (the workflow)

Follow this spine top to bottom. The Reference half below fills in the details.

1. **Flush out the idea with the user.** Ask clarifying questions about the purpose, the task, the
   reward, and the success criteria. Do not start building before the idea is fleshed out.
2. **Confirm the architecture: Scene + ComposedEnv.** State the purpose explicitly — *extensible
   envs* + *separating the logic of a scene sequence from the env definition itself*. The shared
   `verifiers.envs._m_` package carries the env-agnostic machinery; the env adds constraints/rules,
   prompts, dataset, concrete `Scorer`s, and its `Scene` classes.
3. **`/writing-plans` for the env.** Write the plan before touching code.
4. **Build tests for edge-case rollouts** and verify they're handled correctly. **Think hard about
   edge-case reward-hacking rollouts** specifically — what would a model do to game the reward?
5. **Flag any non-obvious deviation** that might surprise the user.
6. **Smoke-test via a subagent.** The building actor launches a subagent to run a smoke test on a
   few cheap models — **deepseek-v4-flash**, **gpt-oss-20b**, **qwen/qwen3-next-80b-a3b-thinking**
   (assume ~$5 OpenRouter spend is OK). The subagent reads a few rollouts, paying special attention
   to:
   - (a) **all relevant parts of the trajectory are logged** — err on the side of "is relevant";
   - (b) **prompts contain the information they should**;
   - (c) **actors are interpreting the trajectory correctly**.
7. **Iterate**: up to 3 subagent iterations, or stop once the subagent is satisfied.
8. **Hand the human the logged dirs** for a full pass.

### Vetting a rule + monitor env (three practices worth keeping)

These tighten step 4 (edge-case reward-hacking) and step 6 (smoke-test) for any env whose reward
splits into a *rule-based verifier* + an *LLM monitor*. They emerged building the parameterized
constraints in `_m_instruction_following_text`; reuse them.

1. **Unbiased reward-hack review (fresh subagent).** The builder is too close to its own logic to
   see the holes. Dispatch a FRESH subagent (Opus) with no attachment to the implementation to hunt
   reward hacks + edge cases and to WRITE tests that try to break each verifier. Triage what it
   finds: genuine verifier bugs — a wrong verdict for the rule's own definition, a crash, an
   infeasible param draw — get fixed; intended "monitor-must-catch" gaps (rule-satisfying gibberish
   or off-topic answers) become documented expectations that the monitor stress-test (practice 3)
   must cover. (This round surfaced `split_sentences` over-splitting abbreviations/ellipses, a
   punctuation word-count hack, accented-letter miscounts, and an infeasible `letter_set` q-without-u
   draw — all fixed.)

2. **Run-once, rescore-many.** The cost is the actor calls; the judge is cheap. Generate rollouts
   ONCE per actor — with reasoning enabled so `reasoning_content` is captured for the cot/both views
   — then `Environment.rescore` across the ENTIRE monitor grid (monitors × views × monitor-prompts).
   Local vLLM monitors make the grid effectively free. Never regenerate actor rollouts just to change
   the judge. (See the "Rescoring" reference section; grid driver `claude_scripts/rescore_grid.py`.)

3. **Monitor stress-test before trusting smoke numbers.** Before believing ANY monitor Pearson/AUC,
   verify the monitor actually rejects rule-satisfying gibberish / off-topic answers — the
   proper-English-and-answers-the-request half it is responsible for. Build adversarial transcripts
   that PASS the rule but should score low, plus genuine good answers that should score high, and
   confirm the monitor separates them. A monitor that doesn't separate makes the constraint look
   reward-hackable and its smoke correlations untrustworthy. (Script:
   `claude_scripts/stress_test_monitors.py`; report: `claude_state/ift-monitor-stress.md`.)

A fourth subtlety: **calibrate difficulty under the SAME actor conditions as the real run.**
Difficulty knobs tuned with actor reasoning OFF are miscalibrated if the real sweep runs with
reasoning ON (a strong reasoning actor complies far more easily). Tune on a representative actor with
reasoning in the mode you'll sweep.

---

# Reference

## Where & how
- Author envs flat in `deps/verifiers/environments/<name>/`. Mark authored in README +
  register in this file. The env_id, the importable module, and the directory share the SAME name
  (e.g. `_m_instruction_following_text`); the loader uses the MODULE name. The PEP 508 distribution
  name in `pyproject.toml` may differ (a leading `_` is forbidden there, so the distribution is
  `m-instruction-following-text`) — but env_id stays the literal module name `_m_instruction_following_text`.
- Multi-file package layout (NOT single .py). The env package contributes only env-specific files;
  the shared `Scene`/`Scorer`/`Judge`/`ComposedEnv` machinery is imported from `verifiers.envs._m_`:
  - `pyproject.toml` (`[tool.hatch.build] include`, `[tool.verifiers.eval]` defaults).
  - `<pkg>/types.py` — StrEnums + frozen dataclasses for boundary/value types (e.g. `ConstraintName`,
    `Difficulty`, `SceneName`, `TaskInfo`). Prefer enums over bare strings everywhere. (`JudgeView`
    and `MsgProvenance` are NOT here — they live in the shared package.)
  - `<pkg>/prompts.py` — `PromptKey(StrEnum)` + `PROMPTS` registry + python template fns (prefer
    python over prompt.json: keeps templates + types + discoverability).
  - `<pkg>/constraints.py` (or domain rules) — pure rule-based verifiers; keyed by an enum.
  - `<pkg>/dataset.py` — builds the HF Dataset; rows = `{question, answer, info}` where
    `info = asdict(task_info)` (the per-example spec dataclass; see "Nesting `info`" below).
  - `<pkg>/scorer_types.py` — env-specific `ScorerName` enum + `ScorerResult` subclasses
    (e.g. `ConstraintScore`) + judge config; the `Scorer`/`ScorerResult` BASE classes are imported
    from `verifiers.envs._m_`.
  - `<pkg>/scorers.py` — concrete env `Scorer`s (e.g. `ConstraintScorer`). The `JudgeScorer` +
    judge-view rendering + score parsing now live in the shared package; import them from there.
  - `<pkg>/env.py` — `load_environment`, the env's `Scene` subclasses (e.g. `TaskScene`), and ONE
    config-driven env (judge view, difficulty filter, weights via config — no near-duplicate env
    classes). Constructs `ComposedEnv(scenes=..., scorers=..., info_type=..., info_enums=..., ...)`.
  - `<pkg>/data/` (small reviewable json); `tests/` (pytest plain fns; ONLY pure logic).
- **The env package contains ONLY env code.** `environments/<name>/<pkg>/` holds the env itself
  (types, prompts, constraints, dataset, scorer_types, scorers, the env's `Scene`s, env, data, tests).
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

## Scenes vs Scorers (the split)
- verifiers gives ONE env→actor hook (`env_response`) + stop conditions (`@vf.stop`);
  `rollout` is `@final`. Don't fight it.
- **Scenes are conversation flow only.** The `Scene` ABC (`verifiers.envs._m_`) =
  `enter(state)->Messages` (the env message(s) that open the scene) + `is_complete(state)->bool`
  (after the actor replies, is the scene done?). Both are abstract — see "Scenes & provenance"
  for why. Scenes do NOT score.
- **Scorers grade the trajectory.** Scoring is trajectory-level by default (most scorers look at
  the whole completion). The `Scorer` ABC has `name: str`, `weight`, and
  `async score(*, prompt, completion, answer, task_info: object, state) -> ScorerResult`. Env
  subclasses narrow `task_info` to their own `TaskInfo` in the override signature.
- `ComposedEnv(scenes, scorers, *, info_type, info_enums=(), **kwargs)` walks `scenes`, records
  per-message provenance in `state["message_scenes"]` (dataclass asdict, index-aligned with the FULL
  conversation = `prompt + completion`), parses `info -> info_type` once (dacite, casting
  `info_enums`) into `state["_task_info"]`, and wraps each `Scorer` into a verifiers reward fn
  (`__name__ = str(scorer.name)`) that appends `asdict(result)` to `state["scorers"]` and returns
  `result.score`. Weights come from `scorer.weight`.
- **Per-scene scoring is the exception.** A scorer that needs one scene filters the full
  conversation with `messages_for_scene(messages, state, scene_idx)` (provenance-based) inside its
  own `score`.
- Multi-message/tool scenes = implement the `is_complete=False` branch of `env_response`
  (currently raises). Per-scene TRAINING credit needs a separate prime-rl `train_sink.py`
  change (eval unaffected — rubric scalars log fine).

## Scenes & provenance
- **The `enter` footgun (why both methods are abstract).** Control flow in `ComposedEnv`:
  `setup_state` sets `scene_idx=0`; **scene 0's opening is the dataset prompt** — verifiers feeds
  `state["prompt"]` directly to the actor. `env_response` runs only *after* an actor reply: it
  checks the current scene's `is_complete`, increments the index, then calls `scenes[idx].enter()`.
  So `scenes[0].enter()` is **never called**, and a silent `enter()->[]` default would only ever
  fire for scenes ≥1 — where `[]` is *wrong* (a later scene opening with no env message hands the
  floor back to the actor with nothing new). Making `enter` abstract forces every scene to declare
  its opening: a scene-0 class writes an explicit `return []` ("opened by the dataset prompt"), later
  scenes return real messages. `is_complete` is abstract too (`True` is a sane default for
  single-message scenes, but it should be stated, not assumed).
- **The message-provenance model (uniform over `prompt + completion`).** Provenance answers, for
  every message: *which scene does it belong to, and who emitted it?* `MsgProvenance` =
  `{scene_idx, scene_name, source}` with `source ∈ {system, dataset, env, actor}`:
  - `system` — the global system prompt (`scene_idx=0`; persists across scenes).
  - `dataset` — the per-example opening request supplied as the initial prompt = scene 0's opening
    (what a scene-0 `enter()` documents by returning `[]`).
  - `env` — emitted by a scene's `enter()` (scenes ≥1).
  - `actor` — emitted by the policy.
  Recording sites: **`setup_state`** tags each `state["prompt"]` message (system→`system`,
  user→`dataset`, both `scene_idx=0`); **`env_response`** tags `enter()` output (`env`);
  **`add_model_response`** tags the actor reply (`actor`). So `state["message_scenes"]` covers the
  WHOLE conversation and scene 0 stops being a special case. Consumed by
  `messages_for_scene(messages, state, scene_idx)`, which zips the full conversation against
  `message_scenes`.

## ScorerResult dataclasses (structured logging)
- `ScorerName(StrEnum)` names each env scorer; the metric column = the enum value (clean, no `__scene`).
- The shared `ScorerResult` base = `{score: float}`. Env subclasses add typed fields then re-declare
  `name` (typically `: ScorerName = <default>`) as the LAST field (dataclass ordering: the only
  defaulted field must come after the non-default ones — else `TypeError: non-default argument
  follows default`). The shared `JudgeScore` takes `name` as a plain `str` field.
- `asdict(result)` serializes the enum to its string value; `state["scorers"]` is a LIST of these
  dicts (each carries its `name`) — NO string-keyed scorer dict.
- The score is intentionally duplicated: verifiers puts it in `metrics` (drives avg_metrics/pass@k)
  AND we keep it in `scorers[i].score` with the full per-scorer detail. Accept the dup.

## Judge / CoT visibility
- The shared `JudgeScorer` (`verifiers.envs._m_`) is a `Scorer` calling `AsyncOpenAI` (OpenRouter
  base_url). `name` is a ctor arg (the env passes e.g. `ScorerName.JUDGE_REQUEST_FOLLOWED`). It sees
  what `render_view(completion, JudgeView)` builds from each assistant msg's `content` and/or
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

## Prompts
- Keep prompt strings in a module-level `write_once(...)` dict keyed by a `StrEnum`; render via
  `strict_format` (raises on any missing/unused `{key}` — a renamed/dropped placeholder fails at call
  time, not silently). Exception: prompts that embed user/CoT text with literal braces use a template
  *function* + plain `.format` of only the named fields (as the instruction-following monitor and the
  confession judge prompt do — the rendered transcript routinely contains `{`/`}`). Both helpers
  (`write_once`, `strict_format`) live in `verifiers/envs/_m_/utils.py`.

## Selective loss masking
- Set `Scene(trainable=False)` and `ComposedEnv` zeroes that scene's `completion_mask`
  (loss-free context) — its `add_model_response` overwrites the just-generated step's
  `tokens["completion_mask"] = [0]*len(...)`, the only place the all-1s on-policy default is changed.
  prime-rl carries those zeros straight into the loss mask. `log_token_masks=True` records a per-step
  `{scene_idx, scene_name, trainable, n_tokens, n_trainable, text_preview}` attribution row in
  `state["token_mask_log"]`. This is how confession training masks the task answer and trains only
  the confession.

## Logging (entire computation trace)
- Write JSON-serializable data to `state[key]` during the rollout, pass `state_columns=[...]`
  to `evaluate()` -> lands in results.jsonl (raises if not serializable; no silent drop).
- Standard columns: `scorers` (list of ScorerResult dicts: score + typed detail incl. judge
  input/output/reasoning/attempts), `scorer_errors` (see fail-loud note), `message_scenes`
  (per-msg provenance over prompt + completion), `raw_responses` (full actor API JSON).
- Custom dated dir + sidecar config: don't fight `path_utils` — a launcher calls
  `env.evaluate(results_path=...)` and writes only the deltas vf's metadata.json lacks
  (datetime, actor backend, short_desc). See `claude_scripts/run_instruction_eval.py`.

## Rescoring (replay-and-rescore, no regeneration)
- `Environment.rescore(source_results_path, results_path=..., state_columns=..., save_results=...)`
  replays saved rollouts and re-runs ONLY the rubric — no actor calls. It loads results.jsonl,
  rebuilds each `State` via `output_to_state` (the inverse of `state_to_output`), runs
  `env.setup_state` (rebuilds `_task_info` etc.) + `_score_state`, and writes new outputs.
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

## Request timeouts (bound stragglers — eval AND rescore)
- **Always set per-request timeouts; the vf default is 3600 s (1 h).** `ClientConfig.timeout`
  defaults to one hour, so a single wedged/rate-limited request holds a concurrency slot for up to an
  hour instead of failing fast. With actor reasoning on (or a flaky OpenRouter provider — e.g.
  nemotron-nano threw 125× 429 in one 350-row sweep) the long tail dominates wall-clock.
- **Actor and judge timeouts are SEPARATE clients.** The actor uses the `ClientConfig` the launcher
  passes to `env.evaluate` (set `timeout=` there — exposed as `run_instruction_eval.py
  --request-timeout`). The judge uses a different `AsyncOpenAI(...)` built in the env's `env.py`
  (`load_environment(..., judge_timeout=...)` → `AsyncOpenAI(timeout=...)`). The actor timeout does
  NOT reach the judge, and **`rescore` makes only judge calls** — so to bound rescore you MUST set
  `judge_timeout` (it threads through `rescore_grid.py --judge-timeout`). This is per-env: each env's
  `env.py` builds its own judge client, so the knob lives there, not in `ComposedEnv`.
- **Tune heuristically to ~95% of requests landing within the timeout.** Set the timeout so roughly
  95% of requests complete inside it and the slow ~5% tail errors out (recorded as `scorer_errors` /
  `monitor_parse_error`, dropped from stats — kept on disk as reward-0 trajectories) rather than
  stalling the run. Eyeball the per-request durations (the tqdm `s/it` and the rollout `timing`
  field) from a smoke run and pick the knee of the distribution; don't over-tighten (genuine
  slow-but-alive reasoning rollouts still need `max_retries` headroom). First-draft values for this
  env: actor `--request-timeout 240`, `judge_timeout 120` (local vLLM judges rarely straggle).

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
  (shared `verifiers.envs._m_`) mutates the client in place. Safe when the env owns a dedicated
  client (our launcher passes a ClientConfig -> a fresh client per process). For a SHARED training
  client this leaks (other rollouts also write `raw_responses`); replace with a per-rollout wrapper there.
- **`info -> TaskInfo` via dacite** needs `config=dacite.Config(cast=[<the StrEnum types>])` to
  convert the serialized strings back into enums. `ComposedEnv` builds this from the `info_enums`
  ctor arg. dacite recurses into nested dataclasses, so the cast list only needs the leaf enum types
  regardless of nesting depth.

## Nesting `info` (the per-example spec)
The verifiers `info` column is framework-load-bearing — it is the ONLY per-example column that
`state_to_output`/`output_to_state` save and restore (so it survives `rescore`), and core reads
`info["tool_defs"]`. Do NOT rename it to `task_info` or anything else: a custom column forwards into
`state` but is dropped on save, breaking replay. Instead, keep the column named `info` and make its
*content* a single top-level `TaskInfo` dataclass that NESTS sub-dataclasses by concern rather than
flattening every field to one level. For `_m_instruction_following_text`:

```python
info = asdict(TaskInfo(
    alpaca=AlpacaProblem(orig_index=..., request_id=..., request=...),  # source-dataset provenance
    constraint=ConstraintSpec(name=..., difficulty=...),               # the imposed task spec
))
# -> {"alpaca": {"orig_index": ..., "request_id": ..., "request": ...},
#     "constraint": {"name": ..., "difficulty": ...}}
```

Why nest: `info` stays self-describing (`info["alpaca"]["orig_index"]` vs a flat soup), each concern
is its own typed dataclass, and adding source-dataset provenance (e.g. the ORIGINAL dataset index,
captured pre-filter/shuffle in the data builder) never collides with task fields. Downstream
consumers (flatten/analysis) read the nested paths.
- **`split_sentences`** (or any naive `[.!?]` splitter) over-splits abbreviations/decimals. Fine
  for pilot constraints; revisit if the model trains against the scorer and exploits it.

## Git / remote
- Our envs + verifiers tweaks live on our OWN fork of verifiers (`GnarlyMshtep/verifiers`), which is
  the submodule's sole remote, named `origin`. We no longer track PrimeIntellect upstream — `main` IS
  our line of development (the old PrimeIntellect remote was removed; their post-fork commits were
  discarded from our `main`). Develop on `main` by default (branch per issue `exp/MSH-###-...` only
  when needed). `git add` only specific files. Push to `origin` so commits link to Linear (webhook
  installed on the fork). Commit trailer: `refs MSH-###` + `Claude-Session:`.
- The prime-rl gitlink for `deps/verifiers` records the verifiers commit SHA; after committing in the
  submodule, bump it (`git -C <prime-rl> add deps/verifiers && commit`) so a fresh clone checks out
  our verifiers HEAD.
- The repo's `deps/research-environments` submodule is PrimeIntellect's dedicated custom-env home;
  we chose our verifiers fork instead because we also want to tweak verifiers itself.

## Sweeps (FYI, not built)
- vf-eval multi-model sweeps only via a TOML with multiple `[[eval]]` blocks; CLI is one
  model per call. We use individual launcher invocations.

## Registry (authored envs)
- `_m_instruction_following_text` (renamed from `instruction_following_text`) — Alpaca request +
  a **parameterized** textual formatting constraint + scoring judge. 5 constraints (controls
  `no_capitals`/`all_sentences_t` + `alternating_xy`/`letter_freq_diff`/`letter_set`), each a
  `(sample_params, render_instruction, verify)` triple with per-example params (sha1-seeded) stored
  in `info` and round-tripped via dacite for reproducible `rescore`. ConstraintScorer + shared
  JudgeScorer over output/cot/both views × immediate/regular monitor prompts; built on the
  `Scene`/`ComposedEnv` scaffold in `verifiers.envs._m_`. Difficulty knobs calibrated by a
  same-conditions smoke (`claude_state/ift-tweak-log.md`); monitor predictiveness analyzed
  logs-only (`claude_scripts/analysis_scripts/instruction_following_text/by_constraint.py`).
