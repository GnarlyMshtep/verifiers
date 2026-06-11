# Building verifiers envs — Matan's way (tentative, v0, 2026-06-11)

> Early and provisional. Written while building the first env (`instruction_following_text`).
> Point future env-building agents here. Expect this to change.

## Where & how
- Author envs flat in `deps/verifiers/environments/<name>/`. Mark authored in README +
  register in this file. env_id = hyphenated name; importable module = name w/ underscores.
- Multi-file package layout (NOT single .py):
  pyproject.toml (`[tool.hatch.build] include`, `[tool.verifiers.eval]` defaults);
  `<pkg>/types.py` (StrEnums + frozen dataclasses for boundary values);
  `<pkg>/prompts.py` (`PromptKey(StrEnum)` + `PROMPTS` registry + python template fns — prefer
  python over prompt.json: keeps templates + types + discoverability);
  `<pkg>/constraints.py` (pure rule-based verifiers); `<pkg>/dataset.py` (HF Dataset:
  question/answer/info); `<pkg>/turn_logic.py` (Turn ABC + ComposedEnv); `<pkg>/rubrics.py`
  (reward fns + judge-view rendering + score parsing); `<pkg>/env.py` (`load_environment`,
  ONE config-driven env — judge view, difficulty filter, weights — no near-duplicate classes);
  `<pkg>/data/` (small reviewable json); `tests/` (pytest plain fns; ONLY pure logic).
- Install into the verifiers venv: `uv pip install --python .venv/bin/python -e environments/<name>`
  (python 3.13 at `deps/verifiers/.venv`; repo conda env is 3.10 and won't work).
- The launcher/scripts need `tyro` + `python-dotenv` in the verifiers venv:
  `uv pip install --python .venv/bin/python tyro python-dotenv`.

## Composable turns
- verifiers gives ONE env→actor hook (`env_response`) + stop conditions (`@vf.stop`);
  `rollout` is `@final`. Don't fight it.
- `Turn` ABC = `enter(state)->Messages` (turn 0 pre-entered by dataset prompt -> enter()==[]),
  `is_complete(state)->bool` (default True; seam for multi-message/tool turns), `reward_fns`.
- `ComposedEnv(MultiTurnEnv)` walks `turns`, records per-message provenance in
  `state["message_turns"]` (dataclass asdict, index-aligned with `state["completion"]`),
  scopes per-turn reward fns via `make_scoped` (filter by provenance), adds env-level global
  reward fns, assembles one `vf.Rubric`. Reward fns are the SAME type; scope = which messages.
- Multi-message/tool turns = implement the `is_complete=False` branch of `env_response`
  (currently raises). Per-turn TRAINING credit needs a separate prime-rl `train_sink.py`
  change (eval unaffected — rubric scalars log fine).

## Gotchas learned (v0)
- `MultiTurnEnv.__init__` wraps your `vf.Rubric` in a `RubricGroup` (adds a monitor rubric).
  So `env.rubric` is a RubricGroup and your funcs live in `env.rubric.rubrics[0]`. Set rubric
  weights at `vf.Rubric(funcs=..., weights=...)` CONSTRUCTION time — setting `env.rubric.weights`
  afterward targets the wrapper and is silently ignored.
- Raw actor-response logging via re-classing `state["client"].__class__ = RawLoggingChatClient`
  mutates the client object in place. Safe when the env owns a dedicated client (our eval
  launcher passes a ClientConfig -> a fresh client per process). For a SHARED training client
  this leaks (other rollouts also write `raw_responses`); replace with a per-rollout wrapper there.
- `split_sentences` is a naive `[.!?]` splitter — abbreviations/decimals over-split. Fine for the
  pilot constraints; revisit if the model trains against the scorer and exploits it.

## Judge / CoT visibility
- Judge = a reward fn calling `AsyncOpenAI` (OpenRouter base_url). It sees what
  `render_view(completion, JudgeView)` builds from each assistant msg's `content` and/or
  `reasoning_content`. Score 0-10 -> [0,1] via `parse_score` (rejects decimals/out-of-range).
  Launch reasoning judges with `extra_body={"reasoning":{"effort":"medium"}}`. vLLM actors need
  `--reasoning-parser` or `reasoning_content` is absent and the cot/both view raises (by design).

## Logging (entire computation trace)
- Write JSON-serializable data to `state[key]` during the rollout, pass `state_columns=[...]`
  to `evaluate()` -> lands in results.jsonl (raises if not serializable; no silent drop).
- Standard columns: `raw_responses` (full actor API JSON), `judge_raw` (full judge prompt+
  response+full_response), `constraint_results`, `message_turns`.
- Custom dated dir + sidecar config: don't fight `path_utils` — a launcher calls
  `env.evaluate(results_path=...)` and writes only the deltas vf's metadata.json lacks
  (datetime, actor backend, short_desc). See `claude_scripts/run_instruction_eval.py`.

## Sweeps (FYI, not built)
- vf-eval multi-model sweeps only via a TOML with multiple `[[eval]]` blocks; CLI is one
  model per call. We use individual launcher invocations.

## Registry (authored envs)
- `instruction_following_text` — instruction following w/ word constraints + scoring judge.
