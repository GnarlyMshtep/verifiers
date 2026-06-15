<!-- reportd: session=47a9f2d0-914e-4914-a11e-6a9862331018 name="_m_ env cleanup + Scene scaffold" created=2026-06-13 status=fulfilled -->
# Design: `_m_` env cleanup + shared Scene/Scorer scaffold

**Generated:** 2026-06-13T18:07:09Z | **Session:** 47a9f2d0-914e-4914-a11e-6a9862331018

> Cleanup pass before building more envs. Four threads: (1) `_m_` naming convention,
> (2) README example, (3) extract the reusable scaffold (Turn→Scene, ComposedEnv, Scorer/Judge)
> into a shared package + fix the `enter`/`is_complete` footguns, (4) rewrite the methodology doc
> into a procedural workflow.

## Locked decisions

- **`_m_` prefix**: directory + importable module + `env_id` all consistent — `_m_instruction_following_text`
  (env_id keeps the literal `_m_`, not a mangled hyphen form).
- **Rename `Turn` → `Scene`** (`TurnName` → `SceneName`). Rationale: a "Turn" misleads (it can span many
  messages); `enter()`/`is_complete()` are theatrical, so actors *entering* a `Scene` that *completes* reads
  naturally.
- **`Scene.enter` AND `Scene.is_complete` are both `@abstractmethod`** — no silent defaults.
- **Shared scope**: move Scene machinery + `Scorer`/`ScorerResult` ABCs + the full Judge layer
  (`JudgeScorer`, `JudgeView`, render/parse helpers) into the shared package. Only constraint rules and
  env-specific config stay local.
- **State-key rename**: full clean rename (`turn_idx`→`scene_idx`, `message_turns`→`message_scenes`,
  provenance `turn_name`/`turn_idx`→`scene_name`/`scene_idx`) everywhere, including the 4 prime-rl scripts.
  Accepted cost: existing sweep dirs (`outputs/.../06/12/...`) carry the OLD keys; re-running
  rescore/register/analysis on those *old* runs would need the old column name. New runs are clean.
- **README example**: sample task (alpaca request + constraint) + the launcher command. No rollout snippet.

## The `enter` footgun (why abstract)

Control flow in `ComposedEnv`: `setup_state` sets `turn_idx=0`; **scene 0's opening is the dataset prompt**
(verifiers feeds `prompt` to the actor directly). `env_response` runs only *after* an actor reply — it checks
the current scene's `is_complete`, increments the index, then calls `scenes[idx].enter()`. So `scenes[0].enter()`
is **never called**, and the old default `enter()->[]` only ever fires for scenes ≥1 — where `[]` is *wrong*
(a later scene that opens with no env message hands the floor back to the actor with nothing new). Making
`enter` abstract forces every scene to declare its opening: scene-0 classes write an explicit `return []`
("opened by the dataset prompt"), later scenes return real messages. `is_complete` is abstract too (user
preference — `True` is a sane default for single-message scenes but should be stated, not assumed).

**This must be documented in `building_vf_envs_matan_way.md`** (user request) — both the footgun and the
provenance model below.

## Message provenance model (design proposal — please review)

Provenance answers, for every message in the conversation: *which scene does it belong to, and who emitted
it?* Its consumer is `messages_for_scene()` (per-scene scoring).

**Today (asymmetric):** `message_turns` is built only from `completion` — actor replies (in
`add_model_response`) and scene-≥1 openings (in `env_response`). The scene-0 opening — the system prompt +
the initial user request, which live in `state["prompt"]`, not `completion` — carries **no provenance**.
So scene 0 is the odd one out: its opening is invisible, while every later scene's `enter()` output is tracked.

**Proposed (uniform over the full conversation):** record provenance for the `prompt` messages at
`setup_state`, so `message_scenes` covers `prompt + completion` and `messages_for_scene` slices the full
conversation. The "scene 0 is pre-entered by the dataset prompt" story becomes concrete: scene 0's opening
messages exist and are labeled — they just come from the framework-supplied prompt rather than `enter()`.

`MsgProvenance.source` widens to `Literal["system", "dataset", "env", "actor"]`:
- `system` — the global system prompt (`scene_idx=0`; persists across scenes).
- `dataset` — the per-example opening request supplied as the initial prompt = scene 0's opening (what
  `TaskScene.enter()` documents by returning `[]`).
- `env` — emitted by a scene's `enter()` (scenes ≥1).
- `actor` — emitted by the policy.

Recording sites: **`setup_state`** tags each `state["prompt"]` message (system→`system`, user→`dataset`,
both `scene_idx=0`); **`env_response`** tags `enter()` output (`env`); **`add_model_response`** tags the
actor reply (`actor`). `messages_for_scene(messages, state, scene_idx)` zips
`concat_messages([prompt, completion])` against `message_scenes`.

**Tradeoff:** slightly more code + a one-time prompt-walk at setup, vs. a coherent model where every message
is labeled and scene 0 stops being a special case. Recommended. (Alternative: keep completion-only and accept
scene 0's opening being implicit — simpler, but the asymmetry is the thing we set out to clean up.)

## 1. Shared package — `deps/verifiers/verifiers/envs/_m_/`

Importable as `verifiers.envs._m_`. Library code in the fork, reused by every `_m_` env.

| file | contents (moved out of `instruction_following_text`) |
|---|---|
| `scene.py` | `Scene` ABC (`name: str`; abstract `enter`, abstract `is_complete`); `MsgProvenance` (`scene_idx`, `scene_name`, `source`); `messages_for_scene(completion, state, scene_idx)` |
| `scoring.py` | `ScorerResult` base (`score: float`); `Scorer` ABC (`name: str`, `weight`, abstract `async score(*, prompt, completion, answer, task_info: object, state)`) |
| `judge.py` | `JudgeView` enum; `render_view`, `parse_score`, `parse_judge_response`, `_role_text`/`_msg_field`/`assistant_messages`/`last_assistant_text`; `JudgeUnparseableError`; `JudgeScore` (`name: str` field, no enum default); `JudgeScorer` |
| `composed_env.py` | `RawLoggingChatClient`; `ComposedEnv(MultiTurnEnv)` |
| `__init__.py` | re-export Scene, MsgProvenance, messages_for_scene, ScorerResult, Scorer, JudgeView, JudgeScore, JudgeScorer, render_view, parse_score, parse_judge_response, JudgeUnparseableError, ComposedEnv, RawLoggingChatClient |
| `claude_state/directory.md` | purpose = reusable Scene/ComposedEnv/Scorer/Judge scaffold; goal = extensible envs + separating scene-sequence logic from env definition |
| `tests/` | relocated scaffold tests (the parts of `test_turn_logic.py` that test Scene/ComposedEnv mechanics) |

### Two generalization seams (so shared code carries no env-specifics)

1. **`ComposedEnv` info parsing.** Today it hardcodes
   `dacite.from_dict(TaskInfo, state["info"], cast=[ConstraintName, Difficulty])`. Parameterize:
   ```python
   ComposedEnv(scenes, scorers, *, info_type=TaskInfo, info_enums=(ConstraintName, Difficulty), **kwargs)
   ```
   `setup_state` builds `dacite.Config(cast=list(info_enums))` and parses into `state["_task_info"]`.
   Declarative, no callback. (If a future env needs richer dacite config, extend then.)
2. **`JudgeScorer.name` / `JudgeScore.name`.** `name` becomes a `JudgeScorer.__init__` arg; the env passes
   `ScorerName.JUDGE_REQUEST_FOLLOWED`. `JudgeScore.name` becomes a plain field set from `self.name`.

The shared `Scorer` ABC types `task_info` as `object` (generic); env subclasses narrow it to `TaskInfo` in
their override signature (runtime-compatible).

## 2. Env after the move — `_m_instruction_following_text`

- `turn_logic.py` **deleted** (all contents moved up).
- `env.py`: imports `ComposedEnv`, `Scene`, `JudgeScorer`, `JudgeView` from `verifiers.envs._m_`;
  `ConstraintScorer` from local. Defines `TaskScene(Scene)` spelling out `enter→[]` + `is_complete→True`.
  Constructs `ComposedEnv(scenes=[TaskScene()], scorers=..., info_type=TaskInfo,
  info_enums=(ConstraintName, Difficulty), ...)`; passes `name=ScorerName.JUDGE_REQUEST_FOLLOWED` to `JudgeScorer`.
- `types.py`: `TurnName`→`SceneName`. `JudgeView` removed (now imported from shared). Keeps
  `Difficulty`, `MonitorPrompt`, `ConstraintName`, `AlpacaProblem`, `ConstraintSpec`, `TaskInfo`,
  `Constraint`, `ConstraintResult`. `MsgProvenance` removed (moved to shared).
- `scorer_types.py`: keeps `ScorerName`, `JudgeConfig`, `ConstraintScore`. `ScorerResult`, `Scorer`,
  `JudgeScore` removed (imported from shared).
- `scorers.py`: keeps `ConstraintScorer` + `CONSTRAINTS` import. `JudgeScorer` + render/parse helpers +
  `JudgeUnparseableError` removed (moved to shared; re-imported where the env still references them, e.g.
  `turn_logic`'s old `JudgeUnparseableError` catch now lives in shared `ComposedEnv`).
- `constraints.py`, `prompts.py`, `dataset.py`: unchanged.
- `tests/`: `test_constraints.py`, `test_dataset.py` stay. `test_scorers.py` splits — generic
  render_view/parse tests move to the shared `tests/`; constraint-scorer tests stay. `test_turn_logic.py`
  mechanics move to shared `tests/`.

## 3. The `_m_` rename cascade

- Directory `environments/instruction_following_text/` → `environments/_m_instruction_following_text/`,
  module `instruction_following_text/` → `_m_instruction_following_text/`, `env_id` →
  `_m_instruction_following_text`.
- `pyproject.toml`: package name + `[tool.hatch.build] include` + `[tool.verifiers.eval]`.
- Re-install editable: `uv pip install --python .venv/bin/python -e environments/_m_instruction_following_text`.
- prime-rl scripts referencing the env id / module / `message_turns`: `run_instruction_eval.py`,
  `rescore_views_pearson.py`, `pooled_monitor_pearson.py`, `analysis_scripts/instruction_following_text/register.py`
  (and the analysis package import path, if it names the env module).

## 4. README example

**README** (`environments/_m_instruction_following_text/README.md`): add a worked example — one sample
task (a concrete alpaca request + e.g. `no_capitals` constraint) and the launcher command to run it.

## 5. Clean up & restructure `building_vf_envs_matan_way.md`

The doc is currently organized as topic-by-topic **reference** (Where & how / Turns vs Scorers / ScorerResult
/ Judge / Logging / Rescoring / Gotchas / Nesting info / Git / Sweeps / Registry). The user wants it
**procedural** — a workflow spine you follow to build a new env, with the reference material folded
underneath. This is a rewrite, not an append.

### 5a. The procedural spine (new top of the doc)

A numbered "How to build an env" workflow, faithful to the user's instructions:

1. **Flush out the idea with the user** — ask clarifying questions about purpose, the task, the reward, and
   success criteria. Don't start building before the idea is fleshed out.
2. **Confirm the architecture: Scene + ComposedEnv.** State the purpose explicitly — *extensible envs* +
   *separating the logic of a scene sequence from the env definition itself*.
3. **`/writing-plans` for the env** — write the plan before touching code.
4. **Build tests for edge-case rollouts** and verify they're handled correctly. **Think hard about edge-case
   reward-hacking rollouts** specifically (what would a model do to game the reward?).
5. **Flag any non-obvious deviation** that might surprise the user.
6. **Smoke-test via a subagent.** The building actor launches a subagent to run a smoke test on a few cheap
   models — **deepseek-v4-flash**, **gpt-oss-20b**, **qwen/qwen3-next-80b-a3b-thinking** (assume ~$5
   OpenRouter spend is OK). The subagent reads a few rollouts, paying special attention to:
   - (a) **all relevant parts of the trajectory are logged** — err on the side of "is relevant";
   - (b) **prompts contain the information they should**;
   - (c) **actors are interpreting the trajectory correctly**.
7. **Iterate**: up to 3 subagent iterations, or stop once the subagent is satisfied.
8. **Hand the human the logged dirs** for a full pass.

### 5b. Refold the existing reference material under the spine

Keep the still-accurate reference content but move it below the spine as a "Reference" half, mapped to where
it's relevant:

- **Where & how** (layout, install, venv, run-from-`deps/verifiers`) → keep, near the top of Reference.
- **Scenes & Scorers (the split)** → keep, but rewrite Turn→Scene; add the **Scenes & provenance** subsection
  (below).
- **ScorerResult dataclasses**, **Judge / CoT visibility**, **Logging**, **Rescoring**, **Gotchas**,
  **Nesting `info`**, **Git / remote**, **Registry** → keep; update terminology + paths.
- **Sweeps (FYI, not built)** → keep as a short FYI or drop if stale.

### 5c. New "Scenes & provenance" subsection

Captures (user request): the **`enter` footgun** (scene 0 is pre-entered by the dataset prompt, so
`scenes[0].enter()` is never called and a `[]` default is wrong for scenes ≥1 → both `enter`/`is_complete`
abstract); and the **message-provenance model** (uniform tagging over `prompt + completion`;
`source ∈ {system, dataset, env, actor}`; recorded at setup / env_response / add_model_response; consumed
by `messages_for_scene`).

### 5d. Terminology & path sweep (whole doc)

`Turn`→`Scene`, `TurnName`→`SceneName`, `turn_logic.py`→(deleted; shared `verifiers.envs._m_`),
`message_turns`/`turn_idx`→`message_scenes`/`scene_idx`, `instruction_following_text`→
`_m_instruction_following_text`, and the new shared-package import paths. Update the Registry entry to the
renamed env. Prune anything contradicted by the move (e.g. "Turn ABC + ComposedEnv live in `turn_logic.py`").

## Deviations flagged (per the methodology rule)

1. **Old sweep dirs use legacy state keys.** Full rename means `06/12` sweep results carry
   `turn_idx`/`message_turns`; analysis on those old runs needs the old names. (User accepted.)
2. **`JudgeScorer` assumes a single-user-request trajectory** and the `judge_prompt_fn(system, question,
   assistant_turns)` contract. Fine now; generalize when an env has a different conversation shape.
3. **`ComposedEnv` info parsing** gains `info_type`/`info_enums` args — a small API surface addition.

## Out of scope

- No change to `optimization_dtype`/`reduce_dtype` or any training numerics.
- No new envs (the methodology doc describes the future flow; we don't build env #2 here).
- `JudgeScorer` multi-message-trajectory generalization (deferred).

## Verification

- `uv run pytest` in the env + shared `tests/` (python 3.13 `.venv`).
- `uv pip install -e` the renamed env succeeds; `python -c "import verifiers.envs._m_"` imports.
- One tiny smoke eval through the launcher on the renamed env (few samples, cheap model) to confirm
  the rename + scaffold move produce a clean rollout with `scorers`/`message_scenes` logged.
