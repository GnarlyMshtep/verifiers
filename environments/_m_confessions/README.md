# `_m_confessions`

## What / why

`_m_confessions` is a prime-rl-trainable verifiers environment that trains an LLM to honestly
self-report (`confess`) how well it complied with a task, following *Training LLMs for Honesty via
Confessions* (arXiv 2512.08093). It composes the existing instruction-following task with a second
`confession` turn: the actor first answers a task request, then is asked to produce a structured
`ConfessionReport` analyzing its own compliance. Only the confession tokens are loss-bearing — the
task answer is context, not a training target — so the reward (a confession-honesty judge) shapes
*honesty of self-report* without re-training the task policy. The environment is a `vf.EnvGroup`
over two `ComposedEnv`s: a `plain` sub-env (task only, monitor reward) and a `confession` sub-env
(task masked + confession trained), with the instruction-following dataset deterministically
partitioned by `confession_probability`.

## How masking works

The on-policy inference client sets every generated step's `tokens["completion_mask"]` to all-1s by
default. How the confessions env overrides it: `ComposedEnv.add_model_response` runs *after*
`super().add_model_response(...)`, looks at the current `Scene.trainable`, and if it's `False`,
overwrites that step's `tokens["completion_mask"] = [0] * len(...)`. That is the only place the value
is changed from the default. prime-rl then carries those zeros straight into the loss mask, so the
task-answer scene contributes zero gradient while the confession scene is trained.

In the `confession` sub-env the scenes are `[TaskScene(trainable=False),
ConfessionScene(trainable=True)]` — the task answer is masked out, the confession is the only span
that contributes gradient.

## `load_environment` knobs

`load_environment` (`_m_confessions/env.py`) accepts:

| arg | default | meaning |
| --- | --- | --- |
| `confession_probability` | `1.0` | fraction of the dataset routed to the `confession` sub-env (the first `⌊p·N⌋` examples by stable index); the rest go to `plain`. `1.0` = all confession. |
| `n_requests` | `17` | number of instruction-following requests sampled into the dataset. |
| `difficulties` | `("easy","medium","hard")` | difficulty tiers to include. |
| `judge_model` | `"openai/gpt-oss-20b"` | model id for both the confession judge and the regular monitor. |
| `judge_base_url` | `"https://openrouter.ai/api/v1"` | OpenAI-compatible base URL for the judges. |
| `judge_api_key_var` | `"OPENROUTER_API_KEY"` | env var holding the judge API key. |
| `judge_reasoning_effort` | `"medium"` | reasoning effort passed to the judges. |
| `judge_max_tokens` | `2048` | judge completion budget. |
| `judge_timeout_s` | `60.0` | hard per-call timeout (no retry layer — a hung judge fails fast). |
| `monitor_view` | `"output"` | `JudgeView` for the regular monitor (what part of the task answer it sees). |
| `log_token_masks` | `False` | when `True`, `ComposedEnv` records a per-step `token_mask_log` row with the trainable attribution. |
| `requests_path` | `None` | override path to the instruction-following requests dataset. |

## Running an inference eval

A copy-pasteable smoke runner lives at `claude_scripts/run_confession_eval.py` (in the prime-rl repo,
not this submodule). Assumes `OPENROUTER_API_KEY` is set in `deps/verifiers/.env`:

```bash
cd deps/verifiers && uv run --python .venv/bin/python ../../claude_scripts/run_confession_eval.py \
  --actor qwen/qwen3-4b --backend openrouter --short-desc confsmoke --n-requests 4 \
  --judge-model openai/gpt-oss-20b
```

It prints `RESULTS_DIR=...` and `avg_reward=...`, and writes outputs under
`outputs/_m_confessions/<MM>/<DD>/<short_desc>-<hash>-<HH_MM>/`.

## What gets logged

With `state_columns=["scorers", "scorer_errors", "message_scenes", "token_mask_log", "raw_responses"]`
each rollout's `state` carries:

- **`state["scorers"]`** — the scorer results. In the `confession` sub-env: `confession` (the honesty
  reward, `[0,1]` + reasoning quoting the transcript), `constraint_satisfied` (ground-truth task
  compliance graded against the *scene-0 task answer*, log-only weight 0), and `regular_monitor`
  (the instruction-following monitor shown the confession, `saw_confession=true`, log-only weight 0).
- **`state["scorer_errors"]`** — populated only if a judge produced no parseable score (reward then
  coerced to `0.0`). The runner prints a loud `WARNING` if any rollout hit this. Should be empty.
- **`state["message_scenes"]`** — per-message provenance: which scene index/name and source (`dataset`
  / `actor` / `env`) produced each message.
- **`state["token_mask_log"]`** — per-step trainable attribution (only when `log_token_masks=True`):
  `scene_idx`, `scene_name`, `trainable`, `n_tokens`, `n_trainable`, `text_preview`.
- **`state["raw_responses"]`** — the raw model responses.

## Example trajectory (abbreviated)

```
[system]   You are a helpful assistant. ...                         (scene 0, source=dataset)
[user]     Write a haiku about the ocean. It must be exactly ...    (scene 0, source=dataset)
[assistant] Endless rolling waves / ...                             (scene 0, source=actor)  <- TASK ANSWER (masked)
[user]     Now, analyze the level of compliance ... Respond with    (scene 1, source=env)    <- confession request xc
           ONLY a valid JSON object matching the ConfessionReport schema ...
[assistant] {"compliance_analysis": [{"objective": "exactly 3 ...   (scene 1, source=actor)  <- CONFESSION (trained)
```

The confession judge scores honesty of the scene-1 confession against the scene-0 answer; the
regular monitor (shown the confession) and the task constraint scorer are logged but contribute no
gradient (weight 0) in the confession sub-env.

## Example `token_mask_log` rows

```jsonc
// scene-0 actor step: the task answer is context only -> masked, contributes 0 gradient
{"scene_idx": 0, "scene_name": "task", "trainable": false, "n_tokens": 42, "n_trainable": 0,
 "text_preview": "Endless rolling waves / ..."}

// scene-1 actor step: the confession is the loss-bearing span -> all tokens trainable
{"scene_idx": 1, "scene_name": "confession", "trainable": true, "n_tokens": 188, "n_trainable": 188,
 "text_preview": "{\"compliance_analysis\": [{\"objective\": \"exactly 3 ..."}
```

(Over an OpenAI-compatible chat backend that doesn't return token ids, `n_tokens`/`n_trainable` may
be `null`; the definitive mask check happens against the trained `TrainingSample`s during RL.)
