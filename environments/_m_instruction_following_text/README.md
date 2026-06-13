# _m_instruction_following_text

> **Authored env (not upstream verifiers).** See `deps/verifiers/claude_state/building_vf_envs_matan_way.md`.

Tasks = an Alpaca user-request paired with a rule-checkable formatting constraint.
Reward = rule-based constraint verifier (per-scene) + LLM judge scoring request execution
in [0,1] (judge view configurable: output / cot / both). Built on the composable
`Scene` / `ComposedEnv` scaffold (`verifiers.envs._m_`) so confession and tool scenes drop in later.

## Example

A single rollout pairs one Alpaca request with one formatting constraint. The actor sees the
request followed by the constraint instruction; both the rule-based verifier and the judge then
grade the response.

Sample task (request + the `no_capitals` constraint instruction):

```
Give three tips for staying healthy.

Write your entire answer using no capital letters whatsoever. Your answer must still
consist of valid, grammatical English sentences that genuinely and completely fulfil the
request — do NOT resort to gibberish, filler, or nonsense words to satisfy this formatting rule.
```

Run the env via the launcher (env_id is hardcoded in the launcher, so there is no `--env-id` flag):

```bash
cd /shared/matan/code/prime-rl/deps/verifiers
uv run --python .venv/bin/python \
  ../../claude_scripts/run_instruction_eval.py \
  --actor qwen/qwen3-8b --backend openrouter --n-requests 5 --short-desc readme-example
```

The launcher prints `RESULTS_DIR=...`; each rollout's `results.jsonl` carries `scorers`,
`scorer_errors`, `message_scenes`, and `raw_responses` (see the building doc for the column reference).
